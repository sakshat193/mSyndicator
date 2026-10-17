import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import httpx
from psycopg import sql
import pytest

from msynd.collect.dispatcher import Dispatcher, HostLocked
from msynd.collect.mmt import HOST, MmtClient
from msynd.db import connect, migrate
from tests.collect.test_dispatcher import FakeClock
from tests.helpers import insert_event, insert_fetch, make_run

NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parents[1] / 'fixtures' / 'mmt'


def runs():
    from msynd.collect.admission import PollRun

    return tuple(PollRun(i, str(12000 + i), date(2026, 10, 7), NOW,
                         1000000 if i == 0 else 0) for i in range(6))


def test_no_cohort_run_starves_beyond_interval_and_cycle():
    from msynd.collect.admission import AdmissionState, next_job

    state = AdmissionState(runs())
    now = NOW
    seen = set()
    for _ in range(4320):
        job = next_job(state, now)
        if job is None:
            now += timedelta(seconds=1)
            continue
        seen.add(job.run.id)
        state = replace(state, runs=tuple(
            replace(run, last_fetch=now) if run.id == job.run.id else run
            for run in state.runs))
        now += timedelta(seconds=5)
        assert max((now - run.last_fetch).total_seconds() for run in state.runs) <= 150
    assert len(seen) == 6


def test_job_order():
    from msynd.collect.admission import AdmissionState, PollRun, next_job

    old = PollRun(1, '12951', date(2026, 10, 7), NOW - timedelta(seconds=600), 0)
    target = replace(old, id=2, last_fetch=NOW - timedelta(seconds=120), priority=1000)
    age = replace(old, id=3, last_fetch=NOW - timedelta(seconds=121), priority=None)
    state = AdmissionState((old, target, age), T_max=600)
    assert next_job(state, NOW).run.id == 1
    state = replace(state, runs=(replace(old, last_fetch=NOW), target, age))
    job = next_job(state, NOW)
    assert job.run.id == 2 and job.reason == 'targeted'
    state = replace(state, runs=(replace(old, last_fetch=NOW), replace(target, last_fetch=NOW), age))
    assert next_job(state, NOW).run.id == 3
    assert next_job(replace(state, runs=()), NOW) is None


def test_stopped_host_issues_nothing():
    from msynd.collect.admission import AdmissionState, next_job

    assert next_job(AdmissionState(runs(), host_state='stopped'), NOW) is None


def test_breaker_waits_then_recovers_with_oldest_status():
    from msynd.collect.admission import AdmissionState, next_job

    items = runs()
    items = (items[0], replace(items[1], last_fetch=NOW - timedelta(seconds=10)))
    state = AdmissionState(items, host_state='breaker', recovery_at=NOW + timedelta(minutes=10))
    assert next_job(state, NOW) is None
    job = next_job(state, NOW + timedelta(minutes=10))
    assert job.reason == 'recovery' and job.run.id == 1


def test_deadline_and_rate_are_parameters():
    from msynd.collect.admission import AdmissionState, next_job

    state = AdmissionState(runs(), rate=1, T_max=600)
    job = next_job(state, NOW + timedelta(seconds=595))
    assert job.reason == 'overdue'
    state = replace(state, rate=.2, T_max=10)
    with pytest.raises(ValueError, match='capacity'):
        next_job(state, NOW + timedelta(seconds=120))


@pytest.mark.parametrize('parameter,value', [('rate', 0), ('rate', float('nan')),
                                            ('h', 0), ('T_max', -1),
                                            ('t_max_factor', float('inf'))])
def test_invalid_budget_rejected(parameter, value):
    from msynd.collect.admission import AdmissionState

    with pytest.raises(ValueError):
        AdmissionState(runs(), **{parameter: value})


def test_main_refuses_without_live_before_db_or_transport(monkeypatch):
    from msynd.collect.__main__ import main

    from msynd.collect import __main__ as collector

    def touched(*a, **k):
        raise AssertionError('touched db or transport')
    monkeypatch.setattr(collector, 'connect', touched)
    monkeypatch.setattr(collector.MmtClient, 'from_env', touched)
    monkeypatch.delenv('MSYND_LIVE', raising=False)
    with pytest.raises(RuntimeError, match='MSYND_LIVE=1'):
        main([])


@pytest.mark.db
def test_state_reads_poll_queue_and_only_due_active_runs(db):
    from msynd.collect.__main__ import load_state

    active = make_run(db)
    scheduled = make_run(db, train='12952')
    future = make_run(db, train='12953', start='2026-10-08')
    done = make_run(db, train='12954')
    db.execute("UPDATE obs.run SET lifecycle='running' WHERE id=%s", (active,))
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (done,))
    for host, purpose, seconds in ((HOST, 'status', -20), ('other', 'status', -10),
                                   (HOST, 'token', -5), (HOST, 'status', 60)):
        insert_fetch(db, run_id=active, host=host, purpose=purpose,
                     started_at=NOW + timedelta(seconds=seconds), outcome='invalid')
    db.execute("INSERT INTO obs.poll_queue VALUES (%s,100,'test',%s)", (active, NOW))
    state = load_state(db, NOW)
    by_id = {run.id: run for run in state.runs}
    assert set(by_id) == {active, scheduled}
    assert future not in by_id and done not in by_id
    assert by_id[active].last_fetch == NOW - timedelta(seconds=20)
    assert by_id[active].priority == 100
    assert by_id[scheduled].priority is None
    assert by_id[scheduled].last_fetch == NOW - timedelta(seconds=120)


@pytest.fixture
def collector_url(db_url):
    # Committed evidence is visible to the real dispatcher; isolate it from other tests.
    name = db_url.rsplit('/', 1)[1] + '_collector'
    admin_url = db_url.rsplit('/', 1)[0] + '/postgres'
    url = db_url.rsplit('/', 1)[0] + '/' + name
    with connect(admin_url) as admin:
        admin.autocommit = True
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
        try:
            migrate.apply(url)
            yield url
        finally:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


@pytest.mark.db
@pytest.mark.parametrize('response', ['ok', 'wrong_date', 'challenge'])
def test_loop_stores_valid_evidence_and_rejects_bad_pages(collector_url, tmp_path, response):
    from msynd.collect.__main__ import run_loop

    body = json.loads((FIXTURES / 'completed_12951.json').read_text())
    halts = []
    def offset(row, prefix, key):
        hours, minutes = map(int, row[key][f'sch{prefix}Time'].split(':'))
        return (row['dayDetails']['dayCnt'] - 1) * 1440 + hours * 60 + minutes
    for i, row in enumerate(body['response']['stations']):
        halts.append((row['station']['code'], None if i == 0 else offset(row, 'Arr', 'arrivalDetails'),
                      None if i == len(body['response']['stations']) - 1 else offset(row, 'Dep', 'departureDetails'),
                      float(row['distance'])))
    with connect(collector_url) as conn:
        run = make_run(conn, halts=halts)
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
    clock = FakeClock()
    clock.base = NOW
    starts = []
    if response == 'challenge':
        async def sleep(seconds):
            clock.t += max(seconds, 1800)
        clock.sleep = sleep
    if response == 'wrong_date':
        body['response']['trainDetails']['startDate'] = '06-10-2026 00:00:00'

    def handle(request):
        starts.append(clock.now())
        assert json.loads(request.content)['findNextRunningDate'] is False
        assert request.url.host == HOST
        return httpx.Response(403, content=b'captcha') if response == 'challenge' else httpx.Response(200, json=body)

    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle), clock=clock)
    client = MmtClient(d)
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(client, conn, disk_path=tmp_path,
                           stop=lambda: clock.t > 3600 if response == 'challenge'
                           else bool(starts) or clock.t >= 60)
    asyncio.run(execute())
    assert d.conn is None
    with connect(collector_url) as conn:
        assert conn.execute('SELECT outcome FROM obs.fetch WHERE run_id=%s', (run,)).fetchall() == [
            ('ok' if response == 'ok' else 'challenge' if response == 'challenge' else 'invalid',)]
        events = conn.execute('SELECT count(*) FROM obs.event_rev WHERE run_id=%s', (run,)).fetchone()[0]
        assert events > 0 if response == 'ok' else events == 0
        assert conn.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == (
            'completed' if response == 'ok' else 'scheduled',)
        assert conn.execute("SELECT count(*) FROM sim.metric WHERE name='disk_pct'").fetchone() == (
            2 if response == 'challenge' else 1,)
        assert conn.execute('SELECT count(*) FROM obs.run').fetchone()[0] >= 2  # Hourly create_runs also ran.


@pytest.mark.db
@pytest.mark.parametrize('error_code,lifecycle', [(1001, 'cancelled'), (9999, 'scheduled')])
def test_loop_not_running_answer_cancels_run(collector_url, tmp_path, error_code, lifecycle):
    from msynd.collect.__main__ import load_state, run_loop

    with connect(collector_url) as conn:
        run = make_run(conn)
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
    clock = FakeClock()
    clock.base = NOW
    starts = []
    def handle(request):
        starts.append(clock.now())
        return httpx.Response(400, json={
            'response': None, 'success': False,
            'error': {'errorCode': error_code, 'errorDescription': 'Train is not running on given date'},
        })
    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle), clock=clock)
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, stop=lambda: bool(starts) or clock.t >= 60)
    asyncio.run(execute())
    with connect(collector_url) as conn:
        assert conn.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == (lifecycle,)
        assert conn.execute('SELECT http_status,outcome FROM obs.fetch WHERE run_id=%s', (run,)).fetchall() == [
            (400, 'invalid')]
        poll = load_state(conn, clock.now())
        harvest = load_state(conn, NOW + timedelta(days=1))
        assert (run in [r.id for r in poll.runs]) is (lifecycle == 'scheduled')
        assert (run in [r.id for r in harvest.harvest]) is (lifecycle == 'scheduled')


@pytest.mark.db
def test_idle_loop_runs_hourly_expiry_and_body_retention(collector_url, tmp_path):
    from msynd.collect.__main__ import run_loop

    clock = FakeClock()
    clock.base = NOW + timedelta(days=2)
    async def sleep(seconds):
        clock.t += max(seconds, 1800)
    clock.sleep = sleep
    with connect(collector_url) as conn:
        run = make_run(conn)
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-11-01'")
        fetch = insert_fetch(conn, host=HOST, started_at=NOW - timedelta(days=2),
                             body_zstd=b'old body')
    requests = []
    d = Dispatcher(HOST, .2, lambda: connect(collector_url),
                   httpx.MockTransport(lambda r: requests.append(r) or httpx.Response(200)), clock=clock)
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, stop=lambda: clock.t > 3600)
    asyncio.run(execute())
    assert requests == []
    with connect(collector_url) as conn:
        assert conn.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('expired_unknown',)
        assert conn.execute('SELECT body_zstd FROM obs.fetch WHERE id=%s', (fetch,)).fetchone() == (None,)
        assert conn.execute("SELECT count(*) FROM sim.metric WHERE name='disk_pct'").fetchone() == (2,)


@pytest.mark.db
def test_loop_cannot_run_alongside_host_owner(collector_url, tmp_path):
    from msynd.collect.__main__ import run_loop

    requests = []
    def transport():
        return httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200))
    async def execute():
        owner = Dispatcher(HOST, .2, lambda: connect(collector_url), transport())
        other = Dispatcher(HOST, .2, lambda: connect(collector_url), transport())
        await owner.start()
        try:
            with connect(collector_url) as conn, pytest.raises(HostLocked):
                await run_loop(MmtClient(other), conn, disk_path=tmp_path)
            assert other.conn is None
        finally:
            await owner.close()
    asyncio.run(execute())
    assert requests == []
    with connect(collector_url) as conn:
        assert conn.execute('SELECT count(*) FROM obs.run').fetchone() == (0,)
        assert conn.execute('SELECT count(*) FROM sim.metric').fetchone() == (0,)


@pytest.mark.db
@pytest.mark.parametrize('initial,response,expected', [
    ('ok', 'invalid', [0, 1]),
    ('ok', 'challenge', [0, 2]),
    ('breaker', 'invalid', [1, 0]),
])
def test_loop_records_dispatcher_start_and_final_request_state(collector_url, tmp_path,
                                                             initial, response, expected):
    from msynd.collect.__main__ import run_loop

    with connect(collector_url) as conn:
        make_run(conn)
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
    clock = FakeClock()
    clock.base = NOW
    starts = []
    def handle(request):
        with connect(collector_url) as conn:
            assert conn.execute('SELECT value FROM sim.metric WHERE name=%s ORDER BY ts',
                                ('dispatcher_state:' + HOST,)).fetchall() == [(expected[0],)]
        starts.append(clock.t)
        return httpx.Response(403, content=b'captcha') if response == 'challenge' else httpx.Response(200, json={})
    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle), clock=clock)
    d.state = initial
    if initial == 'ok' and response == 'invalid':
        d.outcomes.extend([(0, 'invalid')] * 19)
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, stop=lambda: bool(starts) or clock.t >= 60)
    asyncio.run(execute())
    assert starts == [0]
    with connect(collector_url) as conn:
        assert [row[0] for row in conn.execute('SELECT value FROM sim.metric WHERE name=%s ORDER BY ts',
                                              ('dispatcher_state:' + HOST,))] == expected


@pytest.mark.db
@pytest.mark.parametrize('duration,count', [(0, 1), (601, 3)])
def test_loop_records_dispatcher_start_and_five_minute_heartbeat(collector_url, tmp_path, duration, count):
    from msynd.collect.__main__ import run_loop

    clock = FakeClock()
    d = Dispatcher(HOST, .2, lambda: connect(collector_url),
                   httpx.MockTransport(lambda r: pytest.fail('idle collector must not send')), clock=clock)
    with connect(collector_url) as conn:
        before = conn.execute('SELECT clock_timestamp()').fetchone()[0]
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, stop=lambda: clock.t >= duration)
    asyncio.run(execute())
    with connect(collector_url) as conn:
        after = conn.execute('SELECT clock_timestamp()').fetchone()[0]
        rows = conn.execute('SELECT ts,value FROM sim.metric WHERE name=%s ORDER BY ts',
                            ('dispatcher_state:' + HOST,)).fetchall()
        assert len(rows) == count
        assert all(before <= ts <= after and value == 0 for ts, value in rows)


@pytest.mark.db
def test_loop_records_breaker_and_heartbeat_during_retry_wait(collector_url, tmp_path):
    from msynd.collect.__main__ import run_loop

    with connect(collector_url) as conn:
        make_run(conn)
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
    clock = FakeClock()
    clock.base = NOW
    starts = []
    def handle(request):
        starts.append(clock.t)
        return httpx.Response(503) if len(starts) == 1 else httpx.Response(200, json={})
    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle), clock=clock)
    d.outcomes.extend([(0, 'invalid')] * 19)

    async def execute():
        paused, resume = asyncio.Event(), asyncio.Event()
        async def sleep(seconds):
            clock.t += seconds
            if d.state == 'breaker' and not paused.is_set():
                paused.set()
                await resume.wait()
        clock.sleep = sleep
        async def collect():
            with connect(collector_url) as conn:
                await run_loop(MmtClient(d), conn, disk_path=tmp_path, stop=lambda: len(starts) >= 2)
        async def wait_for_metrics(expected):
            deadline = asyncio.get_running_loop().time() + 3
            while True:
                with connect(collector_url) as conn:
                    values = [row[0] for row in conn.execute(
                        'SELECT value FROM sim.metric WHERE name=%s ORDER BY ts', ('dispatcher_state:' + HOST,))]
                if values == expected:
                    return
                if asyncio.get_running_loop().time() >= deadline:
                    assert values == expected
                await asyncio.sleep(.02)
        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(paused.wait(), timeout=3)
            await wait_for_metrics([0, 1])
            clock.t += 300
            await wait_for_metrics([0, 1, 1])
        finally:
            resume.set()
            await task
        await wait_for_metrics([0, 1, 1, 0])
    asyncio.run(execute())
    assert len(starts) == 2


@pytest.mark.db
def test_loop_fairness_includes_dispatcher_start_wait(collector_url, tmp_path):
    from msynd.collect.__main__ import run_loop

    with connect(collector_url) as conn:
        for train in ('12951', '12952'):
            run = make_run(conn, train=train)
            insert_fetch(conn, host=HOST, run_id=run, started_at=NOW - timedelta(seconds=5))
            conn.execute("INSERT INTO obs.poll_queue VALUES (%s,%s,'test',%s)",
                         (run, 1000000 if train == '12951' else 0, NOW))
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
    clock = FakeClock()
    clock.base = NOW
    starts = []
    def handle(request):
        starts.append((json.loads(request.content)['trainNumber'], clock.t))
        return httpx.Response(200, json={})
    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle), clock=clock)
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, T_max=12.5, poll_interval=10,
                           stop=lambda: len(starts) >= 8 or clock.t >= 60)
    asyncio.run(execute())
    assert len(starts) == 8
    previous = {'12951': -5, '12952': -5}
    for train, started in starts:
        assert started - previous[train] <= 15  # Ten-second cadence plus one five-second slot.
        previous[train] = started
    assert all(b[1] - a[1] >= 5 for a, b in zip(starts, starts[1:]))


@pytest.mark.parametrize('args', [
    ['--budget', 'poll=13201'],
    ['--budget', 'poll=5000', '--budget', 'harvest=8201'],
])
def test_cli_rejects_budget_sum_over_13200(args, capsys):
    from msynd.collect.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert '13,200' in capsys.readouterr().err


@pytest.mark.parametrize('value', ['poll=-1', 'other=1', 'poll=1.5', 'poll', 'poll='])
def test_cli_rejects_invalid_budget(value):
    from msynd.collect.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(['--budget', value])
    assert exc.value.code == 2


@pytest.mark.db
def test_loop_budget_exhaustion_sleeps_and_logs_once_per_ist_day(collector_url, tmp_path, caplog):
    from msynd.collect.__main__ import run_loop

    with connect(collector_url) as conn:
        make_run(conn)
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
    clock = FakeClock()
    clock.base = datetime(2026, 10, 7, 18, 28, tzinfo=timezone.utc)
    sleeps = []
    async def sleep(seconds):
        sleeps.append(seconds)
        clock.t += seconds
    clock.sleep = sleep
    d = Dispatcher(HOST, .2, lambda: connect(collector_url),
                   httpx.MockTransport(lambda r: pytest.fail('exhausted budget must not send')),
                   clock=clock, budgets={'poll': 0})
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, stop=lambda: clock.t >= 240)
    asyncio.run(execute())
    assert sleeps == [60, 60, 60, 60]
    messages = [record for record in caplog.records
                if record.name == 'msynd.collect.__main__' and 'budget' in record.message.lower()]
    assert len(messages) == 2
    with connect(collector_url) as conn:
        assert conn.execute('SELECT count(*) FROM obs.fetch').fetchone() == (0,)


def test_poll_due_after_interval_only():
    from msynd.collect.admission import AdmissionState, PollRun, next_job

    run = PollRun(1, '12951', date(2026, 10, 7), NOW)
    state = AdmissionState((run,))
    assert next_job(state, NOW + timedelta(seconds=119)) is None
    job = next_job(state, NOW + timedelta(seconds=120))
    assert job.run == run and job.mode == 'poll' and job.max_attempts == 5
    state = replace(state, poll_interval=30)
    assert next_job(state, NOW + timedelta(seconds=29)) is None
    assert next_job(state, NOW + timedelta(seconds=30)).run == run


@pytest.mark.parametrize('remaining', [0, 1, 5, 6])
def test_harvest_never_delays_due_poll(remaining):
    from msynd.collect.admission import AdmissionState, PollRun, next_job

    poll = PollRun(1, '12951', date(2026, 10, 7), NOW - timedelta(seconds=120-remaining))
    harvest = replace(poll, id=2)
    state = AdmissionState((poll,), harvest=(harvest,))
    job = next_job(state, NOW)
    if remaining == 0:
        assert job.run == poll and job.mode == 'poll'
    elif remaining <= 5:
        assert job is None
        job = next_job(state, NOW + timedelta(seconds=remaining))
        assert job.run == poll and job.mode == 'poll'
    else:
        assert job.run == harvest and job.mode == 'harvest' and job.max_attempts == 1


@pytest.mark.db
def test_harvest_guard_includes_dispatcher_start_wait(collector_url, tmp_path):
    from msynd.collect.__main__ import run_loop

    with connect(collector_url) as conn:
        harvest = make_run(conn, start='2026-10-06')
        first = make_run(conn, train='12952')
        second = make_run(conn, train='12953')
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
        for run, age in ((first, 120), (second, 114)):
            insert_fetch(conn, host=HOST, run_id=run, started_at=NOW-timedelta(seconds=age))
    clock = FakeClock()
    clock.base = NOW
    starts = []
    def handle(request):
        starts.append((clock.t, json.loads(request.content)['trainNumber']))
        return httpx.Response(200, json={})
    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle), clock=clock)
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, poll_slots=2,
                           stop=lambda: len(starts) >= 3 or clock.t >= 60)
    asyncio.run(execute())
    assert starts == [(0, '12952'), (6, '12953'), (11, '12951')]
    with connect(collector_url) as conn:
        assert conn.execute('SELECT mode FROM obs.fetch WHERE run_id=%s', (harvest,)).fetchall() == [('harvest',)]


@pytest.mark.db
def test_scope_excludes_short_routes(db):
    from msynd.collect.__main__ import load_state

    short = make_run(db, train='12951')
    scoped = make_run(db, train='12952')
    for run, km in ((short, 149), (scoped, 150)):
        db.execute('''UPDATE ref.route_stop SET dist_km=LEAST(dist_km,%s)
            WHERE route_version_id=(SELECT route_version_id FROM obs.run WHERE id=%s)''', (km, run))
    assert [r.id for r in load_state(db, NOW).runs] == [scoped]
    assert [r.id for r in load_state(db, NOW + timedelta(days=1)).harvest] == [scoped]


@pytest.mark.db
def test_pass_stays_in_poll_cohort_but_leaves_harvest(db):
    from msynd.collect.__main__ import load_state

    passenger = make_run(db)
    other = make_run(db, train='12952')
    db.execute("UPDATE ref.train SET class='PASS' WHERE number='12951'")
    state = load_state(db, NOW + timedelta(days=1))
    assert {r.id for r in state.runs} == {passenger, other}
    assert [r.id for r in state.harvest] == [other]


@pytest.mark.db
@pytest.mark.parametrize('history,eligible', [
    ([(7, 'cancelled', 400, 'invalid')], False),
    ([(14, 'cancelled', 400, 'invalid')], False),
    ([(21, 'cancelled', 400, 'invalid')], False),
    ([(28, 'cancelled', 400, 'invalid')], False),
    ([(7, 'completed', 200, 'ok'), (14, 'cancelled', 400, 'invalid')], True),
    ([(7, 'cancelled', 400, 'invalid'), (14, 'completed', 200, 'ok')], False),
    ([(35, 'cancelled', 400, 'invalid')], True),
    ([(6, 'cancelled', 400, 'invalid')], True),
    ([(7, 'cancelled', None, None)], True),
    ([(7, 'cancelled', 200, 'ok')], True),
    ([(7, 'scheduled', 400, 'invalid')], True),
    ([(7, 'cancelled', 200, 'invalid'), (14, 'cancelled', 400, 'invalid')], False),
    ([(7, 'cancelled', None, None), (14, 'cancelled', 400, 'invalid')], False),
])
def test_weekday_admission_uses_most_recent_answer_in_last_four_weeks(db, history, eligible):
    from msynd.collect.__main__ import load_state

    run = make_run(db)
    other = make_run(db, train='12952')
    for days, lifecycle, status, outcome in history:
        prior = db.execute('''
            INSERT INTO obs.run(train,start_date,schedule_version_id,route_version_id,lifecycle)
            SELECT train,start_date-%s,schedule_version_id,route_version_id,%s
            FROM obs.run WHERE id=%s RETURNING id
        ''', (days, lifecycle, run)).fetchone()[0]
        if outcome is not None:
            insert_fetch(db, host=HOST, run_id=prior, http_status=status, outcome=outcome,
                         started_at=NOW - timedelta(days=days))
    state = load_state(db, NOW + timedelta(days=1))
    for candidates in (state.runs, state.harvest):
        ids = {r.id for r in candidates}
        assert (run in ids) is eligible
        assert other in ids


@pytest.mark.db
def test_cohort_bounded_and_kept_across_restart(db):
    from msynd.collect.__main__ import load_state

    ids = [make_run(db, train=str(12951+i)) for i in range(8)]
    assert len(load_state(db, NOW).runs) == 6
    for run in ids[:2]:
        insert_fetch(db, host=HOST, run_id=run, mode='poll', outcome='invalid',
                     started_at=NOW - timedelta(seconds=120))
    for _ in range(2):  # A fresh state load has no in-memory cohort to rely on.
        assert {r.id for r in load_state(db, NOW, poll_slots=2).runs} == set(ids[:2])
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (ids[0],))
    assert {r.id for r in load_state(db, NOW, poll_slots=2).runs} == {ids[1], ids[2]}
    assert {r.id for r in load_state(db, NOW + timedelta(minutes=31), poll_slots=2).runs} == {ids[2], ids[3]}


@pytest.mark.db
def test_vacancy_prefers_unseen_train(db):
    from msynd.collect.__main__ import load_state

    seen = make_run(db)
    unseen = make_run(db, train='12952')
    # An unbound pilot status still makes the train seen, but tokens and other hosts do not.
    insert_fetch(db, host=HOST, key='12951/2026-10-06', run_id=None,
                 started_at=NOW - timedelta(days=1), mode='probe')
    insert_fetch(db, host='other', key='12952/2026-10-07', run_id=unseen, started_at=NOW)
    insert_fetch(db, host=HOST, purpose='token', run_id=unseen, started_at=NOW)
    assert [r.id for r in load_state(db, NOW, poll_slots=1).runs] == [unseen]
    assert [r.id for r in load_state(db, NOW + timedelta(days=7), poll_slots=1).runs] == [seen]


@pytest.mark.db
def test_vacancy_prefers_fewest_status_fetches_then_id(db):
    from msynd.collect.__main__ import load_state

    ids = [make_run(db, train=str(12951+i)) for i in range(3)]
    for run, attempts in zip(ids, (3, 1, 1)):
        for _ in range(attempts):
            insert_fetch(db, host=HOST, run_id=run, mode='probe', outcome='invalid',
                         started_at=NOW - timedelta(hours=1))
    assert [r.id for r in load_state(db, NOW, poll_slots=1).runs] == [ids[1]]


@pytest.mark.db
@pytest.mark.parametrize('hours,eligible', [(11.99, False), (12, True), (24, True), (47.99, True), (48, False)])
def test_harvest_window_12h_to_48h(db, hours, eligible):
    from msynd.collect.__main__ import load_state

    run = make_run(db, halts=[('MMCT', None, 0, 0), ('NDLS', 1440+60, None, 150)])
    db.execute('UPDATE obs.run SET resched_min=60 WHERE id=%s', (run,))
    # Multi-day arrival plus rescheduling: Oct 8, 02:00 IST = Oct 7, 20:30 UTC.
    destination = datetime(2026, 10, 7, 20, 30, tzinfo=timezone.utc)
    state = load_state(db, destination + timedelta(hours=hours))
    assert [r.id for r in state.harvest] == ([run] if eligible else [])


@pytest.mark.db
def test_harvest_oldest_destination_then_id_and_lifecycle(db):
    from msynd.collect.__main__ import load_state

    newer = make_run(db)
    older = make_run(db, train='12952', start='2026-10-06')
    tie = make_run(db, train='12953', start='2026-10-06')
    excluded = [make_run(db, train=str(12954+i), start='2026-10-06') for i in range(2)]
    db.execute("UPDATE obs.run SET lifecycle='expired_unknown' WHERE id=%s", (older,))
    db.execute("UPDATE obs.run SET lifecycle='running' WHERE id=%s", (tie,))
    for run, lifecycle in zip(excluded, ('completed', 'cancelled')):
        db.execute('UPDATE obs.run SET lifecycle=%s WHERE id=%s', (lifecycle, run))
    assert [r.id for r in load_state(db, NOW + timedelta(hours=12)).harvest] == [older, tie, newer]


@pytest.mark.db
def test_harvest_one_deferral_to_24h(db):
    from msynd.collect.__main__ import load_state

    run = make_run(db)
    destination = datetime(2026, 10, 7, 10, 10, tzinfo=timezone.utc)
    insert_fetch(db, host=HOST, run_id=run, mode='harvest', started_at=destination + timedelta(hours=12))
    assert load_state(db, destination + timedelta(hours=24, seconds=-1)).harvest == ()
    assert [r.id for r in load_state(db, destination + timedelta(hours=24)).harvest] == [run]
    insert_event(db, run, 'NDLS', 2, 'arr', destination, destination)  # Wrong occurrence is not completion.
    assert [r.id for r in load_state(db, destination + timedelta(hours=24)).harvest] == [run]
    insert_event(db, run, 'NDLS', 1, 'arr', destination, destination)
    assert load_state(db, destination + timedelta(hours=24)).harvest == ()
    db.execute('DELETE FROM obs.event_rev WHERE run_id=%s', (run,))
    insert_fetch(db, host=HOST, run_id=run, mode='harvest', started_at=destination + timedelta(hours=24))
    assert load_state(db, destination + timedelta(hours=25)).harvest == ()
    db.execute('DELETE FROM obs.fetch WHERE run_id=%s', (run,))
    insert_fetch(db, host=HOST, run_id=run, mode='harvest', started_at=destination + timedelta(hours=30))
    assert load_state(db, destination + timedelta(hours=31)).harvest == ()  # Late first fetch: no deferral.


@pytest.mark.db
@pytest.mark.parametrize('outcome', ['transport_error', 'invalid', 'empty', 'inflight', 'challenge'])
def test_harvest_attempt_cap_and_retry_delay(db, outcome):
    from msynd.collect.__main__ import load_state

    run = make_run(db)
    now = NOW + timedelta(days=1)
    for i in range(3):
        insert_fetch(db, host=HOST, run_id=run, mode='harvest', outcome=outcome,
                     started_at=now - timedelta(minutes=30-i*10))
    assert load_state(db, now - timedelta(seconds=1)).harvest == ()
    assert [r.id for r in load_state(db, now).harvest] == [run]
    insert_fetch(db, host=HOST, run_id=run, mode='harvest', outcome=outcome, started_at=now)
    assert load_state(db, now + timedelta(minutes=10)).harvest == ()


@pytest.mark.db
def test_harvest_budget_exhausted_skips_until_next_ist_day(collector_url, tmp_path):
    from msynd.collect.__main__ import run_loop

    clock = FakeClock()
    clock.base = datetime(2026, 10, 7, 18, 29, tzinfo=timezone.utc)
    with connect(collector_url) as conn:
        harvest = make_run(conn, start='2026-10-06')
        poll = make_run(conn, train='12952')
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
        insert_fetch(conn, host=HOST, run_id=poll, started_at=clock.now()-timedelta(seconds=120))
    starts, admissions = [], []
    def handle(request):
        starts.append((clock.t, json.loads(request.content)['trainNumber']))
        return httpx.Response(200, json={})
    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle),
                   clock=clock, budgets={'poll': 100, 'harvest': 0})
    client = MmtClient(d)
    request = client.train_running
    async def recorded(*args, **kwargs):
        admissions.append((clock.t, kwargs['mode'], kwargs['max_attempts']))
        return await request(*args, **kwargs)
    client.train_running = recorded
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(client, conn, disk_path=tmp_path, poll_slots=1, stop=lambda: clock.t >= 130)
    asyncio.run(execute())
    assert [(t, mode) for t, mode, cap in admissions if mode == 'harvest'] == [(5, 'harvest'), (60, 'harvest')]
    assert all(cap == (1 if mode == 'harvest' else 5) for _, mode, cap in admissions)
    assert starts == [(0, '12952'), (120, '12952')]
    with connect(collector_url) as conn:
        assert conn.execute('SELECT count(*) FROM obs.fetch WHERE run_id=%s', (harvest,)).fetchone() == (0,)


@pytest.mark.db
@pytest.mark.parametrize('outcome', ['ok', 'failure'])
def test_harvest_completes_run(collector_url, tmp_path, outcome):
    from msynd.collect.__main__ import run_loop
    from tests.collect.test_store import fixture_run

    with connect(collector_url) as conn:
        run, _ = fixture_run(conn, 'completed_12951')
        conn.execute("UPDATE obs.run SET lifecycle='expired_unknown' WHERE id=%s", (run,))
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-11-01'")
    clock = FakeClock()
    clock.base = NOW + timedelta(days=1, hours=6)
    starts = []
    def handle(request):
        starts.append(clock.t)
        if outcome == 'failure':
            raise httpx.ReadTimeout('offline timeout')
        return httpx.Response(200, content=(FIXTURES / 'completed_12951.json').read_bytes())
    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle), clock=clock)
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, stop=lambda: bool(starts) or clock.t >= 60)
    asyncio.run(execute())
    assert starts == [0]
    with connect(collector_url) as conn:
        assert conn.execute('SELECT mode,outcome FROM obs.fetch WHERE run_id=%s', (run,)).fetchall() == [
            ('harvest', 'ok' if outcome == 'ok' else 'transport_error')]
        assert conn.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == (
            'completed' if outcome == 'ok' else 'expired_unknown',)
        events = conn.execute('SELECT count(*) FROM obs.event_rev WHERE run_id=%s', (run,)).fetchone()[0]
        assert events > 0 if outcome == 'ok' else events == 0


def test_cli_accepts_poll_settings_without_touching_live(monkeypatch):
    from msynd.collect.__main__ import main

    monkeypatch.delenv('MSYND_LIVE', raising=False)
    with pytest.raises(RuntimeError, match='MSYND_LIVE=1'):
        main(['--poll-slots', '3', '--poll-interval', '60'])


@pytest.mark.parametrize('args', [
    ['--poll-slots', '-1'], ['--poll-slots', '1.5'],
    ['--poll-interval', '0'], ['--poll-interval', 'nan'], ['--poll-interval', 'inf'],
])
def test_cli_rejects_invalid_poll_settings(args, capsys):
    from msynd.collect.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert 'poll' in capsys.readouterr().err


@pytest.mark.parametrize('streak,cap', [(0, 3), (1, 2), (2, 1), (3, 1), (8, 1)])
@pytest.mark.parametrize('reason', ['age', 'targeted', 'overdue', 'recovery', 'harvest', 'harvest_recovery'])
def test_every_job_limits_remaining_5xx(streak, cap, reason):
    from msynd.collect.admission import AdmissionState, PollRun, next_job

    run = PollRun(1, '12951', date(2026, 10, 7), NOW - timedelta(seconds=120), streak_5xx=streak)
    state = AdmissionState((run,), T_max=600)
    if reason == 'targeted':
        state = replace(state, runs=(replace(run, priority=100),))
    elif reason == 'overdue':
        state = replace(state, T_max=125)
    elif reason == 'recovery':
        state = replace(state, host_state='breaker', recovery_at=NOW)
    elif reason.startswith('harvest'):
        state = replace(state, runs=(), harvest=(run,),
                        host_state='breaker' if reason == 'harvest_recovery' else 'ok', recovery_at=NOW)
    job = next_job(state, NOW)
    assert job.reason == ('recovery' if reason == 'harvest_recovery' else reason)
    assert job.stop_after_5xx == cap
    assert job.max_attempts == (1 if reason.startswith('harvest') else 5)


@pytest.mark.db
@pytest.mark.parametrize('age,eligible', [(30, False), (60, True), (61, True)])
@pytest.mark.parametrize('inflight_status', [200, 503])
def test_durable_5xx_cooldown_filters_poll_harvest_and_recovery(db, age, eligible, inflight_status):
    from msynd.collect.__main__ import load_state
    from msynd.collect.admission import next_job

    run = make_run(db, start='2026-10-06')
    latest = NOW - timedelta(minutes=age)
    for i, mode in enumerate(('poll', 'probe', 'harvest')):
        insert_fetch(db, host=HOST, run_id=run, mode=mode, outcome='transport_error', http_status=503,
                     started_at=latest - timedelta(seconds=2-i))
    insert_fetch(db, host=HOST, run_id=run, outcome='inflight', http_status=inflight_status,
                 started_at=NOW - timedelta(minutes=1))
    # Unrelated purposes, hosts and future attempts cannot reset the completed MMT streak.
    for kwargs in ({'host': 'other'}, {'purpose': 'token'}, {'started_at': NOW + timedelta(seconds=1)}):
        insert_fetch(db, **({'host': HOST, 'run_id': run, 'started_at': NOW, 'http_status': 200} | kwargs))
    for _ in range(2):  # Reload without carrying any in-memory cooldown state.
        state = load_state(db, NOW)
        assert [r.id for r in state.runs] == ([run] if eligible else [])
        assert [r.id for r in state.harvest] == ([run] if eligible else [])
        for candidates in (state, replace(state, runs=())):
            for recovery in (False, True):
                admission = replace(candidates, host_state='breaker' if recovery else 'ok', recovery_at=NOW)
                job = next_job(admission, NOW)
                if eligible:
                    assert job.run.streak_5xx == 3 and job.stop_after_5xx == 1
                else:
                    assert job is None


@pytest.mark.db
@pytest.mark.parametrize('statuses,streak,cap', [
    ([503], 1, 2), ([503, 503], 2, 1),
    ([200, 503, 503, 503], 3, 1),
    ([503, 503, 503, 200], 0, 3),
    ([503, 503, 503, 400], 0, 3),
    ([503, 503, 503, 429], 0, 3),
    ([503, 503, 503, None], 0, 3),
    ([503, 503, 503, 200, 503], 1, 2),
])
def test_completed_status_attempts_define_trailing_5xx_streak(db, statuses, streak, cap):
    from msynd.collect.__main__ import load_state
    from msynd.collect.admission import next_job

    run = make_run(db, start='2026-10-06')
    for i, status in enumerate(statuses):
        insert_fetch(db, host=HOST, run_id=run, mode='probe', http_status=status,
                     outcome='ok' if status == 200 else 'transport_error' if status is None or status >= 429 else 'invalid',
                     started_at=NOW - timedelta(minutes=61, seconds=len(statuses)-i))
    state = load_state(db, NOW)
    for candidate in (state.runs[0], state.harvest[0]):
        assert candidate.id == run and candidate.streak_5xx == streak
    assert next_job(state, NOW).stop_after_5xx == cap
    assert next_job(replace(state, runs=()), NOW).stop_after_5xx == cap


@pytest.mark.db
def test_streak_uses_fetch_id_to_order_attempts_with_equal_start_time(db):
    from msynd.collect.__main__ import load_state

    run = make_run(db)
    for status in (503, 503, 503, 200, 503):
        insert_fetch(db, host=HOST, run_id=run, http_status=status,
                     outcome='ok' if status == 200 else 'transport_error', started_at=NOW)
    assert load_state(db, NOW).runs[0].streak_5xx == 1


@pytest.mark.db
@pytest.mark.parametrize('streak,expected_attempts', [(0, 3), (1, 2), (2, 1), (3, 1)])
def test_loop_cools_broken_run_and_polls_healthy_run_within_t_max(collector_url, tmp_path, streak, expected_attempts):
    from msynd.collect.__main__ import load_state, run_loop

    with connect(collector_url) as conn:
        broken = make_run(conn)
        healthy = make_run(conn, train='12952')
        conn.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07'")
        for i in range(streak):
            insert_fetch(conn, host=HOST, run_id=broken, outcome='transport_error', http_status=503,
                         started_at=NOW - timedelta(minutes=61, seconds=streak-i))
        insert_fetch(conn, host=HOST, run_id=healthy, started_at=NOW - timedelta(seconds=120))
        conn.execute("INSERT INTO obs.poll_queue VALUES (%s,1000000,'test',%s)", (broken, NOW))
    clock = FakeClock()
    clock.base = NOW
    starts = []
    def handle(request):
        train = json.loads(request.content)['trainNumber']
        starts.append((train, clock.t))
        return httpx.Response(503) if train == '12951' else httpx.Response(200, json={})
    d = Dispatcher(HOST, .2, lambda: connect(collector_url), httpx.MockTransport(handle), clock=clock)
    async def execute():
        with connect(collector_url) as conn:
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, poll_slots=2, T_max=150,
                           stop=lambda: clock.t >= 260)
    asyncio.run(execute())
    assert sum(train == '12951' for train, _ in starts) == expected_attempts
    healthy_starts = [t for train, t in starts if train == '12952']
    assert len(healthy_starts) >= 2
    assert all(b-a <= 150 for a, b in zip([-120] + healthy_starts, healthy_starts))
    assert d.state == 'ok'
    with connect(collector_url) as conn:
        assert broken not in [r.id for r in load_state(conn, clock.now()).runs]
