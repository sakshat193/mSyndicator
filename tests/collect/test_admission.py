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
from tests.helpers import insert_fetch, make_run

NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parents[1] / 'fixtures' / 'mmt'


def runs():
    from msynd.collect.admission import PollRun

    return tuple(PollRun(i, str(12000 + i), date(2026, 10, 7), NOW,
                         1000000 if i == 0 else 0) for i in range(300))


def test_no_run_starves_beyond_tmax():
    from msynd.collect.admission import AdmissionState, next_job

    state = AdmissionState(runs())
    now = NOW
    seen = set()
    # 300 runs * one request / 0.2 per second * 1.25 = 1875 seconds.
    for _ in range(4320):  # Six hours, including the first cycle.
        job = next_job(state, now)
        seen.add(job.run.id)
        state = replace(state, runs=tuple(
            replace(run, last_fetch=now) if run.id == job.run.id else run
            for run in state.runs))
        now += timedelta(seconds=5)
        assert max((now - run.last_fetch).total_seconds() for run in state.runs) <= 1875
    assert len(seen) == 300


def test_job_order():
    from msynd.collect.admission import AdmissionState, PollRun, next_job

    old = PollRun(1, '12951', date(2026, 10, 7), NOW - timedelta(seconds=200), 0)
    target = replace(old, id=2, last_fetch=NOW, priority=1000)
    age = replace(old, id=3, last_fetch=NOW - timedelta(seconds=20), priority=None)
    state = AdmissionState((old, target, age), T_max=100)
    assert next_job(state, NOW).run.id == 1
    state = replace(state, runs=(replace(old, last_fetch=NOW), target, age))
    assert next_job(state, NOW).run.id == 2
    state = replace(state, runs=tuple(replace(r, priority=None) for r in state.runs))
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
    job = next_job(state, NOW + timedelta(seconds=301))
    assert job.reason == 'overdue'
    state = replace(state, rate=.2, T_max=100)
    with pytest.raises(ValueError, match='capacity'):
        next_job(state, NOW)


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
    assert by_id[scheduled].last_fetch == datetime(2026, 10, 6, 18, 30, tzinfo=timezone.utc)


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
            await run_loop(MmtClient(d), conn, disk_path=tmp_path, T_max=12.5,
                           stop=lambda: len(starts) >= 8 or clock.t >= 60)
    asyncio.run(execute())
    assert len(starts) == 8
    previous = {'12951': -5, '12952': -5}
    for train, started in starts:
        assert started - previous[train] <= 12.5
        previous[train] = started
    assert all(b[1] - a[1] >= 5 for a, b in zip(starts, starts[1:]))
