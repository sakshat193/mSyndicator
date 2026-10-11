from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import multiprocessing as mp
import os
import random
import subprocess
import sys
import threading
import time
from uuid import uuid4

import msgpack
import numpy as np
import psycopg
import pytest

from msynd.db.dirty import mark_dirty
from msynd.engine.replay import Params, ReplayOut
from tests.helpers import make_run, insert_event

NOW = datetime(2025, 10, 8, 0, tzinfo=timezone.utc)
PARAMS = Params(n=32, k=2)


def tables(conn, ev):
    return None


def fake_replay(ev, models, params, key):
    # The real loader must merge late/corrected evidence before this pure replay.
    minutes = {x.event: x.minute for x in ev.evidence}
    paths = np.tile([minutes.get(e.idx, e.sched_off) for e in ev.events], (params.k, 1))
    q = np.full((3, 2, 7), len(ev.evidence), dtype=float)
    return ReplayOut(q, np.full(7, len(ev.evidence)), paths, .75, False, (), ())


def count_tables(conn, ev):
    return conn.execute('SELECT count(*) FROM obs.event_rev WHERE run_id=%s', (ev.run_id,)).fetchone()[0]


def count_replay(ev, models, params, key):
    out = fake_replay(ev, models, params, key)
    out.next_q[:] = models
    return out


def normalized_run(value):
    value = dict(value)
    paths = np.frombuffer(value.pop('paths'), dtype='<u2').reshape(value['path_shape'])
    base = value.pop('base_ts')
    value['path_indices'] = paths[:, :, 0].tolist()
    value['path_times'] = (paths[:, :, 1].astype(float) * 6 + base).tolist()
    return value


def broken_replay(*args):
    raise ValueError('bad evidence')


def io_replay(*args):
    raise OSError('temporary IO failure')


@pytest.fixture
def runs(db_url):
    with psycopg.connect(db_url) as conn:
        conn.execute('TRUNCATE obs.dirty_run, sim.frame, sim.metric')
        existing_stations = {row[0] for row in conn.execute('SELECT code FROM ref.station')}
        prefix = uuid4().hex[:10]
        ids = [make_run(conn, train=f'{prefix}{i}', start='2025-10-08') for i in range(3)]
        conn.execute("UPDATE obs.run SET lifecycle='running' WHERE id=ANY(%s)", (ids,))
        bindings = conn.execute('SELECT train,schedule_version_id,route_version_id FROM obs.run WHERE id=ANY(%s)', (ids,)).fetchall()
        conn.commit()
    yield ids
    with psycopg.connect(db_url) as conn:
        conn.execute('DELETE FROM obs.event_rev WHERE run_id=ANY(%s)', (ids,))
        conn.execute('DELETE FROM obs.run WHERE id=ANY(%s)', (ids,))
        for train, schedule, route in bindings:
            conn.execute('DELETE FROM ref.route_stop WHERE route_version_id=%s', (route,))
            conn.execute('DELETE FROM ref.route_version WHERE id=%s', (route,))
            conn.execute('DELETE FROM ref.schedule_stop WHERE version_id=%s', (schedule,))
            conn.execute('DELETE FROM ref.schedule_version WHERE id=%s', (schedule,))
            conn.execute('DELETE FROM ref.train WHERE number=%s', (train,))
        for station in {'MMCT', 'BRC', 'NDLS'} - existing_stations:
            conn.execute('DELETE FROM ref.station WHERE code=%s', (station,))
        conn.execute('TRUNCATE obs.dirty_run, sim.frame, sim.metric')


def payload(conn, gen=None):
    sql = 'SELECT payload FROM sim.frame '
    row = conn.execute(sql + ('ORDER BY gen DESC LIMIT 1' if gen is None else 'WHERE gen=%s'),
                       () if gen is None else (gen,)).fetchone()
    return msgpack.unpackb(bytes(row[0]), raw=False, strict_map_key=False)


def consume(conn, **kwargs):
    from msynd.engine.__main__ import consume
    return consume(conn, fake_replay, tables, PARAMS, **kwargs)


def dirty(conn, ids):
    mark_dirty(conn, ids)
    conn.commit()


def event(conn, run, minute, received=NOW):
    insert_event(conn, run, 'BRC', 1, 'arr', NOW + timedelta(minutes=minute), received)
    mark_dirty(conn, [run])
    conn.commit()


@pytest.mark.db
def test_dirty_race_never_loses_rows(db_url, runs):
    from msynd.engine.__main__ import consume as _consume
    errors = []
    def writer():
        try:
            rng = random.Random(31)
            with psycopg.connect(db_url) as conn:
                for i in range(300):
                    event(conn, rng.choice(runs), i, NOW + timedelta(microseconds=i))
                    time.sleep(rng.random() / 1000)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=writer)
    thread.start()
    final = {}
    with psycopg.connect(db_url) as conn:
        while thread.is_alive() or conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0]:
            conn.commit()
            gen = _consume(conn, count_replay, count_tables, PARAMS)
            if gen is not None:
                final.update(payload(conn, gen)['runs'])
                conn.commit()
        thread.join()
        assert not errors
        for run in runs:
            # Latest merged actual equals the last committed revision for each run.
            actual = conn.execute('SELECT actual_ts FROM obs.event WHERE run_id=%s', (run,)).fetchone()[0]
            value = final[run]
            packed = np.frombuffer(value['paths'], dtype='<u2').reshape(value['path_shape'])
            timestamp = value['base_ts'] + float(packed[0, 1, 1]) * 6
            assert abs(timestamp - actual.timestamp()) <= 3
            count = conn.execute('SELECT count(*) FROM obs.event_rev WHERE run_id=%s', (run,)).fetchone()[0]
            assert value['next_q'] == [float(count)] * 7


@pytest.mark.db
def test_matching_ver_update_during_replay_survives(db_url, runs):
    with psycopg.connect(db_url) as conn, psycopg.connect(db_url) as writer:
        dirty(conn, runs[:1])
        from msynd.engine.__main__ import consume as real_consume
        def racing(ev, models, params, key):
            event(writer, ev.run_id, 10)
            return fake_replay(ev, models, params, key)
        real_consume(conn, racing, tables, PARAMS)
        assert conn.execute('SELECT run_id FROM obs.dirty_run').fetchall() == [(runs[0],)]
        conn.commit()
        consume(conn)
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 0
        assert payload(conn)['runs'][runs[0]]['next_q'] == [1.] * 7


@pytest.mark.db
def test_one_batch_vs_two_identical_payloads(db_url, runs):
    with psycopg.connect(db_url) as conn:
        dirty(conn, runs)
        consume(conn)
        together = payload(conn)['runs']
        conn.commit()
        separate = {}
        for run in runs:
            dirty(conn, [run])
            consume(conn)
            separate.update(payload(conn)['runs'])
            conn.commit()
        assert together == separate


@pytest.mark.db
def test_restart_identical_full_frame(db_url, runs):
    from msynd.engine.__main__ import rebuild
    with psycopg.connect(db_url) as conn:
        rebuild(conn, fake_replay, tables, PARAMS)
        first = payload(conn)
        conn.commit()
    with psycopg.connect(db_url) as conn:
        rebuild(conn, fake_replay, tables, PARAMS)
        second = payload(conn)
        assert first['runs'] == second['runs']
        assert first['versions'] == second['versions']
        assert first['is_full'] and second['is_full']


def _assert_frame_latency(db_url, runs, notifications):
    from msynd.engine.__main__ import run_loop
    ready, stop = threading.Event(), threading.Event()
    errors = []
    def engine():
        try:
            with psycopg.connect(db_url) as conn:
                with ThreadPoolExecutor(4) as pool:
                    run_loop(conn, fake_replay, tables, PARAMS, pool=pool, stop=stop,
                             notifications=notifications, ready=ready)
        except BaseException as exc:
            errors.append(exc)
            ready.set()
    thread = threading.Thread(target=engine)
    thread.start()
    try:
        assert ready.wait(10)
        assert not errors
        with psycopg.connect(db_url, autocommit=True) as api, psycopg.connect(db_url) as writer:
            api.execute('LISTEN sim_frame')
            start_gen = api.execute('SELECT max(gen) FROM sim.frame').fetchone()[0]
            start = time.monotonic()
            event(writer, runs[0], 10)
            if notifications:
                assert list(api.notifies(timeout=2, stop_after=1))
            deadline = start + (2 if notifications else 1.3)
            while time.monotonic() < deadline:
                row = api.execute('SELECT gen FROM sim.frame WHERE gen>%s AND NOT is_full', (start_gen,)).fetchone()
                if row:
                    assert payload(api, row[0])['runs'][runs[0]]['next_q'] == [1.] * 7
                    break
                time.sleep(.01)
            else:
                pytest.fail('delta not visible within fallback bound')
            assert time.monotonic() - start < (2 if notifications else 1.3)
    finally:
        stop.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not errors


@pytest.mark.db
def test_frame_latency_idle_under_2s(db_url, runs):
    _assert_frame_latency(db_url, runs, True)


@pytest.mark.db
def test_notify_loss_bounded(db_url, runs):
    _assert_frame_latency(db_url, runs, False)


@pytest.mark.db
def test_replay_exception_acked_with_error_state(db_url, runs):
    from msynd.engine.__main__ import consume as real_consume
    with psycopg.connect(db_url) as conn:
        dirty(conn, runs[:1])
        real_consume(conn, broken_replay, tables, PARAMS)
        assert payload(conn)['runs'][runs[0]]['state'] == 'error'
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 0
        assert conn.execute("SELECT value FROM sim.metric WHERE name='replay_error'").fetchone()[0] == 1


@pytest.mark.db
def test_transient_failure_keeps_dirty(db_url, runs):
    from msynd.engine.__main__ import consume as real_consume
    with psycopg.connect(db_url) as conn:
        dirty(conn, runs[:1])
        with pytest.raises(OSError):
            real_consume(conn, io_replay, tables, PARAMS)
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM sim.frame').fetchone()[0] == 0


@pytest.mark.db
def test_publication_failure_rolls_back_ack_and_metric(db_url, runs):
    from msynd.engine.__main__ import consume as real_consume
    with psycopg.connect(db_url) as conn:
        dirty(conn, runs[:1])
        conn.execute("CREATE FUNCTION sim.fail_frame() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'publish failed'; END $$")
        conn.execute('CREATE TRIGGER fail_frame BEFORE INSERT ON sim.frame FOR EACH ROW EXECUTE FUNCTION sim.fail_frame()')
        conn.commit()
        try:
            with pytest.raises(psycopg.Error):
                real_consume(conn, broken_replay, tables, PARAMS)
            assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 1
            assert conn.execute('SELECT count(*) FROM sim.metric').fetchone()[0] == 0
            conn.rollback()
        finally:
            conn.execute('DROP TRIGGER fail_frame ON sim.frame')
            conn.execute('DROP FUNCTION sim.fail_frame()')
            conn.commit()


@pytest.mark.db
@pytest.mark.parametrize('lifecycle', ['completed', 'cancelled', 'expired_unknown'])
def test_tombstone_for_retired_run(db_url, runs, lifecycle):
    from msynd.engine.__main__ import rebuild
    with psycopg.connect(db_url) as conn:
        rebuild(conn, fake_replay, tables, PARAMS)
        full = payload(conn)
        assert set(full['runs']) == set(runs)
        conn.commit()
        conn.execute('UPDATE obs.run SET lifecycle=%s WHERE id=%s', (lifecycle, runs[0]))
        dirty(conn, runs)
        consume(conn)
        delta = payload(conn)
        assert delta['tombstones'] == [runs[0]]
        assert set(delta['runs']) == set(runs[1:])
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 0
        for frame in (full, delta):
            for run_id, run in frame['runs'].items():
                assert run['run_id'] == run_id


@pytest.mark.db
def test_late_event_frame_equals_fresh_replay(db_url, runs):
    from msynd.engine.__main__ import rebuild
    with psycopg.connect(db_url) as conn:
        event(conn, runs[0], 10)
        consume(conn)
        # Earlier route event arrives late; correction to BRC follows it.
        insert_event(conn, runs[0], 'MMCT', 1, 'dep', NOW, NOW + timedelta(minutes=1))
        event(conn, runs[0], 15, NOW + timedelta(minutes=2))
        consume(conn)
        incremental = payload(conn)['runs'][runs[0]]
        conn.commit()
        rebuild(conn, fake_replay, tables, PARAMS)
        # Full and delta frames must preserve the same absolute paths.
        assert normalized_run(payload(conn)['runs'][runs[0]]) == normalized_run(incremental)
        assert incremental['next_q'] == [2.] * 7


@pytest.mark.db
def test_spawn_pool_replays_plain_arguments(db_url, runs):
    with psycopg.connect(db_url) as conn:
        dirty(conn, runs)
        with mp.get_context('spawn').Pool(2) as pool:
            consume(conn, pool=pool)
        assert set(payload(conn)['runs']) == set(runs)


@pytest.mark.db
def test_module_entrypoint_spawn_jobs_complete(db_url, runs):
    # runpy's __main__ execution matches python -m; stop after the worker jobs.
    script = '''
import multiprocessing as mp
import runpy
import sys
import psycopg
from msynd import db
from tests.engine.test_loop import PARAMS, fake_replay, tables

def check_jobs(url):
    engine = sys.modules['__main__']
    with psycopg.connect(url) as conn:
        jobs, errors, tombstones = engine._prepare(
            conn, RUNS, tables, PARAMS, fake_replay)
        assert len(jobs) == len(RUNS) and not errors and not tombstones
        with mp.get_context('spawn').Pool(2) as pool:
            results = pool.map_async(engine._job, jobs).get(timeout=8)
        assert {run for run, frame, cached in results} == set(RUNS)
        assert all(frame['state'] == 'schedule_only' and cached is not None
                   and frame['paths'] for run, frame, cached in results)
    raise SystemExit(0)

RUNS = [int(run) for run in sys.argv[1:]]
db.connect = check_jobs
runpy.run_module('msynd.engine', run_name='__main__', alter_sys=True)
'''
    result = subprocess.run(
        [sys.executable, '-c', script, *map(str, runs)],
        env=os.environ | {'DATABASE_URL': db_url, 'ENGINE_WORKERS': '2'},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.db
def test_batch_cap_and_frame_retention(db_url, runs):
    from msynd.engine.__main__ import rebuild
    with psycopg.connect(db_url) as conn:
        conn.execute("INSERT INTO obs.dirty_run(run_id) SELECT generate_series(100000,100200)")
        conn.commit()
        consume(conn)
        assert len(payload(conn)['tombstones']) == 200
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 1
        conn.execute("UPDATE sim.frame SET ts=now()-interval '16 minutes'")
        conn.commit()
        rebuild(conn, fake_replay, tables, PARAMS)
        assert conn.execute('SELECT count(*) FROM sim.frame').fetchone()[0] == 1


def test_encode_run_uint16_and_truncation():
    from msynd.engine.frames import encode_run
    from msynd.engine.evidence import RouteEvent, RunEvidence
    ev = RunEvidence(1, {}, (RouteEvent(0, 'A', 1, 'dep', 3, 0, 0, 0, 0),
                            RouteEvent(1, 'B', 1, 'arr', 9, 1, 10, 20, 0)), (), 0, 0)
    record = {'evidence': ev, 'origin_ts': NOW.timestamp(), 'lifecycle': 'scheduled',
              'last_received': None, 'asof': NOW, 'diverted_from': None, 'part_cancelled': None}
    out = ReplayOut(np.full((2, 2, 7), np.nan), np.full(7, np.nan),
                    np.array([[.2, np.nan], [.4, np.nan]]), .5, True, ('cold_start',), ())
    encoded = encode_run(1, out, record, NOW)
    assert encoded['paths'] == b'\x03\x00\x02\x00\x03\x00\x04\x00'
    assert encoded['path_shape'] == (2, 1, 2)
    assert encoded['state'] == 'schedule_only'
    assert encoded['headline']['surprise'] is True
    assert encoded['flags'] == ('cold_start',)
    assert encoded['next_q'] == [None] * 7
    out.paths[0, 0] = 6553.6
    with pytest.raises(ValueError, match='uint16'):
        encode_run(1, out, record, NOW)


@pytest.mark.db
def test_write_frame_preserves_per_run_time_bases(db):
    from msynd.engine.frames import encode_run, write_frame
    from msynd.engine.evidence import RouteEvent, RunEvidence

    out = ReplayOut(np.full((1, 2, 7), np.nan), np.full(7, np.nan),
                    np.array([[.2], [.4]]), .5, False, (), ())
    encoded = {}
    for run_id, base in ((2, NOW + timedelta(days=6)), (1, NOW)):
        ev = RunEvidence(run_id, {}, (RouteEvent(0, 'A', 1, 'dep', 3, 0, 0, 0, 0),), (), 0, None)
        record = {'evidence': ev, 'origin_ts': base.timestamp(), 'lifecycle': 'scheduled',
                  'last_received': None, 'asof': base, 'diverted_from': None, 'part_cancelled': None}
        encoded[run_id] = encode_run(run_id, out, record, base)

    expected = {run_id: (run['paths'], run['base_ts']) for run_id, run in encoded.items()}
    gen = write_frame(db, True, {}, encoded, [])
    result = payload(db, gen)
    assert result['base_ts'] == NOW.timestamp()
    assert set(result['runs']) == set(encoded)
    for run_id, (paths, base) in expected.items():
        assert result['runs'][run_id]['paths'] == paths
        assert result['runs'][run_id]['base_ts'] == base




@pytest.mark.db
def test_model_change_and_bias_generation_dirty_all_active(db_url, runs, monkeypatch):
    import msynd.engine.__main__ as loop
    from msynd.engine.__main__ import run_loop
    monkeypatch.setattr(loop, '_now', lambda conn: NOW, raising=False)
    with psycopg.connect(db_url) as conn:
        scheduled(conn, runs[0], 180)
        scheduled(conn, runs[1], 30)
    ready, stop = threading.Event(), threading.Event()
    errors = []
    def engine():
        try:
            with psycopg.connect(db_url) as conn, ThreadPoolExecutor(4) as pool:
                run_loop(conn, fake_replay, tables, PARAMS, pool=pool, ready=ready, stop=stop)
        except BaseException as exc:
            errors.append(exc)
            ready.set()
    thread = threading.Thread(target=engine)
    thread.start()
    try:
        assert ready.wait(10) and not errors
        with psycopg.connect(db_url) as writer, psycopg.connect(db_url, autocommit=True) as api:
            model = writer.execute("""INSERT INTO ml.model(role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active)
                VALUES ('run','empirical',1,now(),'[2025-01-01,2025-02-01)','unused','{}','{}',true) RETURNING id""").fetchone()[0]
            writer.execute('NOTIFY model_changed')
            writer.commit()
            for bias in (0, 1):
                if bias:
                    writer.execute("INSERT INTO ml.leg_bias VALUES (%s,1,'A','B',0,0,1)", (model,))
                    writer.commit()  # no notification: generation polling must still find it
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    frame = payload(api)
                    if (not frame['is_full'] and frame['versions']['bias_gen'] == bias
                            and frame['versions']['models'].get('run') == model):
                        assert set(frame['runs']) == set(runs[1:])
                        assert all(frame['runs'][run]['versions']['models']['run'] == model for run in runs[1:])
                        break
                    time.sleep(.01)
                else:
                    pytest.fail('version change did not rebuild active runs')
    finally:
        stop.set()
        thread.join(5)
        with psycopg.connect(db_url) as conn:
            conn.execute('DELETE FROM ml.leg_bias WHERE model_id=%s', (model,))
            conn.execute('DELETE FROM ml.model WHERE id=%s', (model,))
    assert not errors


@pytest.mark.db
def test_periodic_full_frame_reuses_published_runs(db_url, runs, monkeypatch):
    import msynd.engine.__main__ as loop
    from itertools import count
    stop = threading.Event()
    ticks = count(0, 301)
    monkeypatch.setattr(loop, 'monotonic', lambda: next(ticks))
    calls = 0
    def replay_once(ev, models, params, key):
        nonlocal calls
        calls += 1
        return fake_replay(ev, models, params, key)
    class StopAfterOnePass:
        def is_set(self):
            return stop.is_set()
        def wait(self, timeout):
            stop.set()
    with psycopg.connect(db_url) as conn, ThreadPoolExecutor(4) as pool:
        loop.run_loop(conn, replay_once, tables, PARAMS, pool=pool,
                      stop=StopAfterOnePass(), notifications=False)
        assert conn.execute('SELECT count(*) FROM sim.frame WHERE is_full').fetchone()[0] == 2
        assert calls == len(runs)


def scheduled(conn, run_id, departure_minutes):
    # NOW is 05:30 IST; include both the schedule offset and rescheduling.
    conn.execute("UPDATE ref.schedule_stop SET dep_off=60 WHERE idx=0 AND version_id="
                 "(SELECT schedule_version_id FROM obs.run WHERE id=%s)", (run_id,))
    conn.execute("UPDATE obs.run SET lifecycle='scheduled',resched_min=%s WHERE id=%s",
                 (270 + departure_minutes, run_id))


@pytest.mark.db
def test_engine_window_filters_rebuild_and_acknowledges_dirty(db_url, runs, monkeypatch):
    import msynd.engine.__main__ as loop
    monkeypatch.setattr(loop, '_now', lambda conn: NOW, raising=False)
    far, near, running = runs
    replayed = []

    def replay(ev, models, params, key):
        replayed.append(ev.run_id)
        return fake_replay(ev, models, params, key)

    with psycopg.connect(db_url) as conn:
        scheduled(conn, far, 180)
        scheduled(conn, near, 30)
        conn.commit()
        cache = {}
        loop.rebuild(conn, replay, tables, PARAMS, cache=cache)
        assert set(replayed) == {near, running}
        assert set(payload(conn)['runs']) == {near, running}
        assert set(cache) == {near, running}
        conn.commit()
        replayed.clear()
        dirty(conn, runs)
        loop.consume(conn, replay, tables, PARAMS, cache=cache)
        assert set(replayed) == {near, running}
        assert payload(conn)['tombstones'] == [far]
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 0
        assert set(cache) == {near, running}


@pytest.mark.db
@pytest.mark.parametrize('departure_minutes, included', [(-61, False), (-60, True),
                                                      (60, True), (61, False)])
def test_engine_window_boundaries(db_url, runs, monkeypatch, departure_minutes, included):
    import msynd.engine.__main__ as loop
    monkeypatch.setattr(loop, '_now', lambda conn: NOW, raising=False)
    with psycopg.connect(db_url) as conn:
        scheduled(conn, runs[0], departure_minutes)
        dirty(conn, runs[:1])
        consume(conn)
        frame = payload(conn)
        assert (runs[0] in frame['runs']) is included
        assert frame['tombstones'] == ([] if included else runs[:1])
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 0


@pytest.mark.db
def test_window_refresh_tombstones_departed_and_replays_entered(db_url, runs, monkeypatch):
    import msynd.engine.__main__ as loop
    now = NOW
    monkeypatch.setattr(loop, '_now', lambda conn: now, raising=False)
    ticks = iter([0, 301])
    monkeypatch.setattr(loop, 'monotonic', lambda: next(ticks))
    cached = []
    monkeypatch.setattr(loop, 'write_periodic', lambda conn, cache, now, **kw: cached.append(set(cache)))

    class Stop:
        done = False
        def is_set(self):
            nonlocal now
            now = NOW + timedelta(hours=2)
            done, self.done = self.done, True
            return done
        def wait(self, timeout):
            self.done = True

    leaving, entering, running = runs
    with psycopg.connect(db_url) as conn, ThreadPoolExecutor(4) as pool:
        scheduled(conn, leaving, 0)
        scheduled(conn, entering, 180)
        conn.commit()
        loop.run_loop(conn, fake_replay, tables, PARAMS, pool=pool,
                      stop=Stop(), notifications=False)
        frames = [payload(conn, row[0]) for row in conn.execute('SELECT gen FROM sim.frame ORDER BY gen').fetchall()]
        assert set(frames[0]['runs']) == {leaving, running}
        assert any(f['tombstones'] == [leaving] and not f['is_full'] for f in frames)
        assert frames[-1]['is_full'] and set(frames[-1]['runs']) == {entering, running}
        assert cached == [{entering, running}]
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 0


@pytest.mark.db
def test_rebuild_chunks_match_unchunked_payload(db_url, runs, monkeypatch):
    import msynd.engine.__main__ as loop
    monkeypatch.setattr(loop, '_now', lambda conn: NOW, raising=False)
    prepare = loop._prepare
    batches = []

    def prepare_chunk(conn, ids, *args, **kwargs):
        batches.append(len(ids))
        return prepare(conn, ids, *args, **kwargs)

    monkeypatch.setattr(loop, '_prepare', prepare_chunk)
    prefix = uuid4().hex
    with psycopg.connect(db_url) as conn:
        conn.execute("INSERT INTO ref.train(number,name) SELECT %s||i,'chunk' FROM generate_series(1,201) i", (prefix,))
        extra = [row[0] for row in conn.execute('''INSERT INTO obs.run
            (train,start_date,schedule_version_id,route_version_id,lifecycle)
            SELECT %s||i,r.start_date,r.schedule_version_id,r.route_version_id,'running'
            FROM obs.run r CROSS JOIN generate_series(1,201) i WHERE r.id=%s RETURNING id''', (prefix, runs[0]))]
        conn.commit()
        try:
            cache = {}
            loop.rebuild(conn, fake_replay, tables, PARAMS, cache=cache)
            chunked = payload(conn)
            assert batches == [200, 4]
            assert set(cache) == set(runs + extra)
            conn.commit()
            jobs, all_runs, tombstones = prepare(conn, sorted(runs + extra), tables, PARAMS, fake_replay)
            loop._replay_jobs(jobs, all_runs, None)
            loop._publish(conn, True, chunked['versions'], all_runs, tombstones)
            unchunked = payload(conn)
            for key in ('is_full', 'versions', 'base_ts', 'runs', 'tombstones'):
                assert chunked[key] == unchunked[key]
        finally:
            conn.rollback()
            conn.execute('DELETE FROM obs.run WHERE id=ANY(%s)', (extra,))
            conn.execute('DELETE FROM ref.train WHERE number LIKE %s', (prefix + '%',))
            conn.commit()



