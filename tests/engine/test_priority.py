from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from psycopg.types.json import Jsonb

from msynd.engine.evidence import load_run_evidence
from msynd.engine.replay import ReplayOut
from tests.helpers import make_run, insert_event, insert_fetch
from tests.engine.test_loop import runs

NOW = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)
S = {'lag': [0, 10], 'survival': [1, .2], 'tau': 10, 'eps': .1}


def output(dep=0, arr=500, next_times=(100, 110), cold=False):
    q = np.full((3, 2, 7), np.nan)
    q[0, 1] = dep
    q[1, 0] = [90, 100, 105, 110, 115, 120, 130]
    q[2, 0] = arr
    return ReplayOut(q, q[1, 0].copy(), np.zeros((2, 4)), 1, False,
                     ('cold_start',) if cold else (), (), np.array(next_times))


def test_priority_prefers_published_and_wide():
    from msynd.engine.priority import priority
    out = output()
    assert priority(out, S, 130, 0, 30) > priority(out, S, 90, 0, 30) == 0
    narrow = output()
    narrow.next_q[5] = 101
    assert priority(out, S, 130, 0, 30) > priority(narrow, S, 130, 0, 30)
    # All cloud particles, not the two display paths or just P50.
    mixed = output(next_times=(100, 100, 200))
    assert priority(mixed, S, 110, 0, 30) == pytest.approx(20 * .8 * 2 / 3)


def test_age_term_prevents_starvation():
    from msynd.engine.priority import priority
    assert priority(output(), S, 0, 90, 30) == 3
    with pytest.raises(ValueError):
        priority(output(), S, 0, 0, 0)


def test_replay_retains_full_next_cloud_and_endpoint_medians():
    from dataclasses import replace
    from msynd.engine.evidence import Evidence
    from msynd.engine.replay import replay
    from tests.engine.test_replay import fixture
    ev, models, params = fixture()
    ev = replace(ev, evidence=(Evidence(1, True, 201, 'ntes_status'),))
    full = replay(ev, models, params, 7)
    out = replay(ev, models, replace(params, k=2), 7)
    np.testing.assert_array_equal(out.next_times, full.paths[:, 3])
    np.testing.assert_allclose(out.endpoint_p50, np.median(full.paths[:, [0, 3]], axis=0))
    assert np.isnan(out.halt_q[0, 1]).all()


def cached(db, run, out):
    ev = load_run_evidence(db, run, NOW)
    return ({'evidence': ev, 'origin_ts': NOW.timestamp() - 330 * 60}, out)


@pytest.mark.db
def test_forecast_issue_b2_nan_when_stale(db):
    from msynd.engine.priority import write_periodic
    run = make_run(db)
    db.execute("UPDATE obs.run SET lifecycle='running' WHERE id=%s", (run,))
    cache = {run: cached(db, run, output(arr=1000))}
    for station, receipt, minute in [('BRC', -31, 600), ('NDLS', -30, 1100),
                                      ('NDLS', 1, 1200)]:
        db.execute('INSERT INTO obs.ntes_eta_rev VALUES (%s,%s,1,\'arr\',%s,NULL,%s)',
                   (run, station, NOW + timedelta(minutes=minute - 330),
                    NOW + timedelta(minutes=receipt)))
    insert_fetch(db, run_id=run, started_at=NOW)  # cannot confirm stale ETA
    write_periodic(db, cache, NOW, issue=True)
    halts, q, b2 = db.execute('SELECT halt_idx,q,b2 FROM sim.forecast_issue').fetchone()
    assert halts == [1, 2] and len(q) == 14
    assert np.isnan(b2[0]) and b2[1] == 1100
    assert db.execute('SELECT versions FROM sim.forecast_issue').fetchone()[0] == cache[run][0]['evidence'].versions


@pytest.mark.db
def test_forecast_issue_requires_running_but_not_active_p50(db):
    from msynd.engine.priority import write_periodic
    run = make_run(db)
    cache = {run: cached(db, run, output(arr=1000))}
    write_periodic(db, cache, NOW, issue=True)
    assert db.execute('SELECT count(*) FROM sim.forecast_issue').fetchone()[0] == 0
    db.execute("UPDATE obs.run SET lifecycle='running' WHERE id=%s", (run,))
    cache[run] = cached(db, run, output(arr=300))
    write_periodic(db, cache, NOW, issue=True)
    assert db.execute('SELECT halt_idx FROM sim.forecast_issue').fetchone()[0] == [1, 2]


@pytest.mark.db
def test_a1_active_uses_simulated_arrival(db):
    from msynd.engine.priority import write_periodic
    halts = [('MMCT', None, 0, 0), ('BRC', 100, 105, 100), ('NDLS', 300, None, 200)]
    late, unseen, ended, future = [make_run(db, train=str(i), halts=halts) for i in range(4)]
    insert_event(db, late, 'MMCT', 1, 'dep', NOW - timedelta(hours=2), NOW - timedelta(minutes=10))
    insert_event(db, ended, 'NDLS', 1, 'arr', NOW, NOW)
    cache = {late: cached(db, late, output(dep=np.nan, arr=1000, cold=True)),
             unseen: cached(db, unseen, output(arr=300)),
             ended: cached(db, ended, output(arr=np.nan)),
             future: cached(db, future, output(dep=400, arr=1000))}
    write_periodic(db, cache, NOW, metrics=True)
    metrics = dict(db.execute('SELECT name,value FROM sim.metric'))
    assert metrics == dict(a1_active=1, a1_discovered=1, a1_fresh=1,
                          a1_active_1h=1, a1_discovered_1h=1, a1_cold=1)
    # Cancel as of NOW; future withdrawal must not leak into this snapshot.
    train = db.execute('SELECT train FROM obs.run WHERE id=%s', (late,)).fetchone()[0]
    for minute, kind in [(-1, 'cancelled'), (1, 'none')]:
        db.execute('INSERT INTO obs.exception_rev VALUES (%s,\'2026-10-07\',%s,NULL,%s)',
                   (train, Jsonb({'kind': kind}), NOW + timedelta(minutes=minute)))
    write_periodic(db, cache, NOW, metrics=True)
    assert db.execute("SELECT value FROM sim.metric WHERE name='a1_active'").fetchone()[0] == 0
    write_periodic(db, cache, NOW + timedelta(minutes=2), metrics=True)
    assert db.execute("SELECT value FROM sim.metric WHERE name='a1_active' ORDER BY ts DESC LIMIT 1").fetchone()[0] == 1


@pytest.mark.db
def test_poll_age_and_missing_source(db):
    from msynd.engine.priority import write_periodic
    run = make_run(db)
    cache = {run: cached(db, run, output(arr=1000, next_times=(500, 510)))}
    write_periodic(db, cache, NOW, polls=True)
    assert db.execute('SELECT count(*) FROM obs.poll_queue').fetchone()[0] == 0
    write_periodic(db, cache, NOW, polls=True, S=S)
    assert db.execute('SELECT priority FROM obs.poll_queue').fetchone()[0] == 1
    insert_fetch(db, run_id=run, started_at=NOW - timedelta(minutes=60))
    insert_fetch(db, run_id=run, started_at=NOW + timedelta(minutes=1))
    write_periodic(db, cache, NOW, polls=True, S=S)
    assert db.execute('SELECT priority,updated_at FROM obs.poll_queue').fetchone() == (2, NOW)


@pytest.mark.db
def test_a1_inferred_origin_and_empty_metrics(db):
    from msynd.engine.priority import write_periodic
    run = make_run(db)
    out = output(dep=np.nan, arr=1000)
    out.endpoint_p50 = (10, 1000)
    cache = {run: cached(db, run, out)}
    write_periodic(db, cache, NOW, metrics=True)
    assert db.execute("SELECT value FROM sim.metric WHERE name='a1_active'").fetchone()[0] == 1
    write_periodic(db, {}, NOW + timedelta(minutes=1), metrics=True)
    assert db.execute('SELECT count(*) FROM sim.metric WHERE ts=%s AND value=0',
                      (NOW + timedelta(minutes=1),)).fetchone()[0] == 6


@pytest.mark.db
def test_loop_periodic_jobs_reuse_latest_replay(db_url, runs, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import msynd.engine.__main__ as loop
    from tests.engine.test_loop import fake_replay, tables, PARAMS
    import psycopg
    calls = []
    ticks = iter([0, 3601, 3601, 3601])
    monkeypatch.setattr(loop, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(loop, 'write_periodic', lambda conn, cache, now, **kw: calls.append((dict(cache), kw)))
    class Stop:
        done = False
        def is_set(self):
            return self.done
        def wait(self, timeout):
            self.done = True
    with psycopg.connect(db_url) as conn, ThreadPoolExecutor(4) as pool:
        loop.run_loop(conn, fake_replay, tables, PARAMS, pool=pool,
                      stop=Stop(), notifications=False)
        assert len(calls) == 1
        assert set(runs) <= set(calls[0][0])
        assert all(calls[0][1][key] for key in ('polls', 'issue', 'metrics'))
