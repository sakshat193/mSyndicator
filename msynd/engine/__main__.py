"""Offline-capable engine loop; Task 27 supplies the replay model-table source.

consume/rebuild/run_loop own their connection transactions. model_source(conn, ev)
returns ModelTables; replay_fn has the existing replay(ev, models, params, key)
signature. Neither provider clients nor a registry are constructed here.
"""
from dataclasses import replace
from datetime import datetime, time, timezone
import hashlib
import json
import logging
import math
import multiprocessing as mp
from time import monotonic, sleep

import msgpack
import numpy as np
import psycopg
from psycopg.rows import dict_row

from msynd.db.dirty import mark_dirty
from msynd.engine.evidence import load_run_evidence
from msynd.engine.frames import encode_run, write_frame
from msynd.engine.priority import write_periodic
from msynd.engine.replay import Params
from msynd.labels import IST
from msynd.rng import run_key, v_hash

log = logging.getLogger(__name__)
ACTIVE = "lifecycle IN ('scheduled','running')"


def _versions(conn):
    models = conn.execute('SELECT id,role,feature_ver FROM ml.model WHERE active ORDER BY role').fetchall()
    bias = conn.execute('SELECT coalesce(max(b.gen),0) FROM ml.leg_bias b JOIN ml.model m ON m.id=b.model_id WHERE m.active').fetchone()[0]
    return {'models': {role: model for model, role, _ in models}, 'bias_gen': bias,
            'feature_ver': max((feature for _, _, feature in models), default=None)}


def _job(args):
    record, models, params, replay_fn = args
    ev = record['evidence']
    try:
        out = replay_fn(ev, models, params, int(run_key(v_hash(ev.versions), ev.run_id)))
        end = len(ev.events) if ev.stop_event is None else ev.stop_event + 1
        paths = np.asarray(out.paths)[:, :end]
        base = math.floor((record['origin_ts'] + float(np.min(paths)) * 60) / 6) * 6
        return ev.run_id, encode_run(ev.run_id, out, record, datetime.fromtimestamp(base, timezone.utc)), (record, out)
    except (psycopg.Error, OSError, MemoryError):
        raise
    except Exception as exc:
        return ev.run_id, {'state': 'error', 'error': str(exc), 'versions': ev.versions}, None


def _prepare(conn, run_ids, model_source, params, replay_fn):
    jobs, runs, tombstones = [], {}, []
    asof = conn.execute('SELECT clock_timestamp()').fetchone()[0]
    params_hash = hashlib.sha256(json.dumps(
        {'s_tables': params.s_tables, 'v_max': params.v_max, 'ess_min': params.ess_min},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for run_id in run_ids:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute('''SELECT r.*,
                (SELECT max(t) FROM (
                    SELECT received_at AS t FROM obs.event_rev WHERE run_id=r.id AND received_at<=%s
                    UNION ALL SELECT received_at FROM obs.pending_rev WHERE run_id=r.id AND received_at<=%s
                ) receipts) AS last_received FROM obs.run r WHERE id=%s''', (asof, asof, run_id))
            record = cur.fetchone()
        if record is None or record['lifecycle'] == 'completed':
            tombstones.append(run_id)
            continue
        if record['lifecycle'] not in ('scheduled', 'running'):
            runs[run_id] = {'state': record['lifecycle']}
            continue
        try:
            ev = load_run_evidence(conn, run_id, asof)
            # Versions must cover the actual replay N and parameters supplied by the caller.
            ev = replace(ev, versions=ev.versions | {'N': params.n, 'params_hash': params_hash})
            models = model_source(conn, ev)
            record.update(evidence=ev, asof=asof,
                          origin_ts=datetime.combine(record['start_date'], time(), IST).timestamp())
            jobs.append((record, models, params, replay_fn))
        except (psycopg.Error, OSError, MemoryError):
            raise
        except Exception as exc:
            runs[run_id] = {'state': 'error', 'error': str(exc)}
    return jobs, runs, tombstones


def _publish(conn, is_full, versions, runs, tombstones, dirty=()):
    with conn.transaction():
        gen = write_frame(conn, is_full, versions, runs, tombstones)
        errors = sum(run['state'] == 'error' for run in runs.values())
        if errors:
            conn.execute("INSERT INTO sim.metric VALUES (clock_timestamp(),'replay_error',%s)", (errors,))
        for run_id, ver in dirty:
            conn.execute('DELETE FROM obs.dirty_run WHERE run_id=%s AND ver=%s', (run_id, ver))
        conn.execute("DELETE FROM sim.frame WHERE ts < clock_timestamp()-interval '15 minutes'")
        conn.execute('NOTIFY sim_frame')
    return gen


def _replay_jobs(jobs, runs, pool):
    results = map(_job, jobs) if pool is None else pool.map(_job, jobs)
    cached = {}
    for run_id, frame, result in results:
        runs[run_id] = frame
        if result is not None:
            cached[run_id] = result
    return cached


def _update_cache(cache, runs, tombstones, cached):
    if cache is not None:
        for run_id in (*runs, *tombstones):
            cache.pop(run_id, None)
        cache.update(cached)


def consume(conn, replay_fn, model_source, params=Params(), *, pool=None, cache=None):
    """Publish up to 200 dirty runs, then acknowledge only their captured versions."""
    with conn.transaction():
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        dirty = conn.execute('SELECT run_id,ver FROM obs.dirty_run ORDER BY since,run_id LIMIT 200').fetchall()
        if not dirty:
            return None
        versions = _versions(conn)
        jobs, runs, tombstones = _prepare(conn, [run for run, _ in dirty], model_source, params, replay_fn)
    cached = _replay_jobs(jobs, runs, pool)
    gen = _publish(conn, False, versions, runs, tombstones, dirty)
    _update_cache(cache, runs, tombstones, cached)
    return gen


def rebuild(conn, replay_fn, model_source, params=Params(), *, pool=None, cache=None):
    """Startup full replay; dirty rows are left for version-checked consume."""
    with conn.transaction():
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        ids = [row[0] for row in conn.execute(f'SELECT id FROM obs.run WHERE {ACTIVE} ORDER BY id')]
        versions = _versions(conn)
        jobs, runs, tombstones = _prepare(conn, ids, model_source, params, replay_fn)
    cached = _replay_jobs(jobs, runs, pool)
    gen = _publish(conn, True, versions, runs, tombstones)
    _update_cache(cache, runs, tombstones, cached)
    return gen


def _read_frame(conn, gen):
    with conn.transaction():
        data = conn.execute('SELECT payload FROM sim.frame WHERE gen=%s', (gen,)).fetchone()[0]
    return msgpack.unpackb(bytes(data), raw=False, strict_map_key=False)


def run_loop(conn, replay_fn, model_source, params=Params(), *, pool=None,
             stop=None, notifications=True, ready=None, source='ntes_status', T_max=30, F=30):
    """LISTEN before reconciling; one-second poll backs up notification delivery."""
    if pool is None:
        with mp.get_context('spawn').Pool(4) as workers:
            return run_loop(conn, replay_fn, model_source, params, pool=workers,
                            stop=stop, notifications=notifications, ready=ready,
                            source=source, T_max=T_max, F=F)
    if notifications:
        conn.execute('LISTEN dirty')
        conn.execute('LISTEN model_changed')
        conn.commit()
    cache = {}
    current = _read_frame(conn, rebuild(conn, replay_fn, model_source, params, pool=pool, cache=cache))
    runs, versions = current['runs'], current['versions']
    started = monotonic()
    full_at, poll_at, issue_at, metric_at = started + 300, started + 30, started + 3600, started + 60
    if ready is not None:
        ready.set()
    changed = False
    while stop is None or not stop.is_set():
        try:
            with conn.transaction():
                latest = _versions(conn)
                if changed or latest != versions:
                    mark_dirty(conn, [row[0] for row in conn.execute(f'SELECT id FROM obs.run WHERE {ACTIVE}')])
            versions, changed = latest, False
            gen = consume(conn, replay_fn, model_source, params, pool=pool, cache=cache)
            if gen is not None:
                delta = _read_frame(conn, gen)
                runs.update(delta['runs'])
                for run_id in delta['tombstones']:
                    runs.pop(run_id, None)
            tick = monotonic()
            if tick >= full_at:
                _publish(conn, True, versions, runs, [])
                full_at = tick + 300
            polls, issue, metrics = tick >= poll_at, tick >= issue_at, tick >= metric_at
            if polls or issue or metrics:
                # ponytail: CP-3 may revise source='ntes_status', T_max=30 and F=30; absent S skips polls.
                with conn.transaction():
                    now = conn.execute('SELECT clock_timestamp()').fetchone()[0]
                    write_periodic(conn, cache, now, polls=polls, issue=issue, metrics=metrics,
                                   S=params.s_tables.get(source), T_max=T_max, F=F)
                if polls:
                    poll_at = tick + 30
                if issue:
                    issue_at = tick + 3600
                if metrics:
                    metric_at = tick + 60
            # Drain backlog immediately instead of adding a poll delay per batch.
            if gen is not None:
                continue
        except (psycopg.Error, OSError, MemoryError):
            conn.rollback()
            log.exception('transient engine failure; dirty versions retained')
        if notifications:
            notices = list(conn.notifies(timeout=1, stop_after=1))
            changed = any(notice.channel == 'model_changed' for notice in notices)
        elif stop is not None:
            stop.wait(1)
        else:
            sleep(1)

