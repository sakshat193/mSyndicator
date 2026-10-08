"""Periodic engine outputs from the latest replay, without new source requests."""
import math

import numpy as np
from numba import njit
from psycopg.types.json import Jsonb

from msynd.engine.kernel import survival
from msynd.engine.replay import _lag_table


@njit(cache=True)
def _published(now, times, table):
    total = 0.0
    for t in times:
        total += 1 - survival(now - t, table)
    return total / len(times)


def priority(run_out, S, now, age, T_max):
    """Times and ages are run-clock minutes; use the full next-arrival cloud."""
    if not all(math.isfinite(x) for x in (now, age, T_max)) or age < 0 or T_max <= 0:
        raise ValueError('invalid priority clock, age or T_max')
    table = _lag_table(S)
    times = np.asarray(run_out.next_times, dtype=float)
    width = float(run_out.next_q[5] - run_out.next_q[1])
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or not math.isfinite(width) or width < 0:
        raise ValueError('invalid next-arrival cloud or quantiles')
    published = _published(float(now), times, table)
    return published * width + age / T_max


def write_periodic(conn, cache, now, *, polls=False, issue=False, metrics=False,
                   S=None, T_max=30, F=30):
    """Caller owns the transaction; cache maps run IDs to (replay record, output)."""
    counts = dict.fromkeys(('a1_active', 'a1_discovered', 'a1_fresh',
                            'a1_active_1h', 'a1_discovered_1h', 'a1_cold'), 0)
    for run_id, (record, out) in cache.items():
        ev = record['evidence']
        metadata = conn.execute('''SELECT r.lifecycle,
            (SELECT max(received_at) FROM obs.event_rev WHERE run_id=r.id AND received_at<=%s),
            (SELECT max(started_at) FROM obs.fetch WHERE run_id=r.id AND started_at<=%s),
            (SELECT detail->>'kind' FROM obs.exception_rev
             WHERE train=r.train AND start_date=r.start_date AND received_at<=%s
             ORDER BY received_at DESC LIMIT 1)
            FROM obs.run r WHERE id=%s''', (now, now, now, run_id)).fetchone()
        if metadata is None or metadata[0] not in ('scheduled', 'running') or metadata[3] == 'cancelled':
            continue
        clock = (now.timestamp() - record['origin_ts']) / 60
        end = len(ev.events) - 1 if ev.stop_event is None else ev.stop_event
        origin, destination = ev.events[0], ev.events[end]
        actual = {e.event: e.minute for e in ev.evidence if e.actual}
        dep, arr = out.endpoint_p50 if out.endpoint_p50 is not None else (
            out.halt_q[origin.halt_idx, 1, 3],
            out.halt_q[destination.halt_idx, 0, 3] if destination.halt_idx is not None else math.nan)
        dep, arr = actual.get(origin.idx, dep), actual.get(destination.idx, arr)
        # Observed endpoints count as past even though their displayed halt_q is NaN.
        active = (origin.idx in actual or dep <= clock) and destination.idx not in actual and arr > clock
        last, fetched = metadata[1:3]
        if metrics and active:
            counts['a1_active'] += 1
            counts['a1_discovered'] += last is not None
            counts['a1_fresh'] += last is not None and (now - last).total_seconds() <= F * 60
            old = dep <= clock - 60
            counts['a1_active_1h'] += old
            counts['a1_discovered_1h'] += old and last is not None
            counts['a1_cold'] += 'cold_start' in out.flags
        if polls and S is not None and out.next_times is not None:
            # ponytail: no fetch gets T_max age; CP-3 may replace the 30-minute default.
            age = (now - fetched).total_seconds() / 60 if fetched else T_max
            value = priority(out, S, clock, age, T_max)
            conn.execute('''INSERT INTO obs.poll_queue VALUES (%s,%s,'published_width_age',%s)
                ON CONFLICT (run_id) DO UPDATE SET priority=excluded.priority,
                reason=excluded.reason,updated_at=excluded.updated_at''', (run_id, value, now))
        if issue and metadata[0] == 'running':
            # ponytail: issue only finite remaining arrival halts; CP-3 decides any wider issue policy.
            halts = [e for e in ev.events[:end + 1] if e.kind == 'arr'
                     and np.isfinite(out.halt_q[e.halt_idx, 0]).all()]
            if not halts:
                continue
            eta = conn.execute('''SELECT DISTINCT ON (station,occ,kind) station,occ,kind,expected_ts,received_at
                FROM obs.ntes_eta_rev WHERE run_id=%s AND received_at<=%s
                ORDER BY station,occ,kind,received_at DESC''', (run_id, now)).fetchall()
            # ponytail: A3 fixes per-key confirmation at 30 minutes; change only with its contract.
            b2 = {(station, occ, kind): (ts.timestamp() - record['origin_ts']) / 60
                  for station, occ, kind, ts, received in eta
                  if (now - received).total_seconds() <= 30 * 60}
            conn.execute('''INSERT INTO sim.forecast_issue(run_id,issued_at,versions,halt_idx,q,b2,cold_start)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id,issued_at) DO NOTHING''',
                (run_id, now, Jsonb(ev.versions), [e.halt_idx for e in halts],
                 np.concatenate([out.halt_q[e.halt_idx, 0] for e in halts]).tolist(),
                 [b2.get((e.station, e.occ, 'arr'), math.nan) for e in halts],
                 'cold_start' in out.flags))
    if metrics:
        for name, value in counts.items():
            conn.execute('''INSERT INTO sim.metric VALUES (%s,%s,%s)
                ON CONFLICT (ts,name) DO UPDATE SET value=excluded.value''', (now, name, value))
