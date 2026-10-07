"""Msgpack frame codec. Quantiles remain run-clock minutes; paths use UTC seconds."""
import msgpack
import numpy as np
from psycopg.types.json import Jsonb

from msynd.config import settings


def _quantiles(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return float(values) if np.isfinite(values) else None
    return [_quantiles(value) for value in values]


def encode_run(run_id, out, ev, base_ts):
    """ev is DB run metadata plus its RunEvidence; base_ts is a UTC datetime."""
    evidence = ev['evidence']
    end = len(evidence.events) if evidence.stop_event is None else evidence.stop_event + 1
    paths = np.asarray(out.paths, dtype=float)[:, :end]
    indices = np.array([e.route_idx for e in evidence.events[:end]])
    base = base_ts.timestamp()
    ticks = np.rint((paths * 60 + ev['origin_ts'] - base) / 6)
    if (paths.ndim != 2 or paths.shape[1] != end or not np.isfinite(ticks).all()
            or np.any(ticks < 0) or np.any(ticks > 65535)
            or np.any(indices < 0) or np.any(indices > 65535)):
        raise ValueError('display paths outside uint16 frame range')
    packed = np.empty((*paths.shape, 2), dtype='<u2')
    packed[:, :, 0], packed[:, :, 1] = indices, ticks
    last = ev['last_received']
    state = ev['lifecycle']
    if state in ('scheduled', 'running'):
        state = 'schedule_only' if last is None else (
            'observed' if (ev['asof'] - last).total_seconds() <= settings.f_minutes * 60 else 'stale')
    return {'run_id': run_id, 'state': state, 'versions': evidence.versions,
            'origin_ts': ev['origin_ts'], 'base_ts': base,
            # Age is frame.ts - last_received, so restarts don't change evidence payloads.
            'headline': {'last_received': last.timestamp() if last else None,
                         'surprise': bool(out.surprise)},
            'next_q': _quantiles(out.next_q), 'halt_q': _quantiles(out.halt_q),
            'min_ess': float(out.min_ess), 'flags': out.flags,
            'diverted_from': ev['diverted_from'], 'part_cancelled': ev['part_cancelled'],
            'path_shape': packed.shape, 'paths': packed.tobytes()}


def write_frame(conn, is_full, versions, runs, tombstones):
    """Insert only; the caller owns the publication/acknowledgement transaction."""
    bases = [run['base_ts'] for run in runs.values() if 'paths' in run]
    base = min(bases) if bases else None
    encoded = {}
    for run_id, run in sorted(runs.items()):
        run = dict(run)
        if 'paths' in run:
            paths = np.frombuffer(run['paths'], dtype='<u2').reshape(run['path_shape']).copy()
            ticks = paths[:, :, 1].astype(float) + (run['base_ts'] - base) / 6
            if not np.isfinite(ticks).all() or np.any(ticks > 65535):
                raise ValueError('display paths outside uint16 frame range')
            paths[:, :, 1] = np.rint(ticks)
            run.update(base_ts=base, paths=paths.tobytes())
        encoded[run_id] = run
    gen = conn.execute("SELECT nextval(pg_get_serial_sequence('sim.frame','gen'))").fetchone()[0]
    timestamp = conn.execute('SELECT clock_timestamp()').fetchone()[0]
    payload = msgpack.packb({'gen': gen, 'ts': timestamp.timestamp(), 'is_full': is_full,
                            'versions': versions, 'base_ts': base, 'runs': encoded,
                            'tombstones': sorted(tombstones)}, use_bin_type=True)
    conn.execute('INSERT INTO sim.frame(gen,ts,is_full,versions,payload) VALUES (%s,%s,%s,%s,%s)',
                 (gen, timestamp, is_full, Jsonb(versions), payload))
    return gen

