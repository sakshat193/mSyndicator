"""Importable replay worker for spawn pools, including python -m msynd.engine."""
from datetime import datetime, timezone
import math

import numpy as np
import psycopg

from msynd.engine.frames import encode_run
from msynd.rng import run_key, v_hash


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
