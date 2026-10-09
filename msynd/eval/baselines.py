"""Point baselines and the frozen active-model offline replay baseline."""

from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from msynd.engine.cache import _predict
from msynd.engine.evidence import load_run_evidence
from msynd.engine.replay import ModelTables, Params, replay
from msynd.features import FEATURE_VER
from msynd.features.build import segment_queries
from msynd.models.registry import active
from msynd.rng import run_key, v_hash


def b2_eta(conn, run_id, station, occ, kind, t):
    row = conn.execute('''SELECT expected_ts,received_at FROM obs.ntes_eta_rev
        WHERE run_id=%s AND station=%s AND occ=%s AND kind=%s AND received_at<=%s
        ORDER BY received_at DESC LIMIT 1''', (run_id, station, occ, kind, t)).fetchone()
    return row[0] if row is not None and t - row[1] <= timedelta(minutes=30) else None


def b0_schedule(case):
    return case.sched_arr


def b1_persistence(case):
    return case.sched_arr + case.delay


def b2_forecast(case):
    return case.b2


def b3_active(conn, *, models=None, params=None):
    """Freeze model artifacts once; each forecast reads evidence/features at issue.

    Offline model validation uses zero leg bias, so later nightly residuals cannot
    enter historical forecasts (leg_bias has no availability timestamp).
    """
    models = active(conn) if models is None else models
    if params is None:
        data = json.loads((Path(__file__).resolve().parents[2]/"data/params.json").read_text())
        params = Params(s_tables=data["S"])
    frozen_versions = {"models": {role: model[0] for role, model in models.items()},
                       "feature_ver": FEATURE_VER, "bias_gen": 0,
                       "params_hash": hashlib.sha256(json.dumps(
                           {"S": params.s_tables, "v_max": params.v_max, "ess_min": params.ess_min},
                           sort_keys=True, allow_nan=False).encode()).hexdigest()}
    cached = None

    def forecast(case, *, n=None, replicate=0):
        nonlocal cached
        ev = load_run_evidence(conn, case.run_id, case.issued_at)
        queries = segment_queries(conn, case.run_id, case.issued_at)
        rows = queries.to_pylist()
        if not rows:
            raise ValueError("run has no scheduled segments")
        # Only the run's immutable schedule/route identity comes from loaded metadata.
        versions = ev.versions | frozen_versions
        ev = replace(ev, versions=versions)
        count = params.n if n is None else n
        key = (ev, json.dumps(rows, sort_keys=True, default=str), count, replicate)
        if cached is None or cached[0] != key:
            r_model, w_model, o_model = (models[role][1] for role in ("run", "dwell", "origin"))
            r, rlo, rhi = _predict(conn, r_model, queries)
            terminal = rows[-1] | {"idx": rows[-1]["idx"]+1, "sched_dwell": 0.}
            w, _, whi = _predict(conn, w_model, pa.Table.from_pylist(rows+[terminal], schema=queries.schema))
            origin = queries.slice(ev.events[0].halt_idx, 1).append_column("resched_min", pa.array([ev.resched_min]))
            o, olo, ohi = _predict(conn, o_model, origin, delay=0.)
            cold = any(getattr(model, "cold_start", False) for model in (r_model, w_model, o_model))
            tables = ModelTables(r, np.column_stack((rlo.max(axis=1), rhi.max(axis=1))), np.zeros(len(rows)),
                                 w, whi.max(axis=1), o[0, 0], np.array([olo[0, 0], ohi[0, 0]]), cold)
            # ponytail: retain all N paths for exact empirical scores; upgrade to streaming event marginals if eval memory warrants it.
            out = replay(ev, tables, replace(params, n=count, k=count),
                         int(run_key(v_hash(versions | {"N": count}), case.run_id, replicate)))
            cached = key, out.paths
        event = next((e for e in ev.events if e.halt_idx == case.halt_idx and e.kind == "arr"), None)
        last = max((e.event for e in ev.evidence if e.actual), default=-1)
        if event is None or event.idx <= last or (ev.stop_event is not None and event.idx > ev.stop_event):
            return None
        return cached[1][:, event.idx].copy()

    return forecast
