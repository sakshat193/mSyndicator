"""Point baselines and the frozen active-model offline replay baseline."""

from dataclasses import replace
from datetime import timedelta
import hashlib
import json

from msynd.engine.cache import tables_for
from msynd.engine.evidence import load_run_evidence
from msynd.engine.replay import Params, load_s_tables, replay
from msynd.features import FEATURE_VER
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
        params = Params(s_tables=load_s_tables())
    frozen_versions = {"models": {role: model[0] for role, model in models.items()},
                       "feature_ver": FEATURE_VER, "bias_gen": 0,
                       "params_hash": hashlib.sha256(json.dumps(
                           {"S": params.s_tables, "v_max": params.v_max, "ess_min": params.ess_min},
                           sort_keys=True, allow_nan=False).encode()).hexdigest()}
    cached = None

    def forecast(case, *, n=None, replicate=0):
        nonlocal cached
        ev = load_run_evidence(conn, case.run_id, case.issued_at)
        # Only the run's immutable schedule/route identity comes from loaded metadata.
        versions = ev.versions | frozen_versions
        ev = replace(ev, versions=versions)
        tables = tables_for(conn, ev, models, None, asof=case.issued_at)
        count = params.n if n is None else n
        key = (ev, count, replicate)
        if cached is None or cached[0] != key or cached[1] is not tables:
            # ponytail: retain all N paths for exact empirical scores; upgrade to streaming event marginals if eval memory warrants it.
            out = replay(ev, tables, replace(params, n=count, k=count),
                         int(run_key(v_hash(versions | {"N": count}), case.run_id, replicate)))
            cached = key, tables, out.paths
        event = next((e for e in ev.events if e.halt_idx == case.halt_idx and e.kind == "arr"), None)
        last = max((e.event for e in ev.evidence if e.actual), default=-1)
        if event is None or event.idx <= last or (ev.stop_event is not None and event.idx > ev.stop_event):
            return None
        return cached[2][:, event.idx].copy()

    return forecast
