"""Batched model inference; retain a bounded, latest CDF entry per run."""

from collections import OrderedDict
import json

import numpy as np
import pyarrow as pa

from msynd.dist import LEVELS
from msynd.engine.replay import ModelTables
from msynd.features.build import segment_queries, with_delay


_cache = OrderedDict()


def _predict(conn, artifact, queries, delay=None):
    expanded = queries.take(pa.array(np.repeat(np.arange(len(queries)), 16)))
    expanded = with_delay(conn, expanded, np.tile(LEVELS, len(queries)) if delay is None else delay)
    if getattr(artifact, "encoders", None) is not None:
        expanded = artifact.encoders.transform(expanded)
    if hasattr(artifact, "feature_names"):
        expanded = expanded.select(artifact.feature_names)
    return artifact.predict_distribution(expanded)


def tables_for(conn, ev, active, bias_gen, asof=None):
    # Current replay context, read from the caller's DB snapshot; no live source access.
    if asof is None:
        asof = conn.execute("SELECT statement_timestamp()").fetchone()[0]
    shared = segment_queries(conn, ev.run_id, asof)
    rows = shared.to_pylist()
    if not rows:
        raise ValueError("run has no scheduled segments")
    key = (json.dumps(ev.versions, sort_keys=True), bias_gen,
           # Unsaved candidates share id None with the provisional priors; the object tells them apart.
           tuple((role, active[role][0], id(active[role][1])) for role in ("run", "dwell", "origin")),
           json.dumps(rows, sort_keys=True, default=str), ev.events, ev.resched_min)
    cached = _cache.get(ev.run_id)
    if cached is not None and cached[0] == key:
        _cache.move_to_end(ev.run_id)
        return cached[1]
    r_model, w_model, o_model = (active[role][1] for role in ("run", "dwell", "origin"))
    r, r_lo, r_hi = _predict(conn, r_model, shared)
    # Keep schedule indices, including a cancelled head and unused destination W slot.
    terminal = rows[-1] | {"idx": rows[-1]["idx"]+1, "sched_dwell": 0.}
    w, _, w_hi = _predict(conn, w_model, pa.Table.from_pylist(rows+[terminal], schema=shared.schema))
    origin_idx = ev.events[0].halt_idx
    origin = shared.slice(origin_idx, 1)
    origin = origin.append_column("resched_min", pa.array([ev.resched_min]))
    o, o_lo, o_hi = _predict(conn, o_model, origin, delay=0.)
    bias = np.zeros(len(rows))
    if bias_gen is not None:
        model_id = active["run"][0]
        shifts = {(a, b, hour): value for a, b, hour, value in conn.execute("""
            SELECT from_stn,to_stn,hour_bucket,bias_min FROM ml.leg_bias
            WHERE model_id=%s AND gen=%s
        """, (model_id, bias_gen))}
        for i, row in enumerate(rows):
            bias[i] = shifts.get((row["from_stn"], row["to_stn"], int(row["sched_dep"] % 1440 // 180)), 0.)
    # ponytail: max tau across levels; upgrade to per-level tau in ModelTables/kernel if validation warrants it.
    cold_start = any(getattr(m, "cold_start", False) or
                     (hasattr(m, "pooled") and any(str(row["class"]) not in m.pooled for row in rows))
                     for m in (r_model, w_model, o_model))
    tables = ModelTables(r, np.column_stack((r_lo.max(axis=1), r_hi.max(axis=1))), bias,
                         w, w_hi.max(axis=1), o[0, 0], np.array([o_lo[0, 0], o_hi[0, 0]]),
                         cold_start)
    _cache[ev.run_id] = (key, tables)
    _cache.move_to_end(ev.run_id)
    # ponytail: retain 256 runs per process; increase only if measured eviction costs warrant the memory.
    if len(_cache) > 256:
        _cache.popitem(last=False)
    return tables
