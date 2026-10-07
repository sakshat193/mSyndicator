"""Offline G8 prototype; JSON to stdout, gate reports belong to the monitor.

Replay timings exclude worker startup/JIT (reported as setup_seconds).
Rebuild includes pool dispatch and transfer of outputs. Frame timing includes
packing display paths, summaries and msgpack, without compression or DB I/O.
"""
import argparse
from contextlib import closing
from dataclasses import replace
import json
import math
import multiprocessing as mp
from time import perf_counter

import msgpack
import numpy as np

from msynd.dist import LEVELS, from_quantiles
from msynd.engine.evidence import Evidence, RouteEvent, RunEvidence
from msynd.engine.replay import ModelTables, Params, replay
from msynd.rng import run_key, v_hash


def synthetic_run():
    events = [RouteEvent(0, "S0", 1, "dep", 0, 0, 0, 0, 0)]
    evidence = []
    for seg in range(40):
        minute = 62*seg+60
        event = RouteEvent(len(events), f"S{seg+1}", 1, "arr", seg+1, seg+1,
                           minute, 80*(seg+1), seg)
        events.append(event)
        if seg < 20:
            evidence.append(Evidence(event.idx, seg % 2 == 0, minute+8, "ntes_status"))
        if seg < 39:
            events.append(RouteEvent(len(events), event.station, 1, "dep", seg+1, seg+1,
                                     minute+2, event.km, seg+1))
    # ponytail: synthetic LightGBM-shaped quantiles; measure inference with trained artifacts at G8 on the VM.
    r_rows = [from_quantiles(np.array([-8, -5, 0, 2, 5, 10, 15, 25, 40])+
                            .04*np.clip(level, -30, 120), True) for level in LEVELS]
    w_rows = [from_quantiles(np.array([.02, .1, .3, .6, 1, 2, 3, 5, 8])+
                            .005*np.clip(level, 0, 120), False) for level in LEVELS]
    o, lo, hi = from_quantiles([-10, -5, 0, 2, 5, 10, 20, 30, 45], True)
    models = ModelTables(
        np.tile([row[0] for row in r_rows], (40, 1, 1)),
        np.tile(r_rows[0][1:], (40, 1)), np.linspace(-1, 1, 40),
        np.tile([row[0] for row in w_rows], (41, 1, 1)),
        np.full(41, w_rows[0][2]), o, np.array([lo, hi]), False,
    )
    return RunEvidence(0, {"bench": "synthetic-v1"}, tuple(events), tuple(evidence), 0, None), models


def _initialise(n, ready):
    global _ev, _models, _params, _version
    _ev, _models = synthetic_run()
    _params = Params(n=n, k=min(32, n), s_tables={
        "ntes_status": {"lag": [0, 10, 30], "survival": [1, .5, .1], "tau": 60, "eps": .02}
    })
    _version = v_hash(_ev.versions)
    replay(_ev, _models, replace(_params, n=32, k=32), run_key(_version, 0))
    ready.put(True)


def _replay_one(run_id):
    ev = replace(_ev, run_id=run_id)
    start = perf_counter()
    out = replay(ev, _models, _params, run_key(_version, run_id))
    return run_id, perf_counter()-start, out


def build_frame(results, events):
    # ponytail: benchmark-only wire prototype; reuse Task 31's codec once it exists.
    base = math.floor(min(float(out.paths.min()) for _, _, out in results))
    runs = []
    for run_id, _, out in results:
        ticks = np.rint((out.paths.astype(float)-base)*10)
        if not np.isfinite(ticks).all() or np.any(ticks < 0) or np.any(ticks > 65535):
            raise ValueError("display times outside uint16 frame range")
        paths = np.empty((*out.paths.shape, 2), dtype="<u2")
        paths[:, :, 0] = [event.route_idx for event in events]
        paths[:, :, 1] = ticks
        runs.append({"run_id": run_id, "next_q": out.next_q.tolist(),
                     "halt_q": out.halt_q.tolist(), "min_ess": out.min_ess,
                     "surprise": out.surprise, "flags": out.flags,
                     "path_shape": paths.shape, "paths": paths.tobytes()})
    return msgpack.packb({"is_full": True, "base_min": base, "runs": runs}, use_bin_type=True)


def benchmark(runs=2500, workers=4, n=10_000):
    if runs <= 0 or workers <= 0:
        raise ValueError("runs and workers must be positive")
    Params(n=n, k=min(32, n))
    workers = min(workers, runs)
    ctx = mp.get_context("spawn")
    start = perf_counter()
    with closing(ctx.Queue()) as ready:
        with ctx.Pool(workers, initializer=_initialise, initargs=(n, ready)) as pool:
            for _ in range(workers):
                ready.get(timeout=300)
            setup = perf_counter()-start
            start = perf_counter()
            results = pool.map(_replay_one, range(1, runs+1), chunksize=1)
            rebuild = perf_counter()-start
    events = synthetic_run()[0].events
    start = perf_counter()
    frame = build_frame(results, events)
    frame_time = perf_counter()-start
    percentiles = np.quantile([seconds*1000 for _, seconds, _ in results], [.5, .95, .99])
    return {"runs": runs, "workers": workers, "n": n, "segments": 40,
            "events": len(events), "evidence_events": 20,
            "replay_ms": dict(zip(("p50", "p95", "p99"), percentiles.tolist())),
            "setup_seconds": setup, "rebuild_seconds": rebuild,
            "frame_build_seconds": frame_time, "frame_bytes": len(frame)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()
    if min(args.runs, args.workers, args.n) <= 0:
        parser.error("runs, workers and n must be positive")
    print(json.dumps(benchmark(args.runs, args.workers, args.n), allow_nan=False))


if __name__ == "__main__":
    main()
