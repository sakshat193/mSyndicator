import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import msgpack
import numpy as np


def test_bench_smoke_reports_all_measurements_without_gate_file(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    result = subprocess.run(
        [sys.executable, "-m", "msynd.engine.bench", "--runs", "20", "--n", "1000"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert (report["runs"], report["workers"], report["n"], report["segments"]) == (20, 4, 1000, 40)
    assert report["events"] == 80 and report["evidence_events"] == 20
    p = report["replay_ms"]
    assert 0 < p["p50"] <= p["p95"] <= p["p99"]
    for key in ("setup_seconds", "rebuild_seconds", "frame_build_seconds"):
        assert math.isfinite(report[key]) and report[key] > 0
    assert report["frame_bytes"] > 20*32*80*4
    assert not list(tmp_path.iterdir())


def test_bench_workload_has_delay_dependent_masses_and_mixed_first_half_evidence():
    from msynd.engine.bench import synthetic_run

    ev, models = synthetic_run()
    assert len(models.r) == 40 and models.r.shape[1:] == (16, 302)
    assert models.w.shape == (41, 16, 181)
    assert not np.array_equal(models.r[:, 0], models.r[:, -1])
    assert np.allclose(models.r.sum(axis=-1), 1)
    assert np.allclose(models.w.sum(axis=-1), 1)
    assert np.isclose(models.o.sum(), 1)
    assert {e.actual for e in ev.evidence} == {True, False}
    assert len(ev.evidence) == 20
    assert all(ev.events[e.event].seg < 20 for e in ev.evidence)


def test_bench_frame_encodes_uint16_route_and_tenths():
    from msynd.engine.bench import build_frame
    from msynd.engine.evidence import RouteEvent
    from msynd.engine.replay import ReplayOut

    events = (RouteEvent(0, "A", 1, "dep", 0, 0, 100, 0, 0),
              RouteEvent(1, "B", 1, "arr", 1, 1, 101, 1, 0))
    out = ReplayOut(np.zeros((2, 2, 7)), np.arange(7.), np.array([[100., 100.2]]),
                    .8, False, (), ())
    frame = msgpack.unpackb(build_frame([(7, .01, out)], events), raw=False)
    assert frame["base_min"] == 100
    run = frame["runs"][0]
    assert run["run_id"] == 7 and run["path_shape"] == [1, 2, 2]
    assert run["paths"] == struct.pack("<HHHH", 0, 0, 1, 2)
    assert run["next_q"] == list(range(7)) and run["min_ess"] == .8
