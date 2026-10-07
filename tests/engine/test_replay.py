from dataclasses import replace

import numpy as np
import pytest

from msynd.engine.evidence import Evidence, RouteEvent, RunEvidence
from msynd.engine.replay import ModelTables, Params, replay
from msynd.dist import LEVELS
from msynd.rng import RESAMPLE, uniform


def fixture(passes=()):
    events = [RouteEvent(0, "A", 1, "dep", 0, 0, 100, 0, 0)]
    for i, minute in enumerate(passes, 1):
        events.append(RouteEvent(i, f"P{i}", 1, "pass", i, None, minute, minute-100, 0))
    idx = len(events)
    events.extend([
        RouteEvent(idx, "B", 1, "arr", idx, 1, 200, 100, 0),
        RouteEvent(idx+1, "B", 1, "dep", idx, 1, 210, 100, 1),
        RouteEvent(idx+2, "C", 1, "arr", idx+1, 2, 310, 200, 1),
    ])
    ev = RunEvidence(1, {}, tuple(events), (), 0, None)
    r = np.zeros((2, 16, 302)); r[:, :, 61] = 1  # R uniform [0,1)
    w = np.zeros((3, 16, 181)); w[:, :, 10] = 1  # W uniform [10,11)
    o = np.zeros(302); o[61] = 1
    models = ModelTables(r=r, r_tau=np.full((2, 2), 7.), bias=np.zeros(2),
                         w=w, w_tau=np.full(3, 7.), o=o, o_tau=np.array([4., 7.]), cold_start=False)
    params = Params(n=2000, k=2000, s_tables={
        "ntes_status": {"lag": [0, 10], "survival": [1, .5], "tau": 10, "eps": .05}
    })
    return ev, models, params


def test_invariants():
    ev, models, params = fixture()
    # Large negative R residuals exercise the physical floor.
    models.r[:] = 0; models.r[:, :, 0] = 1
    out = replay(ev, models, params, 7)
    assert np.all(np.diff(out.paths, axis=1) >= 0)
    assert np.all(out.paths[:, 2] >= out.paths[:, 1])
    for a, b in ((0, 1), (2, 3)):
        assert np.all(out.paths[:, b]-out.paths[:, a] >= 60*100/params.v_max - 2e-5)


def test_actual_window_all_particles_inside():
    ev, models, params = fixture()
    ev = replace(ev, evidence=(Evidence(1, True, 201, "ntes_status"),))
    out = replay(ev, models, params, 7)
    assert np.all((201 <= out.paths[:, 1]) & (out.paths[:, 1] < 202))
    assert np.isnan(out.halt_q[0]).all() and np.isnan(out.halt_q[1, 0]).all()


def test_infeasible_evidence_flagged_no_nan():
    ev, models, params = fixture()
    ev = replace(ev, evidence=(Evidence(1, True, 110, "ntes_status"),))
    out = replay(ev, models, params, 7)
    assert "data_invariant" in out.flags
    assert np.all(out.paths[:, 1] == 110.5) and np.isfinite(out.paths).all()


def test_deterministic_and_order_free():
    ev, models, params = fixture()
    evidence = (Evidence(0, True, 100, "ntes_status"), Evidence(1, False, 202, "ntes_status"))
    a = replay(replace(ev, evidence=evidence), models, params, 7)
    b = replay(replace(ev, evidence=tuple(reversed(evidence))), models, params, 7)
    c = replay(replace(ev, evidence=evidence), models, params, 7)
    for other in (b, c):
        for name in ("halt_q", "next_q", "paths"):
            assert getattr(a, name).tobytes() == getattr(other, name).tobytes()
        assert (a.pits, a.min_ess, a.flags) == (other.pits, other.min_ess, other.flags)


def test_ess_fallback_triggers(monkeypatch):
    from msynd.engine import kernel

    ev, models, params = fixture()
    models.o[:] = 0; models.o[61:161] = .01  # origin uniform [0,100)
    ev = replace(ev, evidence=(Evidence(1, True, 201, "ntes_status"),))
    attempts = []
    real = kernel.block

    def record(*args):
        attempts.append(args[-1])
        return real(*args)

    monkeypatch.setattr(kernel, "block", record)
    out = replay(ev, models, params, 7)
    assert attempts == [0, 1] and out.min_ess < params.ess_min
    assert "data_invariant" not in out.flags
    assert np.all((201 <= out.paths[:, 1]) & (out.paths[:, 1] < 202))


def test_outputs_shapes_and_strided_paths():
    ev, models, params = fixture()
    all_rows = replay(ev, models, params, 7)
    out = replay(ev, models, replace(params, k=32), 7)
    assert out.paths.shape == (32, 4) and out.paths.dtype == np.float32
    assert out.halt_q.shape == (3, 2, 7) and out.next_q.shape == (7,)
    np.testing.assert_array_equal(out.paths, all_rows.paths[np.arange(32)*2000//32])
    np.testing.assert_allclose(out.halt_q[1, 0], np.quantile(all_rows.paths[:, 1], [.05,.1,.25,.5,.75,.9,.95]))
    np.testing.assert_array_equal(out.next_q, out.halt_q[1, 0])
    assert np.isnan(out.halt_q[0, 0]).all() and np.isnan(out.halt_q[-1, 1]).all()


def test_extreme_delay_replay_no_nan():
    ev, models, params = fixture()
    models.o[:] = 0; models.o[-1] = 1
    ev = replace(ev, evidence=(Evidence(0, True, 1000, "ntes_status"),))
    out = replay(ev, models, params, 7)
    assert np.all((1000 <= out.paths[:, 0]) & (out.paths[:, 0] < 1001))
    assert np.isfinite(out.paths).all() and "data_invariant" not in out.flags


def test_surprise_flag():
    ev, models, params = fixture()
    models.o[:] = 0; models.o[61] = .9999; models.o[161] = .0001
    ev = replace(ev, evidence=(Evidence(0, True, 200, "ntes_status"),))
    out = replay(ev, models, params, 7)
    assert out.surprise and len(out.pits) == 1 and out.pits[0] > .999


def test_multi_pass_monotone():
    ev, models, params = fixture((125, 150, 175))
    models.r[:] = 0; models.r[:, :, 1] = .5; models.r[:, :, -1] = .5
    out = replay(ev, models, params, 7)
    assert np.isfinite(out.paths).all()
    assert np.all(np.diff(out.paths[:, :5], axis=1) >= 0)


def test_chained_remainder_parameters():
    ev, models, params = fixture((125, 150, 175))
    models.bias[0] = 8
    models.r[:] = 0
    for j in range(16):
        models.r[:, j, 61+10*j] = 1  # Q at level j is uniform [10j,10j+1).
    out = replay(ev, models, params, 7)
    expected = np.zeros(7, dtype=np.float32)
    expected[0] = 100 + uniform(np.uint64(7), 0, 0, 0)
    for e in range(1, 5):
        anchor, end = ev.events[e-1], ev.events[4]
        rem = max(.1, end.sched_off-anchor.sched_off)
        r = rem/100
        a = 1 if e == 4 else (ev.events[e].sched_off-anchor.sched_off)/rem
        delay = float(expected[e-1])-anchor.sched_off
        residual = np.interp(delay, LEVELS, 10*np.arange(16)) + uniform(np.uint64(7), 0, e, 0)
        expected[e] = expected[e-1] + a*max(60*(end.km-anchor.km)/params.v_max,
            rem+r*(8+residual))
    np.testing.assert_array_equal(out.paths[0, :5], expected[:5])


def test_passes_sharing_minutes_and_destination_floor():
    ev, models, params = fixture((125, 125, 200))
    out = replay(ev, models, params, 7)
    np.testing.assert_array_equal(out.paths[:, 1], out.paths[:, 2])
    uniforms = np.array([uniform(np.uint64(7), i, 4, 0) for i in range(params.n)])
    expected_arrival = (out.paths[:, 3].astype(float)+.1+.001*uniforms).astype(np.float32)
    np.testing.assert_array_equal(out.paths[:, 4], expected_arrival)
    assert np.isfinite(out.paths).all() and np.all(np.diff(out.paths, axis=1) >= 0)


def test_cancelled_tail_stops_forward_draws():
    ev, models, params = fixture()
    out = replay(replace(ev, stop_event=1), models, params, 7)
    assert np.isfinite(out.paths[:, :2]).all() and np.isnan(out.paths[:, 2:]).all()
    assert np.isnan(out.halt_q[1, 1]).all() and np.isnan(out.halt_q[2]).all()


def test_resched_applied_once_and_cold_start_flag():
    ev, models, params = fixture()
    out = replay(replace(ev, resched_min=60), replace(models, cold_start=True), params, 7)
    assert np.all((160 <= out.paths[:, 0]) & (out.paths[:, 0] < 161))
    assert "cold_start" in out.flags


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        Params(n=10, k=11)


def test_actual_weighting_gathers_upstream_ancestors():
    ev, models, params = fixture()
    models.o[:] = 0; models.o[61:161] = .01
    origins = np.array([100+100*uniform(np.uint64(7), i, 0, 0) for i in range(params.n)], dtype=np.float32)
    bases = origins.astype(float)+100
    weights = np.clip(202-bases, 0, 1)-np.clip(201-bases, 0, 1)
    cumulative = np.cumsum(weights/weights.sum())
    points = (np.arange(params.n)+uniform(np.uint64(7), RESAMPLE, 1, 0))/params.n
    ancestors = np.searchsorted(cumulative, points, side="right")
    out = replay(replace(ev, evidence=(Evidence(1, True, 201, "ntes_status"),)), models,
                 replace(params, ess_min=1e-6), 7)
    np.testing.assert_array_equal(out.paths[:, 0], origins[ancestors])
    assert not np.isnan(out.paths).any()


def test_publication_lag_grid_tail_and_pending_weighting():
    from msynd.engine.kernel import survival
    from msynd.engine.replay import _lag_table

    ev, models, params = fixture()
    lag = _lag_table(params.s_tables["ntes_status"])
    assert survival(-1, lag) == 1
    assert survival(5, lag) == .75
    assert survival(20, lag) == pytest.approx(.05+.45/np.e)
    prior = replay(ev, models, params, 7)
    # All proposals fall on the linear grid: S(205-T) = 1 - .05*(205-T).
    weights = 1-.05*(205-prior.paths[:, 1].astype(float))
    points = (np.arange(params.n)+uniform(np.uint64(7), RESAMPLE, 1, 0))/params.n
    ancestors = np.searchsorted(np.cumsum(weights/weights.sum()), points, side="right")
    out = replay(replace(ev, evidence=(Evidence(1, False, 205, "ntes_status"),)), models, params, 7)
    np.testing.assert_array_equal(out.paths[:, :2], prior.paths[ancestors, :2])
