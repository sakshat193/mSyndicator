from dataclasses import replace
import json

import numpy as np
import pytest

from tests.g7.reference import (
    CASE_NAMES, engine_replicates, enumerate_posterior, lw_reference, make_case,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def references():
    return {}


@pytest.mark.parametrize("n", [10_000, 100_000])
@pytest.mark.parametrize("name", CASE_NAMES)
def test_g7_conditioning(name, n, references, record_property):
    case = make_case(name)
    if name not in references:
        if case.exact:
            reference = enumerate_posterior(case)
            references[name] = reference, {key: 0.0 for key in reference}
        else:
            references[name] = lw_reference(case, n=2_000_000, seed=1234)
    ref, se_ref = references[name]
    estimates, se_rep = engine_replicates(case, n=n, reps=20)
    numbers = {key: {"engine": estimates[key], "reference": value,
                    "se_rep": se_rep[key], "se_ref": se_ref[key],
                    "tolerance": 4*np.hypot(se_rep[key], se_ref[key])}
               for key, value in ref.items()}
    record_property("g7", json.dumps({"case": name, "n": n, "metrics": numbers}))
    assert len([key for key in numbers if key.startswith("cdf_")]) >= 5
    for key, value in ref.items():
        assert abs(estimates[key]-value) <= numbers[key]["tolerance"], (name, n, key, numbers[key])
    if name == "a_upstream":
        assert ref["origin_hi"] == pytest.approx(2/3, abs=1e-14)
    if name == "k_repeated_pending":
        exact = enumerate_posterior(case)["origin_hi"]
        assert exact == pytest.approx(.731059, abs=5e-7)
        tolerance = 4*se_rep["origin_hi"]
        assert abs(estimates["origin_hi"]-exact) <= tolerance
        assert abs(estimates["origin_hi"]-.880797) > tolerance
    if name in ("e_infeasible", "g_degenerate"):
        assert estimates["_fallback_rate"] == 1
    if name == "e_infeasible":
        assert estimates["_invariant_rate"] == 1
    else:
        assert estimates["_invariant_rate"] == 0
    if name == "i_floor_atom":
        assert abs(estimates["floor_atom"]-2/3) <= 4*se_rep["floor_atom"]
    if name == "c_pass":
        independent_key = (name, "independent_subsegments")
        if independent_key not in references:
            references[independent_key] = lw_reference(replace(case, independent_pass=True), n=2_000_000, seed=1234)
        independent, independent_se = references[independent_key]
        limit = max(2, .1*(independent["q90"]-independent["q10"]))
        record_property("a7", json.dumps({
            "limit": limit,
            "mean_error": {key: estimates[key]-independent[key] for key in ("q10", "q50", "q90")},
            "mc_spread": {key: se_rep[key]*np.sqrt(20) for key in ("q10", "q50", "q90")},
            "se_ref": {key: independent_se[key] for key in ("q10", "q50", "q90")},
        }))
        assert all(abs(estimates[key]-independent[key]) <= limit for key in ("q10", "q50", "q90"))
    if name in ("l_negative_origin", "m_rescheduled"):
        expected = 150 if name == "l_negative_origin" else 130
        assert ref["q50"] == pytest.approx(expected, abs=1e-10)


@pytest.mark.db
def test_repeated_pending_uses_assembled_latest_marker(db):
    from datetime import datetime, timedelta
    from msynd.engine.evidence import Evidence, load_run_evidence
    from msynd.labels import IST
    from tests.helpers import make_run

    run = make_run(db, halts=[("MMCT", None, 0, 0), ("BRC", 70, 71, 8/3), ("NDLS", 141, None, 16/3)])
    base = datetime(2026, 10, 7, tzinfo=IST)
    for minute, source in ((29, "ntes_board"), (30, "ntes_status")):
        db.execute("""insert into obs.pending_rev(run_id,station,occ,kind,source,received_at)
            values (%s,'BRC',1,'arr',%s,%s)""", (run, source, base+timedelta(minutes=minute)))
    loaded = load_run_evidence(db, run)
    assert loaded.evidence == (Evidence(1, False, 30, "ntes_status"),)
    case = make_case("k_repeated_pending")
    assembled = replace(case, ev=replace(case.ev, evidence=loaded.evidence))
    direct, direct_se = engine_replicates(case, n=10_000, reps=20)
    merged, merged_se = engine_replicates(assembled, n=10_000, reps=20)
    assert direct == merged and direct_se == merged_se


def test_reference_exact_answers_and_probability_mass():
    assert enumerate_posterior(make_case("a_upstream"))["origin_hi"] == pytest.approx(2/3)
    assert enumerate_posterior(make_case("k_repeated_pending"))["origin_hi"] == pytest.approx(1/(1+np.exp(-1)))
    assert make_case("h_rare").models.o[61] == 5e-7


def test_reference_is_independent_of_production_distribution(monkeypatch):
    from msynd.engine import kernel
    from msynd import dist

    def forbidden(*args, **kwargs):
        raise AssertionError("reference called production probability machinery")

    for name in ("eff_for", "draw", "block", "prior"):
        monkeypatch.setattr(kernel, name, forbidden)
    for name in ("eff_quantile", "eff_cdf_left", "eff_window_mass", "eff_sample_window", "_cdf", "_sample"):
        monkeypatch.setattr(dist, name, forbidden)
    values, errors = lw_reference(make_case("b_bridge"), n=2_000_000, seed=1234)
    assert all(np.isfinite(list(values.values()))) and all(np.isfinite(list(errors.values())))
    assert values["_window_fraction"] == 1 and errors["_window_fraction"] == 0
