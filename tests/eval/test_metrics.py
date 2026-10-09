from datetime import date, timedelta

import numpy as np
import pytest

from tests.eval.test_harness import case


def test_empirical_crps_and_pit_atoms():
    from msynd.eval.metrics import sample_metrics

    result = sample_metrics([0., 10.], 5.)
    assert result["crps"] == 2.5
    assert result["pit"] == .5
    assert sample_metrics([5., 5.], 5.)["pit"] == .5
    assert sample_metrics([5.], 7.)["crps"] == 2.


def test_bootstrap_ci_excludes_zero_on_clear_gain():
    from msynd.eval.metrics import block_bootstrap

    ci = block_bootstrap({date(2026, 9, 21) + timedelta(days=d): [2., 3.] for d in range(14)})
    assert ci.lo == ci.hi == 2.5
    # Whole dates, not individual cases, must be drawn together.
    blocks = {date(2026, 9, 21): [-10.] * 100, date(2026, 9, 22): [10.] * 100}
    ci = block_bootstrap(blocks)
    assert ci.lo == -10. and ci.hi == 10.
    assert ci == block_bootstrap(blocks)


@pytest.mark.parametrize("blocks,reps", [({}, 2000), ({1: [np.nan]}, 2000), ({1: []}, 2000), ({1: [1.]}, 0)])
def test_invalid_bootstrap_input(blocks, reps):
    from msynd.eval.metrics import block_bootstrap

    with pytest.raises(ValueError):
        block_bootstrap(blocks, reps)


def promotion_scores(candidate_loss=9., regression=False, coverage=.8):
    from msynd.eval.harness import score

    # Metric records below supply hand-derived losses and nominal coverage.
    samples = np.arange(101.)
    cases = [case(d, halt_idx=i, station=f"S{i}", actual=float(i), train_class="RAJ" if i == 1 else "EXP")
             for d in range(14) for i in range(2)]
    active = score(cases, lambda c: samples, disruption_days=set())
    candidate = score(cases, lambda c: samples, disruption_days=set())
    # Promotion consumes metric records, so isolate the rule from forecast quality.
    for a, b in zip(active.records, candidate.records):
        a["crps"] = 10.
        b["crps"] = 10.6 if regression and b["class"] == "RAJ" else candidate_loss
        b["coverage"] = {0.8: coverage, 0.9: .9}
    return candidate, active


def test_promotion_rule_rejects_class_regression():
    from msynd.eval.harness import promotion_ok

    candidate, active = promotion_scores(regression=True)
    decision = promotion_ok(candidate, active, now=date(2026, 10, 5))
    assert not decision.accepted
    assert any("RAJ" in reason for reason in decision.reasons)


@pytest.mark.parametrize("loss,coverage,accepted", [(9., .8, True), (9.81, .8, False), (9., .74, False)])
def test_promotion_gain_and_calibration(loss, coverage, accepted):
    from msynd.eval.harness import promotion_ok

    candidate, active = promotion_scores(candidate_loss=loss, coverage=coverage)
    assert promotion_ok(candidate, active, now=date(2026, 10, 5)).accepted is accepted


def test_promotion_requires_matching_cases_and_recent_full_weeks():
    from msynd.eval.harness import promotion_ok

    candidate, active = promotion_scores()
    candidate.records.pop()
    assert not promotion_ok(candidate, active, now=date(2026, 10, 5)).accepted
    candidate, active = promotion_scores()
    assert not promotion_ok(candidate, active, now=date(2026, 11, 2)).accepted


def test_n_curve_selects_smallest_n_using_tail_mc_error():
    from msynd.eval.harness import n_curve

    # Independent replicate offsets with exactly 1/sqrt(N) scale, checked at both tails.
    def forecast(c, *, n, replicate):
        return np.arange(101.) + (replicate - 9.5) * 10. / np.sqrt(n)

    assert n_curve([case(horizon=360)], Ns=(100000, 1000, 10000, 5000), forecaster=forecast) == 100000
    assert n_curve([case(horizon=360)], Ns=(1000,), forecaster=forecast) is None
    with pytest.raises(ValueError, match="6 h"):
        n_curve([case(horizon=120)], forecaster=forecast)


def test_interp_error_against_direct_delay_calls():
    import pyarrow as pa
    from msynd.eval.harness import interp_error

    class CurvedModel:
        role = "run"

        def predict_distribution(self, queries):
            delays = queries["d"].to_numpy()
            masses = np.zeros((len(delays), 302))
            masses[np.arange(len(delays)), 61 + np.minimum(delays**2, 100).astype(int)] = 1.
            shape = (len(delays) // 16, 16)
            return masses.reshape(*shape, 302), np.ones(shape), np.ones(shape)

    query = pa.table({"class": ["EXP"], "d": [0.]})
    result = interp_error(CurvedModel(), query, [2.5])
    assert result["max"] == pytest.approx(6.5)
    assert result["mean"] == pytest.approx(6.5)
