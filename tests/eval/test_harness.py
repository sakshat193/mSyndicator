from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from msynd.labels import IST
from tests.helpers import insert_event, make_run


BASE = datetime(2026, 9, 21, tzinfo=IST)


def case(day=0, **changes):
    from msynd.eval.harness import Case

    values = dict(run_id=day + 1, service_date=(BASE + timedelta(days=day)).date(),
                  issued_at=BASE + timedelta(days=day, minutes=540), halt_idx=1,
                  station="B", occ=1, sched_arr=660., delay=10., actual=670.,
                  train_class="EXP", zone="WR", horizon=120, b2=670.)
    return Case(**(values | changes))


@pytest.mark.db
def test_horizon_bucket_independent_of_outcome(db):
    from msynd.eval.harness import asof_cases

    run = make_run(db, start="2026-09-21", halts=[("A", None, 540, 0),
                   ("B", 660, 665, 100), ("A", 1000, None, 300)])
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (run,))
    insert_event(db, run, "A", 1, "dep", BASE + timedelta(minutes=550), BASE + timedelta(minutes=551))
    insert_event(db, run, "B", 1, "arr", BASE + timedelta(minutes=700), BASE + timedelta(minutes=701))
    insert_event(db, run, "A", 2, "arr", BASE + timedelta(minutes=1100), BASE + timedelta(minutes=1101))
    start, end = BASE + timedelta(minutes=570), BASE + timedelta(minutes=601)
    before = list(asof_cases(db, start, end))
    assert [(c.issued_at.minute, c.halt_idx, c.occ, c.horizon, c.delay) for c in before] == [
        (30, 1, 1, 120, 10.), (30, 2, 2, 360, 10.), (0, 2, 2, 360, 10.)]
    # A later correction must change only the matured outcome, not the issue-time bucket/context.
    insert_event(db, run, "B", 1, "arr", BASE + timedelta(minutes=900), BASE + timedelta(minutes=1200))
    after = list(asof_cases(db, start, end))
    assert [replace(c, actual=0.) for c in before] == [replace(c, actual=0.) for c in after]
    assert before[0].actual == 700. and after[0].actual == 900.


@pytest.mark.parametrize("minutes,want", [(22.5, 30), (37.5, 30), (90, 120),
    (150, 120), (270, 360), (450, 360), (540, 720), (900, 720), (89, None), (-1, None)])
def test_horizon_boundaries(minutes, want):
    from msynd.eval.harness import horizon_bucket

    assert horizon_bucket(minutes) == want


@pytest.mark.db
def test_early_observed_departure_gets_cut_before_scheduled_departure(db):
    from msynd.eval.harness import asof_cases

    run = make_run(db, start="2026-09-21", halts=[("A", None, 555, 0), ("B", 660, None, 100)])
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (run,))
    insert_event(db, run, "A", 1, "dep", BASE + timedelta(minutes=540), BASE + timedelta(minutes=540))
    insert_event(db, run, "B", 1, "arr", BASE + timedelta(minutes=645), BASE + timedelta(minutes=646))
    cases = list(asof_cases(db, BASE + timedelta(minutes=540), BASE + timedelta(minutes=570)))
    assert len(cases) == 1
    assert (cases[0].issued_at, cases[0].horizon, cases[0].delay) == (BASE + timedelta(minutes=540), 120, -15.)


@pytest.mark.db
def test_cases_stop_when_destination_reported_and_labels_must_mature(db):
    from msynd.eval.harness import asof_cases

    run = make_run(db, start="2026-09-21", halts=[("A", None, 540, 0), ("B", 660, None, 100)])
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (run,))
    insert_event(db, run, "A", 1, "dep", BASE + timedelta(minutes=540), BASE + timedelta(minutes=540))
    insert_event(db, run, "B", 1, "arr", BASE + timedelta(minutes=600), BASE + timedelta(minutes=600))
    cases = list(asof_cases(db, BASE + timedelta(minutes=540), BASE + timedelta(minutes=750)))
    assert [c.issued_at for c in cases] == [BASE + timedelta(minutes=540), BASE + timedelta(minutes=570)]
    db.execute("UPDATE obs.run SET lifecycle='running' WHERE id=%s", (run,))
    assert not list(asof_cases(db, BASE, BASE + timedelta(days=1)))
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (run,))
    now = db.execute("SELECT statement_timestamp()").fetchone()[0]
    insert_event(db, run, "B", 1, "arr", now-timedelta(hours=5), now)
    assert not list(asof_cases(db, BASE, BASE + timedelta(days=1)))


def test_identical_eligible_cases_across_baselines():
    from msynd.eval.baselines import b0_schedule, b1_persistence, b2_forecast
    from msynd.eval.harness import score

    cases = [case(), case(1, b2=None), case(2)]
    results = score(cases, {"B0": b0_schedule, "B1": b1_persistence,
                           "B2": b2_forecast, "B3": lambda c: [c.sched_arr + 10.]})
    assert all(s.case_ids == results["B0"].case_ids for s in results.values())
    assert len(results["B0"].case_ids) == 2
    assert results["B0"].overall["p50_mae"] == 10.
    assert results["B1"].overall["p50_mae"] == 0.
    assert results["B2"].overall["p50_mae"] == 0.
    assert results["B3"].overall["p50_mae"] == 0.


@pytest.mark.db
def test_b2_ineligible_when_stale(db):
    from msynd.eval.baselines import b2_eta

    run = make_run(db)
    db.execute("INSERT INTO obs.ntes_eta_rev(run_id,station,occ,kind,expected_ts,received_at) "
               "VALUES (%s,'BRC',1,'arr',%s,%s)", (run, BASE + timedelta(hours=12), BASE))
    assert b2_eta(db, run, "BRC", 1, "arr", BASE + timedelta(minutes=30)) == BASE + timedelta(hours=12)
    assert b2_eta(db, run, "BRC", 1, "arr", BASE + timedelta(minutes=30, seconds=1)) is None
    assert b2_eta(db, run, "BRC", 2, "arr", BASE) is None


def test_disruption_days_derived_from_reference_pits():
    from msynd.eval.harness import score

    cases = [case(d) for d in range(10)] + [case(1, halt_idx=2, station="C")]
    reference = lambda c: [100.] if c.service_date == date(2026, 9, 22) else [c.actual]
    scores = score(cases, lambda c: [c.actual], reference=reference)
    assert scores.disruption_days == {date(2026, 9, 22)}
    assert scores.breakdowns["disruption_day"][True]["n"] == 2
    assert scores.breakdowns["disruption_day"][False]["n"] == 9
    own = score(cases, reference)
    assert own.disruption_days == scores.disruption_days
    early = lambda c: [1000.] if c.service_date == date(2026, 9, 22) else [c.actual]  # PIT 0 tail
    assert score(cases, lambda c: [c.actual], reference=early).disruption_days == {date(2026, 9, 22)}
    assert score(cases, lambda c: [c.actual]).disruption_days == set()
    assert score(cases[:2], lambda c: [0.]).disruption_days == {date(2026, 9, 21)}


def test_explicit_disruption_days_override_reference():
    from msynd.eval.harness import score

    scores = score([case(), case(1)], lambda c: [0.], disruption_days={date(2026, 9, 22)})
    assert scores.disruption_days == {date(2026, 9, 22)}
    assert scores.breakdowns["disruption_day"][True]["n"] == 1
    assert score([case()], lambda c: [0.], disruption_days=set()).disruption_days == set()


def test_scoring_reports_metrics_and_all_breakdowns():
    from msynd.eval.harness import score

    scores = score([case(actual=5., delay=-10.)], lambda c: [0., 10.])
    assert scores.overall["crps"] == 2.5
    assert scores.overall["p50_mae"] == 0.
    assert scores.overall["coverage"] == {0.8: 1., 0.9: 1.}
    assert scores.overall["width"][0.8] == 8.
    assert scores.overall["width"][0.9] == 9.
    assert scores.overall["pinball"][0.05] == pytest.approx(.225)
    assert sum(scores.overall["pit_histogram"]) == 1
    assert set(scores.breakdowns) == {"class", "zone", "delay_band", "horizon", "disruption_day"}
    assert "early" in scores.breakdowns["delay_band"]


def test_p50_mae_is_median_absolute_error():
    from msynd.eval.harness import score

    scores = score([case(d, actual=y) for d, y in enumerate([1., 2., 100.])], lambda c: 0.)
    assert scores.overall["p50_mae"] == 2.


def test_invalid_predictions_and_duplicate_cases_are_rejected():
    from msynd.eval.harness import score

    with pytest.raises(ValueError, match="finite"):
        score([case()], lambda c: [np.nan])
    with pytest.raises(ValueError, match="duplicate"):
        score([case(), case()], lambda c: [0.])
