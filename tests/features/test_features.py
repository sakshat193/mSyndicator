from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pytest

from tests.helpers import make_run, insert_event


def test_encoders_never_see_validation_rows():
    from msynd.features.build import FoldEncoders

    rows = pa.Table.from_pylist([
        dict(split="train", train="A", from_stn="X", to_stn="Y", value=0., observed_running=100., sched_run=120.),
        dict(split="train", train="A", from_stn="X", to_stn="Y", value=10., observed_running=120., sched_run=120.),
        dict(split="valid", train="A", from_stn="X", to_stn="Y", value=1000., observed_running=1., sched_run=120.),
    ])
    enc = FoldEncoders.fit(rows)
    result = enc.transform(rows).to_pylist()
    assert result[0]["train_encoded"] == 5.
    assert result[0]["slack"] == pytest.approx(19.)
    unseen = enc.transform(pa.Table.from_pylist([
        dict(train="unseen", from_stn="Z", to_stn="Y", sched_run=50.)
    ])).to_pylist()[0]
    assert unseen["train_encoded"] == 5.
    assert unseen["slack"] is None
    assert "history_mean" not in unseen


@pytest.mark.db
def test_leakage_received_after_asof_absent(db):
    from msynd.features.build import segment_queries

    run = make_run(db, halts=[("A", None, 540, 0), ("B", 600, 605, 100), ("C", 665, None, 200)])
    base = datetime(2026, 10, 7, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    insert_event(db, run, "A", 1, "dep", base + timedelta(minutes=540), base + timedelta(minutes=541))
    insert_event(db, run, "B", 1, "arr", base + timedelta(minutes=600), base + timedelta(minutes=620))
    before = segment_queries(db, run, base + timedelta(minutes=615)).to_pylist()
    assert before[1]["last_delta_run"] is None
    after = segment_queries(db, run, base + timedelta(minutes=620)).to_pylist()
    assert after[1]["last_delta_run"] == 0.
    assert after[1]["segments_since_observed"] == 1
    # Later corrections must not erase or replace the earlier eligible revision.
    insert_event(db, run, "B", 1, "arr", base + timedelta(minutes=630), base + timedelta(minutes=640))
    assert segment_queries(db, run, base + timedelta(minutes=620)).to_pylist()[1]["last_delta_run"] == 0.
    assert before[0]["km"] == 100.
    assert before[1]["sched_dwell"] == 5.
    assert before[1]["is_last"] is True
    assert "surprises" not in before[0]


@pytest.mark.db
def test_calendar_features_use_sched_dep_plus_d(db):
    from msynd.features.build import segment_queries, with_delay

    run = make_run(db, start="2026-01-31", halts=[("A", None, 1430, 0), ("B", 1490, None, 100)])
    db.execute("update ref.station set zone='NR' where code='A'")
    db.execute("insert into ref.calendar(day,holiday,fog_north) values ('2026-02-01','holiday',true)")
    rows = segment_queries(db, run, datetime(2026, 1, 31, tzinfo=timezone.utc))
    row = with_delay(db, rows, 40.).to_pylist()[0]  # Feb 1 Sunday, 00:30 IST
    assert row["hour_sin"] == pytest.approx(0.13052619222005157)
    assert row["hour_cos"] == pytest.approx(0.9914448613738104)
    assert (row["weekday"], row["month"], row["fog_north"], row["holiday"]) == (6, 2, True, True)
    assert rows.to_pylist()[0]["weekday"] == 5


@pytest.mark.db
def test_splits_rolling_by_week(db):
    from msynd.features.build import Fold, training_frame

    starts = ["2026-09-14", "2026-09-21", "2026-09-28"]
    for i, start in enumerate(starts):
        run = make_run(db, train=str(90000+i), start=start,
                       halts=[("A", None, 540, 0), ("B", 600, 605, 100), ("C", 665, None, 200)])
        db.execute("update obs.run set lifecycle='completed' where id=%s", (run,))
        base = datetime.combine(date.fromisoformat(start), datetime.min.time(), timezone(timedelta(hours=5, minutes=30)))
        for station, kind, minute in [("A", "dep", 550), ("B", "arr", 620+i*100), ("B", "dep", 625+i*100), ("C", "arr", 690+i*100)]:
            insert_event(db, run, station, 1, kind, base+timedelta(minutes=minute), base+timedelta(minutes=minute+1))
    fold = Fold((date(2026,9,14),), (date(2026,9,21),))
    X, y = training_frame(db, fold)
    rows = X.to_pylist()
    assert [r["split"] for r in rows] == ["train", "train", "valid", "valid"]
    assert y.tolist() == [10., 5., 110., 5.]
    assert rows[0]["last_delta_run"] is None  # future arrival cannot enter departure-time context
    assert rows[1]["last_delta_run"] == 10.
    assert rows[2]["train_encoded"] == 7.5  # unseen validation train: training mean
    assert rows[2]["slack"] == pytest.approx(-10.)  # trained pair running time 70, not validation 170
    with pytest.raises(ValueError):
        Fold((date(2026,9,21),), (date(2026,9,14),))
