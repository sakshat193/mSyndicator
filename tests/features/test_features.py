from datetime import date, datetime, timedelta, timezone
import json

import pyarrow as pa
import pytest

from tests.helpers import make_run, insert_event


def test_model_features():
    from msynd.features.build import MODEL_FEATURES

    assert tuple(MODEL_FEATURES) == (
        "class", "zone", "segments_remaining", "is_last", "km", "sched_dep",
        "sched_run", "sched_dwell", "last_delta_run", "segments_since_observed",
        "hour_sin", "hour_cos", "weekday", "month", "fog_north", "holiday",
        "train_encoded", "slack", "d",
    )


def test_encoders_json_roundtrip():
    from msynd.features.build import FoldEncoders

    rows = pa.Table.from_pylist([
        dict(train="A", from_stn="X", to_stn="Y", value=0., observed_running=100., sched_run=120.),
        dict(train="B", from_stn="Y", to_stn="X", value=10., observed_running=130., sched_run=150.),
    ])
    encoders = FoldEncoders.fit(rows)
    data = json.loads(json.dumps(encoders.to_json(), allow_nan=False))
    assert data["pair_p05"] == {"X|Y": 100., "Y|X": 130.}
    restored = FoldEncoders.from_json(data)
    assert restored == encoders
    assert restored.transform(rows).equals(encoders.transform(rows))
    empty = FoldEncoders(0., {}, {})
    assert FoldEncoders.from_json(json.loads(json.dumps(empty.to_json()))) == empty


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
    X, y = training_frame(db, fold, datetime(2026, 10, 1, tzinfo=timezone.utc))
    rows = X.to_pylist()
    assert [r["split"] for r in rows] == ["train", "train", "valid", "valid"]
    assert y.tolist() == [10., 5., 110., 5.]
    assert rows[0]["last_delta_run"] is None  # future arrival cannot enter departure-time context
    assert rows[1]["last_delta_run"] == 10.
    assert all(r["train_encoded"] is None and r["slack"] is None for r in rows[:2])
    assert rows[2]["train_encoded"] == 7.5  # unseen validation train: training mean
    assert rows[2]["slack"] == pytest.approx(-10.)  # trained pair running time 70, not validation 170
    with pytest.raises(ValueError):
        Fold((date(2026,9,21),), (date(2026,9,14),))


def test_training_row_target_never_in_own_encoding():
    from msynd.features.build import FoldEncoders

    rows = [
        dict(start_date=date(2026, 9, 14), split="train", train="A", from_stn="X", to_stn="Y",
             value=0., observed_running=100., sched_run=120.),
        dict(start_date=date(2026, 9, 21), split="train", train="A", from_stn="X", to_stn="Y",
             value=10., observed_running=130., sched_run=120.),
        dict(start_date=date(2026, 9, 15), split="train", train="B", from_stn="Z", to_stn="Y",
             value=20., observed_running=110., sched_run=120.),
        dict(start_date=date(2026, 9, 22), split="train", train="C", from_stn="Z", to_stn="Y",
             value=30., observed_running=140., sched_run=120.),
        dict(start_date=date(2026, 9, 28), split="valid", train="A", from_stn="X", to_stn="Y",
             value=1000., observed_running=1., sched_run=120.),
    ]
    before = FoldEncoders.cross_fit(pa.Table.from_pylist(rows)).to_pylist()
    assert [r["start_date"] for r in before] == [r["start_date"] for r in rows[:4]]
    assert [r["train_encoded"] for r in before] == pytest.approx([210/11, 100/11, 20., 10.])
    assert [r["slack"] for r in before] == [-10., 20., -20., 10.]
    rows[0].update(value=100., observed_running=200.)
    after = FoldEncoders.cross_fit(pa.Table.from_pylist(rows)).to_pylist()
    for i in (0, 2):
        assert after[i] == before[i] | {k: rows[i][k] for k in ("value", "observed_running")}
    assert after[1]["train_encoded"] == pytest.approx(700/11)
    assert after[1]["slack"] == -80.
    assert after[3]["train_encoded"] == 60.  # Unseen train uses only the other week's global mean.
    assert after[3]["slack"] == 10.


def test_single_training_week_encodes_missing():
    from msynd.features.build import FoldEncoders

    rows = pa.Table.from_pylist([
        dict(start_date=date(2026, 9, day), train="A", from_stn="X", to_stn="Y",
             value=float(day), observed_running=100., sched_run=120.) for day in (14, 15)
    ])
    result = FoldEncoders.cross_fit(rows)
    assert result.num_rows == 2
    for name in ("train_encoded", "slack"):
        assert result[name].to_pylist() == [None, None]
        assert result.schema.field(name).type == pa.float64()
    assert FoldEncoders.cross_fit(rows.slice(0, 0)).num_rows == 0


@pytest.fixture
def fold_runs(db):
    runs = []
    for i, start in enumerate(("2026-09-14", "2026-09-21", "2026-09-28")):
        run = make_run(db, train=str(90000+i), start=start,
                       halts=[("A", None, 540, 0), ("B", 600, 605, 100), ("C", 665, None, 200)])
        runs.append(run)
        db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (run,))
        base = datetime.fromisoformat(start).replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
        for station, kind, minute in [("A", "dep", 550), ("B", "arr", 620+i*100),
                                      ("B", "dep", 625+i*100), ("C", "arr", 690+i*100)]:
            insert_event(db, run, station, 1, kind, base+timedelta(minutes=minute),
                         base+timedelta(minutes=minute+1))
    return runs


@pytest.mark.db
def test_validation_rows_use_all_training_rows(db, fold_runs):
    from msynd.features.build import Fold, training_frame

    fold = Fold((date(2026, 9, 14), date(2026, 9, 21)), (date(2026, 9, 28),))
    X, y = training_frame(db, fold, datetime(2026, 10, 1, tzinfo=timezone.utc))
    rows = X.to_pylist()
    assert [r["run_id"] for r in rows] == [run for run in fold_runs for _ in range(2)]
    assert y.tolist() == [10., 5., 110., 5., 210., 5.]
    assert [r["train_encoded"] for r in rows] == [57.5, 57.5, 7.5, 7.5, 32.5, 32.5]
    assert [r["slack"] for r in rows] == pytest.approx([-110., -5., -10., -5., -15., -5.])


@pytest.mark.db
def test_training_frame_asof_is_reproducible(db, fold_runs):
    from msynd.features.build import Fold, training_frame

    fold = Fold((date(2026, 9, 14), date(2026, 9, 21)), (date(2026, 9, 28),))
    asof = datetime(2026, 10, 1, tzinfo=timezone.utc)
    before_X, before_y = training_frame(db, fold, asof)
    assert before_y.tolist() == [10., 5., 110., 5., 210., 5.]
    base = datetime(2026, 9, 14, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    for station, kind, minute in [("A", "dep", 555), ("B", "arr", 640), ("C", "arr", 720)]:
        insert_event(db, fold_runs[0], station, 1, kind, base+timedelta(minutes=minute),
                     asof+timedelta(minutes=1))
    after_X, after_y = training_frame(db, fold, asof)
    assert after_X.equals(before_X)
    assert after_y.tolist() == before_y.tolist()
    later_X, later_y = training_frame(db, fold, asof+timedelta(minutes=1))
    assert later_y.tolist() == [25., 35., 110., 5., 210., 5.]
    assert not later_X.equals(before_X)


def test_training_only_fold():
    from msynd.features.build import Fold

    assert Fold((date(2026, 9, 14),), ()).valid_weeks == ()
    for train, valid in [((), ()), ((date(2026, 9, 15),), ()),
                         ((date(2026, 9, 14),)*2, ())]:
        with pytest.raises(ValueError):
            Fold(train, valid)


@pytest.mark.db
@pytest.mark.parametrize('shared_receipt', [False, True])
@pytest.mark.parametrize('training_only', [False, True])
def test_training_batches_receipts_without_changing_output(db, fold_runs, monkeypatch, shared_receipt, training_only):
    from msynd.features import build

    if shared_receipt:
        db.execute("UPDATE obs.event_rev SET received_at=date_trunc('day',received_at)+interval '23 hours' WHERE kind='dep'")
    fold = build.Fold((date(2026, 9, 14), date(2026, 9, 21)),
                      () if training_only else (date(2026, 9, 28),))
    asof = datetime(2026, 10, 1, tzinfo=timezone.utc)
    expected_X, expected_y = reference_training_frame(db, fold, asof)
    query_calls, delay_calls = [], []
    queries, delay = build.segment_queries, build.with_delay

    def counted_queries(conn, run, receipt):
        query_calls.append((run, receipt))
        return queries(conn, run, receipt)

    def counted_delay(conn, rows, d):
        delay_calls.append(len(rows))
        return delay(conn, rows, d)

    monkeypatch.setattr(build, 'segment_queries', counted_queries)
    monkeypatch.setattr(build, 'with_delay', counted_delay)
    X, y = build.training_frame(db, fold, asof, run_ids=None)
    assert X.equals(expected_X) and y.tolist() == expected_y.tolist()
    n_runs = 2 if training_only else 3
    assert len(query_calls) == len(set(query_calls)) == n_runs*(1 if shared_receipt else 2)
    assert delay_calls == [2 if shared_receipt else 1]*len(query_calls)
    subset_X, subset_y = build.training_frame(db, fold, asof, run_ids=fold_runs[:1])
    assert set(subset_X['run_id'].to_pylist()) == {fold_runs[0]}
    assert subset_y.tolist() == expected_y[:2].tolist()
    assert len(build.training_frame(db, fold, asof, run_ids=[])[0]) == 0


def reference_training_frame(conn, fold, asof):
    from msynd.features.build import (build_live_labels, segment_queries, with_delay,
                                      FoldEncoders, _SCHEMA, run_label)
    from datetime import time
    from msynd.labels import IST
    import numpy as np
    import pyarrow.compute as pc
    labels = {(r["run_id"], r["idx"]): r["value"] for r in
              build_live_labels(conn, asof - timedelta(hours=6), asof).to_pylist() if r["role"] == "run"}
    departures = conn.execute("""
        SELECT r.id,r.start_date,s.idx,s.dep_off,d.actual_ts,d.received_at
        FROM obs.run r JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
        JOIN obs.event_asof(%s) d ON d.run_id=r.id AND d.station=s.station AND d.occ=s.occ AND d.kind='dep'
        ORDER BY r.start_date,r.id,s.idx
    """, (asof,)).fetchall()
    rows = []
    # ponytail: per-departure queries; batch receipt cutoffs if weekly training becomes slow.
    for run, start, idx, scheduled, actual, received in departures:
        week = start - timedelta(days=start.weekday())
        split = "train" if week in fold.train_weeks else "valid" if week in fold.valid_weeks else None
        if split is None or (run, idx) not in labels:
            continue
        queries = segment_queries(conn, run, received)
        selected = queries.filter(pc.equal(queries["idx"], idx))
        d = (actual-datetime.combine(start, time(), IST)).total_seconds()/60-scheduled
        row = with_delay(conn, selected, d).to_pylist()[0]
        value = labels[(run, idx)]
        rows.append(row | {"split": split, "value": value,
                           "observed_running": row["sched_run"]+value})
    schema = _SCHEMA.append(pa.field("d", pa.float64())).append(pa.field("split", pa.string()))
    raw = pa.Table.from_pylist(rows, schema=schema.append(pa.field("value", pa.float64()))
                              .append(pa.field("observed_running", pa.float64())))
    if not rows:
        return pa.Table.from_pylist([], schema=schema), np.empty(0)
    encoders = FoldEncoders.fit(raw)
    train = iter(FoldEncoders.cross_fit(raw).to_pylist())
    valid = iter(encoders.transform(raw.filter(pc.equal(raw["split"], "valid"))).to_pylist())
    encoded = pa.Table.from_pylist([next(train) if r["split"] == "train" else next(valid) for r in rows],
                                  schema=raw.schema)
    return encoded.drop(["value", "observed_running"]), np.asarray([r["value"] for r in rows])
