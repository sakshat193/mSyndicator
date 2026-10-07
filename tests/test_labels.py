from datetime import datetime, timedelta, timezone

import pytest

from msynd.labels import run_label, dwell, origin_residual, composite, build_live_labels
from tests.helpers import make_run, insert_event


def test_clock_counterexample():
    assert run_label(dep_delay_a=15, arr_delay_b=20) == 5
    assert dwell(sched_dwell=10, arr_delay=0, dep_delay=15) == 25
    assert composite(dwell_delta=15, run_delta=5) == 20


def test_resched_plus20():
    assert origin_residual(100, 20, 130) == 10


def test_resched_negative():
    assert origin_residual(100, 60, 150) == -10


@pytest.mark.db
def test_history_never_label(db):
    db.execute("insert into obs.history_daily(train,run_date,station,occ,value_min) values ('12951','2026-10-07','BRC',1,10)")
    assert build_live_labels(db, datetime.now(timezone.utc)).num_rows == 0


@pytest.mark.db
def test_labels_only_matured(db):
    now = datetime(2026, 10, 8, tzinfo=timezone.utc)
    run = make_run(db)
    db.execute("update obs.run set lifecycle='completed' where id=%s", (run,))
    insert_event(db, run, "MMCT", 1, "dep", now - timedelta(hours=7), now - timedelta(hours=7))
    insert_event(db, run, "BRC", 1, "arr", now - timedelta(hours=5), now - timedelta(hours=5))
    result = build_live_labels(db, now - timedelta(hours=6)).to_pylist()
    assert len(result) == 1 and result[0]["role"] == "origin"
    db.execute("update obs.run set lifecycle='running' where id=%s", (run,))
    assert build_live_labels(db, now).num_rows == 0


@pytest.mark.db
def test_label_resched_asof_receipt(db):
    received = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
    actual = datetime(2026, 10, 6, 19, tzinfo=timezone.utc)  # 00:30 IST: residual 30
    run = make_run(db)
    db.execute("update obs.run set lifecycle='completed', resched_min=60 where id=%s", (run,))
    insert_event(db, run, "MMCT", 1, "dep", actual, received)
    db.execute("""insert into obs.exception_rev(train,start_date,detail,received_at)
        values ('12951','2026-10-07','{"resched_min":60}',%s)""",
        (received + timedelta(minutes=1),))
    assert build_live_labels(db, actual + timedelta(hours=7)).to_pylist() == [
        {"run_id": run, "role": "origin", "idx": 0, "value": 30.0}
    ]
