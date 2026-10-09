from datetime import datetime, timedelta, timezone

import pytest

from msynd.labels import run_label, dwell, origin_residual, composite, build_live_labels
from tests.helpers import make_run, insert_event, insert_fetch


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


@pytest.fixture
def completed_run(db):
    base = datetime(2026, 10, 6, 18, 30, tzinfo=timezone.utc)
    run = make_run(db)
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (run,))
    for station, kind, minute in [('MMCT', 'dep', 10), ('BRC', 'arr', 310),
                                  ('BRC', 'dep', 320), ('NDLS', 'arr', 950)]:
        actual = base + timedelta(minutes=minute)
        insert_event(db, run, station, 1, kind, actual, actual + timedelta(minutes=5))
    return run, base + timedelta(days=2)


@pytest.mark.db
def test_unparsed_exception_run_has_no_labels(db, completed_run):
    run, cutoff = completed_run
    expected = build_live_labels(db, cutoff).to_pylist()
    assert {row['role'] for row in expected} == {'origin', 'run', 'dwell'}
    other = make_run(db, train='99999')
    insert_fetch(db, run_id=other, flags=['exception_unparsed'])
    assert build_live_labels(db, cutoff).to_pylist() == expected
    flagged = insert_fetch(db, run_id=run, flags=['exception_unparsed'])
    insert_fetch(db, run_id=run)  # A later clean fetch does not restore eligibility.
    assert build_live_labels(db, cutoff).num_rows == 0
    db.execute("UPDATE obs.fetch SET flags='{}' WHERE id=%s", (flagged,))
    assert build_live_labels(db, cutoff).to_pylist() == expected


@pytest.mark.db
def test_delay_gt_24h_run_keeps_labels(db, completed_run):
    run, cutoff = completed_run
    expected = build_live_labels(db, cutoff).to_pylist()
    assert expected
    insert_fetch(db, run_id=run, flags=['delay_gt_24h'])
    assert build_live_labels(db, cutoff).to_pylist() == expected


@pytest.mark.db
@pytest.mark.parametrize("lifecycle", ["completed", "running"])
def test_frozen_labels_ignore_later_revisions(db, lifecycle):
    asof = datetime(2026, 10, 8, tzinfo=timezone.utc)
    base = datetime(2026, 10, 6, 18, 30, tzinfo=timezone.utc)
    run = make_run(db, halts=[("A", None, 0, 0), ("B", 60, 65, 100), ("A", 120, None, 200)])
    db.execute("UPDATE obs.run SET lifecycle=%s WHERE id=%s", (lifecycle, run))
    for station, occ, kind, minute in [("A", 1, "dep", 10), ("B", 1, "arr", 80),
                                      ("B", 1, "dep", 90), ("A", 1, "arr", 150)]:
        insert_event(db, run, station, occ, kind, base+timedelta(minutes=minute),
                     base+timedelta(minutes=minute+1))
    cutoff = asof - timedelta(hours=6)
    assert build_live_labels(db, cutoff, asof).num_rows == 0  # Wrong destination occurrence.
    later = asof + timedelta(minutes=1)
    insert_event(db, run, "A", 2, "arr", base+timedelta(minutes=150), later)
    assert build_live_labels(db, cutoff, asof).num_rows == 0
    expected = [dict(run_id=run, role=role, idx=idx, value=value) for role, idx, value in
                [("origin", 0, 10.), ("dwell", 1, 10.), ("run", 0, 10.), ("run", 1, 5.)]]
    assert build_live_labels(db, cutoff, later).to_pylist() == expected
    insert_event(db, run, "B", 1, "arr", base+timedelta(minutes=85), later+timedelta(minutes=1))
    assert build_live_labels(db, cutoff, later).to_pylist() == expected
    corrected = build_live_labels(db, cutoff, later+timedelta(minutes=1)).to_pylist()
    assert [r["value"] for r in corrected] == [10., 5., 15., 5.]


@pytest.mark.db
def test_frozen_expiry_requires_updated_at_evidence(db):
    asof = datetime(2026, 10, 8, tzinfo=timezone.utc)
    run = make_run(db)
    db.execute("UPDATE obs.run SET lifecycle='expired_unknown',updated_at=%s WHERE id=%s",
               (asof+timedelta(seconds=1), run))
    insert_event(db, run, "MMCT", 1, "dep", asof-timedelta(hours=8), asof-timedelta(hours=7))
    assert build_live_labels(db, asof-timedelta(hours=6), asof).num_rows == 0
    db.execute("UPDATE obs.run SET updated_at=%s WHERE id=%s", (asof, run))
    assert build_live_labels(db, asof-timedelta(hours=6), asof).num_rows == 1


@pytest.mark.db
def test_frozen_unparsed_exclusion_uses_fetch_receipts(db, completed_run):
    run, asof = completed_run
    cutoff = asof-timedelta(hours=6)
    expected = build_live_labels(db, cutoff, asof).to_pylist()
    assert expected
    flagged = insert_fetch(db, run_id=run, flags=['exception_unparsed'],
                           received_at=asof+timedelta(seconds=1))
    assert build_live_labels(db, cutoff, asof).to_pylist() == expected
    assert build_live_labels(db, cutoff).num_rows == 0
    db.execute("UPDATE obs.fetch SET received_at=%s WHERE id=%s", (asof, flagged))
    assert build_live_labels(db, cutoff, asof).num_rows == 0
