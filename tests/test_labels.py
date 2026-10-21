from datetime import datetime, time, timedelta, timezone

import pyarrow as pa
from msynd.labels import IST

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


@pytest.fixture
def eligibility_runs(db):
    from psycopg.types.json import Jsonb

    asof = datetime(2026, 10, 9, tzinfo=timezone.utc)
    cutoff = asof - timedelta(hours=6)
    runs = []
    for i in range(9):
        run = make_run(db, train=str(81000+i), halts=[('A', None, 0, 0),
                       ('B', 60, 65, 100), ('A', 120, None, 200)])
        runs.append(run)
        db.execute("UPDATE obs.run SET lifecycle='completed',updated_at=%s WHERE id=%s", (asof, run))
        base = datetime(2026, 10, 7, tzinfo=IST)
        for station, occ, kind, minute in [('A', 1, 'dep', 10), ('B', 1, 'arr', 80),
                                          ('B', 1, 'dep', 90), ('A', 2, 'arr', 150)]:
            if i in (0, 3, 4, 8) and station == 'A' and kind == 'arr':
                continue
            actual = cutoff + timedelta(seconds=1) if i == 1 and kind == 'arr' else base+timedelta(minutes=minute)
            insert_event(db, run, station, occ, kind, actual, base+timedelta(minutes=minute+1))
        if i == 0:
            db.execute("UPDATE obs.run SET lifecycle='running' WHERE id=%s", (run,))
        if i in (3, 4):
            db.execute("UPDATE obs.run SET lifecycle='expired_unknown',updated_at=%s WHERE id=%s",
                       (asof+timedelta(seconds=i-3), run))
        if i in (2, 5):
            insert_fetch(db, run_id=run, flags=['exception_unparsed'],
                         received_at=asof+timedelta(seconds=1 if i == 5 else 0))
        if i == 6:
            dep_receipt = base+timedelta(minutes=11)
            for delta, detail in [(-2, {'resched_min': 20}), (-1, {'kind': 'diverted'}),
                                  (0, {'resched_min': 90}), (1, {'resched_min': 100})]:
                db.execute('INSERT INTO obs.exception_rev(train,start_date,detail,received_at) VALUES (%s,%s,%s,%s)',
                           (str(81000+i), base.date(), Jsonb(detail), dep_receipt+timedelta(seconds=delta)))
        if i == 7:
            insert_event(db, run, 'B', 1, 'arr', base+timedelta(minutes=85), asof+timedelta(seconds=1))
        if i == 8:
            insert_event(db, run, 'A', 1, 'arr', base+timedelta(minutes=150), asof)
    return runs, cutoff, asof


@pytest.mark.db
def test_set_labels_equal_original_sql(db, eligibility_runs):
    runs, cutoff, asof = eligibility_runs
    for boundary in (None, asof, asof+timedelta(seconds=1)):
        expected = reference_live_labels(db, cutoff, boundary)
        assert expected.num_rows
        assert build_live_labels(db, cutoff, boundary, run_ids=None).equals(expected)
        subset = [runs[1], runs[3], runs[6], runs[7], runs[8]]
        selected = pa.Table.from_pylist([r for r in expected.to_pylist() if r['run_id'] in subset], schema=expected.schema)
        assert build_live_labels(db, cutoff, boundary, run_ids=subset).equals(selected)


def test_empty_label_selection_does_not_query():
    class NoQueries:
        def execute(self, *a, **kw):
            pytest.fail('empty run selection must not query')

    assert build_live_labels(NoQueries(), datetime.now(timezone.utc), run_ids=[]).num_rows == 0


def reference_live_labels(conn, matured_before, asof=None):
    """Mature labels for finished runs, optionally frozen at a receipt boundary."""
    events = "obs.event" if asof is None else "obs.event_asof(%(asof)s)"
    finished = "r.lifecycle IN ('completed','expired_unknown')" if asof is None else """
        (EXISTS (SELECT 1 FROM ref.schedule_stop destination
          JOIN e ON e.run_id=r.id AND e.station=destination.station
            AND e.occ=destination.occ AND e.kind='arr'
          WHERE destination.version_id=r.schedule_version_id
            AND destination.idx=(SELECT max(idx) FROM ref.schedule_stop
              WHERE version_id=r.schedule_version_id))
         OR (r.lifecycle='expired_unknown' AND r.updated_at <= %(asof)s))
    """
    fetch_cutoff = "" if asof is None else "AND f.received_at <= %(asof)s"
    rows = conn.execute(f"""
        WITH e AS (SELECT * FROM {events})
        SELECT r.id,r.start_date,s.idx,s.arr_off,s.dep_off,
          a.actual_ts,d.actual_ts,d.received_at,
          COALESCE((SELECT (detail->>'resched_min')::integer
            FROM obs.exception_rev x WHERE x.train=r.train AND x.start_date=r.start_date
              AND x.received_at < d.received_at
            ORDER BY x.received_at DESC LIMIT 1),0)
        FROM obs.run r JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
        LEFT JOIN e a ON a.run_id=r.id AND a.station=s.station AND a.occ=s.occ AND a.kind='arr'
          AND a.actual_ts <= %(matured_before)s
        LEFT JOIN e d ON d.run_id=r.id AND d.station=s.station AND d.occ=s.occ AND d.kind='dep'
          AND d.actual_ts <= %(matured_before)s
        WHERE {finished}
          AND NOT EXISTS (SELECT 1 FROM obs.fetch f
            WHERE f.run_id=r.id AND 'exception_unparsed'=ANY(f.flags) {fetch_cutoff})
        ORDER BY r.id,s.idx
    """, {"matured_before": matured_before, "asof": asof}).fetchall()
    labels, previous = [], None
    for run, start, idx, sched_arr, sched_dep, actual_arr, actual_dep, received, resched in rows:
        base = datetime.combine(start, time(), IST)
        arr = (actual_arr - base).total_seconds() / 60 if actual_arr else None
        dep = (actual_dep - base).total_seconds() / 60 if actual_dep else None
        if sched_arr is None and dep is not None:
            labels.append((run, "origin", idx, origin_residual(sched_dep, resched, dep)))
        if sched_arr is not None and sched_dep is not None and arr is not None and dep is not None:
            value = dwell(sched_dep - sched_arr, arr - sched_arr, dep - sched_dep)
            if value >= 0:
                labels.append((run, "dwell", idx, value))
        if previous and previous[0] == run and previous[1] == idx - 1 and arr is not None and previous[2] is not None:
            labels.append((run, "run", idx - 1, run_label(previous[2] - previous[3], arr - sched_arr)))
        previous = (run, idx, dep, sched_dep)
    schema = pa.schema([("run_id", pa.int64()), ("role", pa.string()), ("idx", pa.int16()), ("value", pa.float64())])
    return pa.Table.from_pylist([dict(zip(schema.names, row)) for row in labels], schema=schema)
