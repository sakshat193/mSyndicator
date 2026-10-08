from datetime import date, datetime, timedelta, timezone

import pytest
from psycopg.types.json import Jsonb

from msynd.collect.dispatcher import FetchResult
from msynd.engine.evidence import load_run_evidence
from msynd.labels import IST, build_live_labels
from tests.helpers import insert_event, insert_fetch, make_run

pytestmark = pytest.mark.db
BASE = datetime(2026, 10, 7, tzinfo=IST)


def at(minute):
    return BASE + timedelta(minutes=minute)


def schedule(db, train='12951', days=127, departure=0, arrival=940):
    run = make_run(db, train=train, halts=[('MMCT', None, departure, 0),
                                         ('BRC', departure + 300, departure + 305, 392),
                                         ('NDLS', arrival, None, 1384)])
    version, route = db.execute('SELECT schedule_version_id,route_version_id FROM obs.run WHERE id=%s',
                                (run,)).fetchone()
    db.execute('DELETE FROM obs.run WHERE id=%s', (run,))
    db.execute('UPDATE ref.schedule_version SET days_of_run=%s WHERE id=%s', (days, version))
    return version, route


def apply(db, run, detail, minute=10):
    from msynd.collect.lifecycle import apply_exception

    received = at(minute)
    fetch = insert_fetch(db, run_id=run, received_at=received, started_at=received)
    apply_exception(db, detail, FetchResult(fetch, 'ok', 200, b'', received))
    return fetch


def revision(db, minute, detail):
    db.execute("INSERT INTO obs.exception_rev VALUES ('12951','2026-10-07',%s,NULL,%s)",
               (Jsonb(detail), at(minute)))


def test_creates_only_matching_weekdays(db):
    from msynd.collect.lifecycle import create_runs

    version, route = schedule(db, days=4)  # Wednesday
    schedule(db, train='12952', days=2, arrival=600)  # Tuesday's destination grace has elapsed.
    assert create_runs(db, at(0)) == 1
    assert db.execute('SELECT train,start_date,schedule_version_id,route_version_id,lifecycle FROM obs.run').fetchall() == [
        ('12951', date(2026, 10, 7), version, route, 'scheduled')]
    assert db.execute('SELECT run_id FROM obs.dirty_run').fetchall() == db.execute('SELECT id FROM obs.run').fetchall()
    assert create_runs(db, at(0)) == 0


def test_ist_start_date_at_utc_evening(db):
    from msynd.collect.lifecycle import create_runs

    schedule(db, departure=20, arrival=960)
    schedule(db, train='12952', departure=1380, arrival=2320)
    create_runs(db, datetime(2026, 10, 8, 19, tzinfo=timezone.utc))  # 10-09 00:30 IST
    runs = db.execute('SELECT train,start_date FROM obs.run').fetchall()
    assert ('12951', date(2026, 10, 10)) in runs  # Tomorrow, 00:20 IST, inside 24 hours.
    assert ('12952', date(2026, 10, 9)) in runs   # Today, 23:00 IST.


def test_creation_keeps_24_hour_upper_bound(db):
    from msynd.collect.lifecycle import create_runs

    schedule(db, departure=720, arrival=1320)
    schedule(db, train='12952', departure=719, arrival=1319)
    assert create_runs(db, at(720)) == 3
    assert db.execute('SELECT train,start_date FROM obs.run ORDER BY train,start_date').fetchall() == [
        ('12951', date(2026, 10, 7)), ('12952', date(2026, 10, 7)), ('12952', date(2026, 10, 8))]


@pytest.mark.parametrize('previous_arrival,keep_previous', [(1020, False), (1080, False), (1081, True)])
def test_creation_recovers_departed_runs_within_destination_grace(db, previous_arrival, keep_previous):
    from msynd.collect.lifecycle import create_runs

    # At Thursday 06:00 IST: departure was five hours ago; destination is ten hours ahead.
    schedule(db, days=8, departure=60, arrival=960)
    # Wednesday's arrival was 13 hours ago, exactly 12 hours ago, or just within the grace.
    schedule(db, train='12952', days=4, departure=60, arrival=previous_arrival)
    assert create_runs(db, at(1800)) == 1 + keep_previous
    expected = [('12951', date(2026, 10, 8))]
    if keep_previous:
        expected.append(('12952', date(2026, 10, 7)))
    assert db.execute('SELECT train,start_date FROM obs.run ORDER BY train').fetchall() == expected
    assert create_runs(db, at(1800)) == 0


def test_creation_checks_three_prior_origin_dates(db):
    from msynd.collect.lifecycle import create_runs

    schedule(db, days=4, arrival=5400)  # Wednesday origin, still travelling on Saturday.
    schedule(db, train='12952', days=2, arrival=7000)  # Tuesday origin, outside the three-day lookback.
    assert create_runs(db, at(5040)) == 1  # Saturday 12:00 IST
    assert db.execute('SELECT train,start_date FROM obs.run').fetchall() == [('12951', date(2026, 10, 7))]


def test_creation_selects_valid_schedule_and_latest_route(db):
    from msynd.collect.lifecycle import create_runs

    old, _ = schedule(db)
    db.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07' WHERE id=%s", (old,))
    newer = db.execute('''INSERT INTO ref.schedule_version(train,valid_from,valid_to,days_of_run,source,content_hash)
        VALUES ('12951','2026-10-07','2026-10-07',127,'gtfs','new') RETURNING id''').fetchone()[0]
    future = db.execute('''INSERT INTO ref.schedule_version(train,valid_from,days_of_run,source,content_hash)
        VALUES ('12951','2026-10-09',127,'gtfs','future') RETURNING id''').fetchone()[0]
    for version in (newer, future):
        db.execute('INSERT INTO ref.schedule_stop SELECT %s,idx,station,occ,arr_off,dep_off,dist_km FROM ref.schedule_stop WHERE version_id=%s',
                   (version, old))
        for digest in (b'first', b'latest'):
            route = db.execute("INSERT INTO ref.route_version(schedule_version_id,dist_src,content_hash) VALUES (%s,'gtfs',%s) RETURNING id",
                               (version, digest)).fetchone()[0]
    latest = db.execute('SELECT max(id) FROM ref.route_version WHERE schedule_version_id=%s', (newer,)).fetchone()[0]
    assert create_runs(db, at(0)) == 1
    assert db.execute('SELECT schedule_version_id,route_version_id FROM obs.run').fetchone() == (newer, latest)
    assert create_runs(db, at(1440)) == 1
    assert db.execute('SELECT schedule_version_id FROM obs.run ORDER BY start_date').fetchall() == [(newer,), (old,)]


def test_cancelled_not_created(db):
    from msynd.collect.lifecycle import create_runs

    version, _ = schedule(db, departure=600, arrival=1540)
    db.execute("UPDATE ref.schedule_version SET valid_from='2026-10-07' WHERE id=%s", (version,))
    revision(db, 100, {'kind': 'cancelled'})
    revision(db, 400, {'kind': 'rescheduled', 'resched_min': 60})
    assert create_runs(db, at(300)) == 0  # The later withdrawal is not eligible yet.
    assert db.execute('SELECT start_date FROM obs.run').fetchall() == []
    assert create_runs(db, at(500)) == 1
    assert db.execute("SELECT resched_min FROM obs.run WHERE start_date='2026-10-07'").fetchone() == (60,)


@pytest.mark.parametrize('receipt,delay,now,want', [
    (1000, 0, 1661, 0),   # Destination grace elapsed, but no 12-hour silence.
    (300, 0, 1660, 0),    # Strictly later than arrival + 12 hours.
    (300, 0, 1661, 1),
    (941, 0, 1661, 1),    # Exactly 12 hours without an actual receipt.
    (900, 500, 1800, 0),  # Twice the last observed delay exceeds 12 hours.
    (900, 500, 1941, 1),
    (300, -20, 1661, 1),
])
def test_expiry_rule_both_conditions(db, receipt, delay, now, want):
    from msynd.collect.lifecycle import expire_runs

    run = make_run(db)
    db.execute("UPDATE obs.run SET lifecycle='running' WHERE id=%s", (run,))
    insert_event(db, run, 'BRC', 1, 'arr', at(300 + delay), at(receipt))
    assert expire_runs(db, at(now)) == want
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == (
        'expired_unknown' if want else 'running',)
    assert db.execute('SELECT count(*) FROM obs.dirty_run').fetchone() == (want,)
    assert expire_runs(db, at(now)) == 0


def test_expired_never_completed(db):
    from msynd.collect.lifecycle import expire_runs

    run = make_run(db)
    assert expire_runs(db, at(1661)) == 1
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('expired_unknown',)
    assert build_live_labels(db, at(2500)).num_rows == 0


def test_expiry_preserves_terminal_runs_and_multiday_schedules(db):
    from msynd.collect.lifecycle import expire_runs

    for train, state in [('12951', 'cancelled'), ('12952', 'completed'), ('12953', 'expired_unknown')]:
        run = make_run(db, train=train)
        db.execute('UPDATE obs.run SET lifecycle=%s WHERE id=%s', (state, run))
    make_run(db, train='12954', halts=[('MMCT', None, 0, 0), ('NDLS', 3000, None, 1384)])
    assert expire_runs(db, at(3000)) == 0
    assert expire_runs(db, at(3721)) == 1
    assert db.execute('SELECT train,lifecycle FROM obs.run ORDER BY train').fetchall() == [
        ('12951', 'cancelled'), ('12952', 'completed'), ('12953', 'expired_unknown'), ('12954', 'expired_unknown')]


def test_expiry_uses_current_actuals_asof_now(db):
    from msynd.collect.lifecycle import expire_runs

    run = make_run(db)
    insert_event(db, run, 'BRC', 1, 'arr', at(800), at(900))
    insert_event(db, run, 'BRC', 1, 'arr', at(300), at(901))  # Correct the delay to zero.
    insert_event(db, run, 'BRC', 1, 'dep', at(305), at(1700))  # Future receipt cannot keep it alive.
    assert expire_runs(db, at(1661)) == 1


def test_expiry_uses_interpolated_passing_schedule(db):
    from msynd.collect.lifecycle import expire_runs

    run = make_run(db)
    route = db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()[0]
    db.execute('UPDATE ref.route_stop SET idx=idx+10 WHERE route_version_id=%s AND idx>0', (route,))
    db.execute('UPDATE ref.route_stop SET idx=idx-9 WHERE route_version_id=%s AND idx>0', (route,))
    db.execute("INSERT INTO ref.station(code,name) VALUES ('PASS','Passing')")
    db.execute("INSERT INTO ref.route_stop VALUES (%s,1,'PASS',1,false,196,NULL)", (route,))
    insert_event(db, run, 'PASS', 1, 'pass', at(650), at(700))  # Scheduled 150; delay 500.
    assert expire_runs(db, at(1800)) == 0
    assert expire_runs(db, at(1941)) == 1


@pytest.mark.parametrize('detail,state,resched,diverted,part', [
    ({'kind': 'cancelled'}, 'cancelled', 0, None, None),
    ({'kind': 'rescheduled', 'resched_min': 60}, 'scheduled', 60, None, None),
    ({'kind': 'diverted', 'diverted_from': 'BRC'}, 'scheduled', 0, 'BRC', None),
    ({'kind': 'part_cancelled', 'part_cancelled': [{'from': 'BRC', 'to': 'NDLS'}]},
     'scheduled', 0, None, [{'from': 'BRC', 'to': 'NDLS'}]),
])
def test_exception_marks_dirty(db, detail, state, resched, diverted, part):
    run = make_run(db)
    fetch = apply(db, run, detail)
    assert db.execute('SELECT lifecycle,resched_min,diverted_from,part_cancelled FROM obs.run WHERE id=%s',
                      (run,)).fetchone() == (state, resched, diverted, part)
    assert db.execute('SELECT run_id FROM obs.dirty_run').fetchall() == [(run,)]
    stored, fetch_id, received = db.execute('SELECT detail,fetch_id,received_at FROM obs.exception_rev').fetchone()
    assert stored == {'kind': detail['kind'], 'resched_min': resched, 'diverted_from': diverted, 'part_cancelled': part}
    assert (fetch_id, received) == (fetch, at(10))


def test_exception_revisions_are_cumulative_and_asof(db):
    run = make_run(db)
    apply(db, run, {'kind': 'rescheduled', 'resched_min': 60}, 10)
    apply(db, run, {'kind': 'diverted', 'diverted_from': 'BRC'}, 20)
    assert load_run_evidence(db, run, at(15)).resched_min == 60
    assert load_run_evidence(db, run, at(15)).stop_event is None
    evidence = load_run_evidence(db, run, at(25))
    assert evidence.resched_min == 60 and evidence.stop_event == 1
    insert_event(db, run, 'MMCT', 1, 'dep', at(80), at(30))
    apply(db, run, {'kind': 'part_cancelled', 'part_cancelled': [{'from': 'BRC', 'to': 'NDLS'}]}, 40)
    latest = db.execute('SELECT detail FROM obs.exception_rev ORDER BY received_at DESC LIMIT 1').fetchone()[0]
    assert latest == {'kind': 'part_cancelled', 'resched_min': 60, 'diverted_from': 'BRC',
                      'part_cancelled': [{'from': 'BRC', 'to': 'NDLS'}]}
    assert build_live_labels(db, at(2000)).num_rows == 0
    db.execute("UPDATE obs.run SET lifecycle='expired_unknown' WHERE id=%s", (run,))
    assert build_live_labels(db, at(2000)).to_pylist()[0]['value'] == 20


@pytest.mark.parametrize('kind', ['none', 'rescheduled', 'diverted', 'part_cancelled'])
def test_later_exception_clears_full_cancellation(db, kind):
    run = make_run(db)
    insert_event(db, run, 'MMCT', 1, 'dep', at(1), at(2))
    apply(db, run, {'kind': 'cancelled'}, 10)
    details = {'none': {}, 'rescheduled': {'resched_min': 20}, 'diverted': {'diverted_from': 'BRC'},
               'part_cancelled': {'part_cancelled': [{'from': 'BRC', 'to': 'NDLS'}]}}
    apply(db, run, {'kind': kind, **details[kind]}, 20)
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('running',)


def test_withdrawal_clears_attributes(db):
    run = make_run(db)
    apply(db, run, {'kind': 'rescheduled', 'resched_min': 60, 'diverted_from': 'BRC',
                    'part_cancelled': [{'from': 'BRC', 'to': 'NDLS'}]}, 10)
    apply(db, run, {'kind': 'none'}, 20)
    assert db.execute('SELECT resched_min,diverted_from,part_cancelled FROM obs.run WHERE id=%s',
                      (run,)).fetchone() == (0, None, None)
    evidence = load_run_evidence(db, run)
    assert evidence.resched_min == 0 and evidence.stop_event is None


def test_lifecycle_and_exception_changes_share_transaction(db):
    from msynd.collect.lifecycle import create_runs, expire_runs

    run = make_run(db)
    with pytest.raises(RuntimeError), db.transaction():
        apply(db, run, {'kind': 'rescheduled', 'resched_min': 60})
        assert expire_runs(db, at(1661)) == 1
        assert create_runs(db, at(1440)) == 1
        raise RuntimeError('rollback')
    assert db.execute('SELECT count(*) FROM obs.exception_rev').fetchone() == (0,)
    assert db.execute('SELECT count(*) FROM obs.dirty_run').fetchone() == (0,)
    assert db.execute('SELECT lifecycle,resched_min FROM obs.run').fetchall() == [('scheduled', 0)]


@pytest.mark.parametrize('detail', [
    {'kind': 'bad'}, {'kind': 'rescheduled'}, {'kind': 'rescheduled', 'resched_min': 1.5},
    {'kind': 'diverted', 'diverted_from': 'UNKNOWN'},
    {'kind': 'part_cancelled', 'part_cancelled': [{'from': 'BRC'}]},
    {'kind': 'part_cancelled', 'part_cancelled': [{'from': 'NDLS', 'to': 'BRC'}]},
])
def test_invalid_exception_has_no_partial_writes(db, detail):
    run = make_run(db)
    with pytest.raises(ValueError):
        apply(db, run, detail)
    assert db.execute('SELECT count(*) FROM obs.exception_rev').fetchone() == (0,)
    assert db.execute('SELECT count(*) FROM obs.dirty_run').fetchone() == (0,)
