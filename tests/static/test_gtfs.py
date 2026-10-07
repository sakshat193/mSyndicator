from pathlib import Path
import shutil

import pytest

from msynd.static.gtfs import import_gtfs, train_class, haversine

pytestmark = pytest.mark.db
FIXTURE = Path(__file__).parents[1] / "fixtures/gtfs_mini"


def test_gc_distance_fallback(db):
    report = import_gtfs(db, FIXTURE)
    assert report.trains == 3 and report.stations == 3 and report.missing_coords == 0
    rows = db.execute("select s.dist_km from ref.schedule_stop s join ref.schedule_version v on v.id=s.version_id where v.train='12951' order by s.idx").fetchall()
    expected = haversine(19, 73, 20, 74)
    assert rows[0][0] == 0 and abs(rows[1][0] - expected) < 1e-4
    assert abs(rows[2][0] - 2 * expected) < 1e-4
    assert db.execute("select distinct dist_src from ref.route_version").fetchall() == [("gc",)]


def test_versions_dedup_by_hash(db):
    assert import_gtfs(db, FIXTURE).versions_new == 3
    report = import_gtfs(db, FIXTURE)
    assert report.versions_new == 0 and report.versions_dup == 3


def test_gtfs_times_past_24h(db):
    import_gtfs(db, FIXTURE)
    assert db.execute("select arr_off from ref.schedule_stop s join ref.schedule_version v on v.id=s.version_id where v.train='12951' and s.idx=2").fetchone() == (1570.0,)


def test_repeated_station_gets_occ2(db):
    import_gtfs(db, FIXTURE)
    assert db.execute("select occ from ref.schedule_stop s join ref.schedule_version v on v.id=s.version_id where v.train='12951' and station='AA' order by idx").fetchall() == [(1,), (2,)]


def test_days_of_run_bits(db):
    import_gtfs(db, FIXTURE)
    assert db.execute("select days_of_run from ref.schedule_version where train='12951'").fetchone() == (9,)
    assert db.execute("select fog_north from ref.calendar where day='2026-02-15'").fetchone() == (True,)
    assert db.execute("select fog_north from ref.calendar where day='2026-02-16'").fetchone() == (False,)


def test_class_mapping():
    cases = [("1", "RAJDHANI", "RAJ"), ("1", "SHATABDI", "SHT"), ("1", "VANDE BHARAT", "VB"), ("1", "DURONTO", "DRNT"), ("1", "SUPERFAST", "SF"), ("1", "MAIL", "MEX"), ("1", "EXP", "EXP"), ("1", "MEMU", "PASS"), ("01234", "EXP", "SPL"), ("1", "SPECIAL", "SPL"), ("1", "TRAIN", "UNKNOWN")]
    for number, name, expected in cases:
        assert train_class(number, name) == expected


def test_route_version_is_halts_only_and_immutable(db):
    import_gtfs(db, FIXTURE)
    before = db.execute("select * from ref.route_version order by id").fetchall()
    assert db.execute("select bool_and(halt) from ref.route_stop").fetchone() == (True,)
    import_gtfs(db, FIXTURE)
    assert db.execute("select * from ref.route_version order by id").fetchall() == before


def test_missing_or_inverted_calendar_is_reported_without_inventing_days(db, tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "feed")
    path = tmp_path / "feed/calendar.txt"
    path.write_text("service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nS1,1,0,0,1,0,0,0,20270101,20260101\n")
    report = import_gtfs(db, tmp_path / "feed")
    assert report.trains == 3 and report.versions_new == 0
    assert len(report.skipped_trips) == 3
    assert db.execute("select count(*) from ref.schedule_version").fetchone() == (0,)
