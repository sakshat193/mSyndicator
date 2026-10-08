from datetime import datetime, timedelta, timezone

import pytest

from msynd.engine.evidence import Evidence, canonical_events, load_run_evidence
from tests.helpers import insert_event, make_run

pytestmark = pytest.mark.db
IST = timezone(timedelta(hours=5, minutes=30))
BASE = datetime(2026, 10, 7, tzinfo=IST)


def at(minute):
    return BASE + timedelta(minutes=minute)


def pending(db, run, minute, source="ntes_status", station="BRC", kind="arr", occ=1):
    db.execute("""insert into obs.pending_rev(run_id,station,occ,kind,source,received_at)
        values (%s,%s,%s,%s,%s,%s)""", (run, station, occ, kind, source, at(minute)))


def exception(db, minute, detail):
    from psycopg.types.json import Jsonb

    db.execute("insert into obs.exception_rev(train,start_date,detail,received_at) values ('12951','2026-10-07',%s,%s)",
               (Jsonb(detail), at(minute)))


def test_actual_overrides_pending(db):
    run = make_run(db)
    insert_event(db, run, "BRC", 1, "arr", at(310), at(320))
    pending(db, run, 330)
    assert load_run_evidence(db, run).evidence == (Evidence(1, True, 310, "ntes_status"),)


def test_latest_pending_only_across_sources(db):
    run = make_run(db)
    pending(db, run, 2)
    pending(db, run, 3, "ntes_board")
    assert load_run_evidence(db, run).evidence == (Evidence(1, False, 3, "ntes_board"),)


def test_pending_tie_prefers_mmt_status(db):
    run = make_run(db)
    pending(db, run, 3, "ntes_board")
    pending(db, run, 3, "mmt_status")
    assert load_run_evidence(db, run).evidence == (Evidence(1, False, 3, "mmt_status"),)


def test_asof_excludes_later_received(db):
    run = make_run(db)
    insert_event(db, run, "BRC", 1, "arr", at(600), at(620))
    pending(db, run, 616)
    exception(db, 616, {"resched_min": 60, "diverted_from": "BRC"})
    db.execute("update obs.run set resched_min=60,diverted_from='BRC' where id=%s", (run,))
    ev = load_run_evidence(db, run, at(615))
    assert ev.evidence == () and ev.resched_min == 0 and ev.stop_event is None
    assert load_run_evidence(db, run, at(620)).evidence == (Evidence(1, True, 600, "ntes_status"),)


def test_asof_uses_earlier_actual_revision(db):
    run = make_run(db)
    insert_event(db, run, "BRC", 1, "arr", at(300), at(310))
    insert_event(db, run, "BRC", 1, "arr", at(305), at(330))
    assert load_run_evidence(db, run, at(320)).evidence == (Evidence(1, True, 300, "ntes_status"),)
    assert load_run_evidence(db, run).evidence == (Evidence(1, True, 305, "ntes_status"),)


def test_resched_asof_receipt(db):
    run = make_run(db)
    exception(db, 10, {"resched_min": 20})
    insert_event(db, run, "MMCT", 1, "dep", at(30), at(40))
    exception(db, 40, {"resched_min": 50})  # strict before actual receipt
    exception(db, 41, {"resched_min": 60})
    assert load_run_evidence(db, run, at(20)).resched_min == 20
    assert load_run_evidence(db, run).resched_min == 20


def test_late_event_equals_fresh_assembly(db):
    run = make_run(db)
    insert_event(db, run, "BRC", 1, "arr", at(310), at(320))
    first = load_run_evidence(db, run)
    insert_event(db, run, "MMCT", 1, "dep", at(10), at(330))
    late = load_run_evidence(db, run)
    assert first.evidence == (Evidence(1, True, 310, "ntes_status"),)
    assert late.evidence == (Evidence(0, True, 10, "ntes_status"), Evidence(1, True, 310, "ntes_status"))
    db.execute("delete from obs.event_rev where run_id=%s", (run,))
    insert_event(db, run, "MMCT", 1, "dep", at(10), at(330))
    insert_event(db, run, "BRC", 1, "arr", at(310), at(320))
    assert late == load_run_evidence(db, run)


def test_route_rebind_keeps_keys(db):
    run = make_run(db)
    insert_event(db, run, "BRC", 1, "arr", at(310), at(320))
    before = load_run_evidence(db, run)
    schedule, old_route = db.execute("select schedule_version_id,route_version_id from obs.run where id=%s", (run,)).fetchone()
    route = db.execute("""insert into ref.route_version(schedule_version_id,content_hash,dist_src)
        values (%s,'new','ntes') returning id""", (schedule,)).fetchone()[0]
    db.execute("""insert into ref.route_stop select %s,case when idx=0 then 0 else idx+1 end,
        station,occ,halt,dist_km,sched_off from ref.route_stop where route_version_id=%s""", (route, old_route))
    db.execute("insert into ref.station(code,name) values ('PASS','Passing')")
    db.execute("insert into ref.route_stop values (%s,1,'PASS',1,false,196,NULL)", (route,))
    db.execute("update obs.run set route_version_id=%s where id=%s", (route, run))
    after = load_run_evidence(db, run)
    a, b = before.events[before.evidence[0].event], after.events[after.evidence[0].event]
    assert (a.station, a.occ, a.kind, a.halt_idx) == (b.station, b.occ, b.kind, b.halt_idx)
    assert after.evidence == (Evidence(2, True, 310, "ntes_status"),)
    assert after.events[1].kind == "pass" and after.events[1].sched_off == 150
    assert after.events[1].seg == 0
    assert before.versions["route"] != after.versions["route"]


def test_part_cancelled_tail_sets_stop_event(db):
    run = make_run(db)
    exception(db, 10, {"part_cancelled": [{"from": "BRC", "to": "NDLS"}]})
    pending(db, run, 1000, station="NDLS")
    ev = load_run_evidence(db, run)
    assert ev.stop_event == 1
    assert ev.events[ev.stop_event].station == "BRC"
    assert ev.evidence == ()


def test_part_cancelled_head_starts_at_first_operated_halt(db):
    run = make_run(db)
    exception(db, 10, {"part_cancelled": [{"from": "MMCT", "to": "BRC"}], "resched_min": 20})
    insert_event(db, run, "MMCT", 1, "dep", at(10), at(30))
    insert_event(db, run, "BRC", 1, "dep", at(330), at(340))
    ev = load_run_evidence(db, run)
    assert [(e.station, e.kind, e.halt_idx) for e in ev.events] == [("BRC", "dep", 1), ("NDLS", "arr", 2)]
    assert ev.evidence == (Evidence(0, True, 330, "ntes_status"),)
    assert ev.resched_min == 20 and ev.stop_event is None


def test_repeated_station_occurrences_remain_distinct(db):
    run = make_run(db, halts=[("MMCT", None, 0, 0), ("BRC", 100, 105, 100), ("BRC", 200, None, 200)])
    insert_event(db, run, "BRC", 1, "arr", at(110), at(300))
    insert_event(db, run, "BRC", 2, "arr", at(210), at(300))
    assert load_run_evidence(db, run).evidence == (Evidence(1, True, 110, "ntes_status"), Evidence(3, True, 210, "ntes_status"))


def test_canonical_events_order_and_segment_membership(db):
    run = make_run(db)
    ev = load_run_evidence(db, run)
    assert [(e.idx, e.kind, e.halt_idx, e.seg, e.sched_off) for e in ev.events] == [
        (0, "dep", 0, 0, 0), (1, "arr", 1, 0, 300),
        (2, "dep", 1, 1, 305), (3, "arr", 2, 1, 940)
    ]


def test_missing_run_raises(db):
    with pytest.raises(ValueError, match="run"):
        load_run_evidence(db, -1)


def test_part_cancelled_middle_is_diversion(db):
    run = make_run(db, halts=[("MMCT", None, 0, 0), ("BRC", 100, 105, 100),
                              ("MID", 200, 205, 200), ("NDLS", 300, None, 300)])
    exception(db, 10, {"part_cancelled": [{"from": "BRC", "to": "MID"}]})
    ev = load_run_evidence(db, run)
    assert ev.stop_event == 1 and ev.events[ev.stop_event].kind == "arr"


def test_diversion_uses_latest_eligible_exception(db):
    run = make_run(db)
    db.execute("update obs.run set diverted_from='BRC' where id=%s", (run,))
    exception(db, 10, {"diverted_from": "BRC"})
    exception(db, 20, {})
    assert load_run_evidence(db, run, at(15)).stop_event == 1
    assert load_run_evidence(db, run).stop_event is None


def test_versions_identify_active_models_and_bias(db):
    run = make_run(db)
    model = db.execute("""insert into ml.model(role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active)
        values ('run','empirical',1,now(),'[2026-09-01,2026-10-01)','model','{}','{}',true) returning id""").fetchone()[0]
    db.execute("insert into ml.leg_bias values (%s,4,'MMCT','BRC',0,3,10)", (model,))
    ev = load_run_evidence(db, run)
    schedule, route = db.execute("select schedule_version_id,route_version_id from obs.run where id=%s", (run,)).fetchone()
    assert ev.versions["schedule"] == schedule and ev.versions["route"] == route
    assert ev.versions["models"] == {"run": model}
    assert ev.versions["feature_ver"] == 1 and ev.versions["bias_gen"] == 4


def test_canonical_events_sorts_route_rows(db):
    stops = [
        {"idx": 0, "station": "A", "occ": 1, "halt": True, "halt_idx": 0,
         "arr_off": None, "dep_off": 0, "dist_km": 0, "sched_off": 0},
        {"idx": 1, "station": "P", "occ": 1, "halt": False, "halt_idx": None,
         "arr_off": None, "dep_off": None, "dist_km": 10, "sched_off": 40},
        {"idx": 2, "station": "B", "occ": 1, "halt": True, "halt_idx": 1,
         "arr_off": 100, "dep_off": None, "dist_km": 20, "sched_off": 100}
    ]
    events = canonical_events(list(reversed(stops)))
    assert [(e.station, e.kind, e.sched_off, e.seg) for e in events] == [
        ("A", "dep", 0, 0), ("P", "pass", 40, 0), ("B", "arr", 100, 0)
    ]
