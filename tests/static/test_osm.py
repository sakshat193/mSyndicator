import json

import psycopg
import pytest

from msynd.static import osm
from msynd.static.segments import build_segments
from tests.helpers import make_run


POINTS = [(80, 20), (80.1, 20), (80.1, 20.1)]


def opl_file(tmp_path, points=POINTS, stations="", ways=None):
    path = tmp_path / "rail.opl"
    refs = ",".join(f"n{i}x{lon}y{lat}" for i, (lon, lat) in enumerate(points, 1))
    path.write_text(stations + (ways if ways is not None else f"w1 Trailway=rail N{refs}\n"), encoding="utf-8")
    return path


def route(db, target=21.57, points=(POINTS[0], POINTS[-1]), reverse=False):
    halts = [("AA", None, 0, 0), ("BB", 30, None, target)]
    if reverse:
        halts = [("BB", None, 0, target), ("AA", 30, None, 0)]
    make_run(db, halts=halts)
    for code, name, (lon, lat) in zip(("AA", "BB"), ("Alpha Junction", "Beta Halt"), points):
        db.execute("update ref.station set name=%s,lon=%s,lat=%s where code=%s", (name, lon, lat, code))
    build_segments(db)


def stored(db):
    return db.execute("select stn_lo,stn_hi,polyline,confidence from ref.segment_osm order by stn_lo,stn_hi").fetchall()


def test_parse_opl(tmp_path):
    path = opl_file(tmp_path, stations=(
        "n30518292 Tname=Alpha%20%Junction,name:hi=%0928%%093e%,railway=station,ref=AA%3b%AX x80 y20\n"
        "n8 Trailway=station,name=Bad%zz% x80 y20\n"
        "n9 Trailway=station xnan y20\n"
        "n10 Trailway=halt x67 y20\n"
        "n11 Trailway=halt x80 y38\n"
        "n12 Trailway=station y20\n"
        "garbage\n"
    ), ways=(
        "w4774129 Telectrified=contact_line,gauge=1676,railway=rail,usage=main Nn1x80y20,n2x80.1y20,n3,n4x80.2y20,n5x80.3y20\n"
        "w2 Trailway=narrow_gauge Nn5x80.3y20,n6x80.4y20\n"
        "w3 Trailway=road Nn6x80.4y20,n7x80.5y20\n"
        "w4 Trailway=rail Nn1x80y20,n7xinfy20\n"
        "w5 Trailway=rail Nn1x80y20,n8x99y20\n"
        "w6 Trailway=rail Nn1x80y20,nope\n"
    ))
    points, graph, ends, stations, skipped, service = osm.parse_opl(path)
    assert points[1] == (80, 20)
    assert set(graph[1]) == {2}
    assert set(graph[2]) == {1}
    assert set(graph[4]) == {5}  # Missing location breaks, rather than bridges, the way.
    assert 3 not in points and 7 not in points
    assert {1, 2, 4, 5, 6} <= ends
    assert stations[30518292] == ((80, 20), {
        "name": "Alpha Junction", "name:hi": "ना", "railway": "station", "ref": "AA;AX",
    })
    assert skipped == {"malformed": 4, "non_finite": 2, "out_of_bbox": 3}
    assert service == set()


def test_parse_opl_service_edges(tmp_path):
    path = opl_file(tmp_path, ways=(
        "w1 Trailway=rail,service=siding Nn3x80.02y20,n2x80.01y20,n1x80y20\n"
        "w2 Trailway=narrow_gauge,service=yard Nn3x80.02y20,n4x80.03y20\n"
        "w3 Trailway=rail,service= Nn4x80.03y20,n5x80.04y20\n"
        "w4 Trailway=rail Nn5x80.04y20,n6x80.05y20\n"
        "w5 Trailway=road,service=spur Nn6x80.05y20,n7x80.06y20\n"
        "w6 Trailway=rail,service=crossover Nn6x80.05y20,n9,n10x80.1y20,n11x80.11y20\n"
        "w7 Trailway=rail,service=siding Nn11x80.11y20,n12xinfy20\n"
        "w8 Trailway=rail,service=spur Nn12x80.12y20,n12x80.12y20,n13x80.13y20\n"
    ))
    _, _, _, _, _, service = osm.parse_opl(path)
    assert service == {(1, 2), (2, 3), (3, 4), (6, 9), (9, 10), (10, 11), (12, 12), (12, 13)}


@pytest.mark.parametrize("main_lat,foot_lat", [(20.00054, 20.00054), (20.02, 20.00018)])
def test_anchors_prefer_main_line_and_preserve_split_service_edges(tmp_path, main_lat, foot_lat):
    path = opl_file(tmp_path, ways=(
        "w1 Trailway=rail,service=siding Nn1x80y20.00018,n2x80.01y20.00018\n"
        f"w2 Trailway=rail Nn3x80y{main_lat},n4x80.01y{main_lat}\n"
    ))
    points, graph, _, stations, _, service = osm.parse_opl(path)
    anchors, counts = osm.anchor_stations([
        ("AA", "", 20, 80.004), ("BB", "", 20, 80.006),
    ], stations, points, graph, service)
    assert counts == {"ref": 0, "name": 0, "coord": 2, "no_anchor": 0}
    for code, lon in (("AA", 80.004), ("BB", 80.006)):
        node = anchors[code][0]
        assert points[node] == pytest.approx((lon, foot_lat))
        assert anchors[code][1] == .3
        for neighbor in graph[node]:
            assert (tuple(sorted((node, neighbor))) in service) is (foot_lat == 20.00018)


@pytest.mark.parametrize("points,paths,expected", [
    ([(80, 20), (80.01, 20), (80, 20.000054), (80.01, 20.000054)],
     [(1, 2), (3, 4)], {(1, 3), (2, 4)}),
    ([(80, 20), (80.01, 20), (80, 20.000054), (80.01, 20.000054), (80, 20.000108), (80.01, 20.000108)],
     [(1, 2), (3, 4), (5, 6)], {(1, 3), (1, 5), (3, 5), (2, 4), (2, 6), (4, 6)}),
    ([(80, 20), (80.01, 20), (80, 20.000054), (80, 20.01)],
     [(1, 2), (3, 4)], set()),
    ([(80, 20), (80.00005, 20), (80.0001, 20)], [(1, 2, 3)], set()),
    ([(80, 20), (80.001, 20), (80.001, 20.000054), (80, 20.000054)],
     [(1, 2, 3, 4)], {(1, 4)}),
    ([(80, 20), (80.0001, 20), (80.0001, 20.000054), (80, 20.000054)],
     [(1, 2, 3, 4)], set()),
    ([(80, 20), (80.01, 20), (80, 20.00014), (80.01, 20.00014)],
     [(1, 2), (3, 4)], set()),
    ([(80, 36.99), (80, 37), (80.00016, 36.99), (80.00016, 37)],
     [(1, 2), (3, 4)], {(1, 3), (2, 4)}),
])
def test_link_parallel(tmp_path, points, paths, expected):
    ways = "".join(f"w{i} Trailway=rail N" + ",".join(
        f"n{node}x{points[node-1][0]}y{points[node-1][1]}" for node in path) + "\n"
        for i, path in enumerate(paths, 1))
    points, graph, _, _, _, _ = osm.parse_opl(opl_file(tmp_path, ways=ways))
    before = {(a, b) for a in graph for b in graph[a] if a < b}
    assert osm.link_parallel(points, graph) == len(expected)
    assert {(a, b) for a in graph for b in graph[a] if a < b} - before == expected
    for a, b in expected:
        assert graph[a][b] == graph[b][a]
        assert 0 < graph[a][b] <= .015


@pytest.mark.db
def test_refresh_links_station_feet_on_unconnected_double_line(db, tmp_path):
    route(db, target=.02, points=((80.004, 20), (80.0041, 20.000054)))
    report = osm.refresh(db, opl_file(tmp_path, ways=(
        "w1 Trailway=rail Nn1x80y20,n2x80.01y20\n"
        "w2 Trailway=rail Nn3x80y20.000054,n4x80.01y20.000054\n"
    )))
    assert json.loads(stored(db)[0][2]) == [[80.004, 20], [80.0041, 20.00005]]
    assert stored(db)[0][3] == .3
    assert type(report["links"]) is int and report["links"] == 3
    assert report["failures"] == {"no_anchor": 0, "no_path": 0, "length": 0}


@pytest.mark.db
@pytest.mark.parametrize("points,accepted", [([POINTS[0], POINTS[-1]], True), (POINTS, False)])
def test_short_paths_must_be_direct(db, tmp_path, points, accepted):
    route(db, target=30)
    report = osm.refresh(db, opl_file(tmp_path, points=points, stations=(
        "n100 Trailway=station,ref=AA x80 y20\n"
        "n101 Trailway=station,name=Beta x80.1 y20.1\n"
    )))
    assert bool(stored(db)) is accepted
    assert report["failures"] == {"no_anchor": 0, "no_path": 0, "length": int(not accepted)}
    if accepted:
        assert stored(db)[0][3] == .5


@pytest.mark.db
def test_path_follows_rails_not_chord(db, tmp_path):
    route(db)
    report = osm.refresh(db, opl_file(tmp_path))
    lo, hi, polyline, confidence = stored(db)[0]
    assert (lo, hi, confidence) == ("AA", "BB", pytest.approx(.3))
    assert json.loads(polyline) == [[80, 20], [80.1, 20], [80.1, 20.1]]
    assert report["stations"] == {"ref": 0, "name": 0, "coord": 2, "no_anchor": 0}
    assert report["segments"] == {"1.0": 0, "0.5": 0, "0.3": 1}
    assert report["vertices"] == 3 and report["runtime_s"] >= 0
    assert report["failures"] == {"no_anchor": 0, "no_path": 0, "length": 0}
    assert type(report["links"]) is int and report["links"] == 0


@pytest.mark.db
@pytest.mark.parametrize("stations,confidence,tiers", [
    ("n100 Trailway=station,ref=xx;aa x80 y20\nn101 Trailway=halt,railway:ref=bB x80.1 y20.1\n", 1., (2, 0, 0)),
    ("n100 Trailway=station,ref=AA x80 y20\nn101 Trailway=station,name=Beta%20%Station x80.1 y20.1\n", .5, (1, 1, 0)),
    ("n100 Trailway=station,name=ALPHA%20%Jn. x80 y20\n", .3, (0, 1, 1)),
    ("n99 Trailway=station,ref=AA,name=Alpha x80.06 y20\nn100 Trailway=station,name=Alpha%20%Halt x80 y20\nn101 Trailway=station,ref=BB x80.1 y20.1\n", .5, (1, 1, 0)),
])
def test_confidence_rules(db, tmp_path, stations, confidence, tiers):
    route(db)
    report = osm.refresh(db, opl_file(tmp_path, stations=stations))
    assert stored(db)[0][3] == pytest.approx(confidence)
    assert tuple(report["stations"][tier] for tier in ("ref", "name", "coord")) == tiers


@pytest.mark.db
def test_nearest_name_and_ref_precede_coordinate(db, tmp_path):
    route(db)
    stations = (
        "n100 Trailway=station,ref=AA,name=Unrelated x80.001 y20\n"
        "n101 Trailway=station,name=Alpha x80 y20\n"
        "n102 Trailway=station,name=Beta x80.1 y20.101\n"
        "n103 Trailway=station,name=Beta x80.1 y20.2\n"
    )
    report = osm.refresh(db, opl_file(tmp_path, stations=stations))
    assert report["stations"] == {"ref": 1, "name": 1, "coord": 0, "no_anchor": 0}
    assert stored(db)[0][3] == pytest.approx(.5)


@pytest.mark.db
def test_name_tier_needs_radius_and_similarity(db, tmp_path):
    route(db)
    stations = (
        "n100 Trailway=station,name=Alpha x80.019 y20\n"  # About 2 km from AA.
        "n101 Trailway=station,name=Gamma x80.1006 y20.1\n"  # About 63 m from BB.
    )
    points = [POINTS[0], (80.019, 20), *POINTS[1:]]
    report = osm.refresh(db, opl_file(tmp_path, points=points, stations=stations))
    assert report["stations"] == {"ref": 0, "name": 0, "coord": 2, "no_anchor": 0}
    assert stored(db)[0][3] == pytest.approx(.3)


@pytest.mark.db
@pytest.mark.parametrize("target,points,accepted", [
    (18.75, POINTS, False), (20.54, POINTS, True), (24.2, POINTS, False),
    (3, [(80, 20), (80.03733, 20)], True),
    (3, [(80, 20), (80.03924, 20)], False),
])
def test_path_length_tolerance(db, tmp_path, target, points, accepted):
    route(db, target=target, points=(points[0], points[-1]))
    report = osm.refresh(db, opl_file(tmp_path, points=points))
    assert bool(stored(db)) is accepted
    assert sum(report["failures"].values()) == int(not accepted)


@pytest.mark.db
def test_length_excludes_station_anchor_stubs(db, tmp_path):
    route(db, target=20, points=((79.996, 20), (80.1, 20.105)))
    report = osm.refresh(db, opl_file(tmp_path))
    assert len(stored(db)) == 1
    assert report["failures"]["length"] == 0


@pytest.mark.db
def test_geometry_endpoints_orientation_simplified(db, tmp_path):
    points = [(80 + i * .001, 20) for i in range(201)]
    route(db, target=20.9, points=(points[0], points[-1]), reverse=True)
    osm.refresh(db, opl_file(tmp_path, points=points))
    lo, hi, polyline, _ = stored(db)[0]
    assert (lo, hi) == ("AA", "BB")
    assert " " not in polyline
    assert json.loads(polyline) == [[80, 20], [80.2, 20]]


@pytest.mark.db
@pytest.mark.parametrize("lon", [80.0154321, 80.0048])
def test_geometry_uses_track_feet_without_spurs(db, tmp_path, lon):
    route(db, target=(80.03-lon)*104.49, points=((lon, 20.004), (80.03, 20)))
    report = osm.refresh(db, opl_file(tmp_path, points=[(80, 20), (80.03, 20)]))
    points = json.loads(stored(db)[0][2])
    assert points == [[round(lon, 5), 20], [80.03, 20]]
    assert db.execute("select code,lon,lat from ref.station_osm order by code").fetchall() == [
        ("AA", *points[0]), ("BB", *points[-1]),
    ]
    assert report["snapped"] == 2
    assert report["move_m"]["p50"] == pytest.approx(222.39, abs=.1)
    assert report["move_m"]["p90"] == pytest.approx(400.30, abs=.1)
    assert db.execute("select lon,lat from ref.station where code='AA'").fetchone() == (lon, 20.004)


@pytest.mark.db
@pytest.mark.parametrize("points,ways,reason", [
    (((80, 20), (80.1, 20.12)), None, "no_anchor"),
    ((POINTS[0], POINTS[-1]), "w1 Trailway=rail Nn1x80y20,n2x80.01y20,n99,n3x80.1y20.1,n4x80.11y20.1\n", "no_path"),
    ((POINTS[0], POINTS[0]), None, "no_path"),
])
def test_unroutable_segments_have_no_row(db, tmp_path, points, ways, reason):
    route(db, points=points)
    report = osm.refresh(db, opl_file(tmp_path, ways=ways))
    assert stored(db) == [] and report["failures"][reason] == 1
    if reason == "no_anchor":
        assert report["stations"]["no_anchor"] == 1


def test_contraction_keeps_junctions_way_ends_and_anchors(tmp_path):
    path = opl_file(tmp_path, ways=(
        "w1 Trailway=rail Nn1x80y20,n2x80.02y20,n3x80.04y20,n4x80.06y20,n5x80.08y20,n6x80.1y20\n"
        "w2 Trailway=rail Nn4x80.06y20,n7x80.06y20.02\n"
        "w3 Trailway=rail Nn6x80.1y20,n8x80.12y20\n"
    ))
    points, graph, ends, _, _, _ = osm.parse_opl(path)
    contracted = osm.contract(graph, points, ends | {3})
    assert set(contracted) == {1, 3, 4, 6, 7, 8}
    found = osm.astar(contracted, points, 1, 8, 20)
    assert found is not None
    polyline, km = found
    assert polyline == [(80, 20), (80.02, 20), (80.04, 20), (80.06, 20), (80.08, 20), (80.1, 20), (80.12, 20)]
    assert km == pytest.approx(12.539, abs=.002)
    assert osm.astar(contracted, points, 8, 1, 20)[0] == list(reversed(polyline))
    assert osm.astar(contracted, points, 1, 8, 1) is None


@pytest.mark.db
def test_target_is_median_absolute_route_distance_and_refresh_replaces(db, tmp_path):
    route(db)
    make_run(db, train="2", halts=[("BB", None, 0, 20.54), ("AA", 30, None, 0)])
    make_run(db, train="3", halts=[("AA", None, 0, 0), ("BB", 30, None, 2)])
    build_segments(db)
    db.execute("insert into ref.segment_osm values ('OLD','ROW','[[80,20],[81,20]]',1)")
    db.execute("insert into ref.station_osm values ('OLD',80,20)")
    osm.refresh(db, opl_file(tmp_path))
    assert [row[:2] for row in stored(db)] == [("AA", "BB")]
    assert db.execute("select code from ref.station_osm order by code").fetchall() == [("AA",), ("BB",)]
    db.execute("update ref.route_stop set dist_km=0")
    osm.refresh(db, opl_file(tmp_path))
    assert stored(db) == []
    assert db.execute("select * from ref.station_osm").fetchall() == []


@pytest.mark.db
@pytest.mark.parametrize("table", ["segment_osm", "station_osm"])
def test_refresh_replacement_rolls_back_on_write_failure(db, tmp_path, table):
    route(db)
    db.execute("insert into ref.segment_osm values ('OLD','ROW','[]',1)")
    db.execute("insert into ref.station_osm values ('OLD',80,20)")
    db.execute("""create function pg_temp.reject_osm() returns trigger language plpgsql as
                  $$ begin raise exception 'synthetic write failure'; end $$""")
    db.execute(f"create trigger reject_osm before insert on ref.{table} for each row execute function pg_temp.reject_osm()")
    with pytest.raises(psycopg.errors.RaiseException):
        osm.refresh(db, opl_file(tmp_path))
    assert stored(db) == [("OLD", "ROW", "[]", 1)]
    assert db.execute("select * from ref.station_osm").fetchall() == [("OLD", 80, 20)]


@pytest.mark.db
def test_two_stations_split_one_edge(db, tmp_path):
    route(db, target=1.5, points=((80.015, 20.002), (80.03, 19.998)))
    report = osm.refresh(db, opl_file(tmp_path, points=[(80, 20), (80.06, 20)]))
    assert json.loads(stored(db)[0][2]) == [[80.015, 20], [80.03, 20]]
    assert report["snapped"] == 2


def test_simplify_keeps_small_curves_and_drops_noise():
    assert osm.simplify([(80, 20), (80.01, 20.00018), (80.02, 20)]) == [
        [80, 20], [80.01, 20.00018], [80.02, 20],
    ]
    assert osm.simplify([(80, 20), (80.01, 20.00009), (80.02, 20)]) == [[80, 20], [80.02, 20]]
    assert osm.simplify([(80 + i*.001, 20) for i in range(101)]) == [[80, 20], [80.1, 20]]


def test_anchor_splits_raw_edges_before_contraction(tmp_path):
    points, graph, ends, stations, _, service = osm.parse_opl(opl_file(tmp_path, points=[(80, 20), (80.02, 20)]))
    anchors, counts = osm.anchor_stations([
        ("AA", "", 20.002, 80.005), ("BB", "", 19.998, 80.015), ("CC", "", 20, 80),
    ], stations, points, graph, service)
    a, b = anchors["AA"][0], anchors["BB"][0]
    assert a < 0 and b < 0 and a != b and anchors["CC"][0] == 1
    assert points[a] == pytest.approx((80.005, 20))
    assert points[b] == pytest.approx((80.015, 20))
    assert set(graph[a]) == {1, b} and set(graph[b]) == {a, 2}
    assert graph[a][b] == graph[b][a] == pytest.approx(1.0449, abs=.001)
    contracted = osm.contract(graph, points, ends | {v[0] for v in anchors.values()})
    assert osm.astar(contracted, points, a, b, 1)[1] == pytest.approx(1.0449, abs=.001)


def test_anchor_edge_ties_matched_point_and_radius(tmp_path):
    points, graph, _, stations, _, service = osm.parse_opl(opl_file(tmp_path, ways=(
        "w2 Trailway=rail Nn3x80y20,n4x80.02y20\n"
        "w1 Trailway=rail Nn1x80y20,n2x80.02y20\n"
    ), stations="n100 Trailway=station,ref=AA x80.01 y20\n"))
    anchors, counts = osm.anchor_stations([
        ("AA", "", 20.03, 80.01), ("BB", "", 20.01, 80.01),
    ], stations, points, graph, service)
    assert counts == {"ref": 1, "name": 0, "coord": 0, "no_anchor": 1}
    node = anchors["AA"][0]
    assert points[node] == pytest.approx((80.01, 20))
    assert set(graph[node]) == {1, 2}


@pytest.mark.db
@pytest.mark.parametrize("lon,lat", [(67.99, 20), (98.01, 20), (80, 5.99), (80, 37.01), (float("nan"), 20)])
def test_station_osm_bbox_checks(db, lon, lat):
    with pytest.raises(psycopg.errors.CheckViolation), db.transaction():
        db.execute("insert into ref.station_osm values ('UNKNOWN',%s,%s)", (lon, lat))


@pytest.mark.db
def test_station_osm_only_has_accepted_endpoints(db, tmp_path):
    route(db)
    make_run(db, train="2", halts=[("AA", None, 0, 0), ("CC", 30, None, 100)])
    db.execute("update ref.station set lon=80.1,lat=20.09 where code='CC'")
    build_segments(db)
    report = osm.refresh(db, opl_file(tmp_path))
    assert report["snapped"] == 2
    assert db.execute("select code from ref.station_osm order by code").fetchall() == [("AA",), ("BB",)]


@pytest.mark.db
@pytest.mark.parametrize("lo,hi,confidence", [("BB", "AA", 1), ("AA", "BB", 0), ("AA", "BB", 1.1)])
def test_osm_storage_checks(db, lo, hi, confidence):
    with pytest.raises(psycopg.errors.CheckViolation), db.transaction():
        db.execute("insert into ref.segment_osm values (%s,%s,'[]',%s)", (lo, hi, confidence))
