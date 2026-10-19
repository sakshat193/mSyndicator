import gzip
import json

import pytest

from msynd.static.export import export_network
from msynd.static.segments import build_segments
from tests.helpers import make_run

pytestmark = pytest.mark.db


def chain_network(db, points, distances=(0, 30, 60, 100)):
    make_run(db, train="00001", halts=[("A", None, 0, 0), ("D", 60, None, distances[-1])])
    make_run(db, train="2", halts=[("A", None, 0, distances[0]), ("B", 20, 21, distances[1]),
                                 ("C", 40, 41, distances[2]), ("D", 60, None, distances[3])])
    for code, (lon, lat) in zip("ABCD", points):
        db.execute("update ref.station set lon=%s,lat=%s where code=%s", (lon, lat, code))
    build_segments(db)


def exported(db, tmp_path):
    path, etag = export_network(db, tmp_path)
    return json.loads(gzip.decompress(path.read_bytes())), path, etag


def chord(network):
    return next(s for s in network["segments"] if (s["stn_lo"], s["stn_hi"]) == ("A", "D"))


def test_long_chord_uses_station_chain_without_writing_db(db, tmp_path):
    points = [[0, 0], [.3, .05], [.6, .05], [.9, 0]]
    chain_network(db, points)
    network, path, etag = exported(db, tmp_path)
    assert chord(network)["polyline"] == points
    assert chord(network)["derived"] is True
    assert db.execute("select polyline from ref.segment where stn_lo='A' and stn_hi='D'").fetchone() == (None,)
    body = path.read_bytes()
    _, again, same = exported(db, tmp_path)
    assert again == path and same == etag and again.read_bytes() == body


@pytest.mark.parametrize("points", [
    [[0, 0], [.3, .43], [.6, .43], [.9, 0]],  # About 1.5x the chord.
    [[0, 0], [.03, .005], [.06, .005], [.09, 0]],  # Chord below 20 km.
    [[0, 0], [None, None], [.6, .05], [.9, 0]],  # No complete chain.
    [[0, 0], [.3, .05], [.6, .05], [None, None]],  # Missing chord endpoint.
    [[0, 0], [.45, 0], [.9, 0], [1.7, 0]],  # Chain contains a hop over 80 km.
])
def test_unsuitable_chain_stays_null(db, tmp_path, points):
    chain_network(db, points)
    network, _, _ = exported(db, tmp_path)
    assert chord(network)["polyline"] is None
    assert chord(network)["derived"] is False


@pytest.mark.parametrize("offset,derived", [(.05, True), (.16, False)])
def test_isolated_origin_snaps_only_within_15_km(db, tmp_path, offset, derived):
    make_run(db, train="00001", halts=[("A", None, 0, 0), ("D", 60, None, 100)])
    make_run(db, train="2", halts=[("B", None, 0, 0), ("C", 30, 31, 50), ("D", 60, None, 100)])
    points = [[0, 0], [offset, 0], [.5, 0], [1, 0]]
    for code, (lon, lat) in zip("ABCD", points):
        db.execute("update ref.station set lon=%s,lat=%s where code=%s", (lon, lat, code))
    build_segments(db)
    network, _, _ = exported(db, tmp_path)
    assert chord(network)["polyline"] == (points if derived else None)
    assert chord(network)["derived"] is derived


def test_real_detour_matches_route_track_km(db, tmp_path):
    points = [[0, 0], [.3, .63], [.6, .63], [.9, 0]]
    chain_network(db, points, distances=(0, 78, 112, 190))
    network, _, _ = exported(db, tmp_path)
    assert chord(network)["polyline"] == points
    assert chord(network)["derived"] is True


def test_track_km_is_median_of_absolute_route_distances(db, tmp_path):
    points = [[0, 0], [.3, .63], [.6, .63], [.9, 0]]
    chain_network(db, points, distances=(0, 78, 112, 190))
    make_run(db, train="3", halts=[("A", None, 0, 0), ("D", 60, None, 100)])
    make_run(db, train="4", halts=[("D", None, 0, 600), ("A", 60, None, 0)])
    build_segments(db)
    network, _, _ = exported(db, tmp_path)
    assert chord(network)["polyline"] == points
    assert chord(network)["derived"] is True


def test_chain_shorter_than_track_km_is_still_drawn(db, tmp_path):
    points = [[0, 0], [.3, .05], [.6, .05], [.9, 0]]
    chain_network(db, points)
    db.execute("""update ref.route_stop set dist_km=190 where station='D'
                  and route_version_id=(select id from ref.route_version
                    where schedule_version_id=(select id from ref.schedule_version where train='00001'))""")
    network, _, _ = exported(db, tmp_path)
    assert chord(network)["polyline"] == points
    assert chord(network)["derived"] is True


def test_plain_km_search_finds_direct_chain_beside_tiny_meandering_hops(db, tmp_path):
    direct = [[0, 0], [.3, 0], [.6, 0], [.9, 0]]
    chain_network(db, direct)
    meander = {f"M{i}": (i * .09, .07 if i % 2 else -.07) for i in range(1, 10)}
    make_run(db, train="3", halts=[(code, i * 10, i * 10, i * 18)
                                 for i, code in enumerate(["A", *meander, "D"])])
    for code, (lon, lat) in meander.items():
        db.execute("update ref.station set lon=%s,lat=%s where code=%s", (lon, lat, code))
    build_segments(db)
    network, _, _ = exported(db, tmp_path)
    assert chord(network)["polyline"] == direct
    assert chord(network)["derived"] is True


def test_stored_polyline_is_parsed_and_preserved(db, tmp_path):
    chain_network(db, [[0, 0], [.3, .05], [.6, .05], [.9, 0]])
    points = [[0, 0], [.2, .1], [.9, 0]]
    db.execute("update ref.segment set polyline=%s where stn_lo='A' and stn_hi='D'", (json.dumps(points),))
    network, _, _ = exported(db, tmp_path)
    assert chord(network)["polyline"] == points
    assert chord(network)["derived"] is False
    assert all(isinstance(s["derived"], bool) for s in network["segments"])


def test_invalid_stored_polyline_raises(db, tmp_path):
    chain_network(db, [[0, 0], [.3, .05], [.6, .05], [.9, 0]])
    db.execute("update ref.segment set polyline='invalid' where stn_lo='A' and stn_hi='D'")
    with pytest.raises(json.JSONDecodeError):
        export_network(db, tmp_path)


def test_routes_include_train_identity(db, tmp_path):
    chain_network(db, [[0, 0], [.3, .05], [.6, .05], [.9, 0]])
    db.execute("update ref.train set name='Chain Express',class='EXP' where number='00001'")
    network, _, _ = exported(db, tmp_path)
    route = next(r for r in network["routes"] if r["stops"][1]["station"] == "D")
    assert (route["train"], route["name"], route["class"]) == ("00001", "Chain Express", "EXP")


def test_export_uses_snapped_stations_for_chords_and_chains(db, tmp_path):
    points = [[80, 20], [80.3, 20.05], [80.6, 20.05], [80.9, 20]]
    chain_network(db, points)
    db.execute("insert into ref.station_osm values ('A',80.001,20.001),('C',80.601,20.051)")
    points[0], points[2] = [80.001, 20.001], [80.601, 20.051]
    network, _, _ = exported(db, tmp_path)
    stations = {s["code"]: [s["lon"], s["lat"]] for s in network["stations"]}
    assert [stations[code] for code in "ABCD"] == points
    assert chord(network)["polyline"] == points
    short = next(s for s in network["segments"] if (s["stn_lo"], s["stn_hi"]) == ("A", "B"))
    assert short["polyline"] is None
    assert [stations[short["stn_lo"]], stations[short["stn_hi"]]] == points[:2]
