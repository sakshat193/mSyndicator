import gzip
import json

import pytest

from msynd.static.gtfs import import_gtfs
from msynd.static.segments import build_segments
from msynd.static.export import export_network
from tests.helpers import make_run
from tests.static.test_gtfs import FIXTURE

pytestmark = pytest.mark.db


def test_segment_undirected_shared_by_reverse_route(db):
    make_run(db, train="1", halts=[("AA", None, 0, 0), ("BB", 30, None, 10)])
    make_run(db, train="2", halts=[("BB", None, 0, 0), ("AA", 30, None, 10)])
    assert build_segments(db) == 1
    assert db.execute("select stn_lo,stn_hi from ref.segment").fetchall() == [("AA", "BB")]


def test_route_segment_forward_flag(db):
    make_run(db, train="1", halts=[("AA", None, 0, 0), ("BB", 30, None, 10)])
    make_run(db, train="2", halts=[("BB", None, 0, 0), ("AA", 30, None, 10)])
    build_segments(db)
    assert db.execute("select forward from ref.route_segment order by route_version_id").fetchall() == [(True,), (False,)]


def test_override_applied_and_versioned_by_file_hash(db, tmp_path):
    make_run(db, halts=[("AA", None, 0, 0), ("BB", 30, None, 10)])
    build_segments(db)
    _, first = export_network(db, tmp_path)
    override = tmp_path / "overrides.csv"
    override.write_text('stn_a,stn_b,polyline,confidence\nAA,BB,"[[0,0],[1,1]]",0.9\n')
    build_segments(db, override)
    assert db.execute("select polyline,confidence from ref.segment").fetchone() == ("[[0,0],[1,1]]", pytest.approx(.9))
    _, second = export_network(db, tmp_path)
    assert first != second
    override.write_text('stn_a,stn_b,polyline,confidence\n')
    build_segments(db, override)
    assert db.execute("select polyline,confidence from ref.segment").fetchone() == (None, 0)


def test_export_etag_stable_for_same_content(db, tmp_path):
    import_gtfs(db, FIXTURE)
    build_segments(db)
    path, etag = export_network(db, tmp_path)
    body = path.read_bytes()
    again, same = export_network(db, tmp_path)
    assert again == path and same == etag and path.read_bytes() == body
    network = json.loads(gzip.decompress(body))
    assert len(network["stations"]) == 3 and network["routes"]
    assert all("segments" in route and "stops" in route for route in network["routes"])


def osm_segment(db):
    make_run(db, halts=[("AA", None, 0, 0), ("BB", 30, None, 10)])
    db.execute("insert into ref.segment_osm values ('AA','BB','[[80,20],[80.1,20]]',0.5)")
    db.execute("insert into ref.segment_osm values ('OLD','ROW','[]',0.3)")
    build_segments(db)


def test_override_wins(db, tmp_path):
    osm_segment(db)
    override = tmp_path / "overrides.csv"
    override.write_text('stn_a,stn_b,polyline,confidence\nAA,BB,"[[80,20],[80.05,20.02],[80.1,20]]",0.9\n')
    build_segments(db, override)
    assert db.execute("select polyline,confidence from ref.segment").fetchone() == (
        "[[80,20],[80.05,20.02],[80.1,20]]", pytest.approx(.9),
    )


def test_osm_survives_rebuild_and_unoverride(db, tmp_path):
    osm_segment(db)
    override = tmp_path / "overrides.csv"
    override.write_text('stn_a,stn_b,polyline,confidence\nAA,BB,,0\n')
    build_segments(db, override)
    assert db.execute("select polyline,confidence from ref.segment").fetchone() == (None, 0)
    override.write_text('stn_a,stn_b,polyline,confidence\n')
    for _ in range(2):
        build_segments(db, override)
        assert db.execute("select stn_lo,stn_hi,polyline,confidence from ref.segment").fetchall() == [
            ("AA", "BB", "[[80,20],[80.1,20]]", .5),
        ]
    path, _ = export_network(db, tmp_path, override)
    assert json.loads(gzip.decompress(path.read_bytes()))["segments"][0]["polyline"] == [[80, 20], [80.1, 20]]


@pytest.mark.parametrize("codes", ["ABCD", "ZBCA", "BACD"])
def test_chain_reuses_constituent_polylines(db, tmp_path, codes):
    from tests.static.test_export import exported

    points = [[0, 0], [.3, .05], [.6, .05], [.9, 0]]
    make_run(db, train="1", halts=[(codes[0], None, 0, 0), (codes[-1], 60, None, 100)])
    make_run(db, train="2", halts=[(code, i * 20, i * 20 + 1, km)
                                  for i, (code, km) in enumerate(zip(codes, (0, 30, 60, 100)))])
    for code, (lon, lat) in zip(codes, points):
        db.execute("update ref.station set lon=%s,lat=%s where code=%s", (lon, lat, code))
    build_segments(db)
    curves = [([points[0], [.15, .1], points[1]], codes[:2]),
              ([points[2], [.75, .1], points[3]], codes[2:])]
    for curve, (a, b) in curves:
        if a > b:
            curve = list(reversed(curve))
        db.execute("update ref.segment set polyline=%s where stn_lo=%s and stn_hi=%s",
                   (json.dumps(curve), min(a, b), max(a, b)))
    network, _, _ = exported(db, tmp_path)
    chained = next(s for s in network["segments"] if (s["stn_lo"], s["stn_hi"]) == tuple(sorted((codes[0], codes[-1]))))
    assert chained["derived"] is True
    expected = [[0, 0], [.15, .1], [.3, .05], [.6, .05], [.75, .1], [.9, 0]]
    assert chained["polyline"] == (expected if codes[0] < codes[-1] else list(reversed(expected)))
