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
    override.write_text('stn_a,stn_b,polyline,confidence\nAA,BB,abc,0.9\n')
    build_segments(db, override)
    assert db.execute("select polyline,confidence from ref.segment").fetchone() == ("abc", pytest.approx(.9))
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
