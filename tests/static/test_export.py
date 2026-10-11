import gzip
import json

import pytest

from msynd.static.export import export_network
from msynd.static.segments import build_segments
from tests.helpers import make_run

pytestmark = pytest.mark.db


def chain_network(db, points):
    make_run(db, train="00001", halts=[("A", None, 0, 0), ("D", 60, None, 100)])
    make_run(db, train="2", halts=[("A", None, 0, 0), ("B", 20, 21, 30),
                                 ("C", 40, 41, 60), ("D", 60, None, 100)])
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
])
def test_unsuitable_chain_stays_null(db, tmp_path, points):
    chain_network(db, points)
    network, _, _ = exported(db, tmp_path)
    assert chord(network)["polyline"] is None
    assert chord(network)["derived"] is False


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
