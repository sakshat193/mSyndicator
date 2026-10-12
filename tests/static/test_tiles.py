import gzip
import json
import random
import sqlite3
import struct
import subprocess
import sys

import pytest

from msynd.static import tiles


def varint(value):
    body = bytearray()
    while value >= 128:
        body.append((value & 127) | 128)
        value >>= 7
    body.append(value)
    return bytes(body)


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    return varint((number << 3) | 2) + varint(len(value)) + value


def geometry(rings):
    values, x, y = [], 0, 0
    for ring in rings:
        values.append(9)  # MoveTo(1).
        for i, (nx, ny) in enumerate(ring):
            if i == 1:
                values.append(((len(ring) - 1) << 3) | 2)
            dx, dy = nx - x, ny - y
            values.extend((2 * dx if dx >= 0 else -2 * dx - 1,
                           2 * dy if dy >= 0 else -2 * dy - 1))
            x, y = nx, ny
        values.append(15)  # ClosePath(1); the cursor stays at the last vertex.
    return b"".join(map(varint, values))


def feature(rings, kind=3):
    return (field(1, 42) + field(2, b"\0\0") + field(9, b"unknown")
            + field(3, kind) + field(4, geometry(rings))
            + varint((100 << 3) | 5) + struct.pack("<I", 12345))


def layer(name, features=(), extent=4096):
    body = (field(1, name.encode()) + field(15, 2)
            + b"".join(field(2, f) for f in features)
            + field(3, b"kind") + field(4, field(1, b"river"))
            + field(21, b"unknown layer field"))
    return body if extent is None else body + field(5, extent)


def directory(entries):
    # PMTiles directory columns: ids, runs, lengths, offsets (offset+1, or 0).
    result, previous_id = varint(len(entries)), 0
    for tile_id, offset, length, run in entries:
        result += varint(tile_id - previous_id)
        previous_id = tile_id
    result += b"".join(varint(e[3]) for e in entries)
    result += b"".join(varint(e[2]) for e in entries)
    for i, (_, offset, _, _) in enumerate(entries):
        contiguous = i > 0 and offset == entries[i - 1][1] + entries[i - 1][2]
        result += varint(0 if contiguous else offset + 1)
    return result


def archive(root, bodies, leaves=b"", metadata=None, minzoom=2, maxzoom=2):
    if metadata is None:
        metadata = {
            "name": "Synthetic India", "attribution": "OSM; source credit",
            "description": "test", "planetiler:version": "test",
            "vector_layers": [
                {"id": name, "fields": {"kind": "String"}, "minzoom": 0, "maxzoom": 10}
                for name in ("boundaries", "earth", "landuse", "places", "roads", "water")
            ],
        }
    root = gzip.compress(root, mtime=0)
    meta = gzip.compress(json.dumps(metadata).encode(), mtime=0)
    data = b"".join(bodies)
    header = bytearray(127)
    header[:8] = b"PMTiles\x03"
    offset = 127
    for pos, section in ((8, root), (24, meta), (40, leaves), (56, data)):
        struct.pack_into("<QQ", header, pos, offset, len(section))
        offset += len(section)
    struct.pack_into("<QQQ", header, 72, 3, 2, len(bodies))
    header[96:102] = bytes((1, 2, 2, 1, minzoom, maxzoom))
    struct.pack_into("<iiiiBii", header, 102,
                     600_000_000, 0, 1_060_000_000, 400_000_000,
                     maxzoom, 830_000_000, 200_000_000)
    return bytes(header) + root + meta + leaves + data


BIG = [(-32, 0), (480, 0), (480, 512), (-32, 512)]
TINY = [(700, 0), (716, 0), (716, 16), (700, 16)]
SECOND = [(1000, 100), (1512, 100), (1512, 612), (1000, 612)]
HOLE = [(1100, 200), (1100, 216), (1116, 216), (1116, 200)]


@pytest.mark.parametrize("z,coords,start", [
    (0, [(0, 0)], 0),
    (1, [(0, 0), (0, 1), (1, 1), (1, 0)], 1),
    (2, [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2),
         (2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (2, 0), (3, 0)], 5),
])
def test_hilbert_matches_spec(z, coords, start):
    for tile_id, (x, y) in enumerate(coords, start):
        assert tiles.tileid_to_zxy(tile_id) == (z, x, y)
        assert tiles.zxy_to_tileid(z, x, y) == tile_id


def test_hilbert_z10_roundtrip():
    rng = random.Random(67)
    for tile_id in [349525, 1398100, *[rng.randrange(349525, 1398101) for _ in range(100)]]:
        z, x, y = tiles.tileid_to_zxy(tile_id)
        assert z == 10 and 0 <= x < 1024 and 0 <= y < 1024
        assert tiles.zxy_to_tileid(z, x, y) == tile_id


def test_leaf_directory_runs_and_contiguous_offsets():
    entries = [(5, 0, 20, 2), (7, 20, 30, 1)]
    leaf = gzip.compress(directory(entries), mtime=0)
    source = archive(directory([(5, 0, len(leaf), 0)]), [b"x" * 50], leaves=leaf)
    assert tiles.read_directory(directory(entries)) == entries
    assert [e for group in tiles.directories(source, tiles.read_header(source)) for e in group] == entries


@pytest.mark.parametrize("patch,message", [
    ((0, b"NotTile"), "magic"), ((7, b"\x02"), "version"),
    ((97, b"\x01"), "internal compression"), ((98, b"\x01"), "tile compression"),
    ((99, b"\x02"), "tile type"), ((8, struct.pack("<Q", 1 << 40)), "root.*range"),
])
def test_bad_header(patch, message):
    source = bytearray(archive(directory([]), []))
    offset, value = patch
    source[offset:offset + len(value)] = value
    with pytest.raises(ValueError, match=message):
        tiles.read_header(source)


@pytest.mark.parametrize("body,message", [
    (b"\x80", "truncated varint"), (b"\x80" * 10 + b"\x00", "overlong varint"),
    (b"\xff" * 9 + b"\x02", "overlong varint"), (b"\x81\x00", "overlong varint"),
    (b"\x01\x00", "directory"), (b"\x01\x00\x01\x01\x00", "offset"),
])
def test_bad_directory_varints(body, message):
    with pytest.raises(ValueError, match=message):
        tiles.read_directory(body)


def test_gunzip_is_capped_and_rejects_truncation():
    with pytest.raises(ValueError, match="16 MB"):
        tiles.gunzip(gzip.compress(b"x" * (16 * 1024 * 1024 + 1)))
    with pytest.raises(ValueError, match="gzip"):
        tiles.gunzip(gzip.compress(b"ok")[:-4])


@pytest.mark.parametrize("entry,message", [
    ((5, 100, 20, 1), "tile.*range"), ((5, 1, 20, 0), "leaf.*range"),
])
def test_directory_section_offsets_are_checked(entry, message):
    source = archive(directory([entry]), [b"x" * 50])
    with pytest.raises(ValueError, match=message):
        list(tiles.directories(source, tiles.read_header(source)))


@pytest.mark.parametrize("depth,valid", [(4, True), (5, False)])
def test_directory_depth_limit(depth, valid):
    leaf_parts = [gzip.compress(directory([(5, 0, 1, 1)]), mtime=0)]
    offset = 0
    for _ in range(depth - 2):
        leaf = gzip.compress(directory([(5, offset, len(leaf_parts[-1]), 0)]), mtime=0)
        offset = sum(map(len, leaf_parts))
        leaf_parts.append(leaf)
    root = directory([(5, offset, len(leaf_parts[-1]), 0)])
    source = archive(root, [b"x"], leaves=b"".join(leaf_parts))
    if valid:
        assert [e for group in tiles.directories(source, tiles.read_header(source)) for e in group] == [(5, 0, 1, 1)]
    else:
        with pytest.raises(ValueError, match="depth"):
            list(tiles.directories(source, tiles.read_header(source)))


def test_water_rewrites_only_geometry_and_keeps_other_layers_verbatim():
    water = layer("water", [feature([BIG, TINY, SECOND, HOLE]), feature([TINY], kind=2),
                            feature([TINY], kind=1)])
    earth, roads = layer("earth", [feature([BIG])]), layer("roads", [feature([BIG], kind=2)])
    landuse, places = layer("landuse"), layer("places")
    unknown = field(20, b"unknown tile field")
    source = unknown + b"".join(field(3, l) for l in (earth, water, roads, landuse, places))
    rewritten, report = tiles.rewrite_tile(source)
    expected = unknown + field(3, earth) + field(3, layer("water", [feature([BIG, SECOND])])) + field(3, roads)
    assert rewritten == expected
    assert tiles.decode_geometry(geometry([BIG, TINY, SECOND, HOLE])) == [BIG, TINY, SECOND, HOLE]
    assert report["water_rings_in"] == 4 and report["water_rings_out"] == 2
    assert report["water_vertices_in"] == 16 and report["water_vertices_out"] == 8
    assert report["bytes_per_layer_in"] == {
        "earth": len(earth), "water": len(water), "roads": len(roads),
        "landuse": len(landuse), "places": len(places),
    }
    assert report["bytes_per_layer_out"] == {
        "earth": len(earth), "water": len(layer("water", [feature([BIG, SECOND])])), "roads": len(roads),
    }


@pytest.mark.parametrize("extent,side,kept", [
    (None, 63, False), (None, 64, True), (8192, 127, False), (8192, 128, True),
])
def test_ring_threshold_uses_extent_and_includes_exact_boundary(extent, side, kept):
    square = [(0, 0), (side, 0), (side, side), (0, side)]
    rewritten, report = tiles.rewrite_tile(field(3, layer("water", [feature([square])], extent)))
    assert rewritten == field(3, layer("water", [feature([square])] if kept else [], extent))
    assert report["water_rings_out"] == int(kept)


def test_dropped_exterior_takes_its_holes_and_empty_features_go():
    large_hole = list(reversed(BIG))
    source = field(3, layer("water", [feature([TINY, large_hole]), feature([BIG, large_hole])]))
    rewritten, report = tiles.rewrite_tile(source)
    assert rewritten == field(3, layer("water", [feature([BIG, large_hole])]))
    assert report["water_rings_in"] == 4 and report["water_rings_out"] == 2


@pytest.mark.parametrize("body,message", [
    (b"\x09\x00", "geometry"), (b"\x00", "geometry"),
    (b"\x0b", "geometry"), (b"\x0f", "geometry"),
    (b"\x09\x00\x00\x12\x02\x00\x00\x02", "geometry"),
])
def test_bad_geometry(body, message):
    with pytest.raises(ValueError, match=message):
        tiles.rewrite_tile(field(3, layer("water", [field(3, 3) + field(4, body)])))


@pytest.mark.parametrize("body", [b"\x1a\x80", b"\x1a\x08x", b"\x00", b"\x1a\x02\x0a\x80"])
def test_bad_protobuf_is_a_clear_value_error(body):
    with pytest.raises(ValueError, match="protobuf|varint"):
        tiles.rewrite_tile(body)


def test_unknown_protobuf_groups_are_copied_through():
    group = (varint((50 << 3) | 3) + field(1, 7)
             + varint((51 << 3) | 3) + field(2, b"nested")
             + varint((51 << 3) | 4) + varint((50 << 3) | 4))
    original = feature([BIG, TINY]) + group
    water = layer("water", [original]) + group
    rewritten, _ = tiles.rewrite_tile(group + field(3, water))
    assert rewritten == group + field(3, layer("water", [feature([BIG]) + group]) + group)
    with pytest.raises(ValueError, match="protobuf.*group"):
        tiles.rewrite_tile(group[:-2])


def tiny_archive():
    first = field(3, layer("earth")) + field(3, layer("water", [feature([BIG, TINY, SECOND, HOLE])])) + field(3, layer("places"))
    second = field(3, layer("roads"))
    bodies = [gzip.compress(first, mtime=0), gzip.compress(second, mtime=0)]
    leaf = gzip.compress(directory([(5, 0, len(bodies[0]), 2), (7, len(bodies[0]), len(bodies[1]), 1)]), mtime=0)
    return archive(directory([(5, 0, len(leaf), 0)]), bodies, leaves=leaf), first, second


def test_end_to_end_mbtiles_and_report(tmp_path, capsys, monkeypatch):
    body, first, second = tiny_archive()
    source, output = tmp_path / "in.pmtiles", tmp_path / "out.mbtiles"
    source.write_bytes(body)
    monkeypatch.setattr(sys, "argv", ["tiles", str(source), str(output)])
    tiles.main()
    report = json.loads(capsys.readouterr().out)
    assert report["tiles_in"] == report["tiles_out"] == 3
    assert report["water_rings_in"] == 8 and report["water_rings_out"] == 4
    assert report["water_vertices_in"] == 32 and report["water_vertices_out"] == 16
    assert report["bytes_per_layer_in"]["places"] == 2 * len(layer("places"))
    assert "places" not in report["bytes_per_layer_out"]
    assert report["runtime_s"] >= 0
    with sqlite3.connect(output) as db:
        rows = db.execute("select zoom_level,tile_column,tile_row,tile_data from tiles order by tile_column,tile_row desc").fetchall()
        metadata = dict(db.execute("select name,value from metadata"))
    assert [row[:3] for row in rows] == [(2, 0, 3), (2, 1, 3), (2, 1, 2)]
    expected = field(3, layer("earth")) + field(3, layer("water", [feature([BIG, SECOND])]))
    assert [gzip.decompress(row[3]) for row in rows] == [expected, expected, second]
    assert metadata == {
        "name": "Synthetic India", "format": "pbf", "minzoom": "2", "maxzoom": "2",
        "bounds": "60,0,106,40", "center": "83,20,2", "attribution": "OSM; source credit",
        "json": json.dumps({"vector_layers": [
            {"id": name, "fields": {"kind": "String"}, "minzoom": 0, "maxzoom": 10}
            for name in ("boundaries", "earth", "roads", "water")
        ]}, separators=(",", ":")),
    }
    assert source.read_bytes() == body


def test_run_rewrites_shared_body_once(tmp_path, monkeypatch):
    body, _, _ = tiny_archive()
    source, output = tmp_path / "in.pmtiles", tmp_path / "out.mbtiles"
    source.write_bytes(body)
    rewrite = tiles.rewrite_tile
    seen = []

    def counted(raw):
        seen.append(raw)
        return rewrite(raw)

    monkeypatch.setattr(tiles, "rewrite_tile", counted)
    tiles.clean(source, output)
    assert len(seen) == 2


def test_failed_conversion_rolls_back_and_does_not_clobber(tmp_path):
    source, output = tmp_path / "in.pmtiles", tmp_path / "out.mbtiles"
    bad = gzip.compress(b"\x1a\x80", mtime=0)
    source.write_bytes(archive(directory([(5, 0, len(bad), 1)]), [bad]))
    with pytest.raises(ValueError, match="varint"):
        tiles.clean(source, output)
    assert not output.exists()
    output.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        tiles.clean(source, output)
    assert output.read_bytes() == b"keep me"


def test_overlapping_tile_runs_are_rejected_and_rolled_back(tmp_path):
    source, output = tmp_path / "in.pmtiles", tmp_path / "out.mbtiles"
    compressed = gzip.compress(field(3, layer("earth")), mtime=0)
    source.write_bytes(archive(directory([(5, 0, len(compressed), 2), (6, 0, len(compressed), 1)]), [compressed]))
    with pytest.raises(ValueError, match="duplicate.*tile"):
        tiles.clean(source, output)
    assert not output.exists()


def test_cli_error_is_clear_and_does_not_leave_output(tmp_path):
    source, output = tmp_path / "bad.pmtiles", tmp_path / "out.mbtiles"
    source.write_bytes(b"bad")
    result = subprocess.run([sys.executable, "-m", "msynd.static.tiles", str(source), str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "PMTiles" in result.stderr
    assert "Traceback" not in result.stderr and not output.exists()
