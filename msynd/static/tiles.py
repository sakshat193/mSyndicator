"""Clean a local PMTiles v3 basemap into gzip MVT MBTiles, without network access."""

import argparse
from contextlib import closing
import gzip
import io
import json
import mmap
from pathlib import Path
import sqlite3
import struct
import time
import zlib


# Mirrors the source layers drawn by web/src/basemap.ts.
STYLE_LAYERS = ("earth", "water", "boundaries", "roads", "transit")
# ponytail: one area threshold at every zoom; z10 also feeds overzoom to z13, where a dropped pond is ~32 px wide; add a per-zoom threshold if missed
MIN_RING_PX2 = 16
MAX_GUNZIP = 16 * 1024 * 1024


def _varint(body, pos):
    value = 0
    for i in range(10):
        if pos >= len(body):
            raise ValueError("truncated varint")
        byte = body[pos]
        pos += 1
        if i == 9 and byte > 1:
            raise ValueError("overlong varint")
        value |= (byte & 127) << (7 * i)
        if byte < 128:
            if i and byte == 0:
                raise ValueError("overlong varint")
            return value, pos
    raise ValueError("overlong varint")


def _uvarint(value):
    body = bytearray()
    while value >= 128:
        body.append((value & 127) | 128)
        value >>= 7
    body.append(value)
    return body


def _message_field(number, body):
    return _uvarint((number << 3) | 2) + _uvarint(len(body)) + body


def gunzip(body):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
            result = stream.read(MAX_GUNZIP + 1)
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError(f"invalid gzip body: {exc}") from exc
    if len(result) > MAX_GUNZIP:
        raise ValueError("gunzip exceeds 16 MB")
    return result


def _range(offset, length, size, section):
    if offset < 0 or length < 0 or offset > size or length > size - offset:
        raise ValueError(f"{section} offset/length out of range")


def read_header(body):
    if len(body) < 127:
        raise ValueError("truncated PMTiles v3 header")
    if body[:7] != b"PMTiles":
        raise ValueError("bad PMTiles magic")
    if body[7] != 3:
        raise ValueError("unsupported PMTiles version; require v3")
    for pos, expected, label in ((97, 2, "internal compression (require gzip)"),
                                 (98, 2, "tile compression (require gzip)"),
                                 (99, 1, "tile type (require MVT)")):
        if body[pos] != expected:
            raise ValueError(f"unsupported {label}")
    header = {}
    for pos, section in ((8, "root"), (24, "metadata"), (40, "leaf"), (56, "tile")):
        offset, length = struct.unpack_from("<QQ", body, pos)
        _range(offset, length, len(body), section)
        if length and offset < 127:
            raise ValueError(f"{section} offset/length out of range (inside header)")
        header[section] = offset, length
    header["minzoom"], header["maxzoom"] = body[100], body[101]
    if not 0 <= header["minzoom"] <= header["maxzoom"] <= 26:
        raise ValueError("PMTiles zoom out of range (0..26)")
    header["bounds"] = struct.unpack_from("<iiii", body, 102)
    header["center"] = (*struct.unpack_from("<ii", body, 119), body[118])
    return header


def read_directory(body):
    count, pos = _varint(body, 0)
    if count > (len(body) - pos) // 4:
        raise ValueError("truncated PMTiles directory")
    columns = []
    for _ in range(4):
        values = []
        for _ in range(count):
            value, pos = _varint(body, pos)
            values.append(value)
        columns.append(values)
    if pos != len(body):
        raise ValueError("trailing bytes in PMTiles directory")
    ids, runs, lengths, offsets = columns
    entries, tile_id = [], 0
    for i in range(count):
        if i and ids[i] == 0:
            raise ValueError("PMTiles directory tile ids must increase")
        tile_id += ids[i]
        if lengths[i] == 0:
            raise ValueError("PMTiles directory length must be positive")
        if offsets[i] == 0:
            if i == 0:
                raise ValueError("invalid first PMTiles directory offset")
            offset = entries[-1][1] + entries[-1][2]
        else:
            offset = offsets[i] - 1
        entries.append((tile_id, offset, lengths[i], runs[i]))
    return entries


def directories(body, header):
    """Yield one directory's tile entries at a time; offsets are section-relative."""
    def walk(section, offset, length, depth):
        if depth > 4:
            raise ValueError("PMTiles directory depth exceeds 4")
        start, size = header[section]
        _range(offset, length, size, section)
        entries = read_directory(gunzip(body[start + offset:start + offset + length]))
        first_id = ((1 << (2 * header["minzoom"])) - 1) // 3
        last_id = ((1 << (2 * (header["maxzoom"] + 1))) - 1) // 3
        for tile_id, entry_offset, entry_length, run in entries:
            target = "tile" if run else "leaf"
            _range(entry_offset, entry_length, header[target][1], target)
            if run and not first_id <= tile_id < tile_id + run <= last_id:
                raise ValueError("PMTiles tile id/run out of zoom range")
        yield [entry for entry in entries if entry[3]]
        for _, entry_offset, entry_length, run in entries:
            if run == 0:
                yield from walk("leaf", entry_offset, entry_length, depth + 1)

    yield from walk("root", 0, header["root"][1], 1)


def _rotate(size, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x, y = size - 1 - x, size - 1 - y
        x, y = y, x
    return x, y


def zxy_to_tileid(z, x, y):
    if not 0 <= z <= 26 or not 0 <= x < (1 << z) or not 0 <= y < (1 << z):
        raise ValueError("z/x/y out of range")
    distance, size = 0, (1 << z) >> 1
    while size:
        rx, ry = int(bool(x & size)), int(bool(y & size))
        distance += size * size * ((3 * rx) ^ ry)
        x, y = _rotate(size, x, y, rx, ry)
        size >>= 1
    return ((1 << (2 * z)) - 1) // 3 + distance


def tileid_to_zxy(tile_id):
    if not 0 <= tile_id < ((1 << 54) - 1) // 3:
        raise ValueError("PMTiles tile id out of range")
    z, base = 0, 0
    while tile_id >= base + (1 << (2 * z)):
        base += 1 << (2 * z)
        z += 1
    distance, x, y, size = tile_id - base, 0, 0, 1
    while size < (1 << z):
        rx = 1 & (distance >> 1)
        ry = 1 & (distance ^ rx)
        x, y = _rotate(size, x, y, rx, ry)
        x, y = x + size * rx, y + size * ry
        distance >>= 2
        size <<= 1
    return z, x, y


def _scalar(body, pos, wire):
    if wire == 0:
        return _varint(body, pos)
    if wire not in (1, 2, 5):
        raise ValueError(f"unsupported protobuf wire type {wire}")
    if wire == 2:
        length, pos = _varint(body, pos)
    else:
        length = 8 if wire == 1 else 4
    _range(pos, length, len(body), "protobuf field")
    return body[pos:pos + length], pos + length


def _fields(body):
    """Read protobuf fields, retaining each complete field's original bytes."""
    pos = 0
    while pos < len(body):
        start = pos
        key, pos = _varint(body, pos)
        number, wire = key >> 3, key & 7
        if not 0 < number < (1 << 29):
            raise ValueError("invalid protobuf field number")
        if wire == 3:
            group_start, groups = pos, [number]
            while groups:
                if pos == len(body):
                    raise ValueError("truncated protobuf group")
                key, pos = _varint(body, pos)
                child, child_wire = key >> 3, key & 7
                if not 0 < child < (1 << 29):
                    raise ValueError("invalid protobuf group field number")
                if child_wire == 3:
                    groups.append(child)
                elif child_wire == 4:
                    if child != groups.pop():
                        raise ValueError("mismatched protobuf group end")
                else:
                    _, pos = _scalar(body, pos, child_wire)
            value = body[group_start:pos]
        else:
            value, pos = _scalar(body, pos, wire)
        yield number, wire, value, body[start:pos]


def decode_geometry(body):
    rings, ring, x, y, pos = [], None, 0, 0, 0
    while pos < len(body):
        command, pos = _varint(body, pos)
        op, count = command & 7, command >> 3
        if count == 0 or op not in (1, 2, 7):
            raise ValueError("invalid polygon geometry command")
        if op == 7:
            if count != 1 or ring is None or len(ring) < 3:
                raise ValueError("invalid polygon geometry ClosePath")
            rings.append(ring)
            ring = None
            continue
        if op == 1:
            if count != 1 or ring is not None:
                raise ValueError("invalid polygon geometry MoveTo")
            ring = []
        elif ring is None:
            raise ValueError("polygon geometry LineTo before MoveTo")
        if count > (len(body) - pos) // 2:
            raise ValueError("truncated polygon geometry coordinates")
        for _ in range(count):
            dx, pos = _varint(body, pos)
            dy, pos = _varint(body, pos)
            if dx > 0xffffffff or dy > 0xffffffff:
                raise ValueError("polygon geometry delta out of uint32 range")
            x += (dx >> 1) ^ -(dx & 1)
            y += (dy >> 1) ^ -(dy & 1)
            ring.append((x, y))
    if ring is not None:
        raise ValueError("unclosed polygon geometry ring")
    return rings


def _encode_geometry(rings):
    body, x, y = bytearray(), 0, 0
    for ring in rings:
        body.extend(_uvarint(9))
        for i, (nx, ny) in enumerate(ring):
            if i == 1:
                body.extend(_uvarint(((len(ring) - 1) << 3) | 2))
            dx, dy = nx - x, ny - y
            body.extend(_uvarint(2 * dx if dx >= 0 else -2 * dx - 1))
            body.extend(_uvarint(2 * dy if dy >= 0 else -2 * dy - 1))
            x, y = nx, ny
        body.extend(_uvarint(15))
    return body


def _report():
    return {"bytes_per_layer_in": {}, "bytes_per_layer_out": {},
            "water_rings_in": 0, "water_rings_out": 0,
            "water_vertices_in": 0, "water_vertices_out": 0}


def _water_feature(body, extent, report):
    fields = list(_fields(body))
    kind = 0
    for number, wire, value, _ in fields:
        if number == 3:
            if wire != 0:
                raise ValueError("invalid protobuf feature type")
            kind = value
    if kind != 3:
        return None
    geometries = []
    for number, wire, value, _ in fields:
        if number == 4:
            if wire != 2:
                raise ValueError("invalid protobuf geometry field")
            geometries.append(value)
    if not geometries:
        raise ValueError("missing polygon geometry")
    rings = decode_geometry(b"".join(geometries))
    report["water_rings_in"] += len(rings)
    report["water_vertices_in"] += sum(map(len, rings))
    kept, keep_exterior, have_exterior = [], False, False
    for ring in rings:
        area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:] + ring[:1]))
        large = abs(area2) * 256 ** 2 >= 2 * MIN_RING_PX2 * extent ** 2
        if area2 >= 0:
            keep_exterior, have_exterior = large, True
        elif not have_exterior:
            raise ValueError("polygon geometry hole before exterior")
        if keep_exterior and large:
            kept.append(ring)
    report["water_rings_out"] += len(kept)
    report["water_vertices_out"] += sum(map(len, kept))
    if not kept:
        return None
    geometry = _message_field(4, _encode_geometry(kept))
    rewritten, written = bytearray(), False
    for number, _, _, raw in fields:
        if number == 4:
            if not written:
                rewritten.extend(geometry)
                written = True
        else:
            rewritten.extend(raw)
    return rewritten


def rewrite_tile(body):
    rewritten, report = bytearray(), _report()
    for number, wire, layer, raw in _fields(body):
        if number != 3:
            rewritten.extend(raw)
            continue
        if wire != 2:
            raise ValueError("invalid protobuf Tile.layers field")
        fields = list(_fields(layer))
        name, extent = None, 4096
        for field, field_wire, value, _ in fields:
            if field == 1:
                if field_wire != 2:
                    raise ValueError("invalid protobuf layer name")
                try:
                    name = value.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("invalid protobuf layer name UTF-8") from exc
            elif field == 5:
                if field_wire != 0 or not 0 < value <= 0xffffffff:
                    raise ValueError("invalid protobuf layer extent")
                extent = value
        if name is None:
            raise ValueError("missing protobuf layer name")
        sizes = report["bytes_per_layer_in"]
        sizes[name] = sizes.get(name, 0) + len(layer)
        if name not in STYLE_LAYERS:
            continue
        if name == "water":
            result = bytearray()
            for field, field_wire, value, field_raw in fields:
                if field != 2:
                    result.extend(field_raw)
                else:
                    if field_wire != 2:
                        raise ValueError("invalid protobuf Layer.features field")
                    feature = _water_feature(value, extent, report)
                    if feature is not None:
                        result.extend(_message_field(2, feature))
            rewritten.extend(_message_field(3, result))
        else:
            result = layer
            rewritten.extend(raw)
        sizes = report["bytes_per_layer_out"]
        sizes[name] = sizes.get(name, 0) + len(result)
    return bytes(rewritten), report


def _metadata(body, header, source):
    offset, length = header["metadata"]
    try:
        metadata = json.loads(gunzip(body[offset:offset + length]))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid PMTiles metadata: {exc}") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("vector_layers"), list):
        raise ValueError("PMTiles metadata requires vector_layers array")
    layers = metadata["vector_layers"]
    if any(not isinstance(layer, dict) or not isinstance(layer.get("id"), str) for layer in layers):
        raise ValueError("invalid PMTiles metadata vector_layers")
    name, attribution = metadata.get("name", source.stem), metadata.get("attribution", "")
    if not isinstance(name, str) or not isinstance(attribution, str):
        raise ValueError("PMTiles metadata name/attribution must be strings")

    def coordinate(value):
        return f"{value / 10_000_000:.7f}".rstrip("0").rstrip(".")

    return {"name": name, "format": "pbf", "minzoom": str(header["minzoom"]),
            "maxzoom": str(header["maxzoom"]),
            "bounds": ",".join(map(coordinate, header["bounds"])),
            "center": ",".join([*map(coordinate, header["center"][:2]), str(header["center"][2])]),
            "attribution": attribution,
            "json": json.dumps({"vector_layers": [layer for layer in layers if layer["id"] in STYLE_LAYERS]},
                               separators=(",", ":"))}


def clean(source, output):
    source, output = Path(source), Path(output)
    started, report = time.monotonic(), _report()
    report.update(tiles_in=0, tiles_out=0)
    with source.open("rb") as stream:
        if stream.seek(0, 2) < 127:
            raise ValueError("truncated PMTiles v3 header")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as body:
            header = read_header(body)
            metadata = _metadata(body, header, source)
            # Exclusive creation protects the input and any previous conversion.
            output.open("xb").close()
            try:
                with closing(sqlite3.connect(output)) as db, db:
                    db.execute("BEGIN")
                    db.execute("create table metadata (name text primary key, value text)")
                    db.execute("""create table tiles (zoom_level integer, tile_column integer,
                               tile_row integer, tile_data blob,
                               primary key (zoom_level,tile_column,tile_row))""")
                    db.executemany("insert into metadata values (?,?)", metadata.items())
                    for entries in directories(body, header):
                        # Clustered runs share offsets; release bodies at each directory boundary.
                        cache = {}
                        for tile_id, offset, length, run in entries:
                            key = offset, length
                            if key not in cache:
                                start = header["tile"][0] + offset
                                rewritten, stats = rewrite_tile(gunzip(body[start:start + length]))
                                cache[key] = gzip.compress(rewritten, mtime=0), stats
                            compressed, stats = cache[key]
                            report["tiles_in"] += run
                            for name, value in stats.items():
                                if isinstance(value, dict):
                                    for layer, count in value.items():
                                        totals = report[name]
                                        totals[layer] = totals.get(layer, 0) + count * run
                                else:
                                    report[name] += value * run
                            for addressed_id in range(tile_id, tile_id + run):
                                z, x, y = tileid_to_zxy(addressed_id)
                                db.execute("insert into tiles values (?,?,?,?)", (z, x, (1 << z) - 1 - y, compressed))
                                report["tiles_out"] += 1
            except sqlite3.IntegrityError as exc:
                output.unlink(missing_ok=True)
                raise ValueError("duplicate PMTiles tile id in overlapping directory runs") from exc
            except BaseException:
                output.unlink(missing_ok=True)
                raise
    report["runtime_s"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        report = clean(args.source, args.output)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
