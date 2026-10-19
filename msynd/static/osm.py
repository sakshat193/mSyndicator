import argparse
from collections import defaultdict
import difflib
import heapq
import json
import math
from pathlib import Path
import re
from statistics import quantiles
import time

from msynd.db import connect
from msynd.static.gtfs import haversine


ESCAPE = re.compile(r"%([0-9a-fA-F]{1,6})%")
# ponytail: 15 m is the detail geometry; the web derives a 75 m overview and uses this only past z10
# in the viewport's region (Task 98), as a whole-network 15 m costs +60 % India-zoom frame time.
SIMPLIFY_M = 15


def decode(value):
    if "%" not in value:
        return value
    if "%" in ESCAPE.sub("", value):
        raise ValueError("malformed")

    def codepoint(match):
        number = int(match[1], 16)
        if number > 0x10ffff or 0xd800 <= number <= 0xdfff:
            raise ValueError("malformed")
        return chr(number)

    return ESCAPE.sub(codepoint, value)


def point(lon, lat):
    lon, lat = float(lon), float(lat)
    if not math.isfinite(lon) or not math.isfinite(lat):
        raise ValueError("non_finite")
    if not (68 <= lon <= 98 and 6 <= lat <= 37):
        raise ValueError("out_of_bbox")
    return lon, lat


def node_ref(value):
    if not value.startswith("n"):
        raise ValueError("malformed")
    number, located, location = value[1:].partition("x")
    if not number.isascii() or not number.isdecimal() or int(number) <= 0:
        raise ValueError("malformed")
    if not located:
        return int(number), None
    lon, separator, lat = location.partition("y")
    if not separator:
        raise ValueError("malformed")
    return int(number), point(lon, lat)


def parse_opl(path):
    points, graph, stations, ends = {}, {}, {}, set()
    skipped = {"malformed": 0, "non_finite": 0, "out_of_bbox": 0}
    with Path(path).open("rb") as stream:
        for raw in stream:
            try:
                fields = raw.decode("utf-8").split()
                if not fields or fields[0][0] not in "nwr":
                    raise ValueError("malformed")
                kind, number = fields[0][0], fields[0][1:]
                if not number.isascii() or not number.isdecimal() or int(number) <= 0:
                    raise ValueError("malformed")
                attributes = {}
                for field in fields[1:]:
                    if field[0] in attributes:
                        raise ValueError("malformed")
                    attributes[field[0]] = field[1:]
                tags = {}
                if attributes.get("T"):
                    for tag in attributes["T"].split(","):
                        key, separator, value = tag.partition("=")
                        if not separator or not key:
                            raise ValueError("malformed")
                        tags[decode(key)] = decode(value)
                if kind == "n" and tags.get("railway") in ("station", "halt"):
                    stations[int(number)] = (point(attributes["x"], attributes["y"]), tags)
                elif kind == "w" and tags.get("railway") in ("rail", "narrow_gauge"):
                    refs = [node_ref(value) for value in attributes["N"].split(",")]
                    # Validate the whole line before changing the graph.
                    locations = {}
                    for node, location in refs:
                        if location is not None:
                            if locations.get(node, points.get(node, location)) != location:
                                raise ValueError("malformed")
                            locations[node] = location
                    points.update(locations)
                    previous = None
                    for node, location in refs:
                        if location is None:
                            if previous is not None:
                                ends.add(previous)
                            previous = None
                            continue
                        graph.setdefault(node, {})
                        if previous is None:
                            ends.add(node)
                        elif previous != node:
                            km = distance(points[previous], location)
                            graph[previous][node] = km
                            graph[node][previous] = km
                        previous = node
                    if previous is not None:
                        ends.add(previous)
            except (ValueError, KeyError) as error:
                reason = str(error)
                skipped[reason if reason in skipped else "malformed"] += 1
    return points, graph, ends, stations, skipped


def distance(a, b):
    return haversine(a[1], a[0], b[1], b[0])


def grid_index(points):
    grid = defaultdict(list)
    for node, (lon, lat) in points.items():
        grid[math.floor(lon * 100), math.floor(lat * 100)].append(node)
    return grid


def nearby(grid, points, location, radius):
    lon, lat = location
    dy = radius / 111.195
    dx = dy / math.cos(math.radians(lat))
    for x in range(math.floor((lon - dx) * 100), math.floor((lon + dx) * 100) + 1):
        for y in range(math.floor((lat - dy) * 100), math.floor((lat + dy) * 100) + 1):
            for node in grid.get((x, y), ()):
                km = distance(location, points[node])
                if km <= radius:
                    yield km, node


def normalise(name):
    return "".join(word for word in re.findall(r"[^\W\d_]+", name.lower())
                   if word not in {"junction", "jn", "halt", "station"})


def anchor_stations(stations, osm_stations, points, graph):
    station_points = {node: record[0] for node, record in osm_stations.items()}
    names = {node: normalise(record[1].get("name", "")) for node, record in osm_stations.items()}
    refs = defaultdict(list)
    for node, (_, tags) in osm_stations.items():
        for key in ("railway:ref", "ref"):
            for code in tags.get(key, "").split(";"):
                if code.strip():
                    refs[code.strip().upper()].append(node)
    station_grid, rail_grid = grid_index(station_points), grid_index(points)
    counts = dict.fromkeys(("ref", "name", "coord", "no_anchor"), 0)
    anchors = {}
    next_node = -1
    for code, name, lat, lon in sorted(stations):
        own = (lon, lat)
        if not all(math.isfinite(value) for value in own):
            counts["no_anchor"] += 1
            continue
        matches = [(distance(own, station_points[node]), node) for node in refs.get(code.upper(), ())]
        matched = min((match for match in matches if match[0] <= 5), default=None)
        tier = "ref"
        if matched is None:
            tier, name = "name", normalise(name)
            matched = min((match for match in nearby(station_grid, station_points, own, 1.5)
                           if name and names[match[1]] and difflib.SequenceMatcher(None, name, names[match[1]]).ratio() >= .8),
                          default=None)
        if matched is None:
            tier, location = "coord", own
        else:
            location = station_points[matched[1]]
        # ponytail: edges with both ends > 2 km away are missed; add an edge grid if no_anchor shows it.
        edges = {tuple(sorted((node, neighbor)))
                 for _, node in nearby(rail_grid, points, location, 2) for neighbor in graph[node]}
        scale = 111195.08 * math.cos(math.radians(location[1]))
        best = None
        for a, b in edges:
            x, y = (points[a][0] - location[0]) * scale, (points[a][1] - location[1]) * 111195.08
            dx, dy = (points[b][0] - points[a][0]) * scale, (points[b][1] - points[a][1]) * 111195.08
            length2 = dx * dx + dy * dy
            fraction = min(1, max(0, -(x * dx + y * dy) / length2)) if length2 else 0
            metres2 = (x + fraction * dx) ** 2 + (y + fraction * dy) ** 2
            candidate = (metres2, a, b, fraction)
            if metres2 <= 1000 * 1000 and (best is None or candidate < best):
                best = candidate
        if best is None:
            counts["no_anchor"] += 1
        else:
            _, a, b, fraction = best
            if fraction == 0 or fraction == 1:
                node = a if fraction == 0 else b
            else:
                node, next_node = next_node, next_node - 1
                foot = tuple(points[a][i] + fraction * (points[b][i] - points[a][i]) for i in (0, 1))
                points[node] = foot
                rail_grid[math.floor(foot[0] * 100), math.floor(foot[1] * 100)].append(node)
                del graph[a][b], graph[b][a]
                graph[node] = {a: distance(points[a], foot), b: distance(foot, points[b])}
                graph[a][node], graph[b][node] = graph[node][a], graph[node][b]
            counts[tier] += 1
            anchors[code] = (node, {"ref": 1., "name": .5, "coord": .3}[tier], own)
    return anchors, counts


def contract(graph, points, keep):
    keep = keep | {node for node, edges in graph.items() if len(edges) != 2}
    contracted = {node: [] for node in keep}
    # Consume raw edges so contraction doesn't need a second million-edge visited set.
    for start in keep:
        while graph[start]:
            current, km = graph[start].popitem()
            graph[current].pop(start)
            curve = [points[start], points[current]]
            while current not in keep:
                neighbor, weight = graph[current].popitem()
                graph[neighbor].pop(current)
                km += weight
                current = neighbor
                curve.append(points[current])
            contracted[start].append((current, km, curve, True))
            contracted[current].append((start, km, curve, False))
    return contracted


def astar(graph, points, start, end, target):
    limit = max(1.3 * target, target + 1)
    remaining = distance(points[start], points[end])
    if remaining > limit:
        return None
    queue, costs, previous = [(remaining, 0., start)], {start: 0.}, {}
    while queue:
        _, km, node = heapq.heappop(queue)
        if km != costs[node]:
            continue
        if node == end:
            edges = []
            while node != start:
                parent, curve, forward = previous[node]
                edges.append((curve, forward))
                node = parent
            path = [points[start]]
            for curve, forward in reversed(edges):
                path.extend(curve[1:] if forward else curve[-2::-1])
            return path, km
        for neighbor, weight, curve, forward in graph[node]:
            candidate = km + weight
            if candidate >= costs.get(neighbor, math.inf):
                continue
            heuristic = distance(points[neighbor], points[end])
            if candidate + heuristic <= limit:
                costs[neighbor] = candidate
                previous[neighbor] = node, curve, forward
                heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
    return None


def simplify(points):
    origin = points[0]
    scale = math.cos(math.radians((origin[1] + points[-1][1]) / 2))
    projected = [((lon - origin[0]) * 111195.08 * scale, (lat - origin[1]) * 111195.08)
                 for lon, lat in points]
    keep, pending = {0, len(points) - 1}, [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        x, y = projected[first]
        dx, dy = projected[last][0] - x, projected[last][1] - y
        length2 = dx * dx + dy * dy
        farthest, selected = SIMPLIFY_M * SIMPLIFY_M, None
        for i in range(first + 1, last):
            px, py = projected[i][0] - x, projected[i][1] - y
            fraction = min(1, max(0, (px * dx + py * dy) / length2)) if length2 else 0
            deviation = (px - fraction * dx) ** 2 + (py - fraction * dy) ** 2
            if deviation > farthest:
                farthest, selected = deviation, i
        if selected is not None:
            keep.add(selected)
            pending.extend(((first, selected), (selected, last)))
    return [[round(points[i][0], 5), round(points[i][1], 5)] for i in sorted(keep)]


def refresh(conn, opl):
    started = time.monotonic()
    points, graph, ends, osm_stations, skipped = parse_opl(opl)
    stations = conn.execute("select code,name,lat,lon from ref.station where lat is not null and lon is not null").fetchall()
    anchors, station_counts = anchor_stations(stations, osm_stations, points, graph)
    graph = contract(graph, points, ends | {value[0] for value in anchors.values()})
    points = {node: points[node] for node in graph}
    del osm_stations, ends
    targets = conn.execute("""
        select s.stn_lo,s.stn_hi,
               percentile_cont(0.5) within group (order by abs(b.dist_km-a.dist_km))
        from ref.segment s left join ref.route_segment rs on rs.segment_id=s.id
        left join ref.route_stop a on a.route_version_id=rs.route_version_id and a.idx=rs.idx
        left join ref.route_stop b on b.route_version_id=rs.route_version_id and b.idx=rs.idx+1
        group by s.id order by s.stn_lo,s.stn_hi
    """).fetchall()
    failures = dict.fromkeys(("no_anchor", "no_path", "length"), 0)
    confidence_counts = dict.fromkeys(("1.0", "0.5", "0.3"), 0)
    rows, vertices, snapped = [], 0, set()
    for lo, hi, target in targets:
        if lo not in anchors or hi not in anchors:
            failures["no_anchor"] += 1
            continue
        if target is None or not math.isfinite(target) or target < 0:
            failures["length"] += 1
            continue
        a, b = anchors[lo], anchors[hi]
        if a[0] == b[0]:
            failures["no_path"] += 1
            continue
        result = astar(graph, points, a[0], b[0], target)
        if result is None:
            failures["no_path"] += 1
            continue
        path, km = result
        if abs(km - target) > max(.1 * target, 1):
            failures["length"] += 1
            continue
        curve = simplify(path)
        confidence = min(a[1], b[1])
        rows.append((lo, hi, json.dumps(curve, separators=(",", ":"), allow_nan=False), confidence))
        confidence_counts[str(confidence)] += 1
        vertices += len(curve)
        snapped.update((lo, hi))
    station_rows, moves = [], []
    for code in sorted(snapped):
        node, _, own = anchors[code]
        foot = tuple(round(value, 5) for value in points[node])
        station_rows.append((code, *foot))
        moves.append(1000 * distance(own, foot))
    deciles = quantiles(moves, n=10, method="inclusive") if len(moves) > 1 else moves * 9
    with conn.transaction():
        conn.execute("lock table ref.segment_osm,ref.station_osm in exclusive mode")
        conn.execute("delete from ref.segment_osm")
        conn.execute("delete from ref.station_osm")
        with conn.cursor() as cursor:
            cursor.executemany("insert into ref.segment_osm(stn_lo,stn_hi,polyline,confidence) values (%s,%s,%s,%s)", rows)
            cursor.executemany("insert into ref.station_osm(code,lon,lat) values (%s,%s,%s)", station_rows)
    return {"stations": station_counts, "segments": confidence_counts, "failures": failures,
            "snapped": len(snapped), "move_m": {"p50": deciles[4] if deciles else None, "p90": deciles[8] if deciles else None},
            "vertices": vertices, "runtime_s": time.monotonic() - started, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opl", required=True, type=Path)
    args = parser.parse_args()
    with connect() as conn:
        print(json.dumps(refresh(conn, args.opl), separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
