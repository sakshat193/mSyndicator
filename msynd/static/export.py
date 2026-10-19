import argparse
from collections import defaultdict
import gzip
import hashlib
import heapq
import json
from pathlib import Path
from statistics import median

from msynd.db import connect
from msynd.static.gtfs import haversine
from msynd.static.segments import OVERRIDES, build_segments


def records(conn, query):
    cursor = conn.execute(query)
    names = [column.name for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def segment_geometry(network):
    stations = {s["code"]: s for s in network["stations"]
                if s["lat"] is not None and s["lon"] is not None}
    by_id = {s["id"]: s for s in network["segments"]}
    polylines = {s["id"]: json.loads(s["polyline"]) for s in network["segments"]
                 if s["polyline"] is not None}
    track_km = defaultdict(list)
    for route in network["routes"]:
        stops = {stop["idx"]: stop["dist_km"] for stop in route["stops"]}
        for segment in route["segments"]:
            idx = segment["idx"]
            track_km[segment["segment_id"]].append(abs(stops[idx + 1] - stops[idx]))
    adjacency = {code: [] for code in stations}
    lengths, weights = {}, {}
    for segment in network["segments"]:
        lo, hi = segment["stn_lo"], segment["stn_hi"]
        if lo not in stations or hi not in stations:
            continue
        a, b = stations[lo], stations[hi]
        km = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        lengths[segment["id"]] = km
        weight = weights[segment["id"]] = max(median(track_km.get(segment["id"], [0])), km)
        if km <= 80:
            adjacency[lo].append((hi, segment["id"], weight))
            adjacency[hi].append((lo, segment["id"], weight))
    connected = [code for code, edges in adjacency.items() if edges]
    for lo, a in stations.items():
        if adjacency[lo]:
            continue
        for hi in connected:
            b = stations[hi]
            km = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if km <= 15:
                adjacency[lo].append((hi, None, km))
                adjacency[hi].append((lo, None, km))
    # ponytail: one search per long segment; precompute chains if export time matters.
    for segment in network["segments"]:
        segment["derived"] = False
        if segment["polyline"] is not None:
            segment["polyline"] = polylines[segment["id"]]
            continue
        chord = lengths.get(segment["id"], 0)
        if chord <= 20:
            continue
        target = weights[segment["id"]]
        start, end = segment["stn_lo"], segment["stn_hi"]
        distances, previous = {start: 0}, {}
        queue = [(0, start)]
        while queue:
            distance, station = heapq.heappop(queue)
            if distance > 1.2 * target:
                break
            if distance != distances[station]:
                continue
            if station == end:
                path, current = [], end
                while current != start:
                    parent, edge_id = previous[current]
                    path.append((parent, current, edge_id))
                    current = parent
                if len(path) >= 2:
                    points = []
                    for a, b, edge_id in reversed(path):
                        curve = polylines.get(edge_id)
                        if curve is None:
                            curve = [[stations[code]["lon"], stations[code]["lat"]] for code in (a, b)]
                        elif by_id[edge_id]["stn_lo"] != a:
                            curve = list(reversed(curve))
                        points.extend(curve if not points else curve[1:])
                    segment["polyline"] = points
                    segment["derived"] = True
                break
            for neighbor, edge_id, km in adjacency[station]:
                candidate = distance + km
                if edge_id != segment["id"] and candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = station, edge_id
                    heapq.heappush(queue, (candidate, neighbor))


def export_network(conn, out_dir, overrides_file=OVERRIDES):
    network = {
        "stations": records(conn, """
            select s.code,s.name,coalesce(o.lat,s.lat) as lat,coalesce(o.lon,s.lon) as lon,s.zone
            from ref.station s left join ref.station_osm o on o.code=s.code order by s.code
        """),
        "segments": records(conn, "select id,stn_lo,stn_hi,polyline,confidence from ref.segment order by id"),
        "routes": records(conn, """
            select r.id,r.schedule_version_id,r.dist_src,v.train,t.name,t.class
            from ref.route_version r join ref.schedule_version v on v.id=r.schedule_version_id
            join ref.train t on t.number=v.train order by r.id
        """),
        "overrides_hash": hashlib.sha256(Path(overrides_file).read_bytes()).hexdigest(),
    }
    stops = records(conn, "select route_version_id,idx,station,occ,halt,dist_km,sched_off from ref.route_stop order by route_version_id,idx")
    segments = records(conn, "select route_version_id,idx,segment_id,forward from ref.route_segment order by route_version_id,idx")
    by_route = {route["id"]: route for route in network["routes"]}
    for route in by_route.values():
        route["stops"], route["segments"] = [], []
    for row in stops:
        by_route[row.pop("route_version_id")]["stops"].append(row)
    for row in segments:
        by_route[row.pop("route_version_id")]["segments"].append(row)
    segment_geometry(network)
    body = json.dumps(network, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    etag = hashlib.sha256(body).hexdigest()
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"network-{etag[:12]}.json.gz"
    compressed = gzip.compress(body, mtime=0)
    if not path.exists() or path.read_bytes() != compressed:
        temp = path.with_suffix(".tmp")
        temp.write_bytes(compressed)
        temp.replace(path)
    return path, etag


def report(conn):
    total = conn.execute("select count(*) from ref.train").fetchone()[0]
    scheduled = conn.execute("select count(distinct train) from ref.schedule_version").fetchone()[0]
    drawable = conn.execute("""
        select count(distinct v.train) from ref.schedule_version v
        join ref.route_version r on r.schedule_version_id=v.id
        where not exists (
          select 1 from ref.route_stop s join ref.station t on t.code=s.station
          where s.route_version_id=r.id and (t.lat is null or t.lon is null))
        and (select count(*) from ref.route_segment where route_version_id=r.id)
          = (select count(*)-1 from ref.route_stop where route_version_id=r.id)
    """).fetchone()[0]
    return {
        "trains": total, "scheduled": scheduled, "drawable": drawable,
        "schedule_share": scheduled / total if total else 0,
        "drawable_share": drawable / total if total else 0,
        "confidence_histogram": dict(conn.execute("select confidence,count(*) from ref.segment group by confidence order by confidence").fetchall()),
        "distance_sources": dict(conn.execute("select dist_src,count(*) from ref.route_version group by dist_src order by dist_src").fetchall()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["report", "export"])
    parser.add_argument("--out", default="data/static")
    args = parser.parse_args()
    with connect() as conn:
        build_segments(conn)
        if args.command == "report":
            print(json.dumps(report(conn), indent=2))
        else:
            print(export_network(conn, args.out))


if __name__ == "__main__":
    main()
