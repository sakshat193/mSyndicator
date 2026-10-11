import argparse
from collections import Counter
import gzip
import hashlib
import heapq
import json
from pathlib import Path

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
    adjacency = {code: [] for code in stations}
    lengths = {}
    for segment in network["segments"]:
        lo, hi = segment["stn_lo"], segment["stn_hi"]
        if lo not in stations or hi not in stations:
            continue
        a, b = stations[lo], stations[hi]
        km = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        lengths[segment["id"]] = km
        adjacency[lo].append((hi, segment["id"], km))
        adjacency[hi].append((lo, segment["id"], km))
    # ponytail: one Dijkstra per long segment; precompute in build_segments if export time matters.
    for segment in network["segments"]:
        segment["derived"] = False
        if segment["polyline"] is not None:
            segment["polyline"] = json.loads(segment["polyline"])
            continue
        chord = lengths.get(segment["id"], 0)
        if chord <= 20:
            continue
        start, end = segment["stn_lo"], segment["stn_hi"]
        distances, previous = {start: 0}, {}
        queue = [(0, start)]
        while queue:
            distance, station = heapq.heappop(queue)
            if distance > chord ** 2:
                break
            if distance != distances[station]:
                continue
            if station == end:
                path, km = [end], 0
                while path[-1] != start:
                    parent, edge_km = previous[path[-1]]
                    path.append(parent)
                    km += edge_km
                if len(path) >= 3 and km <= 1.3 * chord:
                    segment["polyline"] = [[stations[code]["lon"], stations[code]["lat"]]
                                           for code in reversed(path)]
                    segment["derived"] = True
                break
            for neighbor, edge_id, km in adjacency[station]:
                candidate = distance + km ** 2
                if edge_id != segment["id"] and candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = (station, km)
                    heapq.heappush(queue, (candidate, neighbor))


def export_network(conn, out_dir, overrides_file=OVERRIDES):
    network = {
        "stations": records(conn, "select code,name,lat,lon,zone from ref.station order by code"),
        "segments": records(conn, "select id,stn_lo,stn_hi,polyline,confidence from ref.segment order by id"),
        "routes": records(conn, """
            select r.id,r.schedule_version_id,r.dist_src,v.train,t.name,t.class
            from ref.route_version r join ref.schedule_version v on v.id=r.schedule_version_id
            join ref.train t on t.number=v.train order by r.id
        """),
        "overrides_hash": hashlib.sha256(Path(overrides_file).read_bytes()).hexdigest(),
    }
    segment_geometry(network)
    stops = records(conn, "select route_version_id,idx,station,occ,halt,dist_km,sched_off from ref.route_stop order by route_version_id,idx")
    segments = records(conn, "select route_version_id,idx,segment_id,forward from ref.route_segment order by route_version_id,idx")
    by_route = {route["id"]: route for route in network["routes"]}
    for route in by_route.values():
        route["stops"], route["segments"] = [], []
    for row in stops:
        by_route[row.pop("route_version_id")]["stops"].append(row)
    for row in segments:
        by_route[row.pop("route_version_id")]["segments"].append(row)
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
