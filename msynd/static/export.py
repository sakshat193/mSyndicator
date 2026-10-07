import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from msynd.db import connect
from msynd.static.segments import OVERRIDES, build_segments


def records(conn, query):
    cursor = conn.execute(query)
    names = [column.name for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def export_network(conn, out_dir, overrides_file=OVERRIDES):
    network = {
        "stations": records(conn, "select code,name,lat,lon,zone from ref.station order by code"),
        "segments": records(conn, "select id,stn_lo,stn_hi,polyline,confidence from ref.segment order by id"),
        "routes": records(conn, "select id,schedule_version_id,dist_src from ref.route_version order by id"),
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
