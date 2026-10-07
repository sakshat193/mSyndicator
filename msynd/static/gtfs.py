import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from msynd.db import connect, migrate

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ImportReport:
    trains: int
    versions_new: int
    versions_dup: int
    stations: int
    missing_coords: int
    skipped_trips: list[dict] = field(default_factory=list)


def train_class(number, name):
    # ponytail: keyword heuristic; use NTES class when schedule refresh supplies it.
    name = name.upper()
    if number.startswith("0") or "SPECIAL" in name or "SPL" in name.split():
        return "SPL"
    for keywords, kind in [(('RAJDHANI',), 'RAJ'), (('SHATABDI',), 'SHT'), (('VANDE',), 'VB'), (('DURONTO',), 'DRNT'), (('SUPERFAST', 'SF'), 'SF'), (('MAIL',), 'MEX'), (('EXP',), 'EXP'), (('PASS', 'MEMU', 'DEMU'), 'PASS')]:
        if any(word in name for word in keywords):
            return kind
    return "UNKNOWN"


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(a)))


def minutes(value):
    h, m, s = map(int, value.split(":"))
    if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError(f"invalid GTFS time: {value}")
    return h * 60 + m + s / 60


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def content_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()).digest()


def import_gtfs(conn, gtfs_dir):
    folder = Path(gtfs_dir)
    routes = {r["route_id"]: r for r in read_csv(folder / "routes.txt")}
    calendars = {r["service_id"]: r for r in read_csv(folder / "calendar.txt")}
    trips = read_csv(folder / "trips.txt")
    times = defaultdict(list)
    for row in read_csv(folder / "stop_times.txt"):
        times[row["trip_id"]].append(row)
    fallback = {}
    datameet = folder / "stations.json"
    if datameet.exists():
        for feature in json.loads(datameet.read_text(encoding="utf-8"))["features"]:
            if feature.get("geometry"):
                lon, lat = feature["geometry"]["coordinates"][:2]
                fallback[feature["properties"]["code"]] = (lat, lon)
    stops, missing = {}, 0
    for row in read_csv(folder / "stops.txt"):
        code = row.get("stop_code") or row["stop_id"]
        coords = (float(row["stop_lat"]), float(row["stop_lon"])) if row.get("stop_lat") and row.get("stop_lon") else fallback.get(code)
        if coords and (not all(math.isfinite(x) for x in coords) or not -90 <= coords[0] <= 90 or not -180 <= coords[1] <= 180):
            raise ValueError(f"invalid coordinates: {code}")
        missing += coords is None
        stops[row["stop_id"]] = (code, coords)
        conn.execute("insert into ref.station(code,name,lat,lon) values (%s,%s,%s,%s) on conflict(code) do update set name=excluded.name,lat=coalesce(excluded.lat,ref.station.lat),lon=coalesce(excluded.lon,ref.station.lon)", (code, row["stop_name"], *(coords or (None, None))))
    new, dup, numbers, skipped = 0, 0, set(), []
    for trip in trips:
        route = routes[trip["route_id"]]
        number = route.get("route_short_name") or route["route_id"]
        numbers.add(number)
        name = route.get("route_long_name") or number
        conn.execute("insert into ref.train(number,name,class) values (%s,%s,%s) on conflict(number) do update set name=excluded.name,class=excluded.class", (number, name, train_class(number, name)))
        calendar = calendars.get(trip["service_id"])
        if calendar is None or calendar["start_date"] > calendar["end_date"]:
            skipped.append({"trip_id": trip["trip_id"], "reason": "missing_calendar" if calendar is None else "inverted_calendar"})
            continue
        rows = sorted(times[trip["trip_id"]], key=lambda r: int(r["stop_sequence"]))
        if len(rows) < 2:
            raise ValueError(f"trip {trip['trip_id']} needs at least two stops")
        has_distances = all(r.get("shape_dist_traveled") for r in rows)
        dist_src, distance = ("gtfs" if has_distances else "gc"), 0.0
        counts, halts, previous, last_time = {}, [], None, -math.inf
        for idx, row in enumerate(rows):
            station, coords = stops[row["stop_id"]]
            if has_distances:
                distance = float(row["shape_dist_traveled"])
            elif previous and coords:
                # ponytail: great-circle km lower-bounds track km; upgrade with NTES timetable distances.
                distance += haversine(*previous, *coords)
            if not math.isfinite(distance) or distance < 0 or (halts and distance < halts[-1][-1]):
                raise ValueError(f"invalid distances: {trip['trip_id']}")
            arrival, departure = minutes(row["arrival_time"]), minutes(row["departure_time"])
            if arrival < last_time or departure < arrival:
                raise ValueError(f"non-monotone times: {trip['trip_id']}")
            last_time, previous = departure, coords
            counts[station] = counts.get(station, 0) + 1
            halts.append((station, counts[station], None if idx == 0 else arrival, None if idx == len(rows)-1 else departure, distance))
        days = sum(int(calendar[key]) << i for i, key in enumerate(("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")))
        first = date.fromisoformat(calendar["start_date"])
        last = date.fromisoformat(calendar["end_date"])
        digest = content_hash([halts, str(first), str(last), days])
        version = conn.execute("insert into ref.schedule_version(train,valid_from,valid_to,days_of_run,source,content_hash) values (%s,%s,%s,%s,'gtfs',%s) on conflict(train,content_hash) do nothing returning id", (number, first, last, days, digest)).fetchone()
        if version is None:
            dup += 1
            continue
        new += 1
        version = version[0]
        route_id = conn.execute("insert into ref.route_version(schedule_version_id,dist_src,content_hash) values (%s,%s,%s) returning id", (version, dist_src, content_hash([halts, dist_src]))).fetchone()[0]
        for idx, (station, occ, arr, dep, km) in enumerate(halts):
            conn.execute("insert into ref.schedule_stop values (%s,%s,%s,%s,%s,%s,%s)", (version, idx, station, occ, arr, dep, km))
            conn.execute("insert into ref.route_stop values (%s,%s,%s,%s,true,%s,%s)", (route_id, idx, station, occ, km, arr if arr is not None else dep))
    holidays = {r["day"]: r["holiday"] for r in read_csv(ROOT / "data/holidays.csv")}
    day = date(2025, 1, 1)
    with conn.cursor() as cursor:
        rows = []
        while day <= date(2027, 12, 31):
            rows.append((day, holidays.get(str(day)), day.month == 12 or day.month == 1 or (day.month == 2 and day.day <= 15)))
            day += timedelta(days=1)
        cursor.executemany("insert into ref.calendar values (%s,%s,%s) on conflict(day) do update set holiday=excluded.holiday,fog_north=excluded.fog_north", rows)
    return ImportReport(len(numbers), new, dup, len(stops), missing, skipped)


def download(source):
    url = f"https://raw.githubusercontent.com/{source['repo']}/{source['sha']}/{source['path']}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["import"])
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    sources = json.loads((ROOT / "data/static_sources.json").read_text())
    archive = args.zip.read_bytes() if args.zip else download(sources["gtfs"])
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            # Copy only known GTFS files; archive paths cannot escape the temporary directory.
            for name in ("routes.txt", "trips.txt", "stops.txt", "stop_times.txt", "calendar.txt"):
                matches = [entry for entry in z.namelist() if Path(entry).name == name]
                if len(matches) != 1:
                    raise ValueError(f"archive must contain one {name}")
                (folder / name).write_bytes(z.read(matches[0]))
        if not args.zip:
            (folder / "stations.json").write_bytes(download(sources["datameet"]))
        migrate.apply()
        with connect() as conn:
            from dataclasses import asdict
            print(json.dumps(asdict(import_gtfs(conn, folder)), indent=2))


if __name__ == "__main__":
    main()
