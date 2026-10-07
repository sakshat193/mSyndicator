from datetime import datetime, timezone


def make_run(db, train="12951", start="2026-10-07", halts=None):
    halts = halts or [("MMCT", None, 0, 0), ("BRC", 300, 305, 392), ("NDLS", 940, None, 1384)]
    db.execute("insert into ref.train(number,name) values (%s,%s) on conflict do nothing", (train, train))
    for station, *_ in halts:
        db.execute("insert into ref.station(code,name) values (%s,%s) on conflict do nothing", (station, station))
    v = db.execute("insert into ref.schedule_version(train,valid_from,days_of_run,source,content_hash) values (%s,'2025-01-01',127,'gtfs',%s) returning id", (train, train.encode())).fetchone()[0]
    route = db.execute("insert into ref.route_version(schedule_version_id,content_hash,dist_src) values (%s,%s,'gtfs') returning id", (v, train.encode())).fetchone()[0]
    counts = {}
    for idx, (station, arr, dep, km) in enumerate(halts):
        counts[station] = counts.get(station, 0) + 1
        occ = counts[station]
        db.execute("insert into ref.schedule_stop values (%s,%s,%s,%s,%s,%s,%s)", (v, idx, station, occ, arr, dep, km))
        db.execute("insert into ref.route_stop values (%s,%s,%s,%s,true,%s,%s)", (route, idx, station, occ, km, dep if arr is None else arr))
    return db.execute("insert into obs.run(train,start_date,schedule_version_id,route_version_id) values (%s,%s,%s,%s) returning id", (train, start, v, route)).fetchone()[0]


def insert_fetch(db, **kw):
    values = {"host": "h", "purpose": "status", "started_at": datetime.now(timezone.utc), "outcome": "ok"} | kw
    columns = ",".join(values)
    return db.execute(f"insert into obs.fetch({columns}) values ({','.join(['%s'] * len(values))}) returning id", tuple(values.values())).fetchone()[0]


def insert_event(db, run_id, station, occ, kind, actual, received, fetch_id=None):
    return db.execute("insert into obs.event_rev(run_id,station,occ,kind,actual_ts,source,fetch_id,received_at) values (%s,%s,%s,%s,%s,'ntes_status',%s,%s) returning id", (run_id, station, occ, kind, actual, fetch_id, received)).fetchone()[0]
