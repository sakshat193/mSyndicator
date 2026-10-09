"""Commit a parsed status page and its dirty markers together."""
from dataclasses import dataclass

from msynd.collect.dispatcher import FetchResult
from msynd.collect.lifecycle import apply_exception
from msynd.collect.status import ParsedStatus
from msynd.db.dirty import mark_dirty
from msynd.static.gtfs import content_hash


@dataclass(frozen=True)
class StoreResult:
    changed: bool


def _enrich(conn, schedule, route, passing):
    if not passing:
        return []
    routes = {}
    for version, station, occ, halt, km, off in conn.execute('''
        SELECT v.id,s.station,s.occ,s.halt,s.dist_km,s.sched_off
        FROM ref.route_version v JOIN ref.route_stop s ON s.route_version_id=v.id
        WHERE v.schedule_version_id=%s ORDER BY v.id,s.idx''', (schedule,)):
        routes.setdefault(version, []).append((station, occ, halt, km, off))
    halts = [row for row in routes[route] if row[2]]
    compatible = [version for version, rows in routes.items()
                  if [row for row in rows if row[2]] == halts]
    stops = {(row[0], row[1]): row for version in compatible for row in routes[version]}
    for station in passing:
        key = (station.station, station.occ)
        if key not in stops:
            stops[key] = (station.station, station.occ, False, float(station.dist_km),
                          None if station.sched_off is None else float(station.sched_off))
    rows = sorted(stops.values(), key=lambda row: row[3])
    if rows == routes[route]:
        return []
    dist_src = conn.execute('SELECT dist_src FROM ref.route_version WHERE id=%s', (route,)).fetchone()[0]
    digest = content_hash([rows, dist_src])
    created = conn.execute('''INSERT INTO ref.route_version(schedule_version_id,content_hash,dist_src)
        VALUES (%s,%s,%s) ON CONFLICT (schedule_version_id,content_hash)
        DO NOTHING RETURNING id''', (schedule, digest, dist_src)).fetchone()
    new_route = created[0] if created else conn.execute('''SELECT id FROM ref.route_version
        WHERE schedule_version_id=%s AND content_hash=%s''', (schedule, digest)).fetchone()[0]
    for idx, (station, occ, halt, km, off) in enumerate(rows):
        conn.execute('INSERT INTO ref.station(code,name) VALUES (%s,%s) ON CONFLICT DO NOTHING', (station, station))
        conn.execute('''INSERT INTO ref.route_stop VALUES (%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT DO NOTHING''', (new_route, idx, station, occ, halt, km, off))
    conn.execute('''INSERT INTO ref.segment(stn_lo,stn_hi,confidence)
        SELECT LEAST(a.station,b.station),GREATEST(a.station,b.station),0
        FROM ref.route_stop a JOIN ref.route_stop b
          ON b.route_version_id=a.route_version_id AND b.idx=a.idx+1
        WHERE a.route_version_id=%s ON CONFLICT DO NOTHING''', (new_route,))
    conn.execute('''INSERT INTO ref.route_segment(route_version_id,idx,segment_id,forward)
        SELECT a.route_version_id,a.idx,s.id,a.station<=b.station
        FROM ref.route_stop a JOIN ref.route_stop b
          ON b.route_version_id=a.route_version_id AND b.idx=a.idx+1
        JOIN ref.segment s ON s.stn_lo=LEAST(a.station,b.station) AND s.stn_hi=GREATEST(a.station,b.station)
        WHERE a.route_version_id=%s ON CONFLICT DO NOTHING''', (new_route,))
    return [r[0] for r in conn.execute('''UPDATE obs.run SET route_version_id=%s,updated_at=now()
        WHERE schedule_version_id=%s AND route_version_id=ANY(%s) AND route_version_id<>%s
        RETURNING id''', (new_route, schedule, compatible, new_route))]


def store_status(conn, run_id, fetch: FetchResult, page: ParsedStatus,
                 source='mmt_status') -> StoreResult:
    if conn.autocommit:
        raise ValueError('store_status requires an evidence transaction')
    if fetch.outcome != 'ok':
        return StoreResult(False)
    with conn.transaction():
        logged = conn.execute('SELECT purpose,outcome,run_id FROM obs.fetch WHERE id=%s', (fetch.fetch_id,)).fetchone()
        if logged is None:
            raise ValueError('unknown fetch')
        if logged[:2] != ('status', 'ok'):
            return StoreResult(False)
        if logged[2] is not None and logged[2] != run_id:
            raise ValueError('fetch belongs to another run')
        # Serialize enrichment and event comparisons for every run of this schedule.
        schedule = conn.execute('SELECT schedule_version_id FROM obs.run WHERE id=%s', (run_id,)).fetchone()
        if schedule is None:
            raise ValueError('unknown run')
        conn.execute('SELECT id FROM ref.schedule_version WHERE id=%s FOR UPDATE', schedule)
        train, start, route, lifecycle = conn.execute('''SELECT train,start_date,route_version_id,lifecycle
            FROM obs.run WHERE id=%s FOR UPDATE''', (run_id,)).fetchone()
        if (train, start) != (page.train, page.start_date):
            raise ValueError('status belongs to another train or origin date')
        if page.flags:
            conn.execute('UPDATE obs.fetch SET flags=%s,run_id=COALESCE(run_id,%s) WHERE id=%s',
                         (sorted(page.flags), run_id, fetch.fetch_id))
        if page.exceptions and logged[2] is None:
            conn.execute('UPDATE obs.fetch SET run_id=%s WHERE id=%s', (run_id, fetch.fetch_id))
        dirty = _enrich(conn, schedule[0], route, page.passing_stations)
        changed = False
        for event in page.events:
            key = (run_id, event.station, event.occ, event.kind)
            previous = conn.execute('''SELECT actual_ts FROM obs.event_rev
                WHERE run_id=%s AND station=%s AND occ=%s AND kind=%s
                ORDER BY received_at DESC,id DESC LIMIT 1''', key).fetchone()
            if previous is None or previous[0] != event.actual_ts:
                conn.execute('''INSERT INTO obs.event_rev(run_id,station,occ,kind,actual_ts,source,fetch_id,received_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (*key, event.actual_ts, source, fetch.fetch_id, fetch.received_at))
                changed = True
        for pending in page.pending:
            conn.execute('''INSERT INTO obs.pending_rev VALUES (%s,%s,%s,%s,%s,%s,%s)''',
                         (run_id, pending.station, pending.occ, pending.kind, source, fetch.fetch_id, fetch.received_at))
            changed = True
        for eta in page.etas:
            conn.execute('INSERT INTO obs.ntes_eta_rev VALUES (%s,%s,%s,%s,%s,%s,%s)',
                         (run_id, eta.station, eta.occ, eta.kind, eta.eta_ts, fetch.fetch_id, fetch.received_at))
            changed = True
        for exception in page.exceptions:
            apply_exception(conn, exception, fetch)
            changed = True
        lifecycle = conn.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run_id,)).fetchone()[0]
        destination = conn.execute('''SELECT station,occ FROM ref.schedule_stop
            WHERE version_id=%s ORDER BY idx DESC LIMIT 1''', schedule).fetchone()
        completed = conn.execute('''SELECT 1 FROM obs.event_rev
            WHERE run_id=%s AND station=%s AND occ=%s AND kind='arr' LIMIT 1''', (run_id, *destination)).fetchone()
        actual = conn.execute('SELECT 1 FROM obs.event_rev WHERE run_id=%s LIMIT 1', (run_id,)).fetchone()
        state = 'completed' if completed else 'running' if actual else lifecycle
        if lifecycle in ('scheduled', 'running', 'expired_unknown') and state != lifecycle:
            conn.execute('UPDATE obs.run SET lifecycle=%s,updated_at=%s WHERE id=%s', (state, fetch.received_at, run_id))
            changed = True
        if changed:
            dirty.append(run_id)
        mark_dirty(conn, dirty)
        return StoreResult(bool(dirty))
