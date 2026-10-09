"""Collector-owned run creation, expiry and cumulative exception revisions."""
from datetime import datetime, time, timedelta

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from msynd.db.dirty import mark_dirty
from msynd.engine.evidence import canonical_events
from msynd.labels import IST


SCHEDULES_SQL = '''
    SELECT DISTINCT ON (v.train,d.day) v.id,v.train,d.day,s.dep_off
    FROM dates d JOIN ref.schedule_version v
      ON v.valid_from<=d.day AND (v.valid_to IS NULL OR v.valid_to>=d.day)
      AND (v.days_of_run & (1 << (extract(isodow FROM d.day)::int-1)))<>0
    JOIN LATERAL (SELECT dep_off FROM ref.schedule_stop WHERE version_id=v.id
                  ORDER BY idx LIMIT 1) s ON true
    ORDER BY v.train,d.day,v.valid_from DESC,v.created_at DESC,v.id DESC
'''
LATEST_ROUTE_SQL = '''
    SELECT id FROM ref.route_version WHERE schedule_version_id=s.id
    ORDER BY created_at DESC,id DESC LIMIT 1
'''


def create_runs(conn, now) -> int:
    if conn.autocommit:
        raise ValueError('create_runs requires an evidence transaction')
    today = now.astimezone(IST).date()
    with conn.transaction():
        # ponytail: recovery scans three prior IST dates; widen for longer-running services.
        runs = conn.execute(f'''
            WITH dates AS (
                SELECT %s::date+n AS day FROM generate_series(-3,1) AS d(n)
            ), schedules AS (
                {SCHEDULES_SQL}
            )
            INSERT INTO obs.run(train,start_date,schedule_version_id,route_version_id,
                                resched_min,diverted_from,part_cancelled,updated_at)
            SELECT s.train,s.day,s.id,r.id,COALESCE((x.detail->>'resched_min')::int,0),
                   x.detail->>'diverted_from',NULLIF(x.detail->'part_cancelled','null'::jsonb),%s
            FROM schedules s JOIN LATERAL (
                {LATEST_ROUTE_SQL}
            ) r ON true JOIN LATERAL (
                SELECT arr_off FROM ref.schedule_stop WHERE version_id=s.id
                ORDER BY idx DESC LIMIT 1
            ) destination ON true LEFT JOIN LATERAL (
                SELECT detail FROM obs.exception_rev WHERE train=s.train AND start_date=s.day
                  AND received_at<=%s ORDER BY received_at DESC LIMIT 1
            ) x ON true
            WHERE (s.day::timestamp AT TIME ZONE 'Asia/Kolkata')
                  + destination.arr_off*interval '1 minute' + interval '12 hours'>%s
              AND (s.day::timestamp AT TIME ZONE 'Asia/Kolkata') + s.dep_off*interval '1 minute'<%s
              AND COALESCE(x.detail->>'kind','none')<>'cancelled'
            ON CONFLICT (train,start_date) DO NOTHING RETURNING id
        ''', (today, now, now, now, now + timedelta(hours=24))).fetchall()
        mark_dirty(conn, [r[0] for r in runs])
        return len(runs)


def expire_runs(conn, now) -> int:
    if conn.autocommit:
        raise ValueError('expire_runs requires an evidence transaction')
    with conn.transaction():
        rows = conn.execute('''
            SELECT r.id,r.start_date,r.schedule_version_id,r.route_version_id,
                   origin.dep_off,destination.arr_off,a.actual_ts,a.received_at,a.station,a.occ,a.kind,
                   CASE a.kind WHEN 'arr' THEN s.arr_off WHEN 'dep' THEN s.dep_off ELSE p.sched_off END
            FROM obs.run r
            JOIN LATERAL (SELECT dep_off FROM ref.schedule_stop WHERE version_id=r.schedule_version_id
                          ORDER BY idx LIMIT 1) origin ON true
            JOIN LATERAL (SELECT arr_off FROM ref.schedule_stop WHERE version_id=r.schedule_version_id
                          ORDER BY idx DESC LIMIT 1) destination ON true
            LEFT JOIN LATERAL (
                SELECT * FROM obs.event_asof(%s) WHERE run_id=r.id
                ORDER BY received_at DESC,actual_ts DESC,id DESC LIMIT 1
            ) a ON true
            LEFT JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id AND s.station=a.station AND s.occ=a.occ
            LEFT JOIN ref.route_stop p ON p.route_version_id=r.route_version_id AND p.station=a.station AND p.occ=a.occ
            WHERE r.lifecycle IN ('scheduled','running')
              AND %s > (r.start_date::timestamp AT TIME ZONE 'Asia/Kolkata')
                       + destination.arr_off*interval '1 minute' + interval '12 hours'
              AND COALESCE(a.received_at,(r.start_date::timestamp AT TIME ZONE 'Asia/Kolkata')
                           + origin.dep_off*interval '1 minute')<=%s-interval '12 hours'
            ORDER BY r.id FOR UPDATE OF r
        ''', (now, now, now)).fetchall()
        expired = []
        for run, start, schedule, route, origin, destination, actual, received, station, occ, kind, offset in rows:
            base = datetime.combine(start, time(), IST)
            if actual is not None and offset is None:
                with conn.cursor(row_factory=dict_row) as cur:
                    stops = cur.execute('''SELECT t.*,h.idx AS halt_idx,h.arr_off,h.dep_off
                        FROM ref.route_stop t LEFT JOIN ref.schedule_stop h
                          ON h.version_id=%s AND h.station=t.station AND h.occ=t.occ
                        WHERE t.route_version_id=%s ORDER BY t.idx''', (schedule, route)).fetchall()
                offset = next(e.sched_off for e in canonical_events(stops)
                              if (e.station, e.occ, e.kind) == (station, occ, kind))
            delay = (actual - base).total_seconds()/60 - offset if actual is not None else 0
            if now > base + timedelta(minutes=destination + max(720, 2*delay)):
                expired.append(run)
        if expired:
            conn.execute("UPDATE obs.run SET lifecycle='expired_unknown',updated_at=%s WHERE id=ANY(%s)",
                         (now, expired))
            mark_dirty(conn, expired)
        return len(expired)


def apply_exception(conn, row, fetch) -> None:
    """The status fetch identifies the run; row contains only exception detail."""
    if conn.autocommit:
        raise ValueError('apply_exception requires an evidence transaction')
    with conn.transaction():
        run = conn.execute('''SELECT r.id,r.train,r.start_date,r.schedule_version_id,r.route_version_id,r.lifecycle
            FROM obs.run r JOIN obs.fetch f ON f.run_id=r.id
            WHERE f.id=%s AND f.purpose='status' AND f.outcome='ok' AND f.received_at=%s
            FOR UPDATE OF r''', (fetch.fetch_id, fetch.received_at)).fetchone()
        if run is None or fetch.outcome != 'ok':
            raise ValueError('exception requires an ok status fetch bound to a run')
        run_id, train, start, schedule, route, state = run
        kind = row.get('kind')
        if kind not in ('cancelled', 'part_cancelled', 'rescheduled', 'diverted', 'none'):
            raise ValueError('invalid exception kind')
        required = {'rescheduled': 'resched_min', 'diverted': 'diverted_from', 'part_cancelled': 'part_cancelled'}
        if kind in required and required[kind] not in row:
            raise ValueError('missing exception attribute')
        previous = conn.execute('''SELECT detail FROM obs.exception_rev WHERE train=%s AND start_date=%s
            AND received_at<=%s ORDER BY received_at DESC LIMIT 1''', (train, start, fetch.received_at)).fetchone()
        detail = {'resched_min': 0, 'diverted_from': None, 'part_cancelled': None}
        if kind != 'none':
            if previous:
                detail.update({k: previous[0][k] for k in detail if k in previous[0]})
            detail.update({k: row[k] for k in detail if k in row})
        if type(detail['resched_min']) is not int:
            raise ValueError('resched_min must be integer minutes')
        stations = conn.execute('SELECT station,halt,idx FROM ref.route_stop WHERE route_version_id=%s ORDER BY idx',
                                (route,)).fetchall()
        if detail['diverted_from'] is not None and detail['diverted_from'] not in [s[0] for s in stations]:
            raise ValueError('diverted_from absent from route')
        part = detail['part_cancelled']
        if part is not None:
            if not isinstance(part, list):
                raise ValueError('part_cancelled must be a list of halt sections')
            halts = [s[0] for s in stations if s[1]]
            for section in part:
                if (not isinstance(section, dict) or set(section) != {'from', 'to'}
                        or section['from'] not in halts or section['to'] not in halts
                        or halts.index(section['from']) >= halts.index(section['to'])):
                    raise ValueError('invalid partial cancellation section')
        detail['kind'] = kind
        # One page can contain several exceptions; its single revision is their cumulative state.
        conn.execute('''INSERT INTO obs.exception_rev VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (train,start_date,received_at) DO UPDATE SET detail=excluded.detail''',
            (train, start, Jsonb(detail), fetch.fetch_id, fetch.received_at))
        latest = conn.execute('''SELECT detail,received_at FROM obs.exception_rev
            WHERE train=%s AND start_date=%s ORDER BY received_at DESC LIMIT 1''', (train, start)).fetchone()
        if latest[1] == fetch.received_at:
            if kind == 'cancelled':
                state = 'cancelled'
            elif state == 'cancelled':
                destination = conn.execute('''SELECT station,occ FROM ref.schedule_stop
                    WHERE version_id=%s ORDER BY idx DESC LIMIT 1''', (schedule,)).fetchone()
                actual = conn.execute('SELECT station,occ,kind FROM obs.event WHERE run_id=%s', (run_id,)).fetchall()
                state = 'completed' if (*destination, 'arr') in actual else 'running' if actual else 'scheduled'
            conn.execute('''UPDATE obs.run SET lifecycle=%s,resched_min=%s,diverted_from=%s,
                part_cancelled=%s,updated_at=%s WHERE id=%s''',
                (state, detail['resched_min'], detail['diverted_from'], Jsonb(part) if part is not None else None,
                 fetch.received_at, run_id))
        mark_dirty(conn, [run_id])
