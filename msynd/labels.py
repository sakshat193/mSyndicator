from datetime import datetime, time, timedelta, timezone

import pyarrow as pa

IST = timezone(timedelta(hours=5, minutes=30))


def run_label(dep_delay_a, arr_delay_b):
    return arr_delay_b - dep_delay_a


def dwell(sched_dwell, arr_delay, dep_delay):
    return sched_dwell + dep_delay - arr_delay


def origin_residual(sched_dep, resched_min, actual_dep):
    return actual_dep - (sched_dep + resched_min)


def composite(dwell_delta, run_delta):
    return dwell_delta + run_delta


def build_live_labels(conn, matured_before, asof=None, *, run_ids=None):
    """Mature labels for finished runs, optionally frozen at a receipt boundary."""
    events = "obs.event" if asof is None else "obs.event_asof(%(asof)s)"
    schema = pa.schema([("run_id", pa.int64()), ("role", pa.string()), ("idx", pa.int16()), ("value", pa.float64())])
    run_ids = None if run_ids is None else list(run_ids)
    if run_ids == []:
        return pa.Table.from_pylist([], schema=schema)
    selected = "" if run_ids is None else "WHERE id=ANY(%(run_ids)s)"
    event_filter = "" if run_ids is None else "WHERE run_id=ANY(%(run_ids)s)"
    finished = "r.lifecycle IN ('completed','expired_unknown')" if asof is None else """
        (done.run_id IS NOT NULL OR (r.lifecycle='expired_unknown' AND r.updated_at <= %(asof)s))"""
    fetch_cutoff = "" if asof is None else "AND f.received_at <= %(asof)s"
    rows = conn.execute(f"""
        WITH e AS (SELECT * FROM {events} {event_filter}),
        runs AS (SELECT * FROM obs.run {selected}),
        destination AS (SELECT DISTINCT ON (version_id) version_id,station,occ
          FROM ref.schedule_stop ORDER BY version_id,idx DESC),
        flagged AS (SELECT DISTINCT f.run_id FROM obs.fetch f
          WHERE 'exception_unparsed'=ANY(f.flags) {fetch_cutoff}),
        exceptions AS (SELECT x.*,lead(received_at) OVER
          (PARTITION BY train,start_date ORDER BY received_at) AS next_received
          FROM obs.exception_rev x)
        SELECT r.id,r.start_date,s.idx,s.arr_off,s.dep_off,
          a.actual_ts,d.actual_ts,d.received_at,
          COALESCE((x.detail->>'resched_min')::integer,0)
        FROM runs r JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
        LEFT JOIN destination dest ON dest.version_id=r.schedule_version_id
        LEFT JOIN e done ON done.run_id=r.id AND done.station=dest.station AND done.occ=dest.occ AND done.kind='arr'
        LEFT JOIN flagged f ON f.run_id=r.id
        LEFT JOIN e a ON a.run_id=r.id AND a.station=s.station AND a.occ=s.occ AND a.kind='arr'
          AND a.actual_ts <= %(matured_before)s
        LEFT JOIN e d ON d.run_id=r.id AND d.station=s.station AND d.occ=s.occ AND d.kind='dep'
          AND d.actual_ts <= %(matured_before)s
        LEFT JOIN exceptions x ON x.train=r.train AND x.start_date=r.start_date
          AND x.received_at < d.received_at AND (x.next_received >= d.received_at OR x.next_received IS NULL)
        WHERE {finished}
          AND f.run_id IS NULL
        ORDER BY r.id,s.idx
    """, {"matured_before": matured_before, "asof": asof, "run_ids": run_ids}).fetchall()
    labels, previous = [], None
    for run, start, idx, sched_arr, sched_dep, actual_arr, actual_dep, received, resched in rows:
        base = datetime.combine(start, time(), IST)
        arr = (actual_arr - base).total_seconds() / 60 if actual_arr else None
        dep = (actual_dep - base).total_seconds() / 60 if actual_dep else None
        if sched_arr is None and dep is not None:
            labels.append((run, "origin", idx, origin_residual(sched_dep, resched, dep)))
        if sched_arr is not None and sched_dep is not None and arr is not None and dep is not None:
            value = dwell(sched_dep - sched_arr, arr - sched_arr, dep - sched_dep)
            if value >= 0:
                labels.append((run, "dwell", idx, value))
        if previous and previous[0] == run and previous[1] == idx - 1 and arr is not None and previous[2] is not None:
            labels.append((run, "run", idx - 1, run_label(previous[2] - previous[3], arr - sched_arr)))
        previous = (run, idx, dep, sched_dep)
    return pa.Table.from_pylist([dict(zip(schema.names, row)) for row in labels], schema=schema)
