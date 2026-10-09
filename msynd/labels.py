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


def build_live_labels(conn, matured_before):
    """Current labels for finished runs; caller supplies the six-hour maturity cutoff."""
    rows = conn.execute("""
        SELECT r.id,r.start_date,s.idx,s.arr_off,s.dep_off,
          a.actual_ts,d.actual_ts,d.received_at,
          COALESCE((SELECT (detail->>'resched_min')::integer
            FROM obs.exception_rev x WHERE x.train=r.train AND x.start_date=r.start_date
              AND x.received_at < d.received_at
            ORDER BY x.received_at DESC LIMIT 1),0)
        FROM obs.run r JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
        LEFT JOIN obs.event a ON a.run_id=r.id AND a.station=s.station AND a.occ=s.occ AND a.kind='arr'
          AND a.actual_ts <= %s
        LEFT JOIN obs.event d ON d.run_id=r.id AND d.station=s.station AND d.occ=s.occ AND d.kind='dep'
          AND d.actual_ts <= %s
        WHERE r.lifecycle IN ('completed','expired_unknown')
          AND NOT EXISTS (SELECT 1 FROM obs.fetch f
            WHERE f.run_id=r.id AND 'exception_unparsed'=ANY(f.flags))
        ORDER BY r.id,s.idx
    """, (matured_before, matured_before)).fetchall()
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
    schema = pa.schema([("run_id", pa.int64()), ("role", pa.string()), ("idx", pa.int16()), ("value", pa.float64())])
    return pa.Table.from_pylist([dict(zip(schema.names, row)) for row in labels], schema=schema)
