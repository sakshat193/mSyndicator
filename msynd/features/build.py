"""Receipt-causal segment features and training-only fold statistics.

Weeks are Monday dates on the origin's IST calendar. training_frame returns
Arrow rows (with train/valid split metadata) and aligned running labels.
History remains quarantined until G4 defines its meaning.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from psycopg.rows import dict_row

from msynd.labels import IST, build_live_labels, run_label


@dataclass(frozen=True)
class Fold:
    train_weeks: tuple[date, ...]
    valid_weeks: tuple[date, ...]

    def __post_init__(self):
        for weeks in (self.train_weeks, self.valid_weeks):
            if not weeks or len(set(weeks)) != len(weeks) or any(w.weekday() != 0 for w in weeks):
                raise ValueError("weeks must be distinct Monday dates, with nonempty train and validation")
        if max(self.train_weeks) >= min(self.valid_weeks):
            raise ValueError("validation weeks must follow all training weeks")


# Explicit types keep all-missing causal context usable by model consumers.
_SCHEMA = pa.schema([
    ("run_id", pa.int64()), ("start_date", pa.date32()), ("train", pa.string()),
    ("class", pa.string()), ("idx", pa.int16()), ("segments_remaining", pa.int16()),
    ("is_last", pa.bool_()), ("from_stn", pa.string()), ("to_stn", pa.string()),
    ("km", pa.float64()), ("sched_dep", pa.float64()), ("sched_run", pa.float64()),
    ("sched_dwell", pa.float64()), ("zone", pa.string()),
    ("last_delta_run", pa.float64()), ("segments_since_observed", pa.int16()),
    ("hour_sin", pa.float64()), ("hour_cos", pa.float64()),
    ("weekday", pa.int8()), ("month", pa.int8()), ("fog_north", pa.bool_()),
    ("holiday", pa.bool_()), ("train_encoded", pa.float64()), ("slack", pa.float64()),
])


def _calendar(conn, rows, delays):
    moments = [datetime.combine(r["start_date"], time(), IST) +
               timedelta(minutes=r["sched_dep"] + float(d)) for r, d in zip(rows, delays)]
    days = sorted({m.date() + timedelta(days=n) for m in moments for n in (-1, 0, 1)})
    calendar = {day: (holiday, fog) for day, holiday, fog in conn.execute(
        "SELECT day,holiday,fog_north FROM ref.calendar WHERE day=ANY(%s)", (days,))}
    for row, moment in zip(rows, moments):
        angle = 2 * math.pi * (moment.hour + moment.minute/60 + moment.second/3600) / 24
        row.update(hour_sin=math.sin(angle), hour_cos=math.cos(angle), weekday=moment.weekday(),
                   month=moment.month,
                   fog_north=bool(calendar.get(moment.date(), (None, False))[1] and
                                  row["zone"] in {"NR", "NCR", "NER", "NWR"}),
                   holiday=any(calendar.get(moment.date()+timedelta(days=n), (None, False))[0]
                               is not None for n in (-1, 0, 1)))


def with_delay(conn, queries, d):
    """Attach delay and recompute calendar features at sched_dep + d (IST)."""
    delays = np.full(len(queries), d, dtype=float) if np.isscalar(d) else np.asarray(d, dtype=float)
    if delays.shape != (len(queries),) or not np.all(np.isfinite(delays)):
        raise ValueError("require one finite delay per query")
    rows = queries.to_pylist()
    _calendar(conn, rows, delays)
    for row, delay in zip(rows, delays):
        row["d"] = float(delay)
    return pa.Table.from_pylist(rows, schema=queries.schema.append(pa.field("d", pa.float64())))


def segment_queries(conn, run_id, asof):
    """Shared halt-pair features; all observed context uses event_asof, never current truth."""
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute("""
            WITH r AS (SELECT * FROM obs.run WHERE id=%s),
            e AS (SELECT * FROM obs.event_asof(%s) WHERE run_id=%s)
            SELECT r.id AS run_id,r.start_date,r.train,t.class,a.idx,
              a.station AS from_stn,b.station AS to_stn,st.zone,
              b.dist_km-a.dist_km AS km,a.dep_off AS sched_dep,
              b.arr_off-a.dep_off AS sched_run,COALESCE(a.dep_off-a.arr_off,0) AS sched_dwell,
              da.actual_ts AS dep_actual,ab.actual_ts AS arr_actual
            FROM r JOIN ref.train t ON t.number=r.train
            JOIN ref.schedule_stop a ON a.version_id=r.schedule_version_id
            JOIN ref.schedule_stop b ON b.version_id=a.version_id AND b.idx=a.idx+1
            JOIN ref.station st ON st.code=a.station
            LEFT JOIN e da ON da.station=a.station AND da.occ=a.occ AND da.kind='dep'
            LEFT JOIN e ab ON ab.station=b.station AND ab.occ=b.occ AND ab.kind='arr'
            ORDER BY a.idx
        """, (run_id, asof, run_id)).fetchall()
    last = None
    for i, row in enumerate(rows):
        dep, arr = row.pop("dep_actual"), row.pop("arr_actual")
        row.update(segments_remaining=len(rows)-i-1, is_last=i == len(rows)-1,
                   last_delta_run=last[1] if last else None,
                   segments_since_observed=row["idx"]-last[0] if last else None,
                   train_encoded=None, slack=None)
        if dep is not None and arr is not None:
            base = datetime.combine(row["start_date"], time(), IST)
            last = (row["idx"], run_label((dep-base).total_seconds()/60-row["sched_dep"],
                    (arr-base).total_seconds()/60-row["sched_dep"]-row["sched_run"]))
    _calendar(conn, rows, [0.] * len(rows))
    return pa.Table.from_pylist(rows, schema=_SCHEMA)


@dataclass
class FoldEncoders:
    mean: float
    train_means: dict
    pair_p05: dict

    @classmethod
    def fit(cls, train_rows):
        rows = [r for r in train_rows.to_pylist() if r.get("split", "train") == "train"]
        if not rows:
            raise ValueError("encoders require training rows")
        values = np.asarray([r["value"] for r in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("encoder targets must be finite")
        mean = float(values.mean())
        trains, pairs = {}, {}
        for r in rows:
            trains.setdefault(r["train"], []).append(r["value"])
            pairs.setdefault((r["from_stn"], r["to_stn"]), []).append(r["observed_running"])
        # ponytail: fixed ten-row shrinkage; tune only inside training folds if validation warrants it.
        return cls(mean, {k: (sum(v)+10*mean)/(len(v)+10) for k, v in trains.items()},
                   {k: float(np.quantile(v, .05)) for k, v in pairs.items()})

    def transform(self, queries):
        rows = queries.to_pylist()
        for r in rows:
            p05 = self.pair_p05.get((r["from_stn"], r["to_stn"]))
            r["train_encoded"] = self.train_means.get(r["train"], self.mean)
            r["slack"] = r["sched_run"]-p05 if p05 is not None else None
        schema = queries.schema
        for name in ("train_encoded", "slack"):
            if name not in schema.names:
                schema = schema.append(pa.field(name, pa.float64()))
        return pa.Table.from_pylist(rows, schema=schema)


def training_frame(conn, fold: Fold):
    """Matured running labels, departure-receipt features, train-only encoders."""
    cutoff = conn.execute("SELECT CURRENT_TIMESTAMP - interval '6 hours'").fetchone()[0]
    labels = {(r["run_id"], r["idx"]): r["value"] for r in
              build_live_labels(conn, cutoff).to_pylist() if r["role"] == "run"}
    departures = conn.execute("""
        SELECT r.id,r.start_date,s.idx,s.dep_off,d.actual_ts,d.received_at
        FROM obs.run r JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
        JOIN obs.event d ON d.run_id=r.id AND d.station=s.station AND d.occ=s.occ AND d.kind='dep'
        ORDER BY r.start_date,r.id,s.idx
    """).fetchall()
    rows = []
    # ponytail: per-departure queries; batch receipt cutoffs if weekly training becomes slow.
    for run, start, idx, scheduled, actual, received in departures:
        week = start - timedelta(days=start.weekday())
        split = "train" if week in fold.train_weeks else "valid" if week in fold.valid_weeks else None
        if split is None or (run, idx) not in labels:
            continue
        queries = segment_queries(conn, run, received)
        selected = queries.filter(pc.equal(queries["idx"], idx))
        d = (actual-datetime.combine(start, time(), IST)).total_seconds()/60-scheduled
        row = with_delay(conn, selected, d).to_pylist()[0]
        value = labels[(run, idx)]
        rows.append(row | {"split": split, "value": value,
                           "observed_running": row["sched_run"]+value})
    schema = _SCHEMA.append(pa.field("d", pa.float64())).append(pa.field("split", pa.string()))
    raw = pa.Table.from_pylist(rows, schema=schema.append(pa.field("value", pa.float64()))
                              .append(pa.field("observed_running", pa.float64())))
    if not rows:
        return pa.Table.from_pylist([], schema=schema), np.empty(0)
    encoders = FoldEncoders.fit(raw)
    return encoders.transform(raw).drop(["value", "observed_running"]), np.asarray([r["value"] for r in rows])


