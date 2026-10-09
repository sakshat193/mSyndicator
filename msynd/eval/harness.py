"""Offline receipt-causal cases; scoring never selects cases using the outcome."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import math

import numpy as np
import pyarrow as pa

from msynd.dist import Eff, LEVELS, QUANTILES, eff_quantile, level_weight
from msynd.engine.evidence import load_run_evidence
from msynd.eval.baselines import b2_eta
from msynd.eval.metrics import block_bootstrap, sample_metrics, samples, summarize
from msynd.features.build import with_delay
from msynd.labels import IST


@dataclass(frozen=True)
class Case:
    run_id: int
    service_date: date
    issued_at: datetime
    halt_idx: int
    station: str
    occ: int
    sched_arr: float
    delay: float
    actual: float
    train_class: str
    zone: str | None
    horizon: int
    b2: float | None = None

    def __post_init__(self):
        if (self.issued_at.utcoffset() is None or self.halt_idx < 0 or self.occ < 1 or
                not np.isfinite([self.sched_arr, self.delay, self.actual]).all() or
                (self.b2 is not None and not math.isfinite(self.b2))):
            raise ValueError("invalid event key, timestamp or nonfinite case times")

    @property
    def key(self):
        return self.run_id, self.issued_at, self.station, self.occ, "arr"


def horizon_bucket(minutes):
    return next((h for h in (30, 120, 360, 720) if .75*h <= minutes <= 1.25*h), None)


def asof_cases(conn, start, end):
    """Issue interval [start,end), UTC/IST-aware datetimes, on the half-hour grid.

    Only completed/expired runs with six-hour-mature actuals supply outcomes.
    Running/remaining status and horizon selection use only issue-time evidence.
    """
    if start.utcoffset() is None or end.utcoffset() is None or start >= end:
        raise ValueError("require an increasing timezone-aware issue interval")
    runs = conn.execute("""SELECT r.id,r.start_date,t.class FROM obs.run r
        JOIN ref.train t ON t.number=r.train
        WHERE r.lifecycle IN ('completed','expired_unknown') ORDER BY r.id""").fetchall()
    matured = conn.execute("SELECT statement_timestamp()-interval '6 hours'").fetchone()[0]
    for run_id, service_date, train_class in runs:
        labels = {(stn, occ): (idx, sched, zone, actual) for idx, stn, occ, sched, zone, actual in
                  conn.execute("""SELECT s.idx,s.station,s.occ,s.arr_off,st.zone,a.actual_ts
                    FROM obs.run r JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
                    JOIN ref.station st ON st.code=s.station
                    JOIN obs.event a ON a.run_id=r.id AND a.station=s.station AND a.occ=s.occ AND a.kind='arr'
                    WHERE r.id=%s AND a.actual_ts<=%s AND s.arr_off IS NOT NULL ORDER BY s.idx""", (run_id, matured))}
        if not labels:
            continue
        base = datetime.combine(service_date, time(), IST)
        departure, first_received = conn.execute("""SELECT s.dep_off,
            (SELECT min(e.received_at) FROM obs.event_rev e WHERE e.run_id=r.id
             AND e.station=s.station AND e.occ=s.occ AND e.kind='dep')
            FROM ref.schedule_stop s JOIN obs.run r ON r.schedule_version_id=s.version_id
            WHERE r.id=%s ORDER BY s.idx LIMIT 1""", (run_id,)).fetchone()
        first_cut = base + timedelta(minutes=departure)
        if first_received is not None:
            first_cut = min(first_cut, first_received)
        cut = max(start, first_cut)
        cut = base + timedelta(minutes=30*math.ceil((cut-base).total_seconds()/1800))
        exceptions = conn.execute("""SELECT received_at,detail FROM obs.exception_rev x JOIN obs.run r
            ON r.train=x.train AND r.start_date=x.start_date WHERE r.id=%s ORDER BY received_at""", (run_id,)).fetchall()
        while cut < end:
            detail = next((detail for received, detail in reversed(exceptions) if received <= cut), {})
            if detail.get("kind") == "cancelled":
                cut += timedelta(minutes=30)
                continue
            ev = load_run_evidence(conn, run_id, cut)
            clock = (cut-base).total_seconds()/60
            actuals = [e for e in ev.evidence if e.actual]
            latest = max(actuals, key=lambda e: e.event, default=None)
            last_idx = latest.event if latest else -1
            delay = latest.minute-ev.events[last_idx].sched_off if latest else 0.
            stop = len(ev.events)-1 if ev.stop_event is None else ev.stop_event
            if last_idx >= stop:
                break
            if clock >= ev.events[0].sched_off + ev.resched_min or last_idx >= 0:
                for event in ev.events:
                    label = labels.get((event.station, event.occ))
                    if event.kind != "arr" or not last_idx < event.idx <= stop or label is None:
                        continue
                    idx, sched, zone, actual = label
                    horizon = horizon_bucket(sched+delay-clock)
                    if horizon is None:
                        continue
                    eta = b2_eta(conn, run_id, event.station, event.occ, "arr", cut)
                    yield Case(run_id, service_date, cut, idx, event.station, event.occ, sched, delay,
                               (actual-base).total_seconds()/60, train_class, zone, horizon,
                               (eta-base).total_seconds()/60 if eta is not None else None)
            cut += timedelta(minutes=30)


@dataclass
class Scores:
    records: list
    overall: dict
    breakdowns: dict
    disruption_days: set

    @property
    def case_ids(self):
        return tuple(r["key"] for r in self.records)


def _delay_band(delay):
    if delay < 0:
        return "early"
    if delay <= 15:
        return "0-15"
    if delay <= 60:
        return "15-60"
    return ">60"


def _disruption_days(records):
    counts = Counter({r["service_date"]: 0 for r in records})
    for r in records:
        counts[r["service_date"]] += r["pit"] < .001 or r["pit"] > .999
    if not any(counts.values()):
        return set()
    # ponytail: scored-case surprise counts; upgrade: stored network surprise counts once the engine persists them.
    ranked = sorted(counts, key=lambda day: (-counts[day], day))
    return {day for day in ranked[:max(1, math.ceil(len(ranked)/10))] if counts[day]}


def score(cases, forecaster, *, reference=None, disruption_days=None):
    """A callable yields samples (a scalar is a point forecast), or None if ineligible.

    A name->callable mapping scores all forecasts on their eligibility intersection.
    B3, when present, supplies reference PITs; otherwise each scored forecast does.
    An explicit disruption_days set, including an empty set, overrides inference.
    """
    multiple = isinstance(forecaster, Mapping)
    forecasters = dict(forecaster) if multiple else {"forecast": forecaster}
    if not forecasters:
        raise ValueError("require at least one forecaster")
    records = {name: [] for name in forecasters}
    ref_records, seen = [], set()
    for case in cases:
        if case.key in seen:
            raise ValueError("duplicate case key")
        seen.add(case.key)
        predictions = {name: f(case) for name, f in forecasters.items()}
        if any(value is None for value in predictions.values()):
            continue
        metadata = dict(key=case.key, service_date=case.service_date, actual=case.actual,
                        **{"class": case.train_class}, zone=case.zone,
                        delay_band=_delay_band(case.delay), horizon=case.horizon)
        for name, values in predictions.items():
            records[name].append(metadata | sample_metrics(values, case.actual))
        if reference is not None and disruption_days is None:
            values = reference(case)
            if values is None:
                raise ValueError("reference forecaster is ineligible on a scored case")
            ref_records.append(metadata | sample_metrics(values, case.actual))
    result = {}
    for name, rows in records.items():
        reference_rows = ref_records if reference is not None else records.get("B3", rows)
        days = set(disruption_days) if disruption_days is not None else _disruption_days(reference_rows)
        groups = {dimension: defaultdict(list) for dimension in ("class", "zone", "delay_band", "horizon", "disruption_day")}
        for row in rows:
            row["disruption_day"] = row["service_date"] in days
            for dimension in groups:
                groups[dimension][row[dimension]].append(row)
        result[name] = Scores(rows, summarize(rows),
                              {dim: {key: summarize(group) for key, group in values.items()}
                               for dim, values in groups.items()}, days)
    return result if multiple else result["forecast"]


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reasons: tuple[str, ...]


def promotion_ok(candidate, active, *, now=None):
    """Read-only decision on the last two complete IST Monday-to-Sunday weeks."""
    now = now or datetime.now(IST)
    day = now.astimezone(IST).date() if isinstance(now, datetime) else now
    end = day-timedelta(days=day.weekday())
    start = end-timedelta(days=14)
    c = {r["key"]: r for r in candidate.records if start <= r["service_date"] < end}
    a = {r["key"]: r for r in active.records if start <= r["service_date"] < end}
    if not c or c.keys() != a.keys():
        return Decision(False, ("require identical eligible cases in the recent two full weeks",))
    if any(c[key]["actual"] != a[key]["actual"] or c[key]["class"] != a[key]["class"] for key in c):
        return Decision(False, ("paired outcomes/classes differ",))
    cs, ac = summarize(list(c.values())), summarize(list(a.values()))
    reasons = []
    if cs["crps"] > .98*ac["crps"]:
        reasons.append("CRPS gain below 2%")
    diff = defaultdict(list)
    for key in c:
        diff[c[key]["service_date"]].append(a[key]["crps"]-c[key]["crps"])
    if block_bootstrap(diff).lo <= 0:
        reasons.append("CRPS gain CI lower bound is not positive")
    if any(abs(cs["coverage"][nominal]-nominal) > .05+1e-12 for nominal in (.8, .9)):
        reasons.append("coverage outside nominal +/-5 points")
    for train_class in {r["class"] for r in a.values()}:
        keys = [key for key in a if a[key]["class"] == train_class]
        if np.mean([c[key]["crps"] for key in keys]) > 1.05*np.mean([a[key]["crps"] for key in keys]) + 1e-12:
            reasons.append(f"class {train_class} CRPS regression exceeds 5%")
    return Decision(not reasons, tuple(reasons))


def n_curve(cases, Ns=(1000, 5000, 10000, 50000, 100000), *, forecaster):
    """Smallest N with <=0.25 min replicate SD at P05/P95 on every 6 h case.

    Use the scoring forecaster, accepting n and an independently salted replicate.
    Twenty independent replays measure error of a single replay, not its average.
    None means no supplied N meets the ceiling.
    """
    cases = [c for c in cases if c.horizon == 360]
    if not cases:
        raise ValueError("require 6 h horizon cases")
    if not Ns or any(not isinstance(n, int) or n <= 0 for n in Ns):
        raise ValueError("require positive integer particle counts")
    for n in sorted(set(Ns)):
        errors = []
        for case in cases:
            q = [np.quantile(samples(forecaster(case, n=n, replicate=r)), [.05, .5, .95]) for r in range(20)]
            errors.append(np.std(q, axis=0, ddof=1))
        if np.max(np.asarray(errors)[:, [0, 2]]) <= .25:
            return n
    return None


def interp_error(model, queries, delays, *, conn=None):
    """Residual-quantile error of level interpolation versus direct delay queries."""
    delays = np.asarray(delays, dtype=float)
    if delays.shape != (len(queries),) or not len(delays) or not np.isfinite(delays).all():
        raise ValueError("require one finite delay per nonempty query")
    if "d" in queries.column_names:
        queries = queries.drop(["d"])
    expanded = queries.take(pa.array(np.repeat(np.arange(len(queries)), len(LEVELS))))

    def predict(d):
        query = with_delay(conn, expanded, d) if conn is not None else expanded.append_column("d", pa.array(d))
        if hasattr(model, "feature_names"):
            query = query.select(model.feature_names)
        return model.predict_distribution(query)

    tabulated = predict(np.tile(LEVELS, len(queries)))
    direct = predict(np.repeat(delays, len(LEVELS)))
    errors = []
    for i, d in enumerate(delays):
        j, weight = level_weight(d)
        quantiles = []
        for distribution, level, w in ((tabulated, j, weight), (direct, 0, 0.)):
            masses, lo, hi = distribution
            signed = model.role != "dwell"
            # Match the engine's current per-item (maximum over levels) tail scales.
            eff = Eff(masses[i], signed, float(lo[i].max()) if signed else 1., float(hi[i].max()),
                      level, w, 0., 1., -np.inf if signed else 0., 0., 1.)
            quantiles.append([eff_quantile(eff, q) for q in QUANTILES[1:-1]])
        errors.append(np.abs(np.asarray(quantiles[0])-quantiles[1]))
    return {"mean": float(np.mean(errors)), "max": float(np.max(errors)), "errors": np.asarray(errors)}
