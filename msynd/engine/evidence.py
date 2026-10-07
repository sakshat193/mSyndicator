from dataclasses import dataclass
from datetime import datetime, time
import hashlib
from pathlib import Path

from psycopg.rows import dict_row

from msynd.config import settings
from msynd.labels import IST


@dataclass(frozen=True)
class RouteEvent:
    idx: int
    station: str
    occ: int
    kind: str
    route_idx: int
    halt_idx: int | None
    sched_off: float
    km: float
    seg: int


@dataclass(frozen=True)
class Evidence:
    event: int
    actual: bool
    minute: float
    source: str


@dataclass(frozen=True)
class RunEvidence:
    run_id: int
    versions: dict
    events: tuple[RouteEvent, ...]
    evidence: tuple[Evidence, ...]
    resched_min: int
    stop_event: int | None


def canonical_events(route_stops) -> tuple[RouteEvent, ...]:
    """Route rows joined to schedule halts; preserve their immutable halt indices."""
    stops = sorted(route_stops, key=lambda s: s["idx"])
    halts = [s for s in stops if s["halt"]]
    if not halts:
        raise ValueError("route has no halts")
    events = []
    previous = halts[0]
    for stop in stops:
        if stop["halt"]:
            kinds = ("dep",) if stop is halts[0] else ("arr",) if stop is halts[-1] else ("arr", "dep")
            for kind in kinds:
                events.append(RouteEvent(len(events), stop["station"], stop["occ"], kind,
                    stop["idx"], stop["halt_idx"], stop[f"{kind}_off"], stop["dist_km"],
                    stop["halt_idx"] - (kind == "arr")))
            previous = stop
        else:
            scheduled = stop["sched_off"]
            if scheduled is None:
                following = next(s for s in halts if s["idx"] > stop["idx"])
                distance = following["dist_km"] - previous["dist_km"]
                fraction = (stop["dist_km"] - previous["dist_km"])/distance if distance else 0
                scheduled = previous["dep_off"] + fraction*(following["arr_off"] - previous["dep_off"])
            events.append(RouteEvent(len(events), stop["station"], stop["occ"], "pass",
                stop["idx"], None, scheduled, stop["dist_km"], previous["halt_idx"]))
    return tuple(events)


def load_run_evidence(conn, run_id: int, asof: datetime | None = None) -> RunEvidence:
    # One statement keeps route, revisions and model metadata in one DB snapshot.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            WITH r AS (SELECT * FROM obs.run WHERE id=%(run)s)
            SELECT r.*,
              (SELECT jsonb_agg(to_jsonb(s)) FROM (
                SELECT t.*,h.idx AS halt_idx,h.arr_off,h.dep_off
                FROM ref.route_stop t LEFT JOIN ref.schedule_stop h
                  ON h.version_id=r.schedule_version_id AND h.station=t.station AND h.occ=t.occ
                WHERE t.route_version_id=r.route_version_id ORDER BY t.idx
              ) s) AS stops,
              (SELECT jsonb_agg(to_jsonb(a)) FROM obs.event_asof(%(cutoff)s) a
                WHERE a.run_id=r.id) AS actuals,
              (SELECT jsonb_agg(to_jsonb(p)) FROM (
                SELECT DISTINCT ON (station,occ,kind) * FROM obs.pending_rev
                WHERE run_id=r.id AND received_at <= %(cutoff)s
                ORDER BY station,occ,kind,received_at DESC,(source='ntes_status') DESC
              ) p) AS pending,
              (SELECT jsonb_agg(to_jsonb(x) ORDER BY x.received_at DESC)
                FROM obs.exception_rev x WHERE x.train=r.train AND x.start_date=r.start_date
                  AND x.received_at <= %(cutoff)s) AS exceptions,
              (SELECT jsonb_agg(to_jsonb(m)) FROM (
                SELECT id,role,feature_ver FROM ml.model WHERE active
              ) m) AS models,
              (SELECT COALESCE(max(b.gen),0) FROM ml.leg_bias b
                JOIN ml.model m ON m.id=b.model_id WHERE m.active) AS bias_gen
            FROM r
        """, {"run": run_id, "cutoff": asof if asof is not None else "infinity"})
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"unknown run {run_id}")

    exceptions = row["exceptions"] or []
    detail = exceptions[0]["detail"] if exceptions else row if asof is None else {}
    cancelled = detail.get("part_cancelled") or []
    diverted = detail.get("diverted_from")
    stops = row["stops"] or []
    boundaries = [diverted] if diverted else []
    # A cancelled head starts at the first operated halt, retaining schedule indices.
    for section in cancelled:
        if stops and section["from"] == stops[0]["station"]:
            start = next(s["idx"] for s in stops if s["station"] == section["to"] and s["halt"])
            stops = [s for s in stops if s["idx"] >= start]
        else:
            boundaries.append(section["from"])
    events = canonical_events(stops)
    stop_event = None
    for station in boundaries:
        boundary = next((e.idx for e in events if e.station == station), None)
        if boundary is None:
            raise ValueError(f"stop station {station} absent from route")
        stop_event = boundary if stop_event is None else min(stop_event, boundary)

    actuals = {(a["station"], a["occ"], a["kind"]): a for a in row["actuals"] or []}
    pending = {(p["station"], p["occ"], p["kind"]): p for p in row["pending"] or []}
    origin = events[0]
    departure = actuals.get((origin.station, origin.occ, origin.kind))
    resched = next((x["detail"].get("resched_min", 0) for x in exceptions
        if departure is None or datetime.fromisoformat(x["received_at"]) < datetime.fromisoformat(departure["received_at"])), 0)
    base = datetime.combine(row["start_date"], time(), IST)
    evidence = []
    for event in events:
        if stop_event is not None and event.idx > stop_event:
            break
        key = (event.station, event.occ, event.kind)
        record = actuals.get(key) or pending.get(key)
        if record:
            actual = key in actuals
            timestamp = datetime.fromisoformat(record["actual_ts" if actual else "received_at"])
            evidence.append(Evidence(event.idx, actual, (timestamp-base).total_seconds()/60, record["source"]))

    models = row["models"] or []
    params = Path(__file__).resolve().parents[2] / "data/params.json"
    versions = {"schedule": row["schedule_version_id"], "route": row["route_version_id"],
        "models": {m["role"]: m["id"] for m in models}, "bias_gen": row["bias_gen"],
        "feature_ver": max((m["feature_ver"] for m in models), default=None),
        "N": settings.n_particles, "params_hash": hashlib.sha256(params.read_bytes()).hexdigest() if params.exists() else None}
    return RunEvidence(run_id, versions, events, tuple(evidence), resched, stop_event)
