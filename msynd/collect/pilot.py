"""MMT pilot; real traffic is enabled only by the existing checkpoint guard."""

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np
from psycopg.rows import dict_row
import zstandard

from msynd.collect.dispatcher import Dispatcher, real_transport
from msynd.collect.mmt import HOST, MmtClient, classify
from msynd.collect.parse_mmt import parse_status
from msynd.collect.railradar import IST
from msynd.collect.status import InvalidResponse
from msynd.db import connect

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "data/mmt_capture/pilot.json"


def cohort_bound(rate, h=1):
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(h) or h <= 0:
        raise ValueError("rate and h must be finite and positive")
    return math.floor(.7 * rate * 120 / h)


@dataclass
class PilotRun:
    train: str
    start: date
    expected: dict

    @property
    def key(self):
        return f"{self.train}/{self.start.isoformat()}"

    def in_window(self, now):
        return any(ts - timedelta(minutes=20) <= now <= ts + timedelta(minutes=60)
                   for ts in self.expected.values())


def inventory(conn, now):
    """Enumerate origin dates, including multi-day runs, from local schedules."""
    rows = conn.execute("""SELECT v.id,v.train,v.valid_from,v.valid_to,v.days_of_run,
        s.station,s.occ,s.arr_off,s.dep_off FROM ref.schedule_version v
        JOIN ref.schedule_stop s ON s.version_id=v.id
        ORDER BY v.valid_from DESC,v.created_at DESC,v.id DESC,s.idx""").fetchall()
    versions = {}
    for version, train, first, last, days, station, occ, arr, dep in rows:
        if version not in versions:
            versions[version] = (train, first, last, days, {})
        events = versions[version][-1]
        for kind, offset in (("arr", arr), ("dep", dep)):
            if offset is not None:
                events[station, occ, kind] = offset
    today = now.astimezone(IST).date()
    result, seen = [], set()
    for train, first, last, days, events in versions.values():
        if not events:
            continue
        for back in range(-1, math.ceil(max(events.values()) / 1440) + 2):
            start = today - timedelta(days=back)
            if (train, start) in seen or start < first or (last is not None and start > last) or not days & (1 << start.weekday()):
                continue
            seen.add((train, start))
            base = datetime.combine(start, time(), IST).astimezone(timezone.utc)
            expected = {key: base + timedelta(minutes=offset) for key, offset in events.items()}
            if min(expected.values()) - timedelta(minutes=20) <= now <= max(expected.values()) + timedelta(minutes=60):
                result.append(PilotRun(train, start, expected))
    return result


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _classify_status(purpose, response):
    outcome = classify(purpose, response)
    if outcome == "ok":
        request = json.loads(response.request.content)
        try:
            parse_status(response.content, request["trainNumber"],
                         datetime.strptime(request["dateOfJourney"], "%d-%m-%Y").date())
        except InvalidResponse:
            return "invalid"
    return outcome


async def run_pilot(client, *, days=7, metadata=METADATA, candidates=None):
    d, clock = client.d, client.d.clock
    bound = cohort_bound(d.rate)
    if not math.isfinite(days) or days <= 0 or bound < 1:
        raise ValueError("pilot needs positive days and capacity for at least one run")
    if Path(metadata).exists():
        raise FileExistsError("archive the previous pilot metadata before starting another pilot")
    session = None
    try:
        d.classify = _classify_status
        await d.start()
        if candidates is None:
            candidates = lambda now: inventory(d.conn, now)
        started = clock.now()
        end = started + timedelta(days=days)
        session = dict(started_at=started.isoformat(), planned_end=end.isoformat(),
                       start_fetch_id=d.conn.execute("SELECT coalesce(max(id),0) FROM obs.fetch").fetchone()[0],
                       polls=0, rate=d.rate, active_curve=[], max_cohort=0)
        runs, cohort, rejected, seen_trains = {}, [], set(), set()
        counts, last_poll = Counter(), {}
        next_hour, next_discovery = started, started
        available = []
        _write_json(metadata, session)
        while clock.now() < end:
            now = clock.now()
            if now >= next_discovery or now >= next_hour:
                available = candidates(now)
                next_discovery = now + timedelta(minutes=1)
            for run in available:
                runs.setdefault(run.key, run)
            hourly = now >= next_hour
            if hourly:
                session["active_curve"].append(dict(at=now.isoformat(), count=sum(
                    min(r.expected.values()) <= now <= max(r.expected.values()) for r in available)))
                next_hour += timedelta(hours=1)
            eligible = [r.key for r in runs.values() if r.key not in rejected and r.in_window(now)]
            cohort = [key for key in cohort if key in eligible]
            if hourly:
                cohort = []
            # Rotate toward unseen trains, then the least-polled runs; keep a cohort between hours.
            choices = sorted((key for key in eligible if key not in cohort),
                             key=lambda key: (runs[key].train in seen_trains, counts[key], key))
            cohort.extend(choices[:bound - len(cohort)])
            session["max_cohort"] = max(session["max_cohort"], len(cohort))
            due = [key for key in cohort if (now - last_poll.get(key, started - timedelta(seconds=120))).total_seconds() >= 120]
            if not due:
                await clock.sleep(min(5, (end - now).total_seconds()))
                continue
            key = min(due, key=lambda k: last_poll.get(k, started - timedelta(seconds=120)))
            run = runs[key]
            session["polls"] += 1
            seen_trains.add(run.train)
            result = await client.train_running(run.train, run.start)
            last_poll[key] = clock.now()
            counts[key] += 1
            if result.outcome == "ok":
                parsed = parse_status(result.body, run.train, run.start)
                for event in parsed.events:
                    run.expected[event.station, event.occ, event.kind] = event.actual_ts
                for eta in parsed.etas:
                    run.expected[eta.station, eta.occ, eta.kind] = eta.eta_ts
            elif result.status == 400:
                try:
                    not_running = json.loads(result.body).get("error", {}).get("errorCode") == 1001
                except (ValueError, AttributeError):
                    not_running = False
                if not_running:
                    rejected.add(key)
            if hourly:
                _write_json(metadata, session)
        now = clock.now()
        if now >= next_hour:
            available = candidates(now)
            session["active_curve"].append(dict(at=now.isoformat(), count=sum(
                min(r.expected.values()) <= now <= max(r.expected.values()) for r in available)))
    finally:
        try:
            if session is not None:
                session["ended_at"] = clock.now().isoformat()
                session["end_fetch_id"] = d.conn.execute("SELECT coalesce(max(id),0) FROM obs.fetch").fetchone()[0]
                _write_json(metadata, session)
        finally:
            await d.close()


def fit_s(intervals, *, unpublished=0, right_censored=(), left_censored=0):
    """Fit the empirical body and a positive exponential tail above P90."""
    if not intervals or any(not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo <= hi for lo, hi in intervals):
        raise ValueError("lag intervals must be finite, nonnegative and nonempty")
    if any(type(n) is not int or n < 0 for n in (unpublished, left_censored)):
        raise ValueError("censoring counts must be nonnegative integers")
    censored = np.asarray(right_censored, dtype=float)
    if censored.ndim != 1 or not np.isfinite(censored).all() or np.any(censored < 0):
        raise ValueError("censored lag bounds must be finite and nonnegative")
    # ponytail: interval midpoints; upgrade to Turnbull when the pilot warrants interval-censored fitting.
    lags = np.array([(lo + hi) / 2 for lo, hi in intervals])
    tail_start = float(np.quantile(lags, .9))
    excess = lags[lags > tail_start] - tail_start
    exposure = float(excess.sum() + np.maximum(0., censored-tail_start).sum())
    tail_events = len(excess) or int(np.count_nonzero(lags == tail_start))
    tau = max(.5, exposure / tail_events)
    # Kaplan-Meier: tied publications precede withdrawals from the risk set.
    times, indices = np.unique(np.r_[lags, censored], return_inverse=True)
    deaths = np.bincount(indices[:len(lags)], minlength=len(times))
    withdrawals = np.bincount(indices[len(lags):], minlength=len(times))
    risk = np.cumsum((deaths+withdrawals)[::-1])[::-1]
    km = np.cumprod(1.-deaths/risk)
    n, z = len(lags) + left_censored + unpublished, 1.959963984540054
    p = unpublished / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    ci = [max(0., center-half), min(1., center+half)]
    if unpublished == 0:
        ci[0] = 0.
    eps = center  # Wilson center keeps a positive never-published mass even with no counterexamples.
    grid = np.unique(np.r_[0., lags[(lags > 0) & (lags < tail_start)], tail_start])
    conditional = []
    for x in grid:
        idx = np.searchsorted(times, x, side="left") - 1
        conditional.append(1. if x == 0 or idx < 0 else float(km[idx]))
    survival = [eps + (1-eps)*s for s in conditional]

    def quantile(s):
        if conditional[-1] > s:
            return float(grid[-1] + tau*math.log(conditional[-1]/s))
        return float(np.interp(s, conditional[::-1], grid[::-1]))

    return dict(lag=grid.tolist(), survival=survival, tau=tau, eps=eps, eps_ci=ci,
                p50=quantile(.5), p90=quantile(.1), tail_start=tail_start,
                resolved=n, unpublished=unpublished, right_censored=len(censored))


def analyse_rows(rows, *, polls, active_curve):
    outcomes, histories, passing = Counter(), {}, set()
    eta_values, eta_confirmations, eta_changes = {}, 0, 0
    pending_after_actual, left_censored = 0, 0
    for row in sorted(rows, key=lambda row: row["received_at"] or row["started_at"]):
        outcome = row["outcome"]
        parsed = None
        if row["purpose"] == "status" and outcome == "ok":
            try:
                train, start = row["key"].split("/")
                parsed = parse_status(zstandard.decompress(row["body_zstd"]), train, date.fromisoformat(start))
            except (InvalidResponse, ValueError, TypeError, AttributeError, zstandard.ZstdError):
                outcome = "invalid"
        outcomes[outcome] += 1
        if parsed is None:
            continue
        history = histories.setdefault(row["key"], dict(actual={}, pending=set(), receipts={},
                                                       order={}, last_poll=None, status=None))
        history["status"] = parsed.status
        received = row["received_at"]
        occurrences = Counter()
        halts = json.loads(zstandard.decompress(row["body_zstd"]))["response"]["stations"]
        for idx, halt in enumerate(halts):
            station = halt["station"]["code"]
            occurrences[station] += 1
            for kind, rank in (("arr", 0), ("dep", 1)):
                history["order"][station, occurrences[station], kind] = (idx, rank)
        for event in parsed.events:
            key = (event.station, event.occ, event.kind)
            if key not in history["actual"]:
                previous = history["last_poll"]
                if previous is not None:
                    history["receipts"][key] = (previous, received)
                else:
                    left_censored += 1
            history["actual"][key] = event.actual_ts
        for event in parsed.pending:
            key = (event.station, event.occ, event.kind)
            if key in history["actual"]:
                pending_after_actual += 1
            else:
                history["pending"].add(key)
        history["last_poll"] = received
        for eta in parsed.etas:
            key = (row["key"], eta.station, eta.occ, eta.kind)
            eta_confirmations += 1
            if key in eta_values and eta_values[key] != eta.eta_ts:
                eta_changes += 1
            eta_values[key] = eta.eta_ts
        if parsed.passing_stations:
            passing.add(parsed.train)
    intervals, unpublished, censored_lags, excluded = [], 0, [], 0
    for history in histories.values():
        for key, receipts in history["receipts"].items():
            actual = history["actual"][key]
            intervals.append([max(0., (ts-actual).total_seconds()/60) for ts in receipts])
        for key in history["pending"] - history["actual"].keys():
            later = [ts for other, ts in history["actual"].items()
                     if history["order"][other] > history["order"][key]]
            if not later:
                excluded += 1
            elif history["status"] == "completed":
                unpublished += 1
            else:
                # Event time <= the earliest later actual, so this is a conservative lag bound.
                censored_lags.append(max(0., (history["last_poll"]-min(later)).total_seconds()/60))
    censored_lags.sort()
    fitted = fit_s(intervals, unpublished=unpublished, right_censored=censored_lags, left_censored=left_censored)
    total = sum(outcomes.values())
    trains = {key.split("/")[0] for key in histories}
    return dict(S={"mmt_status": fitted}, h=total/polls if polls else None,
                requests=total, outcomes=dict(outcomes), rates={k: outcomes[k]/total if total else 0.
                for k in ("empty", "invalid", "challenge")}, lag_intervals=intervals,
                left_censored=left_censored, unpublished=unpublished, right_censored=len(censored_lags),
                censored_lags=censored_lags, excluded=excluded,
                pending_after_actual=pending_after_actual, eta_confirmations=eta_confirmations,
                eta_changes=eta_changes, eta_revision_rate=eta_changes/eta_confirmations if eta_confirmations else 0.,
                distinct_runs=len(histories), trains=len(trains), passing_trains=len(passing),
                passing_prevalence=len(passing)/len(trains) if trains else 0., active_curve=active_curve,
                targets_met=len(histories) >= 40 and len(trains) >= 50)


def analyse(conn, metadata=METADATA, out=ROOT / "docs/gates/G2.md", params=ROOT / "data/params.json"):
    session = json.loads(Path(metadata).read_text(encoding="utf-8"))
    if "ended_at" not in session:
        raise ValueError("pilot has not ended")
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("""SELECT purpose,key,started_at,received_at,outcome,body_zstd
            FROM obs.fetch WHERE host=%s AND id>%s AND id<=%s ORDER BY id""",
            (HOST, session["start_fetch_id"], session["end_fetch_id"]))
        result = analyse_rows(cursor.fetchall(), polls=session["polls"], active_curve=session["active_curve"])
    _write_json(params, dict(S=result["S"], minute="trunc", provisional=False))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# G2: MMT pilot measurements\n\n"
                   "Lag intervals use the previous valid poll without the actual and first flagged actual receipt; "
                   "first sightings without a preceding poll count as published but are excluded from lag fitting. "
                   "A later flagged actual proves an omitted event happened: terminated runs contribute epsilon "
                   "counterexamples; unfinished runs are right-censored at last valid poll minus the later actual. "
                   "Events with no later actual are excluded; ETAs never establish event truth. "
                   "The body uses Kaplan-Meier with interval midpoints, and the exponential tail above observed "
                   "P90 uses censored exposure. These measurements are not a source guarantee. "
                   "The tail remains soft. Minute display uses provisional truncation.\n\n"
                   "Freshness and capacity acceptance remain a user decision. Rerun G7(d) with the "
                   "epsilon CI upper bound and twice the tail scale.\n\n"
                   f"Pilot: {session['started_at']} to {session['ended_at']}.\n\n"
                   "```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--days", type=float, default=7)
    run.add_argument("--rate", type=float, default=.2)
    analysis = commands.add_parser("analyse")
    analysis.add_argument("--out", type=Path, default=ROOT / "docs/gates/G2.md")
    analysis.add_argument("--params", type=Path, default=ROOT / "data/params.json")
    args = parser.parse_args(argv)
    if args.command == "run":
        cohort_bound(args.rate)
        transport = real_transport()
        client = MmtClient(Dispatcher(HOST, args.rate, connect, transport))
        asyncio.run(run_pilot(client, days=args.days))
    else:
        with connect() as conn:
            analyse(conn, out=args.out, params=args.params)


if __name__ == "__main__":
    main()
