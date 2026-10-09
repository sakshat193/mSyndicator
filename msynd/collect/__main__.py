"""Collector entrypoint. Offline callers inject a MockTransport-backed MmtClient."""
import argparse
import asyncio
from datetime import datetime, timedelta
import json
import logging
import os

from msynd.collect.admission import AdmissionState, PollRun, next_job
from msynd.collect.dispatcher import BudgetExhausted, HostStopped
from msynd.collect.lifecycle import create_runs, expire_runs
from msynd.collect.mmt import HOST, MmtClient, classify
from msynd.collect.parse_mmt import parse_status
from msynd.collect.status import InvalidResponse
from msynd.collect.store import store_status
from msynd.config import settings
from msynd.db import connect
from msynd.labels import IST
from msynd.ops.nightly import disk_check, prune_bodies

log = logging.getLogger(__name__)


def load_state(conn, now, *, host_state='ok', rate=.2, T_max=None,
               t_max_factor=1.25, recovery_at=None):
    """Read engine priorities and poll attempts; the caller owns the transaction."""
    # ponytail: aggregate retained attempts once; add a run/start index if CP-3 finds admission too slow.
    rows = conn.execute('''
        SELECT r.id,r.train,r.start_date,COALESCE(f.started_at,origin.departure),q.priority
        FROM obs.run r JOIN LATERAL (
            SELECT (r.start_date::timestamp AT TIME ZONE 'Asia/Kolkata')
                   + (s.dep_off+r.resched_min)*interval '1 minute' AS departure
            FROM ref.schedule_stop s WHERE s.version_id=r.schedule_version_id
            ORDER BY s.idx LIMIT 1
        ) origin ON true LEFT JOIN (
            SELECT run_id,max(started_at) AS started_at FROM obs.fetch
            WHERE host=%s AND purpose='status' AND started_at<=%s
            GROUP BY run_id
        ) f ON f.run_id=r.id LEFT JOIN obs.poll_queue q ON q.run_id=r.id AND q.updated_at<=%s
        WHERE r.lifecycle='running' OR (r.lifecycle='scheduled' AND origin.departure<=%s)
        ORDER BY r.id
    ''', (HOST, now, now, now)).fetchall()
    return AdmissionState(tuple(PollRun(*row) for row in rows), host_state=host_state,
                          rate=rate, T_max=T_max, t_max_factor=t_max_factor,
                          recovery_at=recovery_at)


def _classify_status(purpose, response):
    outcome = classify(purpose, response)
    if outcome == 'ok':
        request = json.loads(response.request.content)
        try:
            parse_status(response.content, request['trainNumber'],
                         datetime.strptime(request['dateOfJourney'], '%d-%m-%Y').date())
        except InvalidResponse:
            return 'invalid'
    return outcome


async def run_loop(client, conn, *, disk_path='/data', T_max=None, t_max_factor=1.25, stop=None):
    d, clock = client.d, client.d.clock
    next_hour = clock.now()
    budget_logged_day = None
    try:
        # Acquire the dispatcher's host lock before lifecycle or retention writes.
        await d.start()
        d.classify = _classify_status
        while stop is None or not stop():
            now = clock.now()
            if now >= next_hour:
                with conn.transaction():
                    create_runs(conn, now)
                    expire_runs(conn, now)
                    _, ok_hours = disk_check(disk_path, conn=conn)
                    prune_bodies(conn, now, ok_hours)
                next_hour = now + timedelta(hours=1)
            recovery_at = now + timedelta(seconds=max(0, d.pause_until - clock.monotonic()))
            with conn.transaction():
                state = load_state(conn, now, host_state=d.state, rate=d.rate,
                                   T_max=T_max, t_max_factor=t_max_factor, recovery_at=recovery_at)
            job = next_job(state, now)
            if job is None:
                await clock.sleep(1)
                continue
            try:
                result = await client.train_running(job.run.train, job.run.start_date, run_id=job.run.id)
            except BudgetExhausted as exc:
                day = clock.now().astimezone(IST).date()
                if budget_logged_day != day:
                    log.warning('MMT %s budget exhausted for IST day %s', exc.mode, day)
                    budget_logged_day = day
                await clock.sleep(60)
                continue
            except HostStopped:
                # Stay alive for housekeeping; a challenge requires a manual process restart.
                log.error('MMT host stopped by a challenge; manual restart required')
                continue
            if result.outcome == 'ok':
                page = parse_status(result.body, job.run.train, job.run.start_date)
                with conn.transaction():
                    store_status(conn, job.run.id, result, page)
    finally:
        await d.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t-max-seconds', type=float)
    parser.add_argument('--t-max-factor', type=float, default=1.25)
    parser.add_argument('--disk-path', default='/data')
    parser.add_argument('--budget', action='append', metavar='MODE=N')
    args = parser.parse_args(argv)
    budgets = {}
    for value in args.budget or ['poll=5000']:
        mode, sep, amount = value.partition('=')
        try:
            amount = int(amount)
        except ValueError:
            parser.error('budget must be MODE=N with N a nonnegative integer')
        if not sep or mode not in ('poll', 'harvest', 'probe', 'backfill') or amount < 0:
            parser.error('budget requires poll|harvest|probe|backfill and N >= 0')
        budgets[mode] = amount
    if sum(budgets.values()) > 13200:
        parser.error('total daily budgets must be <= 13,200')
    if os.environ.get('MSYND_LIVE') != '1':
        raise RuntimeError('collector requires MSYND_LIVE=1 and checkpoint approval')
    # ponytail: G1/G4 are undecided and there is no etrain client; enable only after both gates.
    if os.environ.get('ETRAIN_BACKFILL', '0') == '1':
        raise RuntimeError('etrain backfill is disabled pending G1/G4')
    if settings.live_source != 'mmt_status':
        raise ValueError('collector supports only the approved mmt_status source')
    AdmissionState((), T_max=args.t_max_seconds, t_max_factor=args.t_max_factor)
    # ponytail: F remains the F_MINUTES setting (60); CP-3 decides it. MMT has no board share.
    client = MmtClient.from_env(budgets=budgets)
    with connect() as conn:
        asyncio.run(run_loop(client, conn, disk_path=args.disk_path,
                             T_max=args.t_max_seconds, t_max_factor=args.t_max_factor))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
