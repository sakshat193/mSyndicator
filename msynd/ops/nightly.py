"""Nightly leg bias, retention and backup jobs."""
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess

from msynd.db import connect
from msynd.config import settings
from msynd.dist import Eff, eff_quantile, level_weight
from msynd.engine import cache
from msynd.features.build import segment_queries
from msynd.labels import IST, build_live_labels
from msynd.models import registry

log = logging.getLogger(__name__)


def leg_bias(conn, now, models=None) -> int | None:
    with conn.transaction():
        models = registry.active(conn) if models is None else models
        model_id, artifact = models['run']
        if model_id is None:
            return None
        labels = {(row['run_id'], row['idx']): row['value'] for row in
                  build_live_labels(conn, now - timedelta(hours=6)).to_pylist()
                  if row['role'] == 'run'}
        if not labels:
            return None
        departures = conn.execute('''
            SELECT r.id,r.start_date,s.idx,s.dep_off,d.actual_ts
            FROM obs.run r JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
            JOIN obs.event d ON d.run_id=r.id AND d.station=s.station
              AND d.occ=s.occ AND d.kind='dep'
            WHERE r.id=ANY(%s) AND d.actual_ts >= %s
        ''', (list({run for run, _ in labels}), now - timedelta(days=7)))
        runs = defaultdict(dict)
        for run, start, idx, scheduled, actual in departures:
            if (run, idx) in labels:
                base = datetime.combine(start, time(), IST)
                runs[run][idx] = (actual - base).total_seconds() / 60 - scheduled
        residuals = defaultdict(list)
        for run, delays in runs.items():
            queries = segment_queries(conn, run, now)
            masses, lo, hi = cache._predict(conn, artifact, queries)
            for i, row in enumerate(queries.to_pylist()):
                idx = row['idx']
                if idx not in delays:
                    continue
                j, w = level_weight(delays[idx])
                p50 = eff_quantile(Eff(masses[i], True, lo[i].max(), hi[i].max(),
                                      j, w, 0, 1, -float('inf'), 0, 1), 0.5)
                key = (row['from_stn'], row['to_stn'], int(row['sched_dep'] % 1440 // 180))
                residuals[key].append(labels[(run, idx)] - p50)
        if not residuals:
            return None
        # Serialize generation allocation for this model, including concurrent nightly jobs.
        conn.execute('SELECT id FROM ml.model WHERE id=%s FOR UPDATE', (model_id,))
        gen = conn.execute('SELECT coalesce(max(gen),0)+1 FROM ml.leg_bias WHERE model_id=%s',
                           (model_id,)).fetchone()[0]
        for (a, b, bucket), values in residuals.items():
            n = len(values)
            conn.execute('INSERT INTO ml.leg_bias VALUES (%s,%s,%s,%s,%s,%s,%s)',
                         (model_id, gen, a, b, bucket, median(values) * n / (n + 20), n))
        conn.execute('DELETE FROM ml.leg_bias WHERE model_id=%s AND gen < %s', (model_id, gen - 1))
        return gen


def prune_bodies(conn, now, ok_hours):
    if ok_hours not in (2, 24):
        raise ValueError('ok_hours must be 2 or 24')
    conn.execute('''UPDATE obs.fetch SET body_zstd=NULL
        WHERE body_zstd IS NOT NULL AND (
          started_at < %s OR
          (outcome='ok' AND id %% 100 <> 0 AND started_at < %s))''',
                 (now - timedelta(days=14), now - timedelta(hours=ok_hours)))


def prune_rows(conn, now):
    with conn.transaction():
        conn.execute('DELETE FROM obs.fetch WHERE started_at < %s', (now - timedelta(days=180),))
        conn.execute('DELETE FROM obs.pending_rev WHERE received_at < %s', (now - timedelta(days=180),))
        conn.execute('DELETE FROM obs.ntes_eta_rev WHERE received_at < %s', (now - timedelta(days=35),))
        conn.execute('DELETE FROM sim.forecast_issue WHERE issued_at < %s', (now - timedelta(days=120),))


def disk_check(path, *, conn=None, usage=shutil.disk_usage):
    total, used, _ = usage(path)
    pct = 100 * used / total
    if conn is None:
        with connect() as metric_conn:
            metric_conn.execute("INSERT INTO sim.metric VALUES (clock_timestamp(),'disk_pct',%s)", (pct,))
    else:
        conn.execute("INSERT INTO sim.metric VALUES (clock_timestamp(),'disk_pct',%s)", (pct,))
    if pct >= 70:
        log.warning('disk usage alarm: %.1f%% on %s', pct, path)
    return pct, 2 if pct >= 80 else 24


def backup(*, database_url=None, bucket=None, directory='/data/backups', now=None,
           run=subprocess.run):
    bucket = os.environ.get('OCI_BUCKET_NAME', '') if bucket is None else bucket
    if not bucket:
        raise ValueError('OCI backup bucket is required')
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(IST).date()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    local = directory / f'local-{day}.dump'
    offsite = directory / f'offsite-{day}.dump'
    base = ['pg_dump', '--dbname', database_url or settings.database_url, '--format=custom',
            '--schema=ref', '--schema=ml', '--schema=sim',
            '--exclude-table=sim.frame', '--exclude-table=sim.metric']
    # ponytail: local dump keeps fetch bodies; sanitized fetch copy from the same snapshot if dump size matters
    try:
        for target, extra in ((local, ['--schema=obs']), (offsite, [])):
            partial = target.with_suffix('.partial')
            try:
                run(base + extra + ['--file', str(partial)], check=True)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            if target == local:
                copies = sorted(p for p in directory.glob('local-*.dump')
                                if re.fullmatch(r'local-\d{4}-\d{2}-\d{2}\.dump', p.name))
                for old in copies[:-2]:
                    old.unlink()
        oci = ['oci', 'os', 'object']
        for cadence in ('daily', 'weekly') if day.weekday() == 6 else ('daily',):
            run(oci + ['put', '--bucket-name', bucket, '--name', f'msynd/{cadence}/{day}.dump',
                       '--file', str(offsite), '--force'], check=True)
        result = run(oci + ['list', '--bucket-name', bucket, '--prefix', 'msynd/', '--all',
                            '--output', 'json'], check=True, capture_output=True, text=True)
        objects = [obj for obj in json.loads(result.stdout)['data']
                   if re.fullmatch(r'msynd/(daily|weekly)/\d{4}-\d{2}-\d{2}\.dump', obj['name'])]
        if sum(obj['size'] for obj in objects) > 15_000_000_000:
            log.warning('off-site backup size alarm: exceeds 15 GB')
        for cadence, keep in (('daily', 7), ('weekly', 4)):
            names = sorted(obj['name'] for obj in objects if obj['name'].startswith(f'msynd/{cadence}/'))
            for name in names[:-keep]:
                run(oci + ['delete', '--bucket-name', bucket, '--name', name, '--force'], check=True)
    except Exception:
        log.exception('backup failed')
        raise
    finally:
        offsite.unlink(missing_ok=True)
    return local


def main():
    now = datetime.now(timezone.utc)
    with connect() as conn:
        try:
            with conn.transaction():
                leg_bias(conn, now)
        except Exception:
            log.exception('leg bias failed')
        _, hours = disk_check('/data', conn=conn)
        prune_bodies(conn, now, hours)
        prune_rows(conn, now)
    backup(now=now)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
