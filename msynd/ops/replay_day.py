"""Export stored status bodies and replay a recorded IST day without source traffic."""
import argparse
import base64
from contextlib import nullcontext
import csv
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
import math
import multiprocessing as mp
from pathlib import Path
from time import monotonic, sleep

import numpy as np
from psycopg import sql

from msynd.config import settings
from msynd.db import connect
from msynd.engine.__main__ import consume, _read_frame
from msynd.engine.cache import tables_for
from msynd.engine.priority import write_periodic
from msynd.engine.replay import Params, load_s_tables, replay as replay_run
from msynd.labels import IST
from msynd.models.registry import active
from msynd.ops.pilot_bridge import apply_status, open_source
from msynd.ops.report import a1

REF_TABLES = ('station', 'train', 'schedule_version', 'schedule_stop', 'route_version',
              'route_stop', 'segment', 'route_segment', 'calendar')
FIELDS = ('host', 'purpose', 'key', 'started_at', 'received_at', 'outcome', 'http_status', 'body_zstd')


def export(source, day, out):
    out = Path(out)
    (out / 'ref').mkdir(parents=True, exist_ok=True)
    start = datetime.combine(day, time(), IST)
    end = start + timedelta(days=1)
    manifest = dict(day=day.isoformat(), fetch_count=0, fetch_id_range=None, ref_rows={})
    with open_source(source) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        with (out / 'fetches.jsonl').open('w', encoding='utf-8') as file:
            with conn.cursor(name='replay_day_export') as cur:
                cur.execute('''SELECT id,host,purpose,key,started_at,received_at,outcome,http_status,body_zstd
                    FROM obs.fetch WHERE purpose='status' AND outcome='ok'
                      AND received_at >= %s AND received_at < %s ORDER BY received_at,id''', (start, end))
                for fetch_id, *row in cur:
                    row[3:5] = [value.isoformat() if value is not None else None for value in row[3:5]]
                    row[-1] = base64.b64encode(row[-1]).decode('ascii') if row[-1] is not None else None
                    file.write(json.dumps(dict(zip(FIELDS, row))) + '\n')
                    manifest['fetch_count'] += 1
                    bounds = manifest['fetch_id_range']
                    manifest['fetch_id_range'] = [fetch_id, fetch_id] if bounds is None else [
                        min(bounds[0], fetch_id), max(bounds[1], fetch_id)]
        for table in REF_TABLES:
            name = sql.Identifier('ref', table)
            with conn.cursor() as cur, (out / 'ref' / f'{table}.csv').open('wb') as file:
                with cur.copy(sql.SQL('COPY {} TO STDOUT (FORMAT csv, HEADER)').format(name)) as copy:
                    for block in copy:
                        file.write(block)
            manifest['ref_rows'][table] = conn.execute(
                sql.SQL('SELECT count(*) FROM {}').format(name)).fetchone()[0]
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def _load_ref(conn, directory):
    with conn.transaction():
        for schema, table in [('ref', name) for name in REF_TABLES] + [
                ('obs', 'run'), ('obs', 'fetch'), ('sim', 'frame')]:
            if conn.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(sql.Identifier(schema, table))).fetchone():
                raise ValueError('replay requires an empty migrated database')
        for table in REF_TABLES:
            with (directory / 'ref' / f'{table}.csv').open(encoding='utf-8', newline='') as file:
                columns = next(csv.reader(file))
                query = sql.SQL('COPY {} ({}) FROM STDIN (FORMAT csv)').format(
                    sql.Identifier('ref', table), sql.SQL(',').join(map(sql.Identifier, columns)))
                with conn.cursor() as cur, cur.copy(query) as copy:
                    while block := file.read(65536):
                        copy.write(block)
        sequences = conn.execute('''SELECT ns.nspname,t.relname,a.attname,s.oid,q.seqmin
            FROM pg_class s JOIN pg_sequence q ON q.seqrelid=s.oid
            JOIN pg_depend d ON d.objid=s.oid AND d.classid='pg_class'::regclass AND d.deptype IN ('a','i')
            JOIN pg_class t ON t.oid=d.refobjid JOIN pg_namespace ns ON ns.oid=t.relnamespace
            JOIN pg_attribute a ON a.attrelid=t.oid AND a.attnum=d.refobjsubid''').fetchall()
        for schema, table, column, sequence, minimum in sequences:
            maximum = conn.execute(sql.SQL('SELECT max({}) FROM {}').format(
                sql.Identifier(column), sql.Identifier(schema, table))).fetchone()[0]
            conn.execute('SELECT setval(%s::regclass,%s,%s)',
                         (sequence, maximum if maximum is not None else minimum, maximum is not None))


def _fetches(file):
    for line in file:
        row = json.loads(line)
        for name in ('started_at', 'received_at'):
            if row[name] is not None:
                row[name] = datetime.fromisoformat(row[name])
        if row['body_zstd'] is not None:
            row['body_zstd'] = base64.b64decode(row['body_zstd'], validate=True)
        yield tuple(row[name] for name in FIELDS)


def replay(directory, dsn, *, speed=60, params=None, workers=4) -> dict:
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('speed must be finite and positive')
    if not isinstance(workers, int) or workers < 0:
        raise ValueError('workers must be a nonnegative integer')
    directory = Path(directory)
    day = date.fromisoformat(json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))['day'])
    start = datetime.combine(day, time(), IST)
    params = params if params is not None else Params(
        n=settings.n_particles, k=settings.k_paths, s_tables=load_s_tables())
    if not params.s_tables:
        params = replace(params, s_tables=load_s_tables())
    counts = dict(applied=0, invalid=0, no_version=0)
    cache, pending, framed, latencies = {}, {}, set(), []

    def model_source(conn, ev):
        return tables_for(conn, ev, active(conn), ev.versions['bias_gen'])

    with connect(dsn) as conn:
        _load_ref(conn, directory)
        pool_context = mp.get_context('spawn').Pool(workers) if workers else nullcontext(None)
        with pool_context as pool, (directory / 'fetches.jsonl').open(encoding='utf-8') as file:
            rows = _fetches(file)
            row = next(rows, None)

            def step():
                gen = consume(conn, replay_run, model_source, params, pool=pool, cache=cache)
                committed = monotonic()
                if gen is not None:
                    delta = _read_frame(conn, gen)
                    ids = set(delta['runs']) | set(delta['tombstones'])
                    framed.update(ids)  # a run completed within one fetch only ever appears as a tombstone
                    for run_id in ids:
                        latencies.extend(committed - receipt for receipt in pending.pop(run_id, ()))
                return gen

            if row is not None:
                first = last = row[4]
                wall_start = monotonic()
                minute, hour = first + timedelta(minutes=1), first + timedelta(hours=1)
                # ponytail: replay covers only the polled cohort; at 1.8 s per run (G8-local), a whole-network replay at 60× cannot keep up on the PC; add hourly create_runs for a network-wide day on the VM.
                while row is not None:
                    sim = first + timedelta(seconds=speed * (monotonic() - wall_start))
                    if row[4] <= sim and row[4] <= minute:
                        with conn.transaction():
                            before = dict(conn.execute('SELECT run_id,ver FROM obs.dirty_run'))
                            outcome = apply_status(conn, row)
                            changed = [run_id for run_id, ver in conn.execute('SELECT run_id,ver FROM obs.dirty_run')
                                       if before.get(run_id) != ver]
                        committed = monotonic()
                        for run_id in changed:
                            pending.setdefault(run_id, []).append(committed)
                        counts[outcome] += 1
                        last = row[4]
                        row = next(rows, None)
                    elif minute <= sim and minute < row[4]:
                        with conn.transaction():
                            write_periodic(conn, cache, minute, polls=True, metrics=True, issue=minute >= hour,
                                           S=params.s_tables.get(settings.live_source))
                        if minute >= hour:
                            hour += timedelta(hours=1)
                        minute += timedelta(minutes=1)
                    elif step() is None:
                        wait = (min(row[4], minute) - sim).total_seconds() / speed
                        sleep(min(1, max(0, wait)))
                while step() is not None:
                    pass
                with conn.transaction():
                    write_periodic(conn, cache, last, polls=True, metrics=True, issue=last >= hour,
                                   S=params.s_tables.get(settings.live_source))
        with conn.transaction():
            # A1's sense: a run is discovered once it has an actual event.
            discovered = {row[0] for row in conn.execute('SELECT DISTINCT run_id FROM obs.event_rev')}
            result = counts | {
                'invalid_share': counts['invalid'] / sum(counts.values()) if sum(counts.values()) else 0.0,
                'a1': a1(conn, start, start + timedelta(days=1)),
                'a6': dict(count=len(latencies),
                           p50=float(np.quantile(latencies, .5)) if latencies else None,
                           p95=float(np.quantile(latencies, .95)) if latencies else None,
                           max=max(latencies) if latencies else None),
                'replay_errors': int(conn.execute("SELECT coalesce(sum(value),0) FROM sim.metric WHERE name='replay_error'").fetchone()[0]),
                'runs': conn.execute('SELECT count(*) FROM obs.run').fetchone()[0],
                'runs_discovered': len(discovered),
                'runs_framed': len(framed),
                'runs_unframed': len(discovered - framed),
                'forecast_issues': conn.execute('SELECT count(*) FROM sim.forecast_issue').fetchone()[0],
            }
    (directory / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('export')
    command.add_argument('--source', required=True)
    command.add_argument('--day', required=True, type=date.fromisoformat)
    command.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    export(args.source, args.day, args.out)


if __name__ == '__main__':
    main()
