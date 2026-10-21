"""Fit an inactive collector-data candidate and screen its frozen future window."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import psycopg
from psycopg.types.json import Jsonb

from msynd.config import settings
from msynd.engine.cache import _predict
from msynd.engine.replay import Params, load_s_tables
from msynd.eval.baselines import b0_schedule, b1_persistence, b3_active
from msynd.eval.harness import asof_cases, score
from msynd.eval.metrics import block_bootstrap, sample_metrics, summarize
from msynd.features import FEATURE_VER
from msynd.features.build import Fold, FoldEncoders, MODEL_FEATURES, _frame_from_labels, segment_queries
from msynd.labels import IST, build_live_labels
from msynd.models.empirical import fit_empirical, provisional_models
from msynd.models.lgbm import train_lgbm
from msynd.models.registry import active, load_artifact
from msynd.ops.weekly import MIN_RUNS, MIN_SERVICE_DATES, FLOORS, _support, _role_support, _label_metadata


PHASE_LIMIT_SECONDS = 60
LOCK_KEY = 10220261021
ROLES = ('run', 'dwell', 'origin')


def _now():
    return datetime.now(timezone.utc)


@contextmanager
def _locked(dsn):
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
        acquired = conn.execute('SELECT pg_try_advisory_lock(%s)', (LOCK_KEY,)).fetchone()[0]
        yield conn if acquired else None  # Closing the session also releases its lock after a broken commit.


def _snapshot(dsn):
    conn = psycopg.connect(dsn, connect_timeout=5)
    conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    conn.read_only = True
    return conn


def _probe_ids(ids):
    ids = sorted(set(ids))
    return [ids[i] for i in np.linspace(0, len(ids)-1, min(100, len(ids)), dtype=int)]


def _extract(conn):
    asof = _now()
    started = perf_counter()
    labels = build_live_labels(conn, asof-timedelta(hours=6), asof)
    rows = _label_metadata(conn, labels.to_pylist())
    timings = {'labels': perf_counter()-started}
    support = _support(rows)
    weeks = tuple(sorted({r['start_date']-timedelta(days=r['start_date'].weekday()) for r in rows}))
    fold = Fold(weeks, ()) if weeks else None
    probe_ids = _probe_ids(r['run_id'] for r in rows)
    started = perf_counter()
    probe, _ = _frame_from_labels(conn, fold, asof, labels.filter(pc.is_in(labels['run_id'], value_set=pa.array(probe_ids, type=pa.int64()))))
    timings['feature_probe'] = perf_counter()-started
    modes = {id: set() for id in {r['run_id'] for r in rows}}
    for run, mode in conn.execute('''SELECT DISTINCT e.run_id,f.mode FROM obs.event_rev e
        JOIN obs.fetch f ON f.id=e.fetch_id WHERE e.run_id=ANY(%s) AND e.received_at<=%s''',
        (list(modes), asof)):
        modes[run].add(mode)
    provenance = dict.fromkeys(('poll_only', 'harvest_only', 'mixed', 'unknown'), 0)
    for known in modes.values():
        name = ('unknown' if not known.intersection({'poll', 'harvest'}) else
                'mixed' if len(known) >= 2 else 'poll_only' if 'poll' in known else 'harvest_only')
        provenance[name] += 1
    checks = {'runs': support['runs'] >= MIN_RUNS, 'service_dates': support['service_dates'] >= MIN_SERVICE_DATES,
              'labels_seconds': timings['labels'] <= PHASE_LIMIT_SECONDS,
              'feature_probe_seconds': timings['feature_probe'] <= PHASE_LIMIT_SECONDS}
    report = {'status': 'ready' if all(checks.values()) else 'failed', 'data_cutoff': asof.isoformat(),
              'matured_before': (asof-timedelta(hours=6)).isoformat(), 'support': support,
              'minimum_runs': MIN_RUNS, 'minimum_service_dates': MIN_SERVICE_DATES,
              'floors': dict(FLOORS), 'roles': {role: _role_support([r for r in rows if r['role'] == role], role) for role in ROLES},
              'provenance': provenance, 'probe_runs': probe_ids, 'checks': checks, 'timings': timings}
    return report, labels, rows, fold, probe


def preflight(models_dir, *, dsn=None):
    dsn = dsn or settings.database_url
    with _locked(dsn) as writer:
        if writer is None:
            return {'status': 'locked', 'error': 'shadow job already holds the advisory lock'}
        with _snapshot(dsn) as conn:
            return _extract(conn)[0]


def _params_hash(params):
    return hashlib.sha256(json.dumps({'S': params['s_tables'], 'v_max': params['v_max'], 'ess_min': params['ess_min']},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_entry(path, kind):
    return {'path': str(path), 'kind': kind, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _verify(conn, role, model, entry, probes):
    restored = load_artifact(role, entry['kind'], entry['path'])
    if role == 'origin':
        probes = probes.append_column('resched_min', pa.array([0.]*len(probes)))
    before = _predict(conn, model, probes, delay=0. if role == 'origin' else None)
    after = _predict(conn, restored, probes, delay=0. if role == 'origin' else None)
    if any((a is None) != (b is None) or
           (a is not None and (not np.isfinite(a).all() or not np.array_equal(a, b))) for a, b in zip(before, after)):
        raise ValueError(f'{role} serving parity failed')


def _register(conn, fitted, manifest, report):
    ids = {}
    last = date.fromisoformat(manifest['last_training_start_date'])+timedelta(days=1)
    for role, (kind, model) in fitted.items():
        ids[role] = conn.execute('''INSERT INTO ml.model
            (role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active)
            VALUES (%s,%s,%s,%s,daterange(%s,%s,'[)'),%s,%s,'{}',false) RETURNING id''',
            (role, kind, FEATURE_VER, datetime.fromisoformat(manifest['freeze_time']),
             date.fromisoformat(manifest['train_window'][0]), last, manifest['candidates'][role]['path'], Jsonb(model.tail))).fetchone()[0]
        manifest['candidates'][role]['id'] = ids[role]
    for role, id in ids.items():
        conn.execute('UPDATE ml.model SET metrics=%s WHERE id=%s',
                     (Jsonb(report['roles'][role] | {'shadow': manifest}), id))
    return ids


def fit(models_dir, *, dsn=None):
    dsn = dsn or settings.database_url
    with _locked(dsn) as writer:
        if writer is None:
            return {'status': 'locked', 'error': 'shadow job already holds the advisory lock'}
        with _snapshot(dsn) as conn:
            report, labels, rows, fold, probe = _extract(conn)
            if report['status'] != 'ready':
                return report
            started = perf_counter()
            asof = datetime.fromisoformat(report['data_cutoff'])
            X, y = _frame_from_labels(conn, fold, asof, labels)
            baselines = active(conn)
            sources = {id: (kind, path) for id, kind, path in conn.execute('SELECT id,kind,path FROM ml.model WHERE active')}
            source_bytes = {role: Path(sources[id][1]).read_bytes() for role, (id, model) in baselines.items() if id is not None}
            params = asdict(Params(s_tables=load_s_tables()))
            params.update(params_hash=_params_hash(params), bias_gen=0)
            # ponytail: sixteen segment parity probes; expand if serving regressions warrant broader coverage.
            probes = (probe.drop(['d', 'split']) if len(probe) else
                      segment_queries(conn, report['probe_runs'][0], asof)).slice(0, 16)
            report['timings']['extraction'] = perf_counter()-started
        started = perf_counter()
        fitted, skipped = {}, []
        for role in ROLES:
            if all(c['below_floor'] for c in report['roles'][role]['per_class'].values()):
                skipped.append(role)
                continue
            if role == 'run':
                encoders = FoldEncoders.fit(X.append_column('value', pa.array(y)).append_column(
                    'observed_running', pc.add(X['sched_run'], pa.array(y))))
                fitted[role] = ('lgbm_quantile', train_lgbm(X.select(MODEL_FEATURES), y, role, encoders))
            else:
                fitted[role] = ('empirical', fit_empirical([r for r in rows if r['role'] == role], role))
        report['timings']['training'] = perf_counter()-started
        directory = Path(models_dir).resolve() / f'shadow-{uuid4().hex}'
        committing = False
        try:
            started = perf_counter()
            directory.mkdir(parents=True)
            baseline_entries, candidates = {}, {}
            for role, (id, model) in baselines.items():
                kind = 'empirical' if id is None else sources[id][0]
                path = directory / f'baseline-{role}.json'
                if id is None:
                    provisional_models()[role].save(path)
                else:
                    path.write_bytes(source_bytes[role])
                entry = _file_entry(path, kind) | {'source_id': id}
                _verify(writer, role, model, entry, probes)
                baseline_entries[role] = entry
                if role in fitted:
                    kind, candidate = fitted[role]
                    path = directory / f'candidate-{role}.json'
                    candidate.save(path)
                    entry = _file_entry(path, kind) | {'id': None}
                    _verify(writer, role, candidate, entry, probes)
                else:
                    entry = {key: value for key, value in entry.items() if key != 'source_id'} | {'id': id}
                candidates[role] = entry
            frozen = _now()
            start = frozen.astimezone(IST).date()+timedelta(days=1)
            report['timings']['freeze'] = perf_counter()-started
            manifest = {'candidates': candidates, 'baselines': baseline_entries,
                        'feature_ver': FEATURE_VER, 'model_features': list(MODEL_FEATURES), 'params': params,
                        'support': report['support'], 'roles': report['roles'], 'floors': report['floors'],
                        'skipped_roles': skipped, 'data_cutoff': report['data_cutoff'], 'matured_before': report['matured_before'],
                        'freeze_time': frozen.isoformat(), 'last_training_start_date': max(r['start_date'] for r in rows).isoformat(),
                        'train_window': [min(r['start_date'] for r in rows).isoformat(),
                                         (max(r['start_date'] for r in rows)+timedelta(days=1)).isoformat()],
                        'validation_window': [start.isoformat(), (start+timedelta(days=14)).isoformat()],
                        'timings': report['timings']}
            with writer.transaction():
                ids = _register(writer, fitted, manifest, report)
                committing = True
        except BaseException as exc:
            retained = False
            if committing:
                try:
                    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as check:
                        retained = bool(check.execute('SELECT 1 FROM ml.model WHERE path=ANY(%s) LIMIT 1',
                                                     ([str(path) for path in directory.glob('*.json')],)).fetchone())
                except Exception:
                    retained = True
            if retained:
                raise RuntimeError(f'registration outcome uncertain; files retained at {directory}: {exc}') from exc
            if directory.exists():
                shutil.rmtree(directory)
            raise
        return report | {'status': 'fitted', 'model_ids': ids, 'shadow': manifest}


def _load_frozen(conn, model_id):
    row = conn.execute('SELECT role,kind,path,feature_ver,metrics FROM ml.model WHERE id=%s', (model_id,)).fetchone()
    if row is None or 'shadow' not in row[4]:
        raise ValueError('model has no shadow manifest')
    role, kind, path, feature, metrics = row
    manifest = metrics['shadow']
    if feature != FEATURE_VER or manifest['feature_ver'] != FEATURE_VER or manifest['model_features'] != list(MODEL_FEATURES):
        raise ValueError('feature version or columns differ from frozen manifest')
    candidates, baselines = manifest['candidates'], manifest['baselines']
    if set(candidates) != set(ROLES) or set(baselines) != set(ROLES):
        raise ValueError('manifest is missing a role')
    entry = candidates.get(role, {})
    if role in manifest['skipped_roles'] or (entry.get('id'), entry.get('kind'), entry.get('path')) != (model_id, kind, path):
        raise ValueError('model id is not a candidate in its manifest')
    sets = []
    for entries, id_key in ((candidates, 'id'), (baselines, 'source_id')):
        loaded = {}
        for role, entry in entries.items():
            if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'{role} artifact hash mismatch')
            model = load_artifact(role, entry['kind'], entry['path'])
            if hasattr(model, 'feature_names') and tuple(model.feature_names) != MODEL_FEATURES:
                raise ValueError('artifact feature columns differ from frozen manifest')
            loaded[role] = (entry[id_key], model)
        sets.append(loaded)
    frozen_params = manifest['params']
    if frozen_params['bias_gen'] != 0 or frozen_params['params_hash'] != _params_hash(frozen_params):
        raise ValueError('frozen parameter hash or bias differs')
    params = Params(**{key: frozen_params[key] for key in ('n', 'k', 'v_max', 'ess_min', 's_tables')})
    return manifest, *sets, params


def _case_support(records):
    return {'cases': len(records), 'runs': len({r['key'][0] for r in records}),
            'service_dates': len({r['service_date'] for r in records})}


def _gain_ci(results):
    candidate = results['candidate']
    if _case_support(candidate)['service_dates'] < 7:
        return None
    gains = {}
    for name, rows in results.items():
        if name == 'candidate':
            continue
        baseline = {r['key']: r for r in rows}
        by_date = defaultdict(list)
        for row in candidate:
            by_date[row['service_date']].append(baseline[row['key']]['crps']-row['crps'])
        gains[name] = asdict(block_bootstrap(by_date))
    return gains


def _summaries(records):
    dimensions = ('horizon', 'class', 'zone', 'delay_band', 'disruption_day')
    groups = {dim: defaultdict(list) for dim in dimensions}
    for row in records:
        for dim in dimensions:
            groups[dim][row[dim]].append(row)
    return {'overall': summarize(records), 'breakdowns': {
        dim: {key: summarize(rows) for key, rows in values.items()} for dim, values in groups.items()}}


def score_shadow(model_id, until=None, *, dsn=None):
    with _snapshot(dsn or settings.database_url) as conn:
        manifest, candidates, baselines, params = _load_frozen(conn, model_id)
        now = _now()
        start, end = map(date.fromisoformat, manifest['validation_window'])
        latest = min(now.astimezone(IST).date(), end)
        if until is not None and (until > latest or until < start):
            raise ValueError('--until must be within the available validation window')
        until = latest if until is None else until
        issue_start, issue_end = (datetime.combine(day, time(), IST) for day in (start, until))
        cases = [case for case in asof_cases(conn, issue_start, issue_end)
                 if start <= case.service_date < until and issue_start <= case.issued_at < issue_end] if until > start else []
        results = score(cases, {'candidate': b3_active(conn, models=candidates, params=params),
                                'baseline': b3_active(conn, models=baselines, params=params),
                                'B1': b1_persistence, 'B0': b0_schedule})
        records = {name: scores.records for name, scores in results.items()}
        primary = records['candidate']
        b2_cases = {case.key: case for case in cases if case.b2 is not None}
        b2 = {name: [r for r in rows if r['key'] in b2_cases] for name, rows in records.items()}
        b2['B2'] = [row | sample_metrics(b2_cases[row['key']].b2, row['actual']) for row in b2['candidate']]
        pending = conn.execute('''SELECT count(*) FROM obs.run WHERE start_date >= %s AND start_date < %s
            AND (lifecycle NOT IN ('completed','expired_unknown','cancelled') OR updated_at > %s)''',
            (start, end, now-timedelta(hours=6))).fetchone()[0]
        return {'status': 'awaiting_data' if not primary else 'complete' if until == end and not pending else 'partial',
                'model_id': model_id, 'validation_window': manifest['validation_window'], 'until': until.isoformat(),
                'support': _case_support(primary), 'scores': {name: _summaries(rows) for name, rows in records.items()},
                'crps_gain_ci': _gain_ci(records), 'b2': {'support': _case_support(b2['candidate']),
                'scores': {name: _summaries(rows) for name, rows in b2.items()}, 'crps_gain_ci': _gain_ci(b2)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('preflight', 'fit'):
        commands.add_parser(command).add_argument('--models-dir', required=True, type=Path)
    scoring = commands.add_parser('score')
    scoring.add_argument('--model-id', required=True, type=int)
    scoring.add_argument('--until')
    args = parser.parse_args(argv)
    try:
        report = (score_shadow(args.model_id, date.fromisoformat(args.until) if args.until else None)
                  if args.command == 'score' else {'preflight': preflight, 'fit': fit}[args.command](args.models_dir))
    except Exception as exc:
        report = {'status': 'error', 'error': str(exc)}
    print(json.dumps(report, default=lambda value: value.item(), allow_nan=False))
    return int(report['status'] in {'failed', 'locked', 'error'})


if __name__ == '__main__':
    raise SystemExit(main())
