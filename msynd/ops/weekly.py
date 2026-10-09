"""Fit local live empirical candidates without promoting them."""

import argparse
from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from psycopg.types.json import Jsonb

from msynd.db import connect
from msynd.eval.baselines import b3_active
from msynd.eval.harness import asof_cases, promotion_ok, score
from msynd.features import FEATURE_VER
from msynd.features.build import Fold, FoldEncoders, MODEL_FEATURES, training_frame
from msynd.labels import IST, build_live_labels
from msynd.models.empirical import fit_empirical
from msynd.models.lgbm import train_lgbm
from msynd.models.registry import active


MIN_RUNS = 1000
MIN_SERVICE_DATES = 7
FLOORS = {'run': 100, 'dwell': 100, 'dwell_bucket': 30, 'origin': 200}


def _support(rows):
    return {
        'rows': len(rows),
        'runs': len({r['run_id'] for r in rows}),
        'trains': len({r['train'] for r in rows}),
        'service_dates': len({r['start_date'] for r in rows}),
    }


def fit_empirical_live(conn, models_dir, now) -> dict[str, int]:
    directory = Path(models_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths, ids, reports = [], {}, {}
    try:
        with conn.transaction():
            labels = build_live_labels(conn, now - timedelta(hours=6)).to_pylist()
            metadata = {
                (run, idx): (train, start, name, dwell)
                for run, idx, train, start, name, dwell in conn.execute("""
                    SELECT r.id,s.idx,r.train,r.start_date,t.class,s.dep_off-s.arr_off
                    FROM obs.run r JOIN ref.train t ON t.number=r.train
                    JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
                    WHERE r.id=ANY(%s)
                """, (list({r['run_id'] for r in labels}),))
            }
            for row in labels:
                train, start, name, dwell = metadata[row['run_id'], row['idx']]
                row.update(train=train, start_date=start, **{'class': name})
                if row['role'] == 'dwell':
                    row['sched_dwell'] = dwell
            for role in ('run', 'dwell', 'origin'):
                rows = [r for r in labels if r['role'] == role]
                if not rows:
                    print(f'{role}: skipped (zero labels)')
                    continue
                model = fit_empirical(rows, role)
                model.cold_start = True
                metrics = _support(rows) | {'per_class': {
                    name: _support([r for r in rows if r['class'] == name])
                    for name in sorted({r['class'] for r in rows})
                }}
                dates = [r['start_date'] for r in rows]
                path = directory / f'live-{role}-{uuid4().hex}.json'
                model.save(path)
                paths.append(path)
                ids[role] = conn.execute("""INSERT INTO ml.model
                    (role,kind,feature_ver,trained_at,train_window,path,tail,metrics)
                    VALUES (%s,'empirical',%s,now(),daterange(%s,%s,'[]'),%s,%s,%s)
                    RETURNING id""", (role, FEATURE_VER, min(dates), max(dates), str(path),
                                      Jsonb(model.tail), Jsonb(metrics))).fetchone()[0]
                reports[role] = metrics
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    for role, metrics in reports.items():
        print(f'{role}: model_id={ids[role]} {json.dumps(metrics, sort_keys=True)}')
    return ids


def _role_support(rows, role):
    report = _support(rows) | {'per_class': {}}
    dwells = [r['sched_dwell'] for r in rows] if role == 'dwell' else []
    # Match fit_empirical's training-only quantile buckets.
    edges = np.unique(np.quantile(dwells, [0., .2, .4, .6, .8, 1.])).tolist() if dwells else []
    for name in sorted({r['class'] for r in rows}):
        members = [r for r in rows if r['class'] == name]
        support = _support(members)
        support['below_floor'] = support['runs'] < FLOORS[role]
        if role == 'dwell':
            buckets = {}
            for row in members:
                bucket = str(np.searchsorted(edges[1:-1], row['sched_dwell'], side='right'))
                buckets.setdefault(bucket, []).append(row)
            support['buckets'] = {b: _support(group) | {
                'below_floor': len({r['run_id'] for r in group}) < FLOORS['dwell_bucket']
            } for b, group in buckets.items()}
            support['below_floor'] |= any(b['below_floor'] for b in support['buckets'].values())
        report['per_class'][name] = support
    return report


def evaluate_week(conn, models_dir, now) -> dict:
    """Fit and score frozen weekly candidates without promoting them."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must include a timezone')
    today = now.astimezone(IST).date()
    validation_end = today - timedelta(days=today.weekday())
    validation_start = validation_end - timedelta(weeks=2)
    directory = Path(models_dir).resolve()
    paths = []
    try:
        with conn.transaction():
            labels = build_live_labels(conn, now - timedelta(hours=6), now).to_pylist()
            metadata = {
                (run, idx): (train, start, name, dwell)
                for run, idx, train, start, name, dwell in conn.execute("""
                    SELECT r.id,s.idx,r.train,r.start_date,t.class,s.dep_off-s.arr_off
                    FROM obs.run r JOIN ref.train t ON t.number=r.train
                    JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
                    WHERE r.id=ANY(%s)
                """, (list({r['run_id'] for r in labels}),))
            }
            training = []
            for row in labels:
                train, start, name, dwell = metadata[row['run_id'], row['idx']]
                if start >= validation_start:
                    continue
                row.update(train=train, start_date=start, **{'class': name})
                if row['role'] == 'dwell':
                    row['sched_dwell'] = dwell
                training.append(row)
            weeks = tuple(sorted({r['start_date'] - timedelta(days=r['start_date'].weekday())
                                  for r in training}))
            report = _support(training) | {
                'status': 'insufficient', 'asof': now.isoformat(),
                'validation_window': [validation_start.isoformat(), validation_end.isoformat()],
                'floors': dict(FLOORS), 'roles': {}, 'model_ids': {},
            }
            for role in ('run', 'dwell', 'origin'):
                report['roles'][role] = _role_support([r for r in training if r['role'] == role], role)
            if not weeks or report['runs'] < MIN_RUNS or report['service_dates'] < MIN_SERVICE_DATES:
                return report
            fold = Fold(weeks, (validation_start, validation_start + timedelta(weeks=1)))
            report['train_window'] = [weeks[0].isoformat(), validation_start.isoformat()]
            report['status'] = 'fitted'
            candidate_models = active(conn)
            fitted = {}
            for role, support in report['roles'].items():
                if all(c['below_floor'] for c in support['per_class'].values()):
                    continue
                if role == 'run':
                    X, y = training_frame(conn, fold, now)
                    mask = pc.equal(X['split'], 'train')
                    train_X = X.filter(mask)
                    train_y = y[np.asarray(mask)]
                    encoder_rows = train_X.append_column('value', pa.array(train_y)).append_column(
                        'observed_running', pc.add(train_X['sched_run'], pa.array(train_y)))
                    encoders = FoldEncoders.fit(encoder_rows)
                    model = train_lgbm(train_X.select(MODEL_FEATURES), train_y, 'run', encoders)
                    kind = 'lgbm_quantile'
                else:
                    model = fit_empirical([r for r in training if r['role'] == role], role)
                    kind = 'empirical'
                candidate_models[role] = (None, model)
                fitted[role] = (kind, model)
            start = datetime.combine(validation_start, time(), IST)
            end = min(datetime.combine(validation_end, time(), IST), now)
            result = score(asof_cases(conn, start, end), {
                'candidate': b3_active(conn, models=candidate_models),
                'active': b3_active(conn, models=active(conn)),
            })
            decision = promotion_ok(result['candidate'], result['active'], now=now)
            report['scores'] = json.loads(json.dumps(
                {name: scores.overall for name, scores in result.items()},
                default=lambda value: value.item()))
            report['decision'] = {'accepted': decision.accepted, 'reasons': list(decision.reasons)}
            for role, (kind, model) in fitted.items():
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f'weekly-{role}-{uuid4().hex}.json'
                paths.append(path)  # Include a partial file if save raises.
                model.save(path)
                metrics = report['roles'][role] | {
                    'floors': report['floors'], 'scores': report['scores'],
                    'decision': report['decision'],
                }
                report['model_ids'][role] = conn.execute("""INSERT INTO ml.model
                    (role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active)
                    VALUES (%s,%s,%s,%s,daterange(%s,%s,'[)'),%s,%s,%s,false)
                    RETURNING id""", (role, kind, FEATURE_VER, now, weeks[0], validation_start,
                                      str(path), Jsonb(model.tail), Jsonb(metrics))).fetchone()[0]
            conn.execute("INSERT INTO sim.metric (ts,name,value) VALUES (%s,'weekly_candidate_accepted',%s)",
                         (now, int(decision.accepted)))
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('fit-empirical').add_argument('--models-dir', required=True, type=Path)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--models-dir', required=True, type=Path)
    evaluate.add_argument('--now', type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    with connect() as conn:
        if args.command == 'evaluate':
            print(json.dumps(evaluate_week(conn, args.models_dir, args.now or datetime.now(timezone.utc))))
        else:
            fit_empirical_live(conn, args.models_dir, datetime.now(timezone.utc))


if __name__ == '__main__':
    main()
