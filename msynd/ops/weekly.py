"""Fit local live empirical candidates without promoting them."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import uuid4

from psycopg.types.json import Jsonb

from msynd.db import connect
from msynd.features import FEATURE_VER
from msynd.labels import build_live_labels
from msynd.models.empirical import fit_empirical


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('fit-empirical').add_argument('--models-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    with connect() as conn:
        fit_empirical_live(conn, args.models_dir, datetime.now(timezone.utc))


if __name__ == '__main__':
    main()
