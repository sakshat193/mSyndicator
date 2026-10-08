"""Local-only RSTGCN history import and gated cold-start empirical fitting."""

import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path
from uuid import uuid4
import zipfile

import numpy as np
from psycopg.types.json import Jsonb

from msynd.db import connect
from msynd.features import FEATURE_VER
from msynd.models.empirical import fit_empirical, provisional_models
from msynd.models import registry

EXPECTED_SHA256 = json.loads(
    (Path(__file__).resolve().parents[2] / 'data/static_sources.json').read_text(encoding='utf-8')
)['rstgcn']['sha256']
PREFIX = 'Indian-Railway-Network-and-Delays/'


def import_zip(conn, path):
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != EXPECTED_SHA256:
            raise ValueError('RSTGCN sha256 mismatch')
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive, conn.transaction():
            zones = json.loads(archive.read(PREFIX + 'stations_zones_mapping.json'))
            if not isinstance(zones, dict) or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in zones.items()
            ):
                raise ValueError('invalid station zone mapping')
            conn.execute('CREATE TEMP TABLE rstgcn_import (LIKE obs.history_daily INCLUDING DEFAULTS) ON COMMIT DROP')
            seqs, occs = Counter(), Counter()
            with archive.open(PREFIX + 'train_routes_delays_Sep2024.csv') as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8'))
                if not {'train', 'date', 'station', 'arr_delay'} <= set(reader.fieldnames or []):
                    raise ValueError('missing RSTGCN CSV columns')
                with conn.cursor().copy('COPY rstgcn_import (train,run_date,station,occ,value_min,seq) FROM STDIN') as copy:
                    for row in reader:
                        train, station = row['train'], row['station']
                        day, value = date.fromisoformat(row['date']), float(row['arr_delay'])
                        if not train or not station or not math.isfinite(value) or not 0 <= value <= 32767 or not value.is_integer():
                            raise ValueError('invalid history row')
                        run = (train, day)
                        seqs[run] += 1
                        occs[run, station] += 1
                        copy.write_row((train, day, station, occs[run, station], int(value), seqs[run]))
            conn.execute('''INSERT INTO obs.history_daily
                (train,run_date,station,occ,value_min,seq,fetch_id)
                SELECT train,run_date,station,occ,value_min,seq,NULL FROM rstgcn_import
                ON CONFLICT (train,run_date,station,occ) DO UPDATE
                SET value_min=excluded.value_min,seq=excluded.seq,fetch_id=NULL''')
            with conn.cursor() as cursor:
                cursor.executemany('UPDATE ref.station SET zone=%s WHERE code=%s AND zone IS NULL',
                                   [(zone, station) for station, zone in zones.items()])
            conn.execute('DROP TABLE rstgcn_import')


def build_labels(conn):
    rows = conn.execute('''SELECT h.train,h.run_date,h.seq,h.value_min,COALESCE(t.class,'UNKNOWN')
        FROM obs.history_daily h LEFT JOIN ref.train t ON t.number=h.train
        WHERE h.run_date >= '2024-09-01' AND h.run_date < '2024-10-01' AND h.seq IS NOT NULL
        ORDER BY h.train,h.run_date,h.seq''')
    labels, previous = [], None
    for train, day, seq, value, name in rows:
        run = (train, day)
        common = {'class': name, 'split': 'train' if day.day <= 23 else 'valid'}
        if previous is None or previous[0] != run:
            if value is not None:
                labels.append(dict(common, role='origin', value=value))
        elif seq == previous[1] + 1 and value is not None and previous[2] is not None:
            # ponytail: composite includes dwell; replace with live R after four weeks (Task 38).
            labels.append(dict(common, role='run', value=value - previous[2]))
        previous = (run, seq, value)
    return labels


def validation_log_score(model, labels):
    """Mean log density of signed unit bins and their exponential tails."""
    scores = []
    by_class = {}
    for row in labels:
        if row['role'] != model.role or row['split'] != 'valid':
            continue
        name, value = row['class'], row['value']
        if name not in by_class:
            masses, lo, hi = model.predict_distribution({'class': [name] * 16})
            by_class[name] = masses[0, 0], lo[0, 0], hi[0, 0]
        masses, lo, hi = by_class[name]
        if value < -60:
            score = math.log(masses[0]) - math.log(lo) + (value + 60) / lo
        elif value >= 240:
            score = math.log(masses[-1]) - math.log(hi) - (value - 240) / hi
        else:
            score = math.log(masses[math.floor(value) + 61])
        scores.append(score)
    return float(np.mean(scores)) if scores else None


def fit(conn, models_dir):
    labels = build_labels(conn)
    directory = Path(models_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    priors = provisional_models()
    paths = []
    try:
        with conn.transaction():
            for role in ('origin', 'run'):
                model = fit_empirical(labels, role)
                model.cold_start = True
                candidate = validation_log_score(model, labels)
                provisional = validation_log_score(priors[role], labels)
                path = directory / f'rstgcn-{role}-{uuid4().hex}.json'
                model.save(path)
                paths.append(path)
                model_id = conn.execute('''INSERT INTO ml.model
                    (role,kind,feature_ver,trained_at,train_window,path,tail,metrics)
                    VALUES (%s,'empirical',%s,now(),'[2024-09-01,2024-09-24)',%s,%s,%s) RETURNING id''',
                    (role, FEATURE_VER, str(path), Jsonb(model.tail), Jsonb({
                        'candidate_log_score': candidate, 'provisional_log_score': provisional,
                    }))).fetchone()[0]
                if candidate is not None and candidate > provisional:
                    registry.promote(conn, model_id)
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('import').add_argument('--zip', required=True, type=Path)
    commands.add_parser('fit').add_argument('--models-dir', required=True, type=Path)
    args = parser.parse_args()
    with connect() as conn:
        if args.command == 'import':
            import_zip(conn, args.zip)
        else:
            fit(conn, args.models_dir)


if __name__ == '__main__':
    main()
