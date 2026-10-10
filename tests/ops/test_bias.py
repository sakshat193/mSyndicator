from contextlib import nullcontext
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pytest

from msynd.engine import cache
from msynd.engine.__main__ import _versions
from msynd.engine.evidence import load_run_evidence
from msynd.labels import IST
from msynd.models import registry
from msynd.models.empirical import provisional_models
from msynd.ops import nightly
from tests.helpers import insert_event, make_run

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


class DelayModel:
    feature_names = ('d',)

    def predict_distribution(self, queries):
        rows = queries.to_pylist()
        masses = np.zeros((len(rows), 302))
        for i, row in enumerate(rows):
            # Uniform unit bins: P50 at d=5 is 2.5, at d=10 is 12.5.
            value = {5: 2, 10: 12}.get(row['d'], 100)
            masses[i, value + 61] = 1
        n = len(rows) // 16
        return masses.reshape(n, 16, 302), np.ones((n, 16)), np.ones((n, 16))


def saved_run_model(db, tmp_path):
    model = provisional_models()['run']
    path = tmp_path / f"run-{db.execute('SELECT count(*) FROM ml.model').fetchone()[0]}.json"
    model.save(path)
    return db.execute("""INSERT INTO ml.model
        (role,kind,feature_ver,trained_at,train_window,path,tail,metrics)
        VALUES ('run','empirical',1,now(),'[2026-09-01,2026-10-01)',%s,'{}','{}')
        RETURNING id""", (str(path),)).fetchone()[0]


def models_for(model_id):
    models = {role: (None, model) for role, model in provisional_models().items()}
    models['run'] = (model_id, DelayModel())
    return models


def segment(db, *, train='12951', start='2026-10-07', sched_dep=540,
            delay=7.5, label=20, lifecycle='completed'):
    run = make_run(db, train=train, start=start,
                   halts=[('A', None, sched_dep, 0), ('B', sched_dep + 60, None, 100)])
    db.execute('UPDATE obs.run SET lifecycle=%s WHERE id=%s', (lifecycle, run))
    base = datetime.combine(date.fromisoformat(start), time(), IST)
    dep = base + timedelta(minutes=sched_dep + delay)
    arr = dep + timedelta(minutes=60 + label)
    insert_event(db, run, 'A', 1, 'dep', dep, dep + timedelta(minutes=1))
    insert_event(db, run, 'B', 1, 'arr', arr, arr + timedelta(minutes=1))
    return run


@pytest.mark.db
def test_bias_shrinkage(db, tmp_path):
    model_id = saved_run_model(db, tmp_path)
    for train, delay, label in [('12951', 6, 20), ('12952', 7.5, 40), ('12953', 9, 80)]:
        segment(db, train=train, delay=delay, label=label)
    assert nightly.leg_bias(db, NOW, models_for(model_id)) == 1
    rows = db.execute('SELECT from_stn,to_stn,hour_bucket,bias_min,n FROM ml.leg_bias').fetchall()
    assert len(rows) == 1
    a, b, bucket, bias, n = rows[0]
    assert (a, b, bucket, n) == ('A', 'B', 3, 3)
    # P50s 4.5, 7.5, 10.5 give residuals 15.5, 32.5, 69.5.
    assert bias == pytest.approx(32.5 * 3 / 23)


@pytest.mark.db
def test_bucket_matches_engine(db, tmp_path):
    model_id = saved_run_model(db, tmp_path)
    run = segment(db, sched_dep=1980)
    models = models_for(model_id)
    gen = nightly.leg_bias(db, NOW, models)
    assert gen == 1
    bucket, bias = db.execute('SELECT hour_bucket,bias_min FROM ml.leg_bias').fetchone()
    assert bucket == 3
    assert bias == pytest.approx(12.5 / 21)
    ev = load_run_evidence(db, run, NOW)
    assert cache.tables_for(db, ev, models, gen, asof=NOW).bias.tolist() == [bias]


@pytest.mark.db
def test_window_and_maturity(db, tmp_path):
    model_id = saved_run_model(db, tmp_path)
    models = models_for(model_id)
    segment(db, start='2026-10-03', sched_dep=329, delay=0)
    segment(db, train='12952', start='2026-10-09', sched_dep=1380, delay=0, label=0)
    segment(db, train='12953', lifecycle='running')
    db.execute("INSERT INTO ml.leg_bias VALUES (%s,4,'old','pair',0,9,2)", (model_id,))
    assert nightly.leg_bias(db, NOW, models) is None
    assert db.execute('SELECT gen,bias_min,n FROM ml.leg_bias').fetchall() == [(4, 9, 2)]
    # Departure exactly seven days old and arrival exactly six hours old qualify.
    segment(db, train='12954', start='2026-10-03', sched_dep=330, delay=0, label=0)
    segment(db, train='12955', start='2026-10-09', sched_dep=1350, delay=0, label=0)
    assert nightly.leg_bias(db, NOW, models) == 5
    assert db.execute('SELECT hour_bucket,n FROM ml.leg_bias WHERE gen=5 ORDER BY hour_bucket').fetchall() == [(1, 1), (7, 1)]


@pytest.mark.db
def test_new_model_starts_zero_bias(db, tmp_path):
    run = segment(db)
    old_id = saved_run_model(db, tmp_path)
    registry.promote(db, old_id)
    db.execute("INSERT INTO ml.leg_bias VALUES (%s,3,'A','B',3,12,30)", (old_id,))
    new_id = saved_run_model(db, tmp_path)
    registry.promote(db, new_id)
    assert _versions(db)['bias_gen'] == 0
    models = registry.active(db)
    ev = load_run_evidence(db, run, NOW)
    assert cache.tables_for(db, ev, models, 0, asof=NOW).bias.tolist() == [0.]
    assert nightly.leg_bias(db, NOW) == 1
    assert db.execute('SELECT gen,bias_min,n FROM ml.leg_bias WHERE model_id=%s', (old_id,)).fetchall() == [(3, 12, 30)]
    assert db.execute('SELECT gen,n FROM ml.leg_bias WHERE model_id=%s', (new_id,)).fetchall() == [(1, 1)]
    assert _versions(db)['bias_gen'] == 1
    assert cache.tables_for(db, ev, models, 1, asof=NOW).bias[0] == pytest.approx(20 / 21)


@pytest.mark.db
def test_keeps_previous_generation(db, tmp_path):
    segment(db)
    model_id = saved_run_model(db, tmp_path)
    models = models_for(model_id)
    for gen in (1, 2, 3):
        assert nightly.leg_bias(db, NOW, models) == gen
    assert db.execute('SELECT gen FROM ml.leg_bias ORDER BY gen').fetchall() == [(2,), (3,)]


@pytest.mark.db
def test_provisional_run_model_writes_nothing(db):
    segment(db)
    assert nightly.leg_bias(db, NOW, models_for(None)) is None
    assert nightly.leg_bias(db, NOW) is None
    assert db.execute('SELECT count(*) FROM ml.leg_bias').fetchone()[0] == 0


@pytest.mark.db
def test_bias_failure_still_runs_retention_and_backup(db, tmp_path, monkeypatch, caplog):
    model_id = saved_run_model(db, tmp_path)
    calls = []

    def fail_bias(conn, now):
        calls.append('bias')
        conn.execute("INSERT INTO ml.leg_bias VALUES (%s,1,'A','B',3,12,30)", (model_id,))
        conn.execute('SELECT 1/0')

    def disk(path, *, conn):
        calls.append('disk')
        assert conn.execute('SELECT count(*) FROM ml.leg_bias').fetchone()[0] == 0
        return 10, 24

    monkeypatch.setattr(nightly, 'connect', lambda: nullcontext(db))
    monkeypatch.setattr(nightly, 'leg_bias', fail_bias, raising=False)
    monkeypatch.setattr(nightly, 'disk_check', disk)
    monkeypatch.setattr(nightly, 'prune_bodies', lambda *args: calls.append('bodies'))
    monkeypatch.setattr(nightly, 'prune_rows', lambda *args: calls.append('rows'))
    monkeypatch.setattr(nightly, 'backup', lambda **kwargs: calls.append('backup'))
    nightly.main()
    assert calls == ['bias', 'disk', 'bodies', 'rows', 'backup']
    assert any(record.exc_info and 'bias' in record.message for record in caplog.records)
