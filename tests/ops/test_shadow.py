from dataclasses import replace
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psycopg
from psycopg.types.json import Jsonb
import pytest

from msynd.engine.cache import _predict
from msynd.engine.replay import Params, load_s_tables
from msynd.features import FEATURE_VER
from msynd.features.build import MODEL_FEATURES, segment_queries
from msynd.labels import IST
from msynd.models import registry
from msynd.models.empirical import ModelArtifact, provisional_models
from tests.helpers import insert_event, insert_fetch, make_run


NOW = datetime(2026, 10, 10, 12, tzinfo=IST)


@pytest.fixture
def shadow_data(db_url, tmp_path):
    trains, runs, versions, routes = [], [], [], []
    stations = ['SH_A', 'SH_B', 'SH_C']

    def add(start, mode='poll', lifecycle='completed', train=None):
        train = train or str(82000+len(trains))
        with psycopg.connect(db_url) as conn:
            run = make_run(conn, train=train, start=start,
                           halts=[(stations[0], None, 540, 0), (stations[1], 660, 665, 100),
                                  (stations[2], 1020, None, 300)])
            v, route = conn.execute('SELECT schedule_version_id,route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()
            conn.execute("UPDATE obs.run SET lifecycle=%s,updated_at=%s WHERE id=%s", (lifecycle, NOW-timedelta(days=1), run))
            conn.execute("UPDATE ref.train SET class='EXP' WHERE number=%s", (train,))
            base = datetime.combine(date.fromisoformat(start), time(), IST)
            for i, (station, kind, minute) in enumerate([(stations[0], 'dep', 550), (stations[1], 'arr', 680),
                                                        (stations[1], 'dep', 690), (stations[2], 'arr', 1050)]):
                receipt = base+timedelta(minutes=minute+1)
                fetch = None
                if mode is not None:
                    fetch = insert_fetch(conn, run_id=run, mode=mode[i % len(mode)] if isinstance(mode, list) else mode,
                                         received_at=receipt)
                insert_event(conn, run, station, 1, kind, base+timedelta(minutes=minute), receipt, fetch)
            trains.append(train)
            runs.append(run)
            versions.append(v)
            routes.append(route)
            return run

    try:
        yield add, runs
    finally:
        with psycopg.connect(db_url) as conn:
            ids = [id for id, path in conn.execute('SELECT id,path FROM ml.model') if Path(path).is_relative_to(tmp_path)]
            conn.execute('DELETE FROM ml.leg_bias WHERE model_id=ANY(%s)', (ids,))
            conn.execute('DELETE FROM ml.model WHERE id=ANY(%s)', (ids,))
            for table in ('ntes_eta_rev', 'pending_rev', 'event_rev', 'fetch', 'dirty_run', 'poll_queue'):
                conn.execute(f'DELETE FROM obs.{table} WHERE run_id=ANY(%s)', (runs,))
            conn.execute('DELETE FROM obs.exception_rev WHERE train=ANY(%s)', (trains,))
            conn.execute('DELETE FROM obs.run WHERE id=ANY(%s)', (runs,))
            conn.execute('DELETE FROM ref.route_stop WHERE route_version_id=ANY(%s)', (routes,))
            conn.execute('DELETE FROM ref.route_version WHERE id=ANY(%s)', (routes,))
            conn.execute('DELETE FROM ref.schedule_stop WHERE version_id=ANY(%s)', (versions,))
            conn.execute('DELETE FROM ref.schedule_version WHERE id=ANY(%s)', (versions,))
            conn.execute('DELETE FROM ref.train WHERE number=ANY(%s)', (trains,))
            conn.execute('DELETE FROM ref.station WHERE code=ANY(%s)', (stations,))
        for path in sorted(tmp_path.rglob('*'), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()


def small_shadow(monkeypatch):
    from msynd.ops import shadow, weekly

    monkeypatch.setattr(shadow, 'MIN_RUNS', 1)
    monkeypatch.setattr(shadow, 'MIN_SERVICE_DATES', 1)
    floors = {'run': 1, 'dwell': 1, 'dwell_bucket': 1, 'origin': 1}
    monkeypatch.setattr(shadow, 'FLOORS', floors)
    monkeypatch.setattr(weekly, 'FLOORS', floors)
    monkeypatch.setattr(shadow, '_now', lambda: NOW)
    monkeypatch.setattr(shadow, 'Params', lambda **kw: Params(**({'n': 16, 'k': 16} | kw)))
    return shadow


def baseline(conn, tmp_path, role='run'):
    model = provisional_models()[role]
    path = tmp_path / f'active-{role}.json'
    model.save(path)
    id = conn.execute("""INSERT INTO ml.model
        (role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active)
        VALUES (%s,'empirical',%s,now(),'[2026-09-01,2026-10-01)',%s,'{}','{}',true) RETURNING id""",
        (role, FEATURE_VER, str(path))).fetchone()[0]
    return id, path


def manifest(conn, id):
    return conn.execute("SELECT metrics->'shadow' FROM ml.model WHERE id=%s", (id,)).fetchone()[0]


def update_manifest(conn, id, change):
    value = manifest(conn, id)
    change(value)
    conn.execute("UPDATE ml.model SET metrics=jsonb_set(metrics,'{shadow}',%s) WHERE id=%s", (Jsonb(value), id))


def test_shadow_reuses_weekly_contract():
    from msynd.ops import shadow, weekly

    for name in ('MIN_RUNS', 'MIN_SERVICE_DATES', 'FLOORS', '_support', '_role_support'):
        assert getattr(shadow, name) is getattr(weekly, name)
    assert shadow.PHASE_LIMIT_SECONDS == 60


@pytest.mark.db
def test_preflight_thresholds_no_writes(db_url, shadow_data, tmp_path, monkeypatch, capsys):
    from msynd.ops import shadow

    add, runs = shadow_data
    add('2026-10-07')
    monkeypatch.setattr(shadow, '_now', lambda: NOW)
    monkeypatch.setattr(shadow, 'train_lgbm', lambda *a: pytest.fail('preflight must not fit'))
    with psycopg.connect(db_url) as conn:
        before = conn.execute('SELECT count(*) FROM ml.model').fetchone()[0]
    report = shadow.preflight(tmp_path / 'absent', dsn=db_url)
    assert report['status'] == 'failed'
    assert report['support']['runs'] == 1
    assert not report['checks']['runs'] and not report['checks']['service_dates']
    assert report['floors'] == shadow.FLOORS
    assert set(report['timings']) >= {'labels', 'feature_probe'}
    assert not (tmp_path / 'absent').exists()
    with psycopg.connect(db_url) as conn:
        assert conn.execute('SELECT count(*) FROM ml.model').fetchone()[0] == before
    monkeypatch.setattr(shadow, 'settings', SimpleNamespace(database_url=db_url))
    assert shadow.main(['preflight', '--models-dir', str(tmp_path / 'absent')]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['status'] == 'failed'


@pytest.mark.db
def test_preflight_provenance_and_deterministic_probe(db_url, shadow_data, tmp_path, monkeypatch):
    shadow = small_shadow(monkeypatch)
    add, runs = shadow_data
    for i, mode in enumerate(['poll', 'harvest', ['poll', 'harvest'], None, 'probe', ['harvest', 'backfill']]):
        add(f'2026-10-0{i+1}', mode)
    original = shadow._frame_from_labels
    seen = []

    def probe(conn, fold, asof, labels):
        assert conn.execute('SHOW transaction_isolation').fetchone()[0] == 'repeatable read'
        assert conn.execute('SHOW transaction_read_only').fetchone()[0] == 'on'
        seen.extend(sorted(set(labels['run_id'].to_pylist())))
        return original(conn, fold, asof, labels)

    monkeypatch.setattr(shadow, '_frame_from_labels', probe)
    report = shadow.preflight(tmp_path / 'absent', dsn=db_url)
    assert report['status'] == 'ready'
    assert report['provenance'] == {'poll_only': 1, 'harvest_only': 1, 'mixed': 2, 'unknown': 2}
    assert seen == sorted(runs)
    assert shadow._probe_ids(list(range(201))) == np.linspace(0, 200, 100, dtype=int).tolist()
    assert not (tmp_path / 'absent').exists()


@pytest.mark.db
@pytest.mark.parametrize('command', ['preflight', 'fit'])
def test_phase_breach_stops_job(db_url, shadow_data, tmp_path, monkeypatch, command):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    monkeypatch.setattr(shadow, 'PHASE_LIMIT_SECONDS', -1)
    monkeypatch.setattr(shadow, 'train_lgbm', lambda *a: pytest.fail('timing breach must stop fit'))
    report = getattr(shadow, command)(tmp_path / 'absent', dsn=db_url)
    assert report['status'] == 'failed'
    assert not report['checks']['labels_seconds'] and not report['checks']['feature_probe_seconds']
    assert not (tmp_path / 'absent').exists()


@pytest.mark.db
def test_lock_refuses_another_job(db_url, shadow_data, tmp_path, monkeypatch, capsys):
    shadow = small_shadow(monkeypatch)
    with psycopg.connect(db_url, autocommit=True) as conn:
        assert conn.execute('SELECT pg_try_advisory_lock(%s)', (shadow.LOCK_KEY,)).fetchone()[0]
        for command in ('preflight', 'fit'):
            assert getattr(shadow, command)(tmp_path / 'absent', dsn=db_url)['status'] == 'locked'
        monkeypatch.setattr(shadow, 'settings', SimpleNamespace(database_url=db_url))
        assert shadow.main(['fit', '--models-dir', str(tmp_path / 'absent')]) == 1
        assert json.loads(capsys.readouterr().out.splitlines()[-1])['status'] == 'locked'
    assert not (tmp_path / 'absent').exists()


@pytest.mark.db
def test_fit_freezes_files_and_manifest(db_url, shadow_data, tmp_path, monkeypatch):
    shadow = small_shadow(monkeypatch)
    run = shadow_data[0]('2026-10-07')
    with psycopg.connect(db_url) as conn:
        source_id, source_path = baseline(conn, tmp_path)
    source_bytes = source_path.read_bytes()
    original_labels, original_train = shadow.build_live_labels, shadow.train_lgbm
    calls = []

    def labels(*a, **kw):
        calls.append(1)
        return original_labels(*a, **kw)

    def train(X, y, role, encoders):
        with psycopg.connect(db_url) as conn:
            assert conn.execute("SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND state='idle in transaction'").fetchone()[0] == 0
            assert not conn.execute('SELECT pg_try_advisory_lock(%s)', (shadow.LOCK_KEY,)).fetchone()[0]
        assert tuple(X.column_names) == MODEL_FEATURES
        assert set(encoders.train_means) == {'82000'}
        return original_train(X, y, role, encoders)

    monkeypatch.setattr(shadow, 'build_live_labels', labels)
    monkeypatch.setattr(shadow, 'train_lgbm', train)
    report = shadow.fit(tmp_path / 'models', dsn=db_url)
    assert calls == [1]
    assert report['status'] == 'fitted' and set(report['model_ids']) == {'run', 'dwell', 'origin'}
    with psycopg.connect(db_url) as conn:
        assert conn.execute('SELECT id FROM ml.model WHERE active').fetchall() == [(source_id,)]
        frozen = manifest(conn, report['model_ids']['run'])
        assert frozen == report['shadow']
        assert frozen['feature_ver'] == FEATURE_VER and frozen['model_features'] == list(MODEL_FEATURES)
        assert frozen['validation_window'] == ['2026-10-11', '2026-10-25']
        assert frozen['last_training_start_date'] == '2026-10-07'
        assert frozen['data_cutoff'] == NOW.isoformat()
        assert frozen['matured_before'] == (NOW-timedelta(hours=6)).isoformat()
        assert frozen['freeze_time'] == NOW.isoformat()
        assert frozen['params']['bias_gen'] == 0
        assert frozen['params']['n'] == frozen['params']['k'] == 16
        assert frozen['params']['s_tables'] == load_s_tables()
        for id, role, metrics, active in conn.execute('SELECT id,role,metrics,active FROM ml.model WHERE NOT active'):
            assert frozen['candidates'][role]['id'] == id
            assert metrics == report['roles'][role] | {'shadow': frozen}
        for role, entry in frozen['baselines'].items():
            path = Path(entry['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
            assert entry['source_id'] == (source_id if role == 'run' else None)
            model = registry.load_artifact(role, entry['kind'], path)
            source = registry.load_artifact('run', 'empirical', source_path) if role == 'run' else provisional_models()[role]
            for a, b in zip(_predict(conn, model, segment_queries(conn, run, NOW)),
                            _predict(conn, source, segment_queries(conn, run, NOW))):
                np.testing.assert_array_equal(a, b)
        assert Path(frozen['baselines']['run']['path']).read_bytes() == source_bytes
        candidate = registry.load_artifact('run', 'lgbm_quantile', frozen['candidates']['run']['path'])
        assert candidate.encoders.train_means.keys() == {'82000'}
    assert len(list((tmp_path / 'models').glob('shadow-*'))) == 1


@pytest.mark.db
def test_below_floor_roles_skipped(db_url, shadow_data, tmp_path, monkeypatch):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    monkeypatch.setitem(shadow.FLOORS, 'run', 100)
    monkeypatch.setitem(shadow.FLOORS, 'dwell', 100)
    monkeypatch.setattr(shadow, 'train_lgbm', lambda *a: pytest.fail('below floor'))
    report = shadow.fit(tmp_path / 'models', dsn=db_url)
    assert set(report['model_ids']) == {'origin'}
    assert report['shadow']['skipped_roles'] == ['run', 'dwell']
    assert set(report['shadow']['candidates']) == {'run', 'dwell', 'origin'}
    assert report['shadow']['candidates']['run']['id'] is None


@pytest.mark.db
@pytest.mark.parametrize('failure', ['save', 'parity', 'register'])
def test_failure_before_commit_removes_rows_and_directory(db_url, shadow_data, tmp_path, monkeypatch, failure):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    if failure == 'save':
        def fail(self, path):
            Path(path).write_text('partial')
            raise RuntimeError('injected save')
        monkeypatch.setattr(ModelArtifact, 'save', fail)
    elif failure == 'parity':
        monkeypatch.setattr(shadow, '_predict', lambda *a, **kw: (np.array([np.nan]), None, None))
    else:
        monkeypatch.setattr(shadow, '_register', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('injected register')))
    with pytest.raises((RuntimeError, ValueError), match='injected|parity'):
        shadow.fit(tmp_path / 'models', dsn=db_url)
    assert not list((tmp_path / 'models').glob('shadow-*'))
    with psycopg.connect(db_url) as conn:
        assert conn.execute('SELECT count(*) FROM ml.model').fetchone()[0] == 0


@pytest.mark.db
def test_score_future_window_and_b2_reuses_records(db_url, shadow_data, tmp_path, monkeypatch):
    shadow = small_shadow(monkeypatch)
    add, runs = shadow_data
    training = add('2026-10-07')
    report = shadow.fit(tmp_path / 'models', dsn=db_url)
    id = report['model_ids']['run']
    monkeypatch.setattr(shadow, '_now', lambda: datetime(2026, 10, 26, 12, tzinfo=IST))
    empty = shadow.score_shadow(id, dsn=db_url)
    assert empty['status'] == 'awaiting_data' and empty['support']['cases'] == 0
    future = add('2026-10-11')
    add('2026-10-25')  # Outside the exclusive window end.
    with psycopg.connect(db_url) as conn:
        baseline(conn, tmp_path)  # New active row must never be loaded.
        base = datetime(2026, 10, 11, tzinfo=IST)
        conn.execute("""INSERT INTO obs.ntes_eta_rev(run_id,station,occ,kind,expected_ts,received_at)
            VALUES (%s,'SH_B',1,'arr',%s,%s)""", (future, base+timedelta(minutes=680), base+timedelta(minutes=570)))
    original = shadow.b3_active
    calls = []

    def frozen(conn, *, models, params):
        assert models['run'][0] != id or models['run'][1].encoders
        f = original(conn, models=models, params=params)
        def predict(case):
            assert case.run_id == future and case.service_date == date(2026, 10, 11)
            assert base <= case.issued_at < base+timedelta(days=1)
            calls.append(case.key)
            return f(case)
        return predict

    monkeypatch.setattr(shadow, 'active', lambda *a: pytest.fail('score must not read active'))
    monkeypatch.setattr(shadow, 'b3_active', frozen)
    partial = shadow.score_shadow(id, until=date(2026, 10, 12), dsn=db_url)
    assert partial['status'] == 'partial'
    assert partial['support']['cases'] > partial['b2']['support']['cases'] > 0
    assert len(calls) == 2*partial['support']['cases']
    assert set(partial['scores']) == {'candidate', 'baseline', 'B1', 'B0'}
    assert all(v['overall']['n'] == partial['support']['cases'] for v in partial['scores'].values())
    assert partial['crps_gain_ci'] is None
    complete = shadow.score_shadow(id, dsn=db_url)
    assert complete['status'] == 'complete'
    with psycopg.connect(db_url) as conn:
        conn.execute('UPDATE obs.run SET updated_at=%s WHERE id=%s', (shadow._now(), future))
    assert shadow.score_shadow(id, dsn=db_url)['status'] == 'partial'


@pytest.mark.db
@pytest.mark.parametrize('days', [6, 7])
def test_ci_requires_seven_represented_dates(db_url, shadow_data, tmp_path, monkeypatch, days):
    from tests.eval.test_harness import case

    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    fitted = shadow.fit(tmp_path / 'models', dsn=db_url)
    start = datetime(2026, 10, 11, tzinfo=IST)
    cases = [replace(case(i), service_date=(start+timedelta(days=i)).date(), issued_at=start+timedelta(days=i)) for i in range(days)]
    monkeypatch.setattr(shadow, '_now', lambda: start+timedelta(days=15))
    monkeypatch.setattr(shadow, 'asof_cases', lambda *a: iter(cases))
    monkeypatch.setattr(shadow, 'b3_active', lambda *a, **kw: lambda c: [c.actual])
    report = shadow.score_shadow(fitted['model_ids']['run'], dsn=db_url)
    assert (report['crps_gain_ci'] is not None) == (days == 7)
    if days == 7:
        assert set(report['crps_gain_ci']) == {'baseline', 'B1', 'B0'}


@pytest.mark.db
@pytest.mark.parametrize('rejection', ['hash', 'foreign', 'feature_ver', 'features', 'missing_role', 'params_hash', 'until'])
def test_score_rejections_exit_with_json(db_url, shadow_data, tmp_path, monkeypatch, capsys, rejection):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    fitted = shadow.fit(tmp_path / 'models', dsn=db_url)
    id = fitted['model_ids']['run']
    with psycopg.connect(db_url) as conn:
        if rejection == 'hash':
            Path(fitted['shadow']['baselines']['origin']['path']).write_text('tampered')
        elif rejection == 'foreign':
            id = conn.execute("""INSERT INTO ml.model(role,kind,feature_ver,trained_at,train_window,path,tail,metrics)
                SELECT role,kind,feature_ver,trained_at,train_window,path,tail,metrics FROM ml.model WHERE id=%s RETURNING id""", (id,)).fetchone()[0]
        elif rejection == 'feature_ver':
            update_manifest(conn, id, lambda m: m.update(feature_ver=-1))
        elif rejection == 'features':
            update_manifest(conn, id, lambda m: m.update(model_features=['class']))
        elif rejection == 'missing_role':
            update_manifest(conn, id, lambda m: m['candidates'].pop('dwell'))
        elif rejection == 'params_hash':
            update_manifest(conn, id, lambda m: m['params'].update(v_max=159))
    monkeypatch.setattr(shadow, '_now', lambda: NOW+timedelta(days=30))
    monkeypatch.setattr(shadow, 'settings', SimpleNamespace(database_url=db_url))
    args = ['score', '--model-id', str(id)]
    if rejection == 'until':
        args += ['--until', '2026-10-26']
    assert shadow.main(args) == 1
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result['status'] == 'error' and result['error']


@pytest.mark.db
def test_default_score_before_window_awaits_data(db_url, shadow_data, tmp_path, monkeypatch):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    fitted = shadow.fit(tmp_path / 'models', dsn=db_url)
    monkeypatch.setattr(shadow, 'asof_cases', lambda *a: pytest.fail('window has not opened'))
    report = shadow.score_shadow(fitted['model_ids']['run'], dsn=db_url)
    assert report['status'] == 'awaiting_data' and report['until'] == '2026-10-10'
    assert report['support'] == {'cases': 0, 'runs': 0, 'service_dates': 0}
    with pytest.raises(ValueError, match='until'):
        shadow.score_shadow(fitted['model_ids']['run'], until=date(2026, 10, 12), dsn=db_url)


@pytest.mark.db
def test_freeze_time_follows_training_and_verification(db_url, shadow_data, tmp_path, monkeypatch):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    times = iter([NOW, NOW+timedelta(days=2)])
    monkeypatch.setattr(shadow, '_now', lambda: next(times))
    report = shadow.fit(tmp_path / 'models', dsn=db_url)
    assert report['shadow']['data_cutoff'] == NOW.isoformat()
    assert report['shadow']['freeze_time'] == (NOW+timedelta(days=2)).isoformat()
    assert report['shadow']['validation_window'] == ['2026-10-13', '2026-10-27']


@pytest.mark.db
@pytest.mark.parametrize('phase,ticks', [('labels', [0, 61, 61, 61]), ('feature_probe', [0, 0, 0, 61])])
def test_each_phase_can_independently_fail(db_url, shadow_data, tmp_path, monkeypatch, phase, ticks):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    clock = iter(ticks)
    monkeypatch.setattr(shadow, 'perf_counter', lambda: next(clock))
    report = shadow.preflight(tmp_path / 'absent', dsn=db_url)
    assert report['status'] == 'failed' and report['timings'][phase] == 61
    assert not report['checks'][phase+'_seconds']
    assert not (tmp_path / 'absent').exists()


@pytest.mark.db
@pytest.mark.parametrize('outcome', ['rolled_back', 'committed', 'unreachable'])
def test_commit_error_checks_fresh_connection_before_removing_files(db_url, shadow_data, tmp_path, monkeypatch, outcome):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    original_lock, original_connect = shadow._locked, psycopg.connect
    checks = []

    class Writer:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, *a, **kw):
            return self.conn.execute(*a, **kw)

        @contextmanager
        def transaction(self):
            with self.conn.transaction():
                yield
                if outcome != 'committed':
                    raise psycopg.OperationalError('lost commit acknowledgement')
            raise psycopg.OperationalError('lost commit acknowledgement')

    @contextmanager
    def locked(dsn):
        with original_lock(dsn) as conn:
            yield Writer(conn)

    def connect(*a, **kw):
        if kw.get('autocommit'):
            checks.append(1)
            if outcome == 'unreachable' and len(checks) > 1:
                raise psycopg.OperationalError('cannot confirm commit')
        return original_connect(*a, **kw)

    monkeypatch.setattr(shadow, '_locked', locked)
    monkeypatch.setattr(psycopg, 'connect', connect)
    with pytest.raises((psycopg.OperationalError, RuntimeError), match='files retained' if outcome != 'rolled_back' else 'lost commit'):
        shadow.fit(tmp_path / 'models', dsn=db_url)
    assert len(checks) == 2
    paths = list((tmp_path / 'models').glob('shadow-*'))
    assert bool(paths) == (outcome != 'rolled_back')
    with original_connect(db_url) as conn:
        assert conn.execute('SELECT count(*) FROM ml.model').fetchone()[0] == (3 if outcome == 'committed' else 0)
    if paths:
        assert len(list(paths[0].glob('*.json'))) == 6


def test_lost_lock_connection_does_not_hide_retained_files_error(monkeypatch):
    from msynd.ops import shadow

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, statement, *a):
            if 'pg_try_advisory_lock' not in statement:
                raise psycopg.OperationalError('connection is lost')
            return self

        def fetchone(self):
            return (True,)

    monkeypatch.setattr(psycopg, 'connect', lambda *a, **kw: Connection())
    with pytest.raises(RuntimeError, match='files retained'):
        with shadow._locked('unused'):
            raise RuntimeError('registration uncertain; files retained at model directory')


@pytest.mark.db
def test_origin_only_fit_has_nonempty_serving_probes(db_url, shadow_data, tmp_path, monkeypatch):
    shadow = small_shadow(monkeypatch)
    run = shadow_data[0]('2026-10-07', lifecycle='expired_unknown')
    with psycopg.connect(db_url) as conn:
        conn.execute("DELETE FROM obs.event_rev WHERE run_id=%s AND station<>'SH_A'", (run,))
    predict = shadow._predict

    def nonempty(conn, model, probes, **kw):
        assert len(probes) > 0
        return predict(conn, model, probes, **kw)

    monkeypatch.setattr(shadow, '_predict', nonempty)
    fitted = shadow.fit(tmp_path / 'models', dsn=db_url)
    assert set(fitted['model_ids']) == {'origin'}


@pytest.mark.db
def test_parity_never_opens_another_snapshot_after_cpu_training(db_url, shadow_data, tmp_path, monkeypatch):
    shadow = small_shadow(monkeypatch)
    shadow_data[0]('2026-10-07')
    snapshot, train = shadow._snapshot, shadow.train_lgbm
    trained = []

    def checked_snapshot(*a, **kw):
        assert not trained, 'parity must use the captured snapshot data'
        return snapshot(*a, **kw)

    def cpu(*a, **kw):
        trained.append(True)
        return train(*a, **kw)

    monkeypatch.setattr(shadow, '_snapshot', checked_snapshot)
    monkeypatch.setattr(shadow, 'train_lgbm', cpu)
    assert shadow.fit(tmp_path / 'models', dsn=db_url)['status'] == 'fitted'

