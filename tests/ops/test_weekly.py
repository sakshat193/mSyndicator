from datetime import date, datetime, time, timedelta, timezone
from contextlib import nullcontext
from pathlib import Path
import json

import psycopg
import pytest

from msynd.features import FEATURE_VER
from msynd.labels import IST
from msynd.models.empirical import ModelArtifact, provisional_models
from msynd.models.registry import active
from tests.helpers import insert_event, make_run

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def completed_run(db, train='12951', start='2026-10-07'):
    run = make_run(db, train=train, start=start)
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (run,))
    base = datetime.combine(date.fromisoformat(start), time(), IST)
    for station, kind, offset in [('MMCT', 'dep', 10), ('BRC', 'arr', 315),
                                  ('BRC', 'dep', 322), ('NDLS', 'arr', 960)]:
        actual = base + timedelta(minutes=offset)
        insert_event(db, run, station, 1, kind, actual, actual + timedelta(minutes=1))
    return run


@pytest.mark.db
def test_completed_run_fits_unpromoted_models_with_support(db, tmp_path):
    from msynd.ops.weekly import fit_empirical_live

    completed_run(db)
    db.execute("UPDATE ref.train SET class='SF'")
    # Another version must not supply the bound run's scheduled dwell.
    version = db.execute("""INSERT INTO ref.schedule_version
        (train,valid_from,days_of_run,source,content_hash)
        VALUES ('12951','2026-10-08',127,'gtfs','other') RETURNING id""").fetchone()[0]
    db.execute("""INSERT INTO ref.schedule_stop VALUES
        (%s,1,'BRC',1,300,399,392)""", (version,))
    ids = fit_empirical_live(db, tmp_path / 'models', NOW)
    assert set(ids) == {'run', 'dwell', 'origin'}
    rows = db.execute("""SELECT id,role,kind,feature_ver,path,tail,metrics,active,
        lower(train_window),upper(train_window) FROM ml.model""").fetchall()
    assert len(rows) == 3
    for model_id, role, kind, feature, path, tail, metrics, promoted, lo, hi in rows:
        assert ids[role] == model_id
        assert (kind, feature, promoted) == ('empirical', FEATURE_VER, False)
        assert (lo, hi) == (date(2026, 10, 7), date(2026, 10, 8))
        support = {'rows': 2 if role == 'run' else 1, 'runs': 1, 'trains': 1, 'service_dates': 1}
        assert metrics == support | {'per_class': {'SF': support}}
        model = ModelArtifact.load(path)
        assert model.role == role and model.cold_start and model.tail == tail
        assert Path(path).parent == (tmp_path / 'models').resolve()
        assert Path(path).name.startswith(f'live-{role}-')
        if role == 'dwell':
            assert model.dwell_edges == [5.0]
            assert model.pooled['SF'][7] > .99
        elif role == 'origin':
            assert model.pooled['SF'][71] > .99
        else:
            assert model.pooled['SF'][64] > .495
            assert model.pooled['SF'][66] > .495


@pytest.mark.db
def test_fit_keeps_active_models(db, tmp_path, monkeypatch):
    from msynd.models import registry
    from msynd.ops.weekly import fit_empirical_live

    completed_run(db)
    for role, model in provisional_models().items():
        path = tmp_path / f'active-{role}.json'
        model.save(path)
        db.execute("""INSERT INTO ml.model
            (role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active)
            VALUES (%s,'empirical',%s,now(),'[2026-09-01,2026-10-01)',%s,'{}','{}',true)""",
            (role, FEATURE_VER, str(path)))
    before = active(db)
    monkeypatch.setattr(registry, 'promote', lambda *a: pytest.fail('promotion forbidden'))
    fit_empirical_live(db, tmp_path / 'models', NOW)
    assert active(db) == before


@pytest.mark.db
def test_empty_roles_skipped_and_reported(db, tmp_path, capsys):
    from msynd.ops.weekly import fit_empirical_live

    run = completed_run(db)
    db.execute("DELETE FROM obs.event_rev WHERE run_id=%s AND station<>'MMCT'", (run,))
    ids = fit_empirical_live(db, tmp_path, NOW)
    assert set(ids) == {'origin'}
    output = capsys.readouterr().out
    assert 'run' in output and 'dwell' in output and 'skip' in output.lower()
    assert db.execute('SELECT role FROM ml.model').fetchall() == [('origin',)]
    assert len(list(tmp_path.glob('live-*.json'))) == 1
    db.execute('DELETE FROM obs.event_rev')
    assert fit_empirical_live(db, tmp_path / 'empty', NOW) == {}
    assert not list((tmp_path / 'empty').glob('*.json'))


@pytest.mark.db
def test_insert_failure_rolls_back_models_and_removes_files(db, tmp_path):
    from msynd.ops.weekly import fit_empirical_live

    completed_run(db)
    db.execute("""CREATE FUNCTION pg_temp.reject_dwell() RETURNS trigger LANGUAGE plpgsql AS
        $$ BEGIN RAISE EXCEPTION 'injected insert failure'; END $$""")
    db.execute("""CREATE TRIGGER reject_dwell BEFORE INSERT ON ml.model
        FOR EACH ROW WHEN (NEW.role='dwell') EXECUTE FUNCTION pg_temp.reject_dwell()""")
    keep = tmp_path / 'existing.json'
    keep.write_text('keep', encoding='utf-8')
    with pytest.raises(psycopg.errors.RaiseException, match='injected insert failure'):
        fit_empirical_live(db, tmp_path, NOW)
    assert db.execute('SELECT count(*) FROM ml.model').fetchone()[0] == 0
    assert list(tmp_path.iterdir()) == [keep]


@pytest.mark.db
def test_support_counts_distinct_runs_trains_dates_and_classes(db, tmp_path):
    from msynd.ops.weekly import fit_empirical_live

    first = completed_run(db)
    second = db.execute("""INSERT INTO obs.run
        (train,start_date,schedule_version_id,route_version_id,lifecycle)
        SELECT train,'2026-10-08',schedule_version_id,route_version_id,'completed'
        FROM obs.run WHERE id=%s RETURNING id""", (first,)).fetchone()[0]
    db.execute("""INSERT INTO obs.event_rev(run_id,station,occ,kind,actual_ts,source,received_at)
        SELECT %s,station,occ,kind,actual_ts+interval '1 day',source,received_at+interval '1 day'
        FROM obs.event_rev WHERE run_id=%s""", (second, first))
    completed_run(db, train='12952', start='2026-10-08')
    db.execute("UPDATE ref.train SET class=CASE number WHEN '12951' THEN 'SF' ELSE 'EXP' END")
    fit_empirical_live(db, tmp_path, NOW)
    for role, metrics, lo, hi in db.execute("""SELECT role,metrics,
        lower(train_window),upper(train_window) FROM ml.model"""):
        n = 2 if role == 'run' else 1
        assert metrics == {'rows': 3*n, 'runs': 3, 'trains': 2, 'service_dates': 2,
                           'per_class': {
                               'SF': {'rows': 2*n, 'runs': 2, 'trains': 1, 'service_dates': 2},
                               'EXP': {'rows': n, 'runs': 1, 'trains': 1, 'service_dates': 1}}}
        assert (lo, hi) == (date(2026, 10, 7), date(2026, 10, 9))


@pytest.mark.db
def test_only_finished_and_six_hour_mature_labels_fit(db, tmp_path):
    from msynd.ops.weekly import fit_empirical_live

    run = completed_run(db)
    db.execute("UPDATE obs.run SET lifecycle='running' WHERE id=%s", (run,))
    assert fit_empirical_live(db, tmp_path, NOW) == {}
    db.execute("UPDATE obs.run SET lifecycle='completed' WHERE id=%s", (run,))
    db.execute('DELETE FROM obs.event_rev')
    cutoff = NOW - timedelta(hours=6)
    insert_event(db, run, 'MMCT', 1, 'dep', cutoff, cutoff)
    insert_event(db, run, 'BRC', 1, 'arr', cutoff + timedelta(minutes=1), cutoff)
    assert set(fit_empirical_live(db, tmp_path, NOW)) == {'origin'}


@pytest.mark.db
def test_cli_fits_and_reports_support(db_url, tmp_path, monkeypatch, capsys):
    from msynd.ops import weekly
    from msynd.db import connect

    # Connect only to the fixture's test database, including the CLI boundary.
    with connect(db_url) as conn:
        completed_run(conn)
    monkeypatch.setattr(weekly, 'connect', lambda: connect(db_url))
    try:
        weekly.main(['fit-empirical', '--models-dir', str(tmp_path)])
        output = capsys.readouterr().out
        assert all(role in output for role in ('run', 'dwell', 'origin'))
        assert 'rows' in output and 'service_dates' in output
        assert len(list(tmp_path.glob('live-*.json'))) == 3
    finally:
        with connect(db_url) as conn:
            conn.execute('TRUNCATE ml.model,obs.run,ref.train,ref.station CASCADE')


def small_weekly(monkeypatch):
    from msynd.engine.replay import Params, load_s_tables
    from msynd.eval.baselines import b3_active
    from msynd.ops import weekly

    monkeypatch.setattr(weekly, 'MIN_RUNS', 1)
    monkeypatch.setattr(weekly, 'MIN_SERVICE_DATES', 1)
    monkeypatch.setattr(weekly, 'FLOORS', {'run': 1, 'dwell': 1, 'dwell_bucket': 1, 'origin': 1})
    monkeypatch.setattr(weekly, 'b3_active', lambda conn, **kw: b3_active(
        conn, params=Params(n=32, k=32, s_tables=load_s_tables()), **kw), raising=False)
    return weekly


@pytest.mark.db
@pytest.mark.parametrize('minimum_runs,minimum_dates', [(1000, 1), (1, 7), (1, 1)])
def test_insufficient_support_fits_nothing(db, tmp_path, monkeypatch, minimum_runs, minimum_dates):
    from msynd.ops import weekly

    completed_run(db, start='2026-09-22' if minimum_runs == minimum_dates == 1 else '2026-09-20')
    monkeypatch.setattr(weekly, 'MIN_RUNS', minimum_runs)
    monkeypatch.setattr(weekly, 'MIN_SERVICE_DATES', minimum_dates)
    monkeypatch.setattr(weekly, 'train_lgbm', lambda *a: pytest.fail('must not fit'))
    monkeypatch.setattr(weekly, 'fit_empirical', lambda *a: pytest.fail('must not fit'))
    report = weekly.evaluate_week(db, tmp_path, NOW)
    assert report['status'] == 'insufficient'
    assert report['runs'] == (0 if minimum_runs == minimum_dates == 1 else 1)
    assert db.execute('SELECT count(*) FROM ml.model').fetchone()[0] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.db
def test_candidates_never_see_validation_rows(db, tmp_path, monkeypatch, capsys):
    import numpy as np
    from msynd.features.build import MODEL_FEATURES
    from msynd.models.lgbm import ModelArtifact as LgbmArtifact

    weekly = small_weekly(monkeypatch)
    training = completed_run(db, start='2026-09-20')
    validation = completed_run(db, train='12952', start='2026-09-21')
    completed_run(db, train='12953', start='2026-10-04')
    completed_run(db, train='12954', start='2026-10-05')
    db.execute("UPDATE ref.train SET class='SF'")
    original = weekly.train_lgbm

    def checked_fit(X, y, role, encoders):
        assert tuple(X.column_names) == MODEL_FEATURES
        np.testing.assert_array_equal(y, [5., 3.])
        assert set(encoders.train_means) == {'12951'}
        return original(X, y, role, encoders)

    monkeypatch.setattr(weekly, 'train_lgbm', checked_fit)
    first = weekly.evaluate_week(db, tmp_path / 'first', NOW)
    # A later revision of training truth and a changed validation label must not leak.
    insert_event(db, training, 'BRC', 1, 'arr', NOW - timedelta(days=20), NOW + timedelta(seconds=1))
    db.execute("UPDATE obs.event_rev SET actual_ts=actual_ts+interval '30 minutes' WHERE run_id=%s", (validation,))
    # Compare independent evaluations at the same frozen boundary.
    db.execute('DELETE FROM sim.metric')
    monkeypatch.setattr(weekly, 'connect', lambda: nullcontext(db))
    weekly.main(['evaluate', '--models-dir', str(tmp_path / 'second'), '--now', NOW.isoformat()])
    second = json.loads(capsys.readouterr().out)
    assert first['validation_window'] == ['2026-09-21', '2026-10-05']
    assert first['runs'] == second['runs'] == 1
    rows = db.execute('SELECT role,kind,path,metrics,active,lower(train_window),upper(train_window) FROM ml.model ORDER BY id').fetchall()
    assert len(rows) == 6
    for a, b in zip(rows[:3], rows[3:]):
        role, kind, path, metrics, promoted, lo, hi = a
        assert not promoted and not b[4]
        assert (lo, hi) == (date(2026, 9, 14), date(2026, 9, 21))
        assert set(metrics['scores']) == {'candidate', 'active'}
        assert set(metrics['decision']) == {'accepted', 'reasons'}
        assert json.loads(Path(path).read_text()) == json.loads(Path(b[2]).read_text())
        if role == 'run':
            assert kind == 'lgbm_quantile'
            assert LgbmArtifact.load(path).encoders.train_means.keys() == {'12951'}
        else:
            assert kind == 'empirical'


@pytest.mark.db
def test_below_floor_class_reported(db, tmp_path, monkeypatch):
    weekly = small_weekly(monkeypatch)
    completed_run(db, start='2026-09-20')
    completed_run(db, train='12952', start='2026-09-19')
    db.execute("UPDATE ref.train SET class=CASE number WHEN '12951' THEN 'SF' ELSE 'EXP' END")
    monkeypatch.setattr(weekly, 'FLOORS', {'run': 2, 'dwell': 1, 'dwell_bucket': 2, 'origin': 1})
    monkeypatch.setattr(weekly, 'train_lgbm', lambda *a: pytest.fail('all run classes below floor'))
    baseline = weekly.b3_active
    model_sets = []

    def capture_models(conn, *, models):
        model_sets.append(models)
        return baseline(conn, models=models)

    monkeypatch.setattr(weekly, 'b3_active', capture_models)
    report = weekly.evaluate_week(db, tmp_path, NOW)
    assert set(report['model_ids']) == {'origin'}
    candidate, incumbent = model_sets
    assert candidate['run'] == incumbent['run']
    assert candidate['dwell'] == incumbent['dwell']
    assert candidate['origin'][0] is None and candidate['origin'][1] != incumbent['origin'][1]
    for role in ('run', 'dwell', 'origin'):
        for name in ('SF', 'EXP'):
            support = report['roles'][role]['per_class'][name]
            assert support['runs'] == 1
            assert support['below_floor'] == (role != 'origin')
            if role == 'dwell':
                assert support['buckets']['0']['below_floor']
    assert report['floors'] == weekly.FLOORS


@pytest.mark.db
@pytest.mark.parametrize('failure', ['insert', 'save', 'score', 'metric'])
def test_failure_leaves_no_rows_or_files(db, tmp_path, monkeypatch, failure):
    weekly = small_weekly(monkeypatch)
    completed_run(db, start='2026-09-20')
    db.execute("UPDATE ref.train SET class='SF'")
    keep = tmp_path / 'existing.json'
    keep.write_text('keep', encoding='utf-8')
    if failure in ('insert', 'metric'):
        db.execute("""CREATE FUNCTION pg_temp.reject_weekly() RETURNS trigger LANGUAGE plpgsql AS
            $$ BEGIN RAISE EXCEPTION 'injected failure'; END $$""")
        target = 'ml.model' if failure == 'insert' else 'sim.metric'
        condition = "WHEN (NEW.role='dwell')" if failure == 'insert' else ''
        db.execute(f"""CREATE TRIGGER reject_weekly BEFORE INSERT ON {target}
            FOR EACH ROW {condition} EXECUTE FUNCTION pg_temp.reject_weekly()""")
        error = psycopg.errors.RaiseException
    elif failure == 'score':
        def failed_score(*args, **kwargs):
            assert db.execute('SELECT count(*) FROM ml.model').fetchone()[0] == 0
            assert list(tmp_path.iterdir()) == [keep]
            raise RuntimeError('injected failure')
        monkeypatch.setattr(weekly, 'score', failed_score, raising=False)
        error = RuntimeError
    else:
        def failed_save(self, path):
            Path(path).write_text('partial', encoding='utf-8')
            raise RuntimeError('injected failure')
        monkeypatch.setattr(ModelArtifact, 'save', failed_save)
        error = RuntimeError
    with pytest.raises(error, match='injected failure'):
        weekly.evaluate_week(db, tmp_path, NOW)
    assert db.execute('SELECT count(*) FROM ml.model').fetchone()[0] == 0
    assert db.execute('SELECT count(*) FROM sim.metric').fetchone()[0] == 0
    assert list(tmp_path.iterdir()) == [keep]


@pytest.mark.db
def test_scores_on_identical_cases(db, tmp_path, monkeypatch):
    from msynd.eval.harness import promotion_ok, score

    weekly = small_weekly(monkeypatch)
    completed_run(db, start='2026-09-20')
    validation = completed_run(db, train='12952', start='2026-09-21')
    db.execute("UPDATE ref.train SET class='SF'")
    results = []
    baseline = weekly.b3_active
    model_sets = []

    def capture_models(conn, *, models):
        model_sets.append(models)
        return baseline(conn, models=models)

    def paired_score(cases, forecasters):
        assert db.execute('SELECT count(*) FROM ml.model').fetchone()[0] == 0
        result = score(cases, forecasters)
        results.append(result)
        return result

    monkeypatch.setattr(weekly, 'score', paired_score, raising=False)
    monkeypatch.setattr(weekly, 'b3_active', capture_models)
    report = weekly.evaluate_week(db, tmp_path, NOW)
    assert report['scores']
    proposed, current = model_sets
    assert all(proposed[role][0] is None and proposed[role][1] != current[role][1]
               for role in ('run', 'dwell', 'origin'))
    assert set(proposed['run'][1].encoders.train_means) == {'12951'}
    candidate, incumbent = results[0]['candidate'], results[0]['active']
    assert candidate.case_ids == incumbent.case_ids and candidate.case_ids
    assert {row['key'][0] for row in candidate.records} == {validation}
    assert report['scores'] == json.loads(json.dumps(
        {'candidate': candidate.overall, 'active': incumbent.overall}, default=lambda value: value.item()))
    decision = promotion_ok(candidate, incumbent, now=NOW)
    assert report['decision'] == {'accepted': decision.accepted, 'reasons': list(decision.reasons)}
    for role, metrics in db.execute('SELECT role,metrics FROM ml.model'):
        assert metrics == json.loads(json.dumps(report['roles'][role] | {
            'floors': report['floors'], 'scores': report['scores'], 'decision': report['decision']}))


@pytest.mark.db
@pytest.mark.parametrize('accepted', [False, True])
def test_never_promotes(db, tmp_path, monkeypatch, accepted):
    from msynd.eval.harness import Decision
    from msynd.models import registry

    weekly = small_weekly(monkeypatch)
    completed_run(db, start='2026-09-20')
    monkeypatch.setattr(registry, 'promote', lambda *a: pytest.fail('promotion forbidden'))
    if accepted:
        monkeypatch.setattr(weekly, 'promotion_ok', lambda *a, **kw: Decision(True, ()))
    report = weekly.evaluate_week(db, tmp_path, NOW)
    assert report['model_ids']
    assert db.execute('SELECT count(*) FROM ml.model WHERE active').fetchone()[0] == 0
    rows = db.execute('SELECT * FROM sim.metric').fetchall()
    assert rows == [(NOW, 'weekly_candidate_accepted', int(accepted))]
    assert report['decision']['accepted'] is accepted
