from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

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
