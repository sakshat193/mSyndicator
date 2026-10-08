from datetime import date
import hashlib
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def archive(tmp_path, monkeypatch):
    from msynd.ops import rstgcn

    path = tmp_path / "mini.zip"
    with zipfile.ZipFile(path, "w") as z:
        for source in (Path(__file__).parents[1] / "fixtures/rstgcn_mini").iterdir():
            z.write(source, "Indian-Railway-Network-and-Delays/" + source.name)
    monkeypatch.setattr(rstgcn, "EXPECTED_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


@pytest.mark.db
def test_import_seq_occ_and_idempotent(db, archive):
    from msynd.ops.rstgcn import import_zip

    import_zip(db, archive)
    before = db.execute("SELECT train,run_date,station,occ,value_min,seq,fetch_id FROM obs.history_daily ORDER BY train,run_date,seq").fetchall()
    assert len(before) == 8
    assert before[:3] == [
        ("00123", date(2024, 9, 1), "AAA", 1, 20, 1, None),
        ("00123", date(2024, 9, 1), "BBB", 1, 30, 2, None),
        ("00123", date(2024, 9, 1), "AAA", 2, 40, 3, None),
    ]
    assert [r[5] for r in before] == [1, 2, 3, 1, 2, 1, 2, 3]
    import_zip(db, archive)
    assert db.execute("SELECT train,run_date,station,occ,value_min,seq,fetch_id FROM obs.history_daily ORDER BY train,run_date,seq").fetchall() == before


@pytest.mark.db
def test_zone_fill_never_overwrites(db, archive):
    from msynd.ops.rstgcn import import_zip

    db.execute("INSERT INTO ref.station(code,name,zone) VALUES ('AAA','a',NULL),('BBB','b','WR'),('CCC','c',NULL)")
    import_zip(db, archive)
    assert db.execute("SELECT code,zone FROM ref.station ORDER BY code").fetchall() == [("AAA", "NR"), ("BBB", "WR"), ("CCC", "ER")]


def test_bad_sha_refuses(tmp_path):
    from msynd.ops.rstgcn import import_zip

    path = tmp_path / "bad.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="sha256"):
        import_zip(None, path)  # Refuse before parsing or touching the DB.


@pytest.mark.db
def test_labels_origin_and_composite(db, archive):
    from msynd.ops.rstgcn import import_zip, build_labels
    from msynd.labels import build_live_labels
    from datetime import datetime, timezone

    import_zip(db, archive)
    db.execute("INSERT INTO ref.train(number,name,class) VALUES ('00123','one','SF')")
    db.execute("""INSERT INTO obs.history_daily(train,run_date,station,occ,value_min,seq) VALUES
        ('old','2024-09-02','AAA',1,99,NULL),
        ('outside','2024-10-01','AAA',1,99,1),
        ('gap','2024-09-30','AAA',1,4,1),
        ('gap','2024-09-30','BBB',1,9,3)""")
    labels = build_labels(db)
    assert [(r['class'], r['value'], r['split']) for r in labels if r['role'] == 'origin'] == [
        ('SF', 20, 'train'), ('SF', 20, 'train'), ('UNKNOWN', 20, 'valid'), ('UNKNOWN', 4, 'valid')]
    assert [(r['value'], r['split']) for r in labels if r['role'] == 'run'] == [(10, 'train'), (10, 'train'), (10, 'train'), (0, 'valid'), (0, 'valid')]
    assert build_live_labels(db, datetime.now(timezone.utc)).num_rows == 0


@pytest.mark.db
def test_fit_marks_cold_start_and_promotes_only_on_gain(db, archive, tmp_path):
    from msynd.ops.rstgcn import import_zip, fit
    from msynd.models.empirical import ModelArtifact
    from msynd.models.registry import active

    import_zip(db, archive)
    fit(db, tmp_path / 'models')
    rows = db.execute("SELECT role,kind,path,metrics,active,lower(train_window),upper(train_window) FROM ml.model ORDER BY role").fetchall()
    assert len(rows) == 2
    for role, kind, path, metrics, promoted, lo, hi in rows:
        assert kind == 'empirical'
        model = ModelArtifact.load(path)
        assert model.cold_start
        assert lo == date(2024, 9, 1) and hi == date(2024, 9, 24)
        assert promoted == (metrics['candidate_log_score'] > metrics['provisional_log_score'])
        assert promoted == (role == 'origin')
        assert model.pooled['UNKNOWN'][81] > .99 if role == 'origin' else model.pooled['UNKNOWN'][71] > .99
    assert active(db)['dwell'][0] is None
    # A tied prior with no training/validation must never replace an active model.
    origin_id = active(db)['origin'][0]
    db.execute('DELETE FROM obs.history_daily')
    fit(db, tmp_path / 'empty')
    assert active(db)['origin'][0] == origin_id
    assert db.execute('SELECT count(*) FROM ml.model WHERE active').fetchone()[0] == 1
