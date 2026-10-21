import numpy as np
import pyarrow as pa
import pytest

from msynd.models import registry
from msynd.models.empirical import provisional_models


@pytest.mark.parametrize('role', ['run', 'dwell', 'origin'])
def test_shared_loader_roundtrip(tmp_path, role):
    model = provisional_models()[role]
    path = tmp_path / 'model.json'
    model.save(path)
    restored = registry.load_artifact(role, 'empirical', path)
    queries = pa.Table.from_pylist([{'class': 'EXP', 'sched_dwell': 5.}]*16)
    for a, b in zip(model.predict_distribution(queries), restored.predict_distribution(queries)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='role'):
        registry.load_artifact('dwell' if role != 'dwell' else 'run', 'empirical', path)
    for bad_role, kind in [('composite', 'empirical'), (role, 'neural')]:
        with pytest.raises(ValueError, match='unsupported'):
            registry.load_artifact(bad_role, kind, path)


@pytest.mark.db
def test_active_uses_shared_loader(db, monkeypatch):
    seen = []
    model = provisional_models()['run']
    monkeypatch.setattr(registry, 'load_artifact', lambda *args: seen.append(args) or model)
    id = db.execute("""INSERT INTO ml.model
        (role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active)
        VALUES ('run','empirical',1,now(),'[2026-09-01,2026-10-01)','frozen','{}','{}',true)
        RETURNING id""").fetchone()[0]
    assert registry.active(db)['run'] == (id, model)
    assert seen == [('run', 'empirical', 'frozen')]
