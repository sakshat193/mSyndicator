from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.eval.test_harness import BASE, case
from tests.helpers import insert_event, make_run


@pytest.mark.db
def test_b3_active_replays_receipt_causal_evidence(db):
    from msynd.eval.baselines import b3_active
    from msynd.engine.replay import Params

    run = make_run(db, start="2026-09-21", halts=[("A", None, 540, 0), ("B", 660, None, 100)])
    insert_event(db, run, "A", 1, "dep", BASE + timedelta(minutes=550), BASE + timedelta(minutes=551))
    forecast = b3_active(db, params=Params(n=128, k=32))
    c = case(run_id=run, issued_at=BASE + timedelta(minutes=570))
    before = forecast(c)
    assert before.shape == (128,)
    assert np.isfinite(before).all()
    # A correction received later must not enter the historical model/features/replay.
    insert_event(db, run, "A", 1, "dep", BASE + timedelta(minutes=600), BASE + timedelta(minutes=590))
    np.testing.assert_array_equal(forecast(c), before)
    changed = forecast(replace(c, issued_at=BASE + timedelta(minutes=600)))
    assert np.median(changed) > np.median(before) + 40.
    assert forecast(c, n=64, replicate=1).shape == (64,)
    assert not np.array_equal(forecast(c, replicate=1), before)


@pytest.mark.db
@pytest.mark.parametrize("change", ["registry", "params"])
def test_frozen_forecaster_ignores_later_registry_and_parameter_metadata(db, monkeypatch, change):
    from msynd.eval.baselines import b3_active
    from msynd.engine.replay import Params

    run = make_run(db, start="2026-09-21", halts=[("A", None, 540, 0), ("B", 660, None, 100)])
    forecast = b3_active(db, params=Params(n=64, k=32))
    c = case(run_id=run)
    before = forecast(c)
    if change == "registry":
        db.execute("""INSERT INTO ml.model(role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active)
            VALUES ('run','empirical',9,now(),'[2026-09-01,2026-09-15)','unused','{}','{}',true)""")
    else:
        real_read = Path.read_bytes
        monkeypatch.setattr(Path, "read_bytes", lambda path: b"changed parameter file" if path.name == "params.json" else real_read(path))
    np.testing.assert_array_equal(forecast(c), before)
