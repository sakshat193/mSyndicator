from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from tests.helpers import make_run, insert_event


@pytest.mark.db
def test_cache_hit_on_same_v(db):
    from msynd.engine import cache
    from msynd.engine.evidence import load_run_evidence
    from msynd.models.empirical import provisional_models

    run = make_run(db, halts=[("A", None, 540, 0), ("B", 600, 605, 100), ("C", 665, None, 200)])
    models = {r: (None, m) for r, m in provisional_models().items()}
    ev = load_run_evidence(db, run)
    a = cache.tables_for(db, ev, models, 0)
    assert cache.tables_for(db, ev, models, 0) is a
    assert (a.r.shape, a.w.shape, a.o.shape) == ((2, 16, 302), (3, 16, 181), (302,))
    assert a.cold_start
    assert a.w[1, 0, :2].sum() == 0.
    assert a.w[1, 0, 2:5].sum() > 0.
    b = cache.tables_for(db, ev, models, 1)
    assert b is not a
    assert cache.tables_for(db, replace(ev, versions=ev.versions | {"N": 2000}), models, 1) is not b
    # Context change must miss even when V is unchanged.
    now = datetime.now(timezone.utc)
    insert_event(db, run, "A", 1, "dep", now, now)
    insert_event(db, run, "B", 1, "arr", now, now)
    assert cache.tables_for(db, ev, models, 1) is not b


@pytest.mark.db
def test_cache_batches_levels_bias_and_lgbm_tail_max(db):
    from msynd.engine import cache
    from msynd.engine.evidence import load_run_evidence
    from msynd.models.lgbm import train_lgbm
    from msynd.models.empirical import provisional_models
    import pyarrow as pa

    run = make_run(db, halts=[("A", None, 540, 0), ("B", 600, 605, 100), ("C", 665, None, 200)])
    db.execute("UPDATE ref.train SET class='X' WHERE number='12951'")
    levels = np.tile([-30,-10,0,5,10,15,20,30,45,60,90,120,180,240,360,600], 30)
    model = train_lgbm(pa.table({"class": ["X"] * len(levels), "d": levels}),
                       levels * .1 + np.tile(np.arange(30), 16), "run")
    id = db.execute("""INSERT INTO ml.model(role,kind,feature_ver,trained_at,train_window,path,tail,metrics)
        VALUES ('run','lgbm_quantile',1,now(),'[2026-09-01,2026-10-01)','unused','{}','{}') RETURNING id""").fetchone()[0]
    db.execute("INSERT INTO ml.leg_bias VALUES (%s,7,'A','B',3,12,30)", (id,))
    models = {r: (None, m) for r, m in provisional_models().items()}
    models["run"] = (id, model)
    ev = load_run_evidence(db, run)
    result = cache.tables_for(db, ev, models, 7)
    masses, lo, hi = model.predict_distribution(pa.table({"class": ["X"] * 32, "d": np.tile(cache.LEVELS, 2)}))
    np.testing.assert_array_equal(result.r, masses)
    np.testing.assert_array_equal(result.r_tau, np.column_stack((lo.max(axis=1), hi.max(axis=1))))
    np.testing.assert_array_equal(result.bias, [12., 0.])
    assert cache.tables_for(db, ev, models, 8).bias.tolist() == [0., 0.]
    dwell = train_lgbm(pa.table({"class": ["X"] * len(levels), "d": levels}),
                      np.maximum(0., levels*.1), "dwell")
    models["dwell"] = (id+1000, dwell)
    result = cache.tables_for(db, ev, models, 7)
    masses, _, hi = dwell.predict_distribution(pa.table({"class": ["X"] * 48, "d": np.tile(cache.LEVELS, 3)}))
    np.testing.assert_array_equal(result.w, masses)
    np.testing.assert_array_equal(result.w_tau, hi.max(axis=1))


@pytest.mark.db
def test_origin_has_zero_delay_and_unseen_empirical_is_cold_start(db):
    from msynd.engine import cache
    from msynd.engine.evidence import load_run_evidence
    from msynd.models.empirical import fit_empirical
    from msynd.models.lgbm import train_lgbm
    import pyarrow as pa

    run = make_run(db)
    db.execute("UPDATE ref.train SET class='X' WHERE number='12951'")
    empirical = {role: (None, fit_empirical(pa.table({"class": ["Y"], "value": [1.],
                     "sched_dwell": [5.]}), role)) for role in ("run", "dwell", "origin")}
    ev = load_run_evidence(db, run)
    assert cache.tables_for(db, ev, empirical, 0).cold_start
    levels = np.tile(cache.LEVELS, 30)
    origin = train_lgbm(pa.table({"class": ["X"] * len(levels), "d": levels}), levels*.2, "origin")
    empirical["origin"] = (123, origin)
    result = cache.tables_for(db, ev, empirical, 0)
    m, lo, hi = origin.predict_distribution(pa.table({"class": ["X"] * 16, "d": [0.] * 16}))
    np.testing.assert_array_equal(result.o, m[0,0])
    np.testing.assert_array_equal(result.o_tau, [lo[0,0], hi[0,0]])


@pytest.mark.db
def test_cache_preserves_head_cancelled_indices(db):
    from msynd.engine import cache
    from msynd.engine.evidence import load_run_evidence
    from msynd.models.empirical import provisional_models
    from msynd.engine.replay import replay, Params

    run = make_run(db, halts=[("A", None, 540, 0), ("B", 600, 605, 100), ("C", 665, None, 200)])
    db.execute("UPDATE obs.run SET part_cancelled='[{\"from\":\"A\",\"to\":\"B\"}]' WHERE id=%s", (run,))
    ev = load_run_evidence(db, run)
    result = cache.tables_for(db, ev, {r: (None,m) for r,m in provisional_models().items()}, 0)
    assert result.r.shape == (2,16,302)
    assert result.w.shape == (3,16,181)
    assert replay(ev, result, Params(n=32, k=2), 42).paths.shape == (2,2)
