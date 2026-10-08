import math

import numpy as np
import pyarrow as pa
import psycopg
import pytest

from msynd import dist


def query(classes, dwells=None):
    return pa.table({"class": np.repeat(classes, 16),
                     "sched_dwell": np.repeat(dwells or [0.] * len(classes), 16),
                     "d": np.tile(dist.LEVELS, len(classes))})


def test_empirical_masses_sum_to_one_and_signed_o(tmp_path):
    from msynd.models.empirical import fit_empirical, ModelArtifact

    for role in ("run", "origin", "dwell"):
        values = [-70., -10., 0., 240., 260.] if role != "dwell" else [0., 1., 10., 180., 200.]
        labels = pa.table({"class": ["X"] * 5, "value": values, "sched_dwell": [5.] * 5})
        model = fit_empirical(labels, role)
        masses, lo, hi = model.predict_distribution(query(["X"], [5.]))
        assert masses.shape == (1, 16, 181 if role == "dwell" else 302)
        np.testing.assert_allclose(masses.sum(axis=-1), 1.)
        if role != "dwell":
            prior_lo = .5*math.exp(-12) if role == "run" else 0.
            prior_bin = .5*(math.exp(-9/5)-math.exp(-10/5)) if role == "run" else 0.
            assert masses[0, 0, 0] == pytest.approx(.99*.2 + .01*prior_lo)
            assert masses[0, 0, 51] == pytest.approx(.99*.2 + .01*prior_bin)  # [-10,-9)
            assert lo[0, 0] == 10.
            assert hi[0, 0] == 10.
        else:
            assert lo is None
            assert hi[0, 0] == 10.
        path = tmp_path / (role + ".json")
        model.save(path)
        restored = ModelArtifact.load(path)
        for a, b in zip(model.predict_distribution(query(["X"], [5.])),
                        restored.predict_distribution(query(["X"], [5.]))):
            if a is not None:
                np.testing.assert_array_equal(a, b)


def test_dwell_buckets_and_fallback(tmp_path):
    from msynd.models.empirical import fit_empirical, ModelArtifact

    labels = pa.table({"class": ["X"] * 5 + ["Y"], "value": [1., 2., 3., 4., 5., 90.],
                       "sched_dwell": [0., 10., 20., 30., 40., 0.]})
    model = fit_empirical(labels, "dwell")
    assert model.dwell_edges == [0., 10., 20., 30., 40.]
    m, _, _ = model.predict_distribution(query(["X", "Y", "X", "NEW"], [20., 20., 100., 5.]))
    assert m[0, 0, 3] == pytest.approx(.99)
    assert m[1, 0, 90] == pytest.approx(.99 + .01*math.exp(-45)*(1-math.exp(-.5)))  # empty Y bucket -> Y pool
    np.testing.assert_allclose(m[2, 0, 1:6], [.198] * 5)  # outside fitted edges -> pool
    assert m[3, 0, :2].sum() == 0.  # unseen class -> half sched dwell + Exp(2)
    assert m[3, 0, 2] == pytest.approx(1-math.exp(-.25))
    path = tmp_path / "w.json"
    model.save(path)
    assert ModelArtifact.load(path).dwell_edges == model.dwell_edges


def test_provisional_flags_cold_start():
    from msynd.models.empirical import provisional_models

    models = provisional_models()
    assert set(models) == {"run", "dwell", "origin"}
    assert all(m.cold_start for m in models.values())
    r, lo, hi = models["run"].predict_distribution(query(["X"]))
    assert lo[0, 0] == hi[0, 0] == 5.
    assert r[0, 0, :61].sum() == pytest.approx(.5)
    assert r[0, 0, -1] == pytest.approx(.5*math.exp(-48), rel=1e-12, abs=0.)
    w, _, hi = models["dwell"].predict_distribution(query(["X"], [10.]))
    assert hi[0, 0] == 2.
    assert w[0, 0, :5].sum() == 0.
    assert w[0, 0, 5] == pytest.approx(1-math.exp(-.5))
    o, _, hi = models["origin"].predict_distribution(query(["X"]))
    assert hi[0, 0] == 10.
    assert o[0, 0, :61].sum() == pytest.approx(1-math.exp(-.2))


def test_empty_and_training_only_fallback():
    from msynd.models.empirical import fit_empirical

    empty = fit_empirical(pa.table({"class": [], "value": [], "sched_dwell": []}), "dwell")
    assert empty.cold_start
    assert empty.predict_distribution(query(["X"], [5.]))[0][0,0,5] > 0.
    model = fit_empirical(pa.Table.from_pylist([
        {"class": "X", "value": 1., "sched_dwell": 5., "split": "train"},
        {"class": "X", "value": 99., "sched_dwell": 500., "split": "valid"},
    ]), "dwell")
    assert model.dwell_edges == [5.]
    m = model.predict_distribution(query(["X"], [5.]))[0][0,0]
    assert m[1] == pytest.approx(.99)
    assert m[5] == pytest.approx(.01*math.exp(-1.25)*(1-math.exp(-.5)))


@pytest.mark.parametrize("role", ["run", "origin", "dwell"])
def test_unseen_bin_has_nonzero_mass(role):
    from msynd.models.empirical import fit_empirical

    model = fit_empirical([{"class": "X", "value": 0., "sched_dwell": 5.}], role)
    m, lo, hi = model.predict_distribution(query(["X"], [5.]))
    index = 100 if role == "dwell" else 161
    scale = {"run": 5., "origin": 10., "dwell": 2.}[role]
    shift = {"run": 0., "origin": -2., "dwell": 2.5}[role]
    factor = .5 if role == "run" else 1.
    expected = .01*factor*math.exp(-(100-shift)/scale)*(1-math.exp(-1/scale))
    assert m[0, 0, index] > 0.
    assert m[0, 0, index] == pytest.approx(expected, rel=1e-12, abs=0.)
    assert hi[0, 0] == scale
    if role != "dwell":
        assert lo[0, 0] == (5. if role == "run" else .5)
    assert not model.cold_start


@pytest.mark.parametrize("role", ["run", "origin", "dwell"])
def test_smoothed_masses_sum_to_one(role):
    from msynd.models.empirical import fit_empirical

    values = [0., 0., 180., 200.] if role == "dwell" else [-80., 0., 240., 280.]
    model = fit_empirical([{"class": "X", "value": v, "sched_dwell": d}
                           for v, d in zip(values, [0., 10., 20., 30.])], role)
    for masses in [model.pooled["X"], *model.buckets.get("X", {}).values()]:
        assert np.all(np.isfinite(masses))
        assert np.all(np.asarray(masses) >= 0.)
        assert sum(masses) == pytest.approx(1.)


def test_dwell_smoothing_uses_each_histograms_mean_training_dwell():
    from msynd.models.empirical import fit_empirical

    rows = [{"class": "X", "value": 0., "sched_dwell": d} for d in range(11)]
    rows.append({"class": "X", "value": 99., "sched_dwell": 500., "split": "valid"})
    model = fit_empirical(rows, "dwell")
    # Pool mean is 5; first bucket contains dwells 0 and 1, so its mean is .5.
    assert model.pooled["X"][0] == pytest.approx(.99)
    assert model.pooled["X"][5] == pytest.approx(.01*math.exp(-1.25)*(1-math.exp(-.5)))
    m, _, hi = model.predict_distribution(query(["X", "X"], [0., 100.]))
    assert m[0, 0, 0] == pytest.approx(.99 + .01*(1-math.exp(-.375)))
    assert m[0, 0, 1] == pytest.approx(.01*(math.exp(-.375)-math.exp(-.875)))
    np.testing.assert_array_equal(m[1, 0], model.pooled["X"])
    assert np.all(hi == 2.)


@pytest.mark.parametrize("role,value", [("bad", 0.), ("dwell", -1.), ("run", float("nan"))])
def test_invalid_labels(role, value):
    from msynd.models.empirical import fit_empirical

    with pytest.raises(ValueError):
        fit_empirical(pa.table({"class": ["X"], "value": [value], "sched_dwell": [5.]}), role)


@pytest.mark.db
def test_promote_single_active_and_notifies(db, db_url, tmp_path):
    from msynd.models.empirical import provisional_models
    from msynd.models import registry

    # Transactional NOTIFY must be observable from another connection after promotion.
    db.autocommit = True
    ids = []
    with psycopg.connect(db_url, autocommit=True) as listener:
        listener.execute("LISTEN model_changed")
        try:
            for role in ("run", "run", "dwell"):
                path = tmp_path / f"{role}-{len(ids)}.json"
                provisional_models()[role].save(path)
                ids.append(db.execute("""INSERT INTO ml.model
                    (role,kind,feature_ver,trained_at,train_window,path,tail,metrics)
                    VALUES (%s,'empirical',1,now(),'[2026-09-01,2026-10-01)',%s,'{}','{}') RETURNING id
                    """, (role, str(path))).fetchone()[0])
            registry.promote(db, ids[0])
            registry.promote(db, ids[2])
            registry.promote(db, ids[1])
            assert db.execute("SELECT id FROM ml.model WHERE active ORDER BY id").fetchall() == [(ids[1],), (ids[2],)]
            active = registry.active(db)
            assert active["run"][0] == ids[1]
            assert active["dwell"][1].cold_start
            assert active["origin"][0] is None  # day-1 fallback
            assert len(list(listener.notifies(timeout=2, stop_after=3))) == 3
            with pytest.raises(RuntimeError):
                with db.transaction():
                    registry.promote(db, ids[0])
                    raise RuntimeError("abort outer transaction")
            assert registry.active(db)["run"][0] == ids[1]
            assert list(listener.notifies(timeout=.1)) == []
            with pytest.raises(ValueError):
                registry.promote(db, -1)
            assert registry.active(db)["run"][0] == ids[1]
        finally:
            db.execute("DELETE FROM ml.model WHERE id=ANY(%s)", (ids,))


def test_dwell_prior_mass_below_scheduled():
    from msynd.models.empirical import _prior

    masses, _, hi = _prior("dwell", 10.)
    assert masses[:5].sum() == 0.
    assert np.all(masses[5:10] > 0.)
    assert masses.sum() == pytest.approx(1.)
    assert hi == 2.


def test_dwell_prior_zero_dwell():
    from msynd.models.empirical import _prior

    masses, _, hi = _prior("dwell", 0.)
    expected = np.r_[np.exp(-np.arange(180)/2)*(1-math.exp(-.5)), math.exp(-90)]
    np.testing.assert_allclose(masses, expected, rtol=1e-12, atol=0.)
    assert hi == 2.
