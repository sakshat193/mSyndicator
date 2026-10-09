import math
import json

import numpy as np
import pyarrow as pa
import pytest

from msynd import dist
from msynd.models.lgbm import ModelArtifact, train_lgbm, predict_masses


def training(role="run"):
    # Two classes with different label spreads; no validation data enters fitting.
    z = (np.arange(1000) + .5) / 1000
    y = np.concatenate((-20 + 40*z, -40 + 80*z))
    if role == "dwell":
        y += 40
    X = pa.table({"class": ["RAJ"]*1000 + ["PASS"]*1000,
                  "d": np.zeros(2000)})
    return X, y


def queries():
    return pa.table({"class": ["RAJ"]*16 + ["PASS"]*16,
                     "d": np.tile(dist.LEVELS, 2)})


def test_quantile_reproduction_half_bin():
    X, y = training()
    artifact = train_lgbm(X, y, "run")
    q = artifact.predict_quantiles(queries())
    masses, lo, hi = artifact.predict_distribution(queries())
    assert masses.shape == (2, 16, 302)
    np.testing.assert_allclose(q[0, 0, [2, 6]], [-16, 16], atol=1, rtol=0)
    np.testing.assert_allclose(q[1, 0, [2, 6]], [-32, 32], atol=1, rtol=0)
    np.testing.assert_allclose(lo, np.maximum(.5, (q[:, :, 2]-q[:, :, 0])/math.log(5)))
    np.testing.assert_allclose(hi, np.maximum(.5, (q[:, :, -1]-q[:, :, -3])/math.log(5)))
    np.testing.assert_allclose(masses.sum(axis=-1), 1, atol=1e-12)
    np.testing.assert_array_equal(predict_masses(artifact, queries()), masses)
    for row in range(2):
        for level in (0, 15):
            eff = dist.Eff(masses[row, level], True, lo[row, level], hi[row, level],
                           0, 0, 0, 1, -math.inf, 0, 1)
            reproduced = [dist.eff_quantile(eff, u) for u in dist.QUANTILES]
            np.testing.assert_allclose(reproduced, q[row, level], atol=.5, rtol=0)
    for name, labels in (("RAJ", y[:1000]), ("PASS", y[1000:])):
        a, b, c, d = np.quantile(labels, [.02, .1, .9, .98])
        assert artifact.tail[name] == pytest.approx({"lo": max(.5, (b-a)/math.log(5)),
                                                   "hi": max(.5, (d-c)/math.log(5))})
    # Training metadata must never change the predictive distribution.
    artifact.tail["RAJ"] = {"lo": 500., "hi": 500.}
    np.testing.assert_array_equal(predict_masses(artifact, queries()), masses)


def test_signed_o_negative_labels_roundtrip(tmp_path):
    X, y = training()
    artifact = train_lgbm(X, y - 15, "origin")
    path = tmp_path / "model.json"  # The value stored in ml.model.path.
    artifact.save(path)
    assert "encoders" not in json.loads(path.read_text(encoding="utf-8"))
    restored = ModelArtifact.load(path)
    assert restored.encoders is None
    before = artifact.predict_distribution(queries())
    after = restored.predict_distribution(queries())
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert restored.tail == artifact.tail
    assert restored.role == "origin"
    assert np.all(restored.predict_quantiles(queries())[:, :, 4] < 0)
    assert np.all(after[0][:, :, :61].sum(axis=-1) > .5)


def test_lgbm_artifact_saves_encoders(tmp_path):
    from msynd.features.build import FoldEncoders

    encoders = FoldEncoders.fit(pa.Table.from_pylist([
        dict(train="A", from_stn="X", to_stn="Y", value=0., observed_running=100.),
        dict(train="B", from_stn="Y", to_stn="Z", value=10., observed_running=130.),
    ]))
    X, y = training()
    artifact = train_lgbm(X, y, "run", encoders=encoders)
    assert artifact.encoders == encoders
    path = tmp_path / "model.json"
    artifact.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["encoders"] == encoders.to_json()
    restored = ModelArtifact.load(path)
    assert restored.encoders == encoders
    np.testing.assert_array_equal(predict_masses(restored, queries()), predict_masses(artifact, queries()))


@pytest.mark.parametrize("column", ["run_id", "start_date", "train", "idx", "from_stn", "to_stn", "split"])
def test_train_lgbm_rejects_identifier_columns(column):
    X, y = training()
    X = X.append_column(column, pa.array(["identifier"] * len(y)))
    with pytest.raises(ValueError, match=column):
        train_lgbm(X, y, "run")


def test_noncrossing_after_sort():
    X, y = training("dwell")
    artifact = train_lgbm(X, y, "dwell")
    # Real trained boosters deliberately reordered to simulate quantile crossing.
    artifact.boosters = artifact.boosters[::-1]
    q = artifact.predict_quantiles(queries())
    assert np.all(np.diff(q, axis=-1) >= 0)
    assert np.all(q[:, :, 0] < q[:, :, -1])
    masses, lo, hi = artifact.predict_distribution(queries())
    assert masses.shape == (2, 16, 181)
    assert np.all(masses >= 0)
    np.testing.assert_allclose(masses.sum(axis=-1), 1, atol=1e-12)
    assert lo is None
    assert np.all(hi >= .5)
    assert all(set(tail) == {"hi"} for tail in artifact.tail.values())


def test_invalid_training_and_query_inputs():
    X, y = training()
    with pytest.raises(ValueError):
        train_lgbm(X, y, "other")
    with pytest.raises(ValueError):
        train_lgbm(X, y[:-1], "run")
    with pytest.raises(ValueError):
        train_lgbm(X, np.full_like(y, np.nan), "run")
    with pytest.raises(ValueError):
        train_lgbm(X, y, "dwell")
    artifact = train_lgbm(X, y, "run")
    with pytest.raises(ValueError):
        predict_masses(artifact, queries().slice(0, 15))
    with pytest.raises(ValueError):
        predict_masses(artifact, queries().drop(["d"]))


def test_tail_floor_on_narrow_labels():
    X, y = training()
    artifact = train_lgbm(X, y/100, "run")
    _, lo, hi = artifact.predict_distribution(queries())
    np.testing.assert_array_equal(lo, .5)
    np.testing.assert_array_equal(hi, .5)
    assert all(tail == {"lo": .5, "hi": .5} for tail in artifact.tail.values())
