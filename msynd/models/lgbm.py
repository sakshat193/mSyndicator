"""Quantile boosters over named columns (Arrow Table or mapping).

Queries are already expanded, with 16 consecutive delay-level rows per item.
Class/category encodings are fitted on training rows and saved with the model.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile

import lightgbm as lgb
import numpy as np

from msynd import dist
from msynd.features.build import FoldEncoders


def _columns(X):
    columns = X.to_pydict() if hasattr(X, "to_pydict") else dict(X)
    columns = {name: np.asarray(values) for name, values in columns.items()}
    if "class" not in columns or any(v.ndim != 1 for v in columns.values()):
        raise ValueError("require named one-dimensional features including class")
    if len({len(v) for v in columns.values()}) != 1:
        raise ValueError("feature lengths differ")
    return columns


def _matrix(columns, names, categories):
    if set(columns) != set(names):
        raise ValueError("query features differ from training features")
    arrays = []
    for name in names:
        values = columns[name]
        if name in categories:
            codes = {value: i for i, value in enumerate(categories[name])}
            arrays.append([codes.get(str(value), -1) for value in values])
        else:
            arrays.append(values.astype(float))
    matrix = np.column_stack(arrays).astype(float)
    if np.any(np.isinf(matrix)):
        raise ValueError("infinite feature values")
    return matrix


@dataclass
class ModelArtifact:
    role: str
    boosters: list
    feature_names: tuple
    categories: dict
    tail: dict
    encoders: FoldEncoders | None = None

    def predict_quantiles(self, Xq):
        columns = _columns(Xq)
        matrix = _matrix(columns, self.feature_names, self.categories)
        if len(matrix) % len(dist.LEVELS):
            raise ValueError("require 16 consecutive query rows per item")
        if not len(matrix):
            return np.empty((0, 16, 9))
        q = np.column_stack([model.predict(matrix, num_threads=1) for model in self.boosters])
        return np.sort(q, axis=-1).reshape(-1, 16, 9)

    def predict_distribution(self, Xq):
        """Return masses and per-query tail scales; W has no lower tail."""
        q = self.predict_quantiles(Xq)
        signed = self.role != "dwell"
        masses = np.empty((*q.shape[:2], dist.SIGNED_NB if signed else dist.NONNEG_NB))
        lo = np.empty(q.shape[:2]) if signed else None
        hi = np.empty(q.shape[:2])
        for index in np.ndindex(q.shape[:2]):
            masses[index], lower, hi[index] = dist.from_quantiles(q[index], signed)
            if signed:
                lo[index] = lower
        return masses, lo, hi

    def save(self, path):
        """Atomically save to the filesystem path recorded in ml.model.path."""
        path = Path(path)
        data = {"role": self.role, "feature_names": self.feature_names,
                "categories": self.categories, "tail": self.tail,
                "boosters": [model.model_to_string() for model in self.boosters]}
        if self.encoders is not None:
            data["encoders"] = self.encoders.to_json()
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(data, stream, allow_nan=False)
            except BaseException:
                stream.close()
                temporary.unlink()
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["role"] not in {"run", "dwell", "origin"} or len(data["boosters"]) != 9:
            raise ValueError("invalid quantile artifact")
        return cls(data["role"], [lgb.Booster(model_str=s) for s in data["boosters"]],
                   tuple(data["feature_names"]), data["categories"], data["tail"],
                   FoldEncoders.from_json(data["encoders"]) if "encoders" in data else None)


def train_lgbm(X, y, role, encoders=None):
    if role not in {"run", "dwell", "origin"}:
        raise ValueError("role must be run, dwell or origin")
    columns = _columns(X)
    identifiers = set(columns) & {"run_id", "start_date", "train", "idx", "from_stn", "to_stn", "split"}
    if identifiers:
        raise ValueError(f"identifier columns are not model inputs: {', '.join(sorted(identifiers))}")
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or len(y) != len(columns["class"]) or not np.all(np.isfinite(y)):
        raise ValueError("require one finite label per training row")
    if role == "dwell" and np.any(y < 0):
        raise ValueError("dwell labels must be nonnegative")
    if any(value is None for value in columns["class"]):
        raise ValueError("missing training class")
    names = tuple(columns)
    categories = {name: sorted(set(map(str, values))) for name, values in columns.items()
                  if name == "class" or values.dtype.kind in "OUS"}
    matrix = _matrix(columns, names, categories)
    dataset = lgb.Dataset(matrix, label=y, feature_name=list(names),
                          categorical_feature=[i for i, name in enumerate(names) if name in categories],
                          free_raw_data=False)
    # ponytail: fixed 100-round CPU baseline; tune rounds with rolling validation if it underfits.
    params = {"objective": "quantile", "verbosity": -1, "num_threads": 1,
              "seed": 0, "deterministic": True, "force_col_wise": True}
    boosters = [lgb.train(params | {"alpha": float(alpha)}, dataset, num_boost_round=100)
                for alpha in dist.QUANTILES]
    tail = {}
    classes = np.asarray(list(map(str, columns["class"])))
    for name in categories["class"]:
        a, b, c, d = np.quantile(y[classes == name], [.02, .1, .9, .98])
        tail[name] = {"hi": max(.5, float(d-c)/math.log(5))}
        if role != "dwell":
            tail[name]["lo"] = max(.5, float(b-a)/math.log(5))
    return ModelArtifact(role, boosters, names, categories, tail, encoders)


def predict_masses(artifact, Xq):
    return artifact.predict_distribution(Xq)[0]
