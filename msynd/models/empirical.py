"""Class-pooled fixed-bin models and explicit day-1 priors."""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import numpy as np

from msynd import dist


def _histogram(values, signed, prior):
    edges = np.arange(-60, 241) if signed else np.arange(181)
    counts = np.bincount(np.searchsorted(edges, values, side="right"),
                         minlength=302) if signed else np.bincount(
                             np.searchsorted(edges, values, side="right") - 1, minlength=181)
    # ponytail: fixed mix weight; upgrade: fit it on validation log-score.
    return (.99 * counts / len(values) + .01 * prior).tolist()


def _prior(role, dwell):
    edges = np.arange(181) if role == "dwell" else np.arange(-60, 241)
    if role == "run":
        cdf = np.empty(len(edges))
        negative = edges < 0
        cdf[negative] = .5 * np.exp(edges[negative] / 5)
        cdf[~negative] = 1 - .5 * np.exp(-edges[~negative] / 5)
        survival = np.empty(len(edges))
        survival[negative] = 1-cdf[negative]
        survival[~negative] = .5 * np.exp(-edges[~negative] / 5)
        lo, hi = 5., 5.
    else:
        # ponytail: fixed half-dwell edge; upgrade: fit the edge from live W labels.
        shift, hi = (dwell/2, 2.) if role == "dwell" else (-2., 10.)
        cdf = -np.expm1(-np.maximum(0., edges-shift) / hi)
        survival = np.exp(-np.maximum(0., edges-shift) / hi)
        lo = .5
    interior = np.where(cdf[:-1] < .5, np.diff(cdf), -np.diff(survival))
    masses = np.r_[cdf[0], interior, survival[-1]] if role != "dwell" else np.r_[interior, survival[-1]]
    return masses, lo, hi


@dataclass
class ModelArtifact:
    role: str
    pooled: dict
    buckets: dict
    dwell_edges: list
    tail: dict
    cold_start: bool = False

    def predict_distribution(self, Xq):
        rows = Xq.to_pylist() if hasattr(Xq, "to_pylist") else [dict(zip(Xq, v)) for v in zip(*Xq.values())]
        if len(rows) % 16:
            raise ValueError("require 16 consecutive query rows per item")
        masses, lower, upper = [], [], []
        for row in rows:
            name = str(row["class"])
            dwell = float(row.get("sched_dwell", 0.))
            if not np.isfinite(dwell) or dwell < 0:
                raise ValueError("scheduled dwell must be finite and nonnegative")
            m = None
            if self.role == "dwell" and self.dwell_edges and self.dwell_edges[0] <= dwell <= self.dwell_edges[-1]:
                bucket = np.searchsorted(self.dwell_edges[1:-1], dwell, side="right")
                m = self.buckets.get(name, {}).get(str(bucket))
            if m is None:
                m = self.pooled.get(name)
            if m is None:
                m, lo, hi = _prior(self.role, dwell)
            else:
                lo, hi = self.tail[name].get("lo", .5), self.tail[name]["hi"]
            masses.append(m)
            lower.append(lo)
            upper.append(hi)
        n = len(rows) // 16
        return (np.asarray(masses).reshape(n, 16, 181 if self.role == "dwell" else 302),
                None if self.role == "dwell" else np.asarray(lower).reshape(n, 16),
                np.asarray(upper).reshape(n, 16))

    def save(self, path):
        path = Path(path)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(vars(self), stream, allow_nan=False)
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
        model = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        if model.role not in {"run", "dwell", "origin"}:
            raise ValueError("invalid empirical role")
        return model


def fit_empirical(labels, role):
    """Labels carry class/value and, for W, scheduled dwell; only training rows enter."""
    if role not in {"run", "dwell", "origin"}:
        raise ValueError("role must be run, dwell or origin")
    rows = labels.to_pylist() if hasattr(labels, "to_pylist") else list(labels)
    rows = [r for r in rows if r.get("role", role) == role and r.get("split", "train") == "train"]
    values = np.asarray([r["value"] for r in rows], dtype=float)
    if not np.all(np.isfinite(values)) or (role == "dwell" and np.any(values < 0)):
        raise ValueError("labels must be finite; dwell must be nonnegative")
    if any(r["class"] is None for r in rows):
        raise ValueError("missing training class")
    dwells = np.asarray([r["sched_dwell"] for r in rows], dtype=float) if role == "dwell" else None
    if dwells is not None and (not np.all(np.isfinite(dwells)) or np.any(dwells < 0)):
        raise ValueError("scheduled dwell must be finite and nonnegative")
    edges = np.unique(np.quantile(dwells, [0., .2, .4, .6, .8, 1.])).tolist() if len(rows) and role == "dwell" else []
    pooled, buckets, tails = {}, {}, {}
    classes = np.asarray([str(r["class"]) for r in rows])
    signed = role != "dwell"
    for name in sorted(set(classes)):
        selected = classes == name
        y = values[selected]
        # ponytail: mean-dwell prior; upgrade: mix the prior at predict time using the query dwell
        dwell = float(dwells[selected].mean()) if role == "dwell" else 0.
        prior, prior_lo, prior_hi = _prior(role, dwell)
        pooled[name] = _histogram(y, signed, prior)
        excess = y[y >= (240 if signed else 180)] - (240 if signed else 180)
        tails[name] = {"hi": max(prior_hi, float(excess.mean())) if len(excess) else prior_hi}
        if signed:
            excess = -60-y[y < -60]
            tails[name]["lo"] = max(prior_lo, float(excess.mean())) if len(excess) else prior_lo
        if role == "dwell":
            bucket_ids = np.searchsorted(edges[1:-1], dwells[selected], side="right")
            buckets[name] = {}
            for b in np.unique(bucket_ids):
                members = bucket_ids == b
                prior, _, _ = _prior(role, float(dwells[selected][members].mean()))
                buckets[name][str(b)] = _histogram(y[members], False, prior)
    return ModelArtifact(role, pooled, buckets, edges, tails, not rows)


def provisional_models():
    return {role: ModelArtifact(role, {}, {}, [], {}, True) for role in ("run", "dwell", "origin")}
