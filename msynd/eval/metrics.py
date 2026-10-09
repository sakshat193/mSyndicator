"""Empirical forecast scores and service-date block confidence intervals."""

from dataclasses import dataclass

import numpy as np

from msynd.dist import QUANTILES


def samples(values):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("require nonempty finite one-dimensional forecast samples")
    return np.sort(values)


def sample_metrics(forecast, actual):
    x = samples(forecast)
    if not np.isfinite(actual):
        raise ValueError("require a finite actual")
    levels = QUANTILES[1:-1]
    q = np.quantile(x, levels)
    residual = actual - q
    # E|X-y| - E|X-X'|/2; sorting avoids a quadratic pairwise matrix.
    crps = np.abs(x - actual).mean() - np.dot(2*np.arange(len(x)) - len(x) + 1, x)/len(x)**2
    return dict(crps=float(max(0., crps)),
                pinball=dict(zip(levels, np.maximum(levels*residual, (levels-1)*residual))),
                coverage={.8: float(q[1] <= actual <= q[5]), .9: float(q[0] <= actual <= q[6])},
                width={.8: float(q[5]-q[1]), .9: float(q[6]-q[0])},
                p50_ae=float(abs(q[3]-actual)),
                # Mid-PIT handles point baselines and physical-floor atoms without random jitter.
                pit=float((np.count_nonzero(x < actual) + .5*np.count_nonzero(x == actual))/len(x)))


def summarize(records):
    """P50 MAE is the spec's median absolute error, not the arithmetic mean."""
    if not records:
        return {"n": 0}
    result = {"n": len(records), "crps": float(np.mean([r["crps"] for r in records])),
              "p50_mae": float(np.median([r["p50_ae"] for r in records])),
              "pit_histogram": tuple(np.histogram([r["pit"] for r in records], bins=np.linspace(0, 1, 11))[0])}
    for name in ("pinball", "coverage", "width"):
        result[name] = {level: float(np.mean([r[name][level] for r in records])) for level in records[0][name]}
    return result


@dataclass(frozen=True)
class CI:
    lo: float
    hi: float


def block_bootstrap(diff_by_date, reps=2000):
    """Resample entire service dates, keeping every paired case in a drawn block."""
    blocks = [np.atleast_1d(np.asarray(values, dtype=float)) for _, values in sorted(diff_by_date.items())]
    if (not blocks or not isinstance(reps, int) or reps <= 0 or
            any(b.ndim != 1 or not len(b) or not np.isfinite(b).all() for b in blocks)):
        raise ValueError("require nonempty finite date blocks and positive integer reps")
    totals = np.array([b.sum() for b in blocks])
    counts = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(0)
    means = np.empty(reps)
    for i in range(reps):
        draw = rng.integers(len(blocks), size=len(blocks))
        means[i] = totals[draw].sum()/counts[draw].sum()
    lo, hi = np.quantile(means, [.025, .975])
    return CI(float(lo), float(hi))
