"""Frozen pre-speedup probability implementation: independent equivalence oracle."""
import math

from numba import njit
import numpy as np

from msynd.dist import Eff, _cdf, _raw_q, _sample


@njit
def _old_raw_q(m, below, above, signed, tau_lo, tau_hi, p, upper):
    low, high = (-60.0, 240.0) if signed else (0.0, 180.0)
    offset = 1 if signed else 0
    if upper:
        if m[-1] > 0 and p <= m[-1]:
            return high + tau_hi*(math.log(m[-1])-math.log(p)) if p > 0 else math.inf
        for i in range(len(m)-2, offset-1, -1):
            if m[i] > 0 and p < above[i]:
                return low+(i-offset)+1-(p-above[i+1])/m[i]
        u = 1-p
        return low+tau_lo*(math.log(u)-math.log(m[0])) if signed and u > 0 else -math.inf
    if signed and m[0] > 0 and p < m[0]:
        return low+tau_lo*(math.log(p)-math.log(m[0])) if p > 0 else -math.inf
    for i in range(offset, len(m)-1):
        if m[i] > 0 and p < below[i]:
            previous = below[i-1] if i else 0.0
            return low+(i-offset)+(p-previous)/m[i]
    v = 1-p
    return high+tau_hi*(math.log(m[-1])-math.log(v)) if m[-1] > 0 and v > 0 else math.inf


@njit
def _old_raw_interp(e, p, upper):
    m, below, above, signed, lo, hi, j, w, b, a, f, c, s = e
    q = _old_raw_q(m[j],below[j],above[j],signed,lo,hi,p,upper)
    if w == 0:
        return q
    return (1-w)*q+w*_old_raw_q(m[j+1],below[j+1],above[j+1],signed,lo,hi,p,upper)


@njit
def _old_cdf(e, x):
    if e[9] == 0:
        return (0.0, 1.0) if x <= e[8] else (1.0, 0.0)
    if x <= e[8]+e[9]*e[10]:
        return 0.0, 1.0
    y = (x-e[8])/e[9]
    if y <= e[10]:
        return 0.0, 1.0
    target = (y-e[11])/e[12]
    if target <= _old_raw_interp(e,0.0,False):
        return 0.0, 1.0
    if target >= _old_raw_interp(e,0.0,True):
        return 1.0, 0.0
    m, below, above, signed, lo, hi, j, w, b, a, f, c, s = e
    other = j+1 if w > 0 else j
    upper_mass = min(m[j,-1], m[other,-1])
    if upper_mass > 0 and target >= _old_raw_interp(e,upper_mass,True):
        anchor = (240.0 if signed else 180.0)+hi*((1-w)*math.log(m[j,-1])+w*math.log(m[other,-1]))
        survival = math.exp((anchor-target)/hi)
        return 1-survival, survival
    lower_mass = min(m[j,0],m[other,0]) if signed else 0.0
    if lower_mass > 0:
        anchor = -60-lo*((1-w)*math.log(m[j,0])+w*math.log(m[other,0]))
        if target <= anchor+lo*math.log(lower_mass):
            cdf = math.exp((target-anchor)/lo)
            return cdf, 1-cdf
    upper = target > _old_raw_interp(e,.5,False)
    left, right = 0.0, .5
    for _ in range(60):
        mid = (left+right)/2
        q = _old_raw_interp(e,mid,upper)
        if (q < target) != upper:
            left = mid
        else:
            right = mid
    p = (left+right)/2
    return (1-p,p) if upper else (p,1-p)


def test_random_eff_matches_frozen_cdf_and_raw_quantile():
    rng = np.random.default_rng(19)
    for number in range(200):
        signed = number % 3 != 0
        m = rng.exponential(size=(2, 302 if signed else 181))
        m[rng.random(m.shape) < .7] = 0
        if number % 4 == 0:
            m[:, [0, -1]] = 0
        m /= m.sum(axis=1, keepdims=True)
        e = Eff(m, signed, *rng.uniform(.5, 20, 2), 0,
                0 if number % 2 == 0 else rng.uniform(.01, .99),
                rng.uniform(-100, 100), 0 if number % 19 == 0 else rng.uniform(.1, 2),
                -math.inf if number % 5 == 0 else rng.uniform(0, 40),
                rng.uniform(-20, 30), rng.uniform(.1, 3))
        args = e.args
        for level in range(2):
            for upper in (False, True):
                cumulative = args[2 if upper else 1][level]
                probabilities = [0, 1, .5, 1e-13, 1-1e-13, *rng.random(5),
                                 *cumulative[rng.integers(0, len(cumulative), 5)]]
                for p in probabilities:
                    p = min(1., float(p))
                    raw_args = (m[level], args[1][level], args[2][level], signed,
                                e.tau_lo, e.tau_hi, p, upper)
                    np.testing.assert_allclose(_raw_q(*raw_args), _old_raw_q(*raw_args),
                                               atol=1e-9, rtol=0, err_msg=f"Eff {number}")
        targets = [e.b, e.b+e.a*e.f, -math.inf, math.inf]
        low, high = (-60, 240) if signed else (0, 180)
        targets += [e.b+e.a*max(e.f, e.c+e.s*x)
                    for x in (low-20, low, low+1, high-1, high, high+20, *rng.uniform(low, high, 8))]
        for x in targets:
            if not math.isnan(x):
                np.testing.assert_allclose(_cdf(args, x), _old_cdf(args, x),
                                           atol=1e-9, rtol=0, err_msg=f"Eff {number}, x={x}")


def test_sample_uses_precomputed_window_cdf_and_mass():
    m = np.zeros(181)
    m[:10] = .1
    e = Eff(m, False, 1, 1, 0, 0, 0, 1, 5, 0, 1)
    assert _sample(e.args, 5, 6, .7, (0., 1.), .6) == 5
    np.testing.assert_allclose(_sample(e.args, 5, 6, .95, (0., 1.), .6), 5.7, atol=1e-12)
