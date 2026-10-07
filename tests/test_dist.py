import numpy as np
import pytest

from msynd.dist import (Eff, SIGNED_NB, NONNEG_NB, LEVELS, QUANTILES, from_quantiles,
                        level_weight, eff_quantile, eff_cdf_left, eff_window_mass, eff_sample_window)


def eff(masses, signed=True, **kwargs):
    return Eff(masses, signed, kwargs.pop("tau_lo", 4), kwargs.pop("tau_hi", 7), 0, 0, 0, 1, kwargs.pop("f", -np.inf), kwargs.pop("c", 0), 1, **kwargs)


def test_rare_window_conditional_uniform():
    m = np.zeros(SIGNED_NB); m[61] = 5e-7; m[111] = 1-5e-7
    e = eff(m)
    assert eff_window_mass(e, 0, 1) == pytest.approx(5e-7, rel=1e-6)
    xs = np.array([eff_sample_window(e, 0, 1, u) for u in np.random.default_rng(7).random(100000)])
    assert xs.min() >= 0 and xs.max() < 1 and abs(xs.mean()-.5) < .005


def test_floor_atom_left_limit():
    m = np.zeros(NONNEG_NB); m[:10] = .1
    e = eff(m, signed=False, f=5)
    assert eff_window_mass(e, 5, 6) == pytest.approx(.6)
    assert eff_window_mass(e, 4, 5) == 0
    assert eff_window_mass(e, 6, 7) == pytest.approx(.1)
    xs = np.array([eff_sample_window(e, 5, 6, u) for u in np.random.default_rng(7).random(60000)])
    assert abs(np.mean(xs == 5)-5/6) < .01


def test_cdf_quantile_consistency():
    rng = np.random.default_rng(19)
    us = (np.arange(10000)+.5)/10000
    for _ in range(50):
        m = rng.dirichlet(np.ones(SIGNED_NB), size=2)
        e = Eff(m, True, 3, 9, 0, rng.random(), rng.normal()*50, rng.uniform(.1, 2), rng.uniform(-20, 50), rng.normal()*30, rng.uniform(.1, 2))
        samples = np.array([eff_quantile(e, u) for u in us])
        for x in np.linspace(samples[0], samples[-1], 20):
            f, s = eff_cdf_left(e, x)
            assert f+s == pytest.approx(1)
            assert abs(np.mean(samples < x)-f) <= .00011


def test_overflow_tail_mean_tau():
    m = np.zeros(SIGNED_NB); m[-1] = 1
    e = eff(m, tau_hi=7)
    xs = np.array([eff_quantile(e, u) for u in (np.arange(100000)+.5)/100000])
    assert np.all(np.isfinite(xs)) and np.mean(xs) == pytest.approx(247, abs=.1)


def test_underflow_tail_signed():
    m = np.zeros(SIGNED_NB); m[0] = 1
    e = eff(m, tau_lo=4)
    xs = np.array([eff_quantile(e, u) for u in (np.arange(100000)+.5)/100000])
    assert np.mean(xs) == pytest.approx(-64, abs=.1)


def test_lower_tail_does_not_extrapolate_across_sparse_support_gap():
    m = np.zeros(SIGNED_NB); m[0] = .5; m[71] = .5
    e = eff(m, tau_lo=4)
    for x in (-60, -55, 0, 10):
        assert eff_cdf_left(e, x) == pytest.approx((.5, .5))
    assert eff_window_mass(e, -55, 0) == 0


def test_all_underflow_has_bounded_upper_support_and_floor_atom():
    m = np.zeros(SIGNED_NB); m[0] = 1
    assert eff_cdf_left(eff(m), -55) == pytest.approx((1, 0))
    e = eff(m, f=5)
    assert eff_window_mass(e, 5, 6) == pytest.approx(1)
    assert eff_sample_window(e, 5, 6, .7) == 5


def test_endpoint_interpolation_ignores_zero_weight_infinite_tail():
    m = np.zeros((2, SIGNED_NB)); m[0, 0] = 1; m[1, -1] = 1
    e = Eff(m, True, 4, 7, 0, 1, 0, 1, -np.inf, 0, 1)
    assert eff_quantile(e, 0) == 240
    assert eff_cdf_left(e, 239) == (0, 1)


def test_lgbm_roundtrip_half_bin():
    q = np.array([-20, -10, 0, 25, 15, 40, 60, 80, 100], dtype=float)
    m, lo, hi = from_quantiles(q, True)
    e = eff(m, tau_lo=lo, tau_hi=hi)
    assert np.max(np.abs(np.array([eff_quantile(e, u) for u in QUANTILES])-np.sort(q))) <= .5


def test_negative_origin_residual_supported():
    m = np.zeros(SIGNED_NB); m[50] = .5; m[51] = .5
    e = Eff(m, True, 4, 7, 0, 0, 100, 1, -np.inf, 60, 1)
    assert eff_window_mass(e, 150, 151) > 0
    assert eff_quantile(e, .5) == pytest.approx(150)


def test_level_interpolation_same_u():
    m = np.zeros((16, SIGNED_NB)); m[:, 61] = 1; m[4, 61] = 0; m[4, 71] = 1
    j, w = level_weight(7.5)
    e = Eff(m, True, 4, 7, j, w, 0, 1, -np.inf, 0, 1)
    assert eff_quantile(e, .25) == pytest.approx(5.25)
    assert level_weight(700) == (15, 0)
    assert level_weight(-500) == (0, 0)


def test_extreme_level_and_tails_finite():
    m = np.zeros(SIGNED_NB); m[-1] = 1
    e = eff(m, tau_hi=7)
    lo = 240 - 7*np.log(1e-12)
    expected = 1e-12 * -np.expm1(-1/7)
    assert eff_window_mass(e, lo, lo+1) == pytest.approx(expected, rel=1e-6, abs=0)
    assert np.isfinite(eff_sample_window(e, lo, lo+1, .9))
