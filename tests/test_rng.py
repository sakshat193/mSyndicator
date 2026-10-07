import numpy as np

from msynd.rng import splitmix64, uniform, run_key, v_hash


def test_known_answer():
    assert splitmix64(np.uint64(0)) == 0xE220A8397B1DCDAF


def test_uniform_open_interval_determinism_and_batch_independence():
    key = run_key(v_hash({"n": 10000, "params_hash": "abc"}), 17)
    full = np.array([uniform(key, i, 3, 0) for i in range(1000)])
    halves = np.concatenate([np.array([uniform(key, i, 3, 0) for i in range(start, start + 500)]) for start in (0, 500)])
    assert np.array_equal(full, halves) and np.all((full > 0) & (full < 1))
    assert run_key(np.uint64(123), 17, 0) != run_key(np.uint64(123), 17, 1)
    assert v_hash({"b": 2, "a": 1}) == v_hash({"a": 1, "b": 2})


def test_chi_square_uniformity():
    from scipy.stats import chisquare

    samples = np.array([uniform(np.uint64(1234567), i, 7, 0) for i in range(100000)])
    assert chisquare(np.histogram(samples, bins=20, range=(0, 1))[0]).pvalue > 1e-4
