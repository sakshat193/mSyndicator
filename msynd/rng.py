import hashlib
import json

from numba import njit
import numpy as np

RESAMPLE = 0xFFFFFFFF


@njit
def splitmix64(x):
    z = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit
def run_key(v_hash, run_id, replicate=0):
    return splitmix64(splitmix64(np.uint64(v_hash) ^ np.uint64(run_id)) ^ np.uint64(replicate))


@njit
def uniform(key, particle, event_idx, attempt):
    if particle < 0 or particle > 0xFFFFFFFF or event_idx < 0 or event_idx > 0xFFFFFF or attempt < 0 or attempt > 255:
        raise ValueError("RNG counter out of range")
    packed = (np.uint64(event_idx) << np.uint64(40)) | (np.uint64(attempt) << np.uint64(32)) | np.uint64(particle)
    bits = splitmix64(np.uint64(key) ^ packed) >> np.uint64(11)
    return min((bits + 0.5) * 2.0**-53, np.nextafter(1.0, 0.0))


def v_hash(versions):
    encoded = json.dumps(versions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return np.uint64(int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little"))
