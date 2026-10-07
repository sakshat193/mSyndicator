# G8 (local prototype) — engine bench

**Status: measured, not a gate verdict.** G8 proper runs on the Oracle VM with all services up (CP-3 / Task 33).
Machine: user's Windows PC, 12 logical CPUs, 4 workers. Commit 2bf64bb.
Command: `python -m msynd.engine.bench --runs 100 --workers 4 --n 10000` (synthetic: 40 segments, 80 events,
20 evidence items alternating actual/pending, LightGBM-shaped masses). 100 runs instead of 2,500 to keep wall time short.

| Metric | §9.4 estimate | 1fdf20d (Task 19) | 2bf64bb (after hot-path fixes) |
|---|---|---|---|
| Replay p50 / p95 / p99 | 10–20 ms | 13.3 s / 38.6 s / 40.2 s | 1.79 s / 2.38 s / 2.43 s |
| Rebuild, 100 runs, 4 workers | — | 519 s | 40.9 s |
| Rebuild, 2,500 runs (projected ×25) | 10–15 s | ≈ 3.6 h | ≈ 17 min |
| Frame build / size | — | 0.11 s / 1.57 MB | 0.014 s / 1.57 MB |

## Profile (one run, N = 10k, single process)

| Evidence | 1fdf20d | 2bf64bb |
|---|---|---|
| none (forward draws only) | 2.17 s | 0.82 s |
| pending only | 1.6 s | 0.85 s |
| actual only | 39.65 s | 1.88 s |
| all | 13.4 s | 1.63 s |

All time is inside njit kernels. Fixed so far (no behaviour change; frozen-old comparison tests):
- `_raw_q` linear bin scans → `searchsorted`;
- direct left-limit CDF when w = 0; bracketed Illinois search when w > 0 (`_cdf` 53 µs → 4 µs);
- CDF/window computed once per particle and reused for PIT and sampling;
- event-only Eff parameters built once per event.

## Remaining bottleneck

Forward draws: ≈ 1 µs per particle-event (0.8 s for 80 events × 10k), against ≈ 25 ns implied by §9.4.
The §9.4 estimate assumed vectorised inversion; per-particle quantile lookup does not reach it.
Next options in §9.7 order: caching/batching (e.g. per-event quantile tables over u), smaller N via the §4.2 rule,
then a native port only with a comparative prototype. Decision deferred to G8 on the VM.

G7 re-run on 2bf64bb (two commits touched `dist.py`): 33 passed in 126 s.
