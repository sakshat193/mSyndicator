# G8 (VM bench) — engine on the Oracle A1

**Status: measured, not the gate verdict.** G8 proper needs all services up and the evidence-commit-to-API latency (Task 33).
Machine: A1 4 OCPU / 24 GB (aarch64), image `msynd:local` (repo b5f9e73), 4 workers. 2026-10-10.
Command: `python -m msynd.engine.bench --runs 100 --workers 4 --n 10000` (same synthetic runs as `G8-local.md`).

| Metric | §9.4 estimate | PC (2bf64bb) | VM |
|---|---|---|---|
| Replay p50 / p95 / p99 | 10–20 ms | 1.79 / 2.38 / 2.43 s | 0.86 / 0.89 / 0.90 s |
| Rebuild, 100 runs | — | 40.9 s | 21.6 s (+12.1 s setup/JIT) |
| Rebuild, 2,500 runs (×25) | 10–15 s | ≈ 17 min | ≈ 9 min |
| Frame build / size | — | 0.014 s / 1.57 MB | 0.015 s / 1.57 MB |

Reading: steady state is fine at MMT rates. At most 0.2 status requests/s means ≤ 0.2 dirty runs/s, against
≈ 4.6 replays/s on 4 workers (≈ 5 % load). A cold full rebuild is minutes, not seconds: §9.4's estimate is about 50× low.
Startup can serve the last frame while rebuilding. Apply §9.7 only if Task 33's measured latency needs it.
