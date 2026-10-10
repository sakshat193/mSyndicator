# Phase 4 — replay-day end-to-end (Task 33)

**Status: replay-day e2e passes on the VM (run 2); Phase 4 exit still gated.** G8 with all services up and the 7-day live run need the collector (CP-3).
Day: pilot day 2026-10-09 (`data/replay_days/2026-10-09`, 9,800 `ok` bodies), fed at a requested 60× through
parsers → store → engine (4 workers, N = 10,000) on a fresh test DB. `uv run pytest -m slow tests/e2e`.

| Metric | PC run 1 (before d8a7fcf) | VM run 1 (ef469f1) | VM run 2 (4a6452b) |
|---|---|---|---|
| Fetches applied / invalid / no version | 9,800 / 0 / 0 | 9,800 / 0 / 0 | 9,800 / 0 / 0 |
| Runs / discovered / framed / unframed | 448 / 403 / 448 / 0 | 448 / 403 / 448 / 0 | 448 / 403 / 448 / 0 |
| A1 discovered, mean (min) | 0.928 (0.816) | 0.934 (0.840) | 0.934 (0.840) |
| A1 freshly observed, mean (min) | 0.170 (0.012) | 0.165 (0.015) | 0.166 (0.015) |
| A1 cold-start share | 0.999 | 0.999 | 0.999 |
| A6 p50 / p95 / max (s) | 2.53 / 5.33 / 36.5 | 2.29 / 4.67 / 18.3 | 2.27 / 4.58 / 18.2 |
| Replay errors | 942 | 28 | **0** |
| Forecast issues | 3,639 | 4,266 | 4,271 |
| Speed achieved | 19.1× | 23.4× | 23.7× |
| Wall: apply / periodic / engine (s) | 871 / 909 / 2,454 | 279 / 602 / 2,591 | 277 / 605 / 2,546 |

**Replay errors.**
- PC run 1: MMT pass times from another timetable fell outside their GTFS halts (31 of 213 pilot runs). Fixed in d8a7fcf and ef469f1.
- VM run 1: all 28 come from one run (12348), replayed 28 times. GTFS gives PNE dep and BHP arr the same minute (1044), and `_route` rejected zero scheduled running time.
  - 50 of 10,593 schedule versions have such a segment.
  - Fixed in b952a09: zero is allowed, and the kernel floors the scale at 0.1 min. VM run 2 passes the e2e with 0 errors (57 min).

**60× is not reachable at this scope, and live does not need it.** The engine alone takes 2,591 s of wall time for 81,120 simulated seconds (31×), before apply and periodic work.
- Live runs at 1×, where the same day costs ≈ 3 % of 4 workers. A6 here was measured under 23× compression, so it is pessimistic.
- The compression mostly hurts the harness, not latency. Apply §9.7 only if live G8 shows a bottleneck.

**A1 freshness is a pilot artefact.** The pilot polled a small rotating cohort at ≤ 0.3 req/s, so the low freshly-observed share is the pilot's polling budget, not the pipeline. `target_passes` is false for the same reason; A1 is judged in the 7-day live run.

**Still gated (🛑 Task 33):**
- G8 on the VM with collector, engine, api and caddy up.
- 7 days live, recording A1, A2, A6 and that `sim.forecast_issue` populates.
