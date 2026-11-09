# Plan: dashboard on the pilot's live data (rev 2, 2026-10-09)

Goal: open a local map showing the trains the G2 pilot is polling, with forecast streaks, using only data already
collected. No new MMT traffic, no writes to the pilot DB `msynd`, nothing leaves the PC.

## What exists (impl/v1 @ 3779a04)
- API (Task 34): `/api/live` WebSocket frames, `/api/run/{id}`, `/api/search`, `/api/static/network`.
- Engine: `python -m msynd.engine` → `main()` → `run_loop` (Task 24). The registry falls back to `provisional_models()`
  when `ml.model` is empty, so an empty view DB works at a cold start.
- Store and lifecycle (Tasks 20–21) turn a parsed MMT page into evidence + dirty runs.
- Web (Task 35): `frames.ts` (FrameStore, gap resync), `position.ts` (pos parity). There is no map yet (Task 36 is open).
- Pilot data: raw MMT bodies in `msynd.obs.fetch.body_zstd`, about 160 runs parse cleanly so far (dry G2, 09:30).

## Architecture
```
msynd (pilot, read-only) ──bridge──▶ msynd_view ──engine──▶ sim.frame ──API (127.0.0.1:8000)──▶ vite dev (5173)
```

## Steps
| # | Who | What |
|---|-----|------|
| 1 | monitor | Create `msynd_view` in the same container; `migrate.apply`; copy `ref.*` via `pg_dump -n ref msynd \| psql msynd_view` (read-only on msynd). |
| 2 | Codex (task F1) | `msynd/ops/pilot_bridge.py`: a read-only connection to the source DSN (`conn.read_only = True`) reads `ok` status rows after a cursor id, in `id` order. For each row: find or create the run (train/date → `obs.run`) → `parse_mmt.parse_status` → `store.store_status` into the target DSN. `--follow` re-reads every 60 s. The cursor is kept in the target DB. Tests: synthetic bodies, idempotent rerun, source opened read-only, no HTTP. |
| 3 | monitor | `python -m msynd.static.export` → `data/static`, served by the API. |
| 4 | monitor | Run the engine on `msynd_view` with **pool 2 and N = 2,000** (the PC was low on memory overnight; N = 10k × 4 workers is VM sizing). Needs only env or flags already in `run_loop`; if missing, a 2-line `main()` arg in F1. |
| 5 | Codex (task F2 = minimal Task 36) | `web/src/main.ts`, `layers.ts`: MapLibre with a plain dark background, network `PathLayer`, heads as a `ScatterplotLayer` at the P50 position via `pos()`, K streaks as a `TripsLayer`, and click → `/api/run/{id}` panel (P10/P50/P90 per remaining halt). The vite proxy sends `/api` to 127.0.0.1:8000. Tests: `timeline.test.ts` (LIVE offset, zone) as in Task 36. The full timeline, status bar and K fallback come later with the real Task 36. |
| 6 | monitor | Start the API with `uvicorn --host 127.0.0.1` and `npm --prefix web run dev`; run a Playwright smoke test (load, WS connects, a train is clickable) on localhost only. |

Effort: 2 Codex tasks (F1, F2), each about one session, and about 1 h of monitor steps.
Memory note: run the engine and Codex at different times on this PC, or with pool 1.

## Your decisions
- **A. Create `msynd_view` and read pilot bodies from `msynd` (read-only)?** The pilot is unaffected; the DB volume grows by about 2× the evidence size (small: 9 MB of raw bodies so far).
- **B. Basemap:** no basemap now (recommended; only the train network is drawn on a dark background), or download the India Protomaps PMTiles extract (about 1 GB from build.protomaps.com) for Task 36 proper.
- **C. Expectation:** "live" means only the pilot's cohort, about 16 trains at a time, refreshed about every 2 min with up to ~2 min of lag from the bridge. An all-India view needs the real collector, which can't run until after the pilot (16 Oct) and CP-3.
- **D. Plan amendment:** F1 (the bridge) is not in the plan; F2 is a cut-down Task 36. As monitor I'd add F1 as a housekeeping amendment and run F2 as "Task 36 part 1", leaving Task 36's boxes unticked until the full version.
