# m-syndicator v1 — Live All-India Train Tracker: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution in this repo:** Codex implements exactly one task per `codex exec` session. Claude (the monitor) launches each session, reviews and verifies it, runs every live or gated step itself, and ticks the boxes. See "How this plan runs".

**Goal:** Build the v1 tracker described in spec v2.2: collector → evidence DB → block-wise importance-resampling replay engine → frame log → API → map dashboard. Every Phase 0 gate gets measured and recorded.

**Architecture:** One Python package `msynd` with entrypoints `collect`, `engine`, `api.app` and `ops.*`. It runs on Postgres 16 under Docker Compose on an Oracle A1 VM, behind Caddy. The dashboard is Vite + TypeScript. Evidence is stored as append-only revisions with `received_at`. Each dirty run is replayed in full from its evidence, using a counter-based RNG. Frames are kept in a durable log ordered by `gen`.

**Tech stack:** Python 3.12 via uv, psycopg 3, httpx, selectolax, zstandard, numpy, numba, lightgbm, onnxruntime, pyarrow, fastapi, uvicorn, msgpack; Postgres 16; Caddy 2; Vite, TypeScript, MapLibre, deck.gl, PMTiles, vitest.

**Spec:** `docs/superpowers/specs/2026-10-07-live-train-tracker-design.md`, which Task 0 brings to v2.2. Reviews are in `docs/superpowers/reviews/`; v2.1-review findings are cited as R3-n.

---

## How this plan runs (Codex implements, Claude monitors)

**Prerequisites (monitor, once):** Docker Desktop must be running. Start the test database with `docker compose up -d db`, which listens on `localhost:55432`. Work on branch `impl/v1`.

**Per task loop (monitor):**
1. Write `.codex-runs/prompt-NN.txt` (gitignored) from the template below and launch Codex in the background from Git Bash (the Bash tool, `run_in_background`):
   `codex exec -C "B:\m-syndicator" -s workspace-write -c sandbox_workspace_write.network_access=true -o .codex-runs/task-NN.md - < .codex-runs/prompt-NN.txt`
   If the Windows sandbox blocks uv or Docker, stop and ask the user before relaxing it.
2. When it finishes, read `.codex-runs/task-NN.md`. If it starts with `BLOCKED:`, resolve the blocker (or ask the user), then use `codex exec resume --last "<answer>"`.
3. **Review gate** (all must hold, else send a numbered fix list via `codex exec resume --last`):
   - exactly one new commit `task NN: <title>`, touching only the task's files plus `uv.lock`/`package-lock.json`;
   - `uv run pytest -q` is green; for web tasks, `npm --prefix web test` is green too;
   - the tests pin the cases the task lists. On correctness-core tasks (12–18, 20, 29, 31), flip one critical line by hand, confirm a test fails, then revert;
   - minimal code: no speculative layers, and deliberate ceilings carry a `# ponytail:` comment;
   - no real HTTP path to NTES or etrain bypasses the `MSYND_LIVE` guard (Task 3).
4. Tick the task's boxes in this plan and commit the plan update.

**Codex prompt template:**
```
Implement Task NN of docs/superpowers/plans/2026-10-07-live-train-tracker.md.
Read AGENTS.md, then the plan's "Global Constraints", "Contract amendments (v2.2)" and Task NN,
then the spec sections Task NN cites. Implement ONLY Task NN, tests first, following its steps.
Run the task's verify commands and `uv run pytest -q`. Commit as `task NN: <title>`.
Never edit the spec or the plan. Never send HTTP to enquiry.indianrail.gov.in, etrain.info, api.railradar.in, makemytrip.com or goibibo.com hosts.
If a step needs a decision the plan does not make, stop and start your final message with "BLOCKED:".
```

**🛑 Checkpoints** are steps that need the user (accounts, legal or data-use decisions, starting live traffic, hardware measurements) or wall-clock time. At each one the monitor stops and asks the user. Codex never runs a 🛑 step or anything that needs `MSYND_LIVE=1`.

---

## Global Constraints

These are copied from the spec; every task implicitly includes them.

- Python 3.12, with dependencies locked by `uv`. Runtime dependencies: `httpx selectolax zstandard psycopg[binary,pool] numpy numba lightgbm onnxruntime pyarrow fastapi uvicorn websockets msgpack`. Import: the `osmium-tool` binary. (The `kaggle oci-cli` and `networkx` extras were removed unused in Task 101.) Dev: `pytest`. **No new runtime dependency without a plan change.**
- Postgres 16. Timestamps are UTC `timestamptz` and are displayed in IST. Offsets are run-clock minutes (float), counted from 00:00 IST on the origin start date. Migrations are numbered SQL files applied by `msynd/db/migrate.py`.
- Deliberately absent: PostGIS, Redis, Kafka, persisted particle state, Rust.
- HTTP: one Dispatcher per host. Request **starts** are spaced ≥ 1/rate apart, with no burst and one request in flight. Tokens, session pages and retries all count. `MMT_RATE=0.2` (live source, spec §18 2026-10-08); `NTES_RATE`/`ETRAIN_RATE` are unused (not approved). Changing a rate is the user's decision. The user agent is `m-syndicator/<ver> (personal research)`.
- `V_MAX = 160` km/h. Every prediction distance is timetable km. The physical floor is `60 × km / V_MAX` minutes.
- Event keys are `(run_id, station, occ, kind)`, never a sequence number. Forecasts address halts by `schedule_stop.idx`. The canonical order at one station is `arr < pass < dep`.
- As-of rule: anything computed as of *t* uses only revisions with `received_at ≤ t`. An expected time never becomes an actual.
- Delay levels are `[-30,-10,0,5,10,15,20,30,45,60,90,120,180,240,360,600]`. Quantiles are P5, P10, P25, P50, P75, P90, P95. Defaults: N = 10,000; K = 32 display paths. Surprise means PIT < 0.001 or > 0.999.
- The engine runs Numba `@njit` kernels first. Any native port must wait for a measured G8 bottleneck.
- Frames are msgpack plus permessage-deflate. Each display-path event is `uint16 route_idx` + `uint16` time in 0.1 min from the frame base.
- Dashboard: Vite + TypeScript, MapLibre, deck.gl `PathLayer`/`TripsLayer`/`ScatterplotLayer`, and a self-hosted Protomaps PMTiles extract of India in a dark style.
- Personal use only. Raw source data never leaves the VM or the user's PC unless G1 allows it.
- **Live-traffic guard:** the real httpx transport for any source (MMT, RailRadar) is built only when `MSYND_LIVE=1`. Tests use `httpx.MockTransport`.

## Contract amendments (v2.2)

These resolve the v2.1 review. Task 0 writes them into the spec. Every task treats them as overriding the v2.1 text.

- **A1 Signed O (R3-2).** There are two bin formats.
  - `SIGNED`: edges −60…240, i.e. underflow + 300 unit bins + overflow = **302 bins**. Used by **R and O**.
  - `NONNEG`: edges 0…180, i.e. 180 unit bins + overflow = **181 bins**. Used by **W** only.
  - Both `SIGNED` tails are exponential: underflow → `−60 − τ_lo·E`, overflow → `240 + τ_hi·E`, with E ~ Exp(1). This replaces "underflow returns the lower edge" for R as well; the physical floor still applies afterwards.
  - `ml.model.tail` is stored as `{"<class>": {"lo": τ_lo, "hi": τ_hi}}`.
  - LightGBM tails: `s_hi = (q.98 − q.90)/ln 5` anchored at q.98, and `s_lo = (q.10 − q.02)/ln 5` anchored at q.02, each floored at 0.5 min.
  - The neural heads are 302-way for R and O, and 181-way for W.
- **A2 Effective event-time distribution (R3-1).** Every drawn event time has the form `T = b + a·max(f, c + s·Q_d(u))`, where:
  - `b` is the particle's base time;
  - `a, s > 0` are pass scalings, and 1 otherwise;
  - `f` is the floor: `60·km/V_MAX` for running, 0 for dwell, −∞ for the origin;
  - `c` is the shift: scheduled minutes + bias (R), 0 (W), `resched_min` (O);
  - `Q_d(u) = (1−w)·Q_j(u) + w·Q_{j+1}(u)` interpolates quantiles between delay levels.

  `G(x) = P(T < x)` is the matching **left-limit CDF**. Computing it inverts `Q_d` by bisection, in the CDF coordinate below the median and in the survival coordinate above it. In the joint exponential tails it uses the closed form.

  Truncation to a window `[lo, hi)`:
  - mass `W = G(hi) − G(lo)`, computed as `S(lo) − S(hi)` in the upper half (never as `1 − F`);
  - draw `T = Q_eff(G(lo) + u·W)`, clamped below `hi`.

  Any `W > 0` is a valid window: the event is drawn conditionally and weighted by W. `W = 0` gives that particle zero weight. Only when **every** particle has zero weight does the flagged override apply (set T to the observation, raise `data_invariant`). The 1e-6 "impossible" threshold is deleted. Atoms follow from the left-limit definition: D ~ U[0,10) with a floor at 5 gives `P(T ∈ [5,6)) = 0.6`.
- **A3 Per-key ETA confirmation (R3-3).** Every `ok` status response appends one `obs.ntes_eta_rev` row for **every ETA present on the page**, whether changed or not.
  - B2 at *t* is the latest row for that key with `received_at ≤ t`, and it is eligible only if `t − received_at ≤ 30 min`.
  - A missing key gets no confirmation. Token or session fetches never write ETA rows. `obs.fetch.run_id` stays, but only for poll age.
  - `obs.ntes_eta_rev` is kept for **35 days**.
  - `sim.forecast_issue` gains `b2 real[]` (run clock, NaN = ineligible at issue), so A3 never needs old ETA rows. Phase 2 measures the row volume.
- **A4 G2 cohort and tail (R3-4).**
  - At least 40 **distinct** runs over the week, in rotating cohorts. The **simultaneous** cohort is `n_sim ≤ floor(0.7 · 0.3 · 120 / h)`, which is 12 at h = 2. The other 30 % of the budget goes to boards, exceptions, sessions and retries.
  - S fit: lags are interval-censored; runs unfinished at pilot end are right-censored, not "never published"; ε gets a 95 % Wilson CI.
  - A hard bound (S = 0 beyond some lag) requires an **independently documented** source guarantee. A pilot with zero counterexamples never qualifies.
  - Sensitivity: rerun G7(d) with ε at its CI upper bound and with the tail scale ×2.
- **A5 FK retention (R3-5).** Every FK to `obs.fetch` is `ON DELETE SET NULL`. There is no cascade into evidence. `obs.fetch` rows are kept 180 days.
- **A6 Evidence merge per event key, as of t (R3 additions).**
  - If any actual revision exists, use the latest one; pending markers for that key are ignored.
  - Otherwise use **one** marker: the latest pending marker across sources (ties go to `ntes_status`), with its source's S.
  - Repeated markers are never multiplied, and cross-source joint likelihood is not modelled (conservative). Witness: 0.731059, not 0.880797.
- **A7 Pass approximation acceptance.**
  - Prior: the pass time is the time-fraction point of the segment's full R draw, `b = t_dep(A)`, `a = sched_to_pass/sched_run`. The remainder after pass k is `T_P + max(f_rem, sched_rem + r·(bias + Q_d'(u)))`, with `r = sched_rem/sched_run` and d' = the delay at the pass (slot 3+k).
  - G7(c) compares this against a reference model with independent proportional sub-segment draws. Criterion: next-halt P10/P50/P90 error ≤ max(2 min, 10 % of the reference P10–P90 width), with MC error (spread across replicates) reported separately from approximation error (mean difference).
  - If it fails, pass events are excluded from conditioning (kept only as display anchors), and §18 records the decision.
- **A8 MC tolerance.** Use 20 independent replicates (a replicate index salted into the RNG key). Compare CDF values on ≥ 5 grid points and discrete marginals, plus quantiles for continuous cases. Tolerance = `4·sqrt(se_rep² + se_ref²)`, with `se_rep = sd/√20`.
  - The reference is exact enumeration where every condition is bin-aligned. Otherwise it is iid likelihood weighting over ≥ 2 M prior samples, with a computed standard error. This deliberately uses a different algorithm from the engine.
- **A9 Bounded dirty consume with acknowledge-after-publish.** `obs.dirty_run` gains `ver bigint` from sequence `obs.dirty_seq`. Every upsert sets a new `ver`.
  - The engine reads `run_id, ver … ORDER BY since LIMIT 200` and replays them. Then, in **one** transaction, it inserts the delta frame, deletes only the rows whose `(run_id, ver)` still match, and sends `NOTIFY sim_frame`.
  - Transient DB or IO failure: nothing is acknowledged, so the rows are retried.
  - A deterministic replay exception is acknowledged with run state `error` and metric `replay_error`.
- **A10 A6 endpoint.** A6 runs from the evidence commit until the API process has read the delta frame containing it (server side, before the WebSocket send).
  - Test: idle, with NOTIFY working → < 2 s.
  - With NOTIFY disabled → ≤ 1 s engine poll + 5 s API poll + replay time.
- **A11 Small additions.**
  - `obs.fetch.outcome` adds `inflight`: the row is inserted at request start and updated on completion. On startup, a dispatcher waits until the last logged start + 1/rate.
  - V adds `params_hash`, covering the S per source, V_MAX and ESS threshold in `data/params.json`.
  - Particle state is an `N × E` float32 event-time matrix that is gathered on resample, replacing the `anc` field (`# ponytail:` upgrade path: ancestor indices if G8 flags the copies).
  - `part_cancelled`: a cancelled tail ends the simulation at the first cancelled halt. A cancelled head starts at the first operated halt, with O applied to its scheduled departure. A cancelled middle is treated as a diversion.

## Review Focus

These are inputs the spec implies but no gate exercises directly. Each has a pinned test in the owning task.

1. **Wrong-run page:** NTES answers with a different train or start date than requested (for example the previous day's instance). The outcome must be `invalid` and nothing is stored. → Task 10 `test_status_for_other_start_date_is_invalid`.
2. **Midnight, multi-day and IST date boundaries:** IST page times past midnight, runs > 24 h, and an origin date that differs between IST and UTC (18:30–24:00 UTC). Expected: run-clock times are monotone and the start date is the IST date. → Task 4 `test_gtfs_times_past_24h`, Task 10 `test_multiday_offsets_monotone`, Task 21 `test_ist_start_date_at_utc_evening`.
3. **Repeated stations (occ = 2):** a route visits a station twice. Expected: events map to the right occurrence and keys never collide. → Task 10 `test_repeated_station_occ`, Task 20 `test_repeated_station_keys`.
4. **Late, out-of-order and corrected revisions:** an event received after later events, or a correction to an earlier time. Expected: the engine output equals a fresh replay on the same evidence. → Task 16 `test_late_event_equals_fresh_assembly`, Task 31 `test_late_event_frame_equals_fresh_replay`.
5. **Extreme and early delays:** delays > 600 min or > 24 h, and negative delays or residuals. Expected: kept, flagged, sampled through tails, never NaN. → Task 10 `test_delay_over_24h_kept_flagged`, Task 14 `test_extreme_level_and_tails_finite`, Task 17 `test_extreme_delay_replay_no_nan`.

---

## File map

```
AGENTS.md  pyproject.toml  uv.lock  compose.yaml  Dockerfile  Caddyfile  .env.example  .gitignore
data/params.json  data/segment_overrides.csv  data/holidays.csv  data/static_sources.json
docs/sources.md  docs/gates/G{1..8}.md
msynd/__init__.py              VERSION
msynd/config.py                env settings (os.environ + defaults)
msynd/db/__init__.py           connect(), listen helpers
msynd/db/migrate.py            msynd/db/migrations/0001_init.sql (+ later numbered files)
msynd/db/dirty.py              mark_dirty(conn, run_ids)  ← the only writer of obs.dirty_run
msynd/labels.py  msynd/rng.py  msynd/dist.py  msynd/pos.py
msynd/static/gtfs.py  msynd/static/segments.py  msynd/static/export.py  msynd/static/osm.py
msynd/collect/__main__.py  dispatcher.py  clients.py  capture.py  parse_ntes.py  parse_etrain.py
msynd/collect/validate.py  store.py  lifecycle.py  admission.py  pilot.py
msynd/features/__init__.py (FEATURE_VER)  msynd/features/build.py
msynd/models/empirical.py  lgbm.py  registry.py
msynd/eval/harness.py  baselines.py  metrics.py
msynd/engine/__main__.py  evidence.py  kernel.py  replay.py  cache.py  frames.py  priority.py  bench.py
msynd/api/app.py
msynd/ops/report.py  nightly.py  weekly.py  export.py
tests/... (mirrors package; tests/g7/, tests/e2e/, tests/fixtures/{ntes,etrain,pos_cases.json})
web/ (Vite + TS: src/main.ts, src/position.ts, src/frames.ts, src/timeline.ts, src/layers.ts, src/panel.ts)
kaggle/kernel.py (Phase 6 only)
```

## Milestones and checkpoints

| Milestone | Tasks | Ends with |
|---|---|---|
| M0 Foundations | 0–5 | Schema, dispatcher and static network, all offline |
| M1 Source access | 6–11 | 🛑 CP-1 (G1: MMT, personal gathering) → 🛑 discovery → fixtures (G3) → parser → 🛑 CP-2 (7-day G2 pilot at 0.2 req/s) |
| M2 Correctness core | 12–19 | 🛑 G7 gate passes. Runs during the pilot |
| M3 Collector | 20–25 | 🛑 CP-3: G2 decisions, G5 VM, 7-day Phase 2 live run, Phase 2 exit |
| M4 Models and eval | 26–30 | Phase 3 exit once there are ≥ 4 weeks of live data; G4 is gated |
| M5 Engine service | 31–33 | 🛑 G8 on the VM, Phase 4 exit |
| M6 Dashboard | 34–37 | 🛑 A7 on the user's laptop, Phase 5 exit |
| M7 Ops, GPU, acceptance | 38–40 | Gated on G1/G6; ongoing |

---

## M0 — Foundations

### Task 0 (monitor): Housekeeping and spec v2.2

**Files:** spec, `docs/superpowers/plans/2026-10-07-live-train-tracker.md`, the untracked review files.

- [x] `git switch -c impl/v1`. Commit the two untracked v2.1 review files.
- [x] Copy this plan to `docs/superpowers/plans/2026-10-07-live-train-tracker.md`.
- [x] Edit the spec to v2.2. Apply A1–A11 in their sections (§3.2, §3.3, §3.4, §4.1, §6, §9.2, §9.5, §12, §15 G2/G7) and add an "R3" table to §19 mapping R3-1…5 and the additions to A1–A11.
- [x] Run `python docs/superpowers/reviews/2026-10-07-live-train-tracker-v2.1-checks.py`; expect 11 checks to pass.
- [x] Commit: `Revise spec to v2.2 after third review; add implementation plan`.

### Task 1: Project skeleton, test DB harness, AGENTS.md

**Files:** create `pyproject.toml`, `.gitignore`, `.env.example`, `compose.yaml` (only the `db` service for now), `AGENTS.md`, `msynd/__init__.py`, `msynd/config.py`, `msynd/db/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`.

**Interfaces:**
- Produces `msynd.config.settings`: a frozen dataclass read from env with fields `database_url, ntes_rate=1.0, etrain_rate=0.2, live=False, n_particles=10000, k_paths=32, f_minutes=60`.
- Produces `msynd.db.connect(url=None) -> psycopg.Connection` and the pytest fixtures `db` (one connection per test, inside a transaction that is rolled back) and `db_url` (a fresh database per session, migrated).

Steps:
- [x] `pyproject.toml`: project `msynd`, `requires-python = ">=3.12,<3.13"`, the runtime dependencies from Global Constraints, optional groups `ops` and `import`, dev `pytest`, and pytest markers `db`, `slow`, `live`. Run `uv python install 3.12 && uv sync`.
- [x] `.gitignore`: `.venv/ __pycache__/ .codex-runs/ web/node_modules/ web/dist/ data/replay_days/ *.pmtiles .env`.
- [x] `compose.yaml`: service `db` from `postgres:16` with env `POSTGRES_USER/PASSWORD/DB=msynd`, port `55432:5432`, and a named volume.
- [x] `tests/conftest.py`:
  - `MSYND_TEST_DSN` defaults to `postgresql://msynd:msynd@localhost:55432/postgres`;
  - the session fixture creates the database `msynd_test_<pid>`, runs `migrate.apply(url)` (stubbed until Task 2) and drops it at the end;
  - the `db` fixture yields a connection inside a rolled-back transaction;
  - auto-skip `live` tests unless `MSYND_LIVE=1`.
- [x] `AGENTS.md` (≤ 40 lines) states:
  - one task per session; tests first;
  - commands `uv run pytest -q`, `uv run pytest -q -m "not db"`, `npm --prefix web test`;
  - never set `MSYND_LIVE` and never contact NTES or etrain;
  - minimal code: stdlib or an installed dependency first, no single-implementation abstractions, a `# ponytail:` comment on deliberate ceilings;
  - never edit the spec or the plan;
  - commit as `task NN: title`;
  - stop with `BLOCKED:` on any undecided choice.
- [x] `tests/test_smoke.py`: `@pytest.mark.db def test_db(db): assert db.execute("select 1").fetchone() == (1,)`.
- [x] Verify: `docker compose up -d db; uv run pytest -q` passes.
- [x] Commit `task 01: project skeleton and test db harness`.

### Task 2: Schema migration 0001 (spec §6 + A3, A5, A9, A11)

**Files:** create `msynd/db/migrate.py`, `msynd/db/migrations/0001_init.sql`, `msynd/db/dirty.py`, `tests/db/test_schema.py`, `tests/helpers.py`.

**Interfaces:**
- `migrate.apply(url)` applies every unapplied `NNNN_*.sql` in order, each in its own transaction, and records it in `public.schema_migrations(version text primary key, applied_at timestamptz)`.
- `dirty.mark_dirty(conn, run_ids: list[int]) -> None` runs `INSERT … ON CONFLICT (run_id) DO UPDATE SET ver = nextval('obs.dirty_seq'), since = LEAST(obs.dirty_run.since, excluded.since)` followed by `NOTIFY dirty`. It must be called inside the caller's evidence transaction.
- `tests/helpers.py`: `make_run(db, train='12951', start='2026-10-07', halts=[('MMCT',None,0,0),('BRC',300,305,392),('NDLS',940,None,1384)]) -> run_id`, `insert_fetch(db, **kw) -> id`, `insert_event(db, run_id, station, occ, kind, actual, received, fetch_id=None) -> id`.

Steps:
- [x] Write `0001_init.sql`: §6 verbatim, with these changes:
  - every `REFERENCES obs.fetch` becomes `REFERENCES obs.fetch ON DELETE SET NULL`;
  - `obs.fetch.outcome` gets `CHECK (outcome IN ('inflight','ok','empty','invalid','transport_error','challenge'))`;
  - add `CREATE SEQUENCE obs.dirty_seq` and `obs.dirty_run.ver bigint NOT NULL DEFAULT nextval('obs.dirty_seq')`;
  - add `sim.forecast_issue.b2 real[] NOT NULL` and `cold_start boolean NOT NULL DEFAULT false`;
  - add `ref.route_version.dist_src text NOT NULL CHECK (dist_src IN ('gtfs','gc','ntes'))`, which is part of the route content hash;
  - add comments on `ntes_eta_rev` (A3) and `ml.model.tail` (A1).
- [x] Write the tests first:

```python
pytestmark = pytest.mark.db

def test_event_asof_filters_before_latest(db):          # spec §14 "As-of query"
    r = make_run(db)
    insert_event(db, r, 'BRC', 1, 'arr', actual=ist('2026-10-07 09:50'), received=ist('2026-10-07 10:00'))
    insert_event(db, r, 'BRC', 1, 'arr', actual=ist('2026-10-07 09:55'), received=ist('2026-10-07 10:30'))
    asof = db.execute("select actual_ts from obs.event_asof(%s) where run_id=%s", (ist('2026-10-07 10:15'), r)).fetchone()[0]
    cur = db.execute("select actual_ts from obs.event where run_id=%s", (r,)).fetchone()[0]
    assert asof == ist('2026-10-07 09:50') and cur == ist('2026-10-07 09:55')

def test_fetch_retention_sets_null_never_cascades(db):   # A5
    f = insert_fetch(db, started_at=utcnow() - timedelta(days=181), outcome='ok')
    e = insert_event(db, make_run(db), 'BRC', 1, 'arr', actual=utcnow(), received=utcnow(), fetch_id=f)
    db.execute("delete from obs.fetch where id=%s", (f,))
    assert db.execute("select fetch_id from obs.event_rev where id=%s", (e,)).fetchone() == (None,)

def test_mark_dirty_bumps_ver(db):                        # A9
    r = make_run(db); mark_dirty(db, [r]); v1 = ver(db, r); mark_dirty(db, [r])
    assert ver(db, r) > v1

def test_event_kind_order(db):
    assert db.execute("select 'arr'::obs.event_kind < 'pass' and 'pass'::obs.event_kind < 'dep'").fetchone()[0]

def test_one_active_model_per_role(db): ...  # second active row for role 'run' raises UniqueViolation
def test_outcome_check(db): ...             # outcome 'weird' raises CheckViolation
```
- [x] Run them; expect failures. Implement. Run them again; expect a pass.
- [x] Commit `task 02: schema migration 0001`.

### Task 3: Dispatcher, the single HTTP owner per host (spec §7.1, A11)

**Files:** create `msynd/collect/dispatcher.py`, `tests/collect/test_dispatcher.py`.

**Interfaces (produces):**
```python
class HostLocked(Exception): ...
class HostStopped(Exception): ...          # challenge seen; manual restart required
@dataclass
class FetchResult: fetch_id: int; outcome: str; status: int | None; body: bytes; received_at: datetime
class Dispatcher:
    def __init__(self, host: str, rate: float, conn_factory, transport: httpx.AsyncBaseTransport,
                 clock: Clock = RealClock(), rand: random.Random | None = None,
                 classify: Callable[[str, httpx.Response], str] | None = None,   # purpose,resp -> ok|invalid|challenge
                 on_empty: Callable[[], Awaitable[None]] | None = None): ...
    async def start(self) -> None          # pg_try_advisory_lock(hashtext('http:'||host)) on a held connection, else HostLocked
    async def request(self, purpose: str, method: str, url: str, *, key: str | None = None,
                      run_id: int | None = None, **httpx_kwargs) -> FetchResult
    state: str                             # 'ok' | 'breaker' | 'stopped'
def real_transport() -> httpx.AsyncHTTPTransport   # raises RuntimeError unless os.environ.get('MSYND_LIVE') == '1'
```

Rules (spec §7.1):
- One `asyncio.Lock` per host. The next start is `max(now, last_start + 1/rate)`.
- Insert an `obs.fetch` row with `outcome='inflight'` at start; update it on completion.
- `transport_error` covers timeouts, 5xx and 429. Retry with backoff `min(300, 2·2ⁿ)·U(0.5, 1.5)`, or `Retry-After` if larger, up to 5 attempts; each retry is a new spaced and logged request.
- `empty` (200 with a blank body): call `on_empty()` once, then retry once.
- `challenge`: state `stopped`, raise `HostStopped`.
- Breaker: when more than 20 % of ≥ 20 outcomes in a rolling 5 min are non-`ok`, set state `breaker` and sleep 10 min.
- On `start()`, wait until `max(started_at) + 1/rate` from `obs.fetch` for this host.
- Set the UA header.

Steps:
- [x] Write the tests (fake clock: `monotonic()` and `async sleep(dt)` advance a counter; `ListLog` or the DB fixture records rows):

```python
async def test_starts_spaced_with_tokens_retries_sessions(db_url):
    clock = FakeClock()
    script = cycle([resp(200, b'tok'), resp(503, headers={'Retry-After': '3'}), resp(200, b''), resp(200, b'<ok/>')])
    d = Dispatcher('h', 1.0, conn_factory(db_url), MockTransport(lambda r: next(script)), clock=clock,
                   on_empty=noop_async)
    await d.start()
    for i in range(150): await d.request('status', 'GET', f'https://h/x?{i}')
    starts = fetch_started_monotonic(d)                      # from FakeClock stamps recorded per row
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(starts, starts[1:]))
    assert max_in_window(starts, 60.0) <= 60

async def test_second_process_cannot_lock(db_url): ...       # second Dispatcher('h').start() raises HostLocked
async def test_challenge_stops_host(db_url): ...             # classify -> 'challenge'; next request raises HostStopped; state=='stopped'
async def test_breaker_pauses_10_min(db_url): ...            # 20 invalid of 20 -> state 'breaker', next start >= +600 s
async def test_empty_refreshes_session_once_then_retries_once(db_url): ...  # on_empty called once; 2 rows: empty, ok
async def test_every_request_logged_with_outcome(db_url): ...  # no row left 'inflight' after completion
def test_real_transport_requires_live_flag(monkeypatch): ...   # RuntimeError without MSYND_LIVE=1
async def test_restart_respects_last_start(db_url): ...        # row started 0.2 s ago -> first start waits 0.8 s
```
- [x] Run them; expect failures. Implement. Run them again; expect a pass.
- [x] Commit `task 03: dispatcher`.

### Task 4: GTFS static import (spec §3.1, §7.6; Phase 1)

**Files:** create `msynd/static/gtfs.py`, `data/static_sources.json` (pinned GTFS commit SHA and datameet SHA), `data/holidays.csv`, `tests/static/test_gtfs.py`, `tests/fixtures/gtfs_mini/*.txt` (hand-written, 3 trains).

**Interfaces:**
- `python -m msynd.static.gtfs import [--zip PATH]` downloads the pinned zip, or reads `--zip`.
- `import_gtfs(conn, gtfs_dir) -> ImportReport(trains, versions_new, versions_dup, stations, missing_coords)`.
- Produces rows in `ref.station`, `ref.train`, `ref.schedule_version`, `ref.schedule_stop`, a halts-only `ref.route_version`/`ref.route_stop`, and `ref.calendar` for 2025-01-01..2027-12-31 (`fog_north` = Dec 1–Feb 15; holidays from the CSV).

Rules:
- **Distance:**
  - If `stop_times.txt` has `shape_dist_traveled`, use it (converted to km) with `dist_src='gtfs'`.
  - Otherwise, use the cumulative great-circle distance between stations with `dist_src='gc'`. It is a lower bound on track km, so the physical floor stays safe. Mark it `# ponytail:` with NTES-schedule km as the upgrade path.
  - When the schedule refresh (Task 22) parses NTES distances, the store creates a new route version with `dist_src='ntes'` (immutable versions; runs are rebound). The Phase 1 report prints the share of each `dist_src`.
- Times past `24:00:00` become run-clock minutes > 1440.
- `days_of_run`: bit0 = Mon … bit6 = Sun, from `calendar.txt` at the origin.
- Content hash: sha256 over the canonical halt tuples. Deduplicate via `UNIQUE (train, content_hash)`.
- `occ` counts repeats of a station within a version.
- Class comes from keywords in the train name: RAJDHANI→RAJ, SHATABDI→SHT, VANDE→VB, DURONTO→DRNT, SUPERFAST/SF→SF, MAIL→MEX, EXP→EXP, PASS/MEMU/DEMU→PASS, SPECIAL/SPL or a number starting with '0'→SPL, else UNKNOWN. Mark this `# ponytail:` keyword heuristic, with NTES class as the upgrade path.
- Station coordinates come from GTFS, with gaps filled from datameet `stations.json`.

Steps:
- [x] Tests on `gtfs_mini`:
  - `test_gc_distance_fallback` (fixture without `shape_dist_traveled` → `dist_src='gc'`, km is monotone and ≤ the hand-computed haversine sum + 1e-6);
  - `test_versions_dedup_by_hash` (importing twice → `versions_new == 0`);
  - `test_gtfs_times_past_24h` (a stop at `26:10:00` → `arr_off == 1570.0`);
  - `test_repeated_station_gets_occ2`;
  - `test_days_of_run_bits` (Mon+Thu → `0b0001001`);
  - `test_class_mapping`;
  - `test_route_version_is_halts_only_and_immutable`.
- [x] Implement; tests pass. Then the monitor runs the real import once and records the report counts in `docs/gates/phase1.md`.
- [x] Commit `task 04: gtfs static import`.

### Task 5: Segments, overrides, network export, Phase 1 report (spec §3.1, §7.6, §10)

**Files:** create `msynd/static/segments.py`, `msynd/static/export.py`, `data/segment_overrides.csv` (header only: `stn_a,stn_b,polyline,confidence`), `tests/static/test_segments.py`.

**Interfaces:**
- `build_segments(conn) -> int` creates an undirected `ref.segment (stn_lo < stn_hi)` for each consecutive route-stop pair (polyline NULL, confidence 0), plus `ref.route_segment(forward)`. It then applies the overrides CSV.
- `export_network(conn, out_dir) -> (path, etag)` writes `network-<sha12>.json.gz` containing stations, segments (polyline, confidence), route versions (stops, km, `sched_off`, halt flag) and route→segment membership.
- `python -m msynd.static.export report` prints the share of trains with a schedule version and drawable geometry (straight lines count), plus the confidence histogram.

Steps:
- [x] Tests:
  - `test_segment_undirected_shared_by_reverse_route`;
  - `test_route_segment_forward_flag`;
  - `test_override_applied_and_versioned_by_file_hash`;
  - `test_export_etag_stable_for_same_content`.
- [x] Implement; tests pass. Commit `task 05: segments and network export`.

---

## M1 — Source access

### Task 6: G1 data-use draft → 🛑 CP-1

**Files:** create `docs/sources.md`.

- [x] Codex (network allowed, terms pages only; no enquiry endpoints) records, for each source, the URL, retrieval date and quoted relevant terms: NTES (including the Disclaimer page), etrain (Tripozo terms), the GTFS repo licence, OSM ODbL, Protomaps and datameet. Each is mapped against four uses: automated collection, model training, export to Kaggle, portfolio display. Unknowns are marked "unclear", never assumed.
- [x] Commit `task 06: G1 data-use draft`.
- [x] **🛑 CP-1 (user):** decided 2026-10-08: live source MakeMyTrip RIS, personal gathering only at 0.2 req/s; NTES/etrain not approved; Kaggle export not approved (spec §18). Decide G1 per use and record it in spec §18. Explicitly approve:
  - (a) protocol discovery and fixture capture, ≈ 150 requests at ≤ 0.3 req/s;
  - (b) the 7-day G2 pilot at ≤ 0.3 req/s;
  - (c) whether etrain backfill and Kaggle export are allowed.

  Without (a) and (b), M1 stops and M2 continues alone.

### Task 7: Capture CLI → 🛑 MMT discovery

Live source since CP-1 is MakeMyTrip RIS (spec §18, 2026-10-08): `MMT_HOST = rails-ris.makemytrip.com`, `MMT_RATE = 0.2`.

**Files:** create `msynd/collect/capture.py`, `tests/collect/test_capture.py`.

**Interfaces:** `python -m msynd.collect.capture plan.json --out <dir> [--rate 0.2]`. `plan.json` is a list of `{name, purpose, method, url, json?}`, one Dispatcher per host. It writes `<name>.body` plus `manifest.json` (status, response headers minus `Set-Cookie`, sha256, received_at, outcome). The transport is `real_transport()`, so it runs only with `MSYND_LIVE=1`.

- [x] Tests with MockTransport:
  - the files and manifest are written;
  - `Set-Cookie` and any `auth`/`token`/`csrf`-like header values are redacted;
  - starts are spaced ≥ 1/rate;
  - without `MSYND_LIVE=1` the CLI exits before any request.
- [x] Implement; commit `task 07: capture cli`.
- [x] **🛑 Monitor (live; the user confirms first):** `MSYND_LIVE=1 python -m msynd.collect.capture discovery.json --out data/mmt_capture/discovery/` at 0.2 req/s, ≤ 10 requests:
  - the autosuggest for 12951;
  - livestatus for 12951 today and for one train not yet started.

  Record h (expected 1), required headers or cookies, and the response shape in `docs/gates/G3.md`. Raw bodies stay under the gitignored `data/mmt_capture/`.

### Task 8: Shared status types and MMT client (spec §2, §7.1)

**Files:** create `msynd/collect/status.py` and `msynd/collect/mmt.py`; modify `msynd/collect/railradar.py` (import the moved types), `msynd/collect/dispatcher.py` (non-2xx replies other than 429/5xx go to the source `classify` when one is given, else stay `invalid`; user ruling 2026-10-08) and `tests/collect/test_dispatcher.py` and `msynd/config.py` (`mmt_rate=0.2` from `MMT_RATE`); create `tests/collect/test_mmt.py`.

**Interfaces:**
```python
# status.py — moved unchanged from railradar.py, plus Eta and ParsedStatus.etas
class InvalidResponse(ValueError): ...
Event, Pending, PassingStation                      # as in railradar.py
@dataclass(frozen=True)
class Eta: station: str; occ: int; kind: str; eta_ts: datetime
ParsedStatus(..., flags, etas: tuple[Eta, ...] = ())
# mmt.py
HOST = "rails-ris.makemytrip.com"
class MmtClient:
    def __init__(self, d: Dispatcher): ...
    @classmethod
    def from_env(cls): ...                           # Dispatcher(HOST, settings.mmt_rate, ..., real_transport())
    async def train_running(self, train: str, start: date, run_id=None) -> FetchResult
        # one POST /api/ris/train/livestatus/v2, json {"trainNumber": train,
        #   "dateOfJourney": start.strftime("%d-%m-%Y"), "findNextRunningDate": False}
def classify(purpose, response) -> str               # ok | invalid | challenge (403, HTML body, captcha marker)
```
Any headers that discovery records as required are added here. Without the Task 7 discovery, use only `Content-Type: application/json` and the UA.

- [x] Tests:
  - exactly one POST per enquiry, with the exact JSON body;
  - `findNextRunningDate` is always false;
  - `from_env` raises without `MSYND_LIVE=1`;
  - `classify` maps 403 or an HTML body to `challenge` and malformed JSON to `invalid`;
  - a 403 through the Dispatcher with the MMT classifier stops the host; without a classifier a 404 stays `invalid`;
  - the RailRadar tests still pass after the type move.
- [x] Implement; commit `task 08: shared status types and mmt client`.

### Task 9: G3 fixture capture (monitor, live 🛑)

- [x] After the user confirms, capture ≤ 40 livestatus bodies at 0.2 req/s:
  - normal running; not started; completed;
  - cancelled, rescheduled and diverted (if found);
  - a multi-day train and a repeated-station train (from GTFS);
  - one passing through midnight IST; one with a delay > 3 h.
- [x] Copy the chosen bodies, trimmed to the fields the parser reads and labelled, into `tests/fixtures/mmt/`. Record in `docs/gates/G3.md` which edge cases were captured and which could not be found.

### Task 10: MMT parser and semantic validation (spec §7.1 Validation, §7.4, A3)

**Files:** create `msynd/collect/parse_mmt.py`, `tests/collect/test_parse_mmt.py`, `tests/fixtures/mmt/<name>.expected.json` (golden files; the monitor hand-checks the first version of each).

**Interfaces:** `parse_status(body: bytes, train: str, start: date) -> ParsedStatus` (from `status.py`); `validate` raises `InvalidResponse`.

Rules:
- An actual exists only where the station's `arrived`/`departed` flag is true. A time without its flag is an `Eta` (A3), never an actual. `distanceCovered`, `currentStation` and `totalLateMins` never become events.
- The train and journey date must match the request. A response for another date (for example `findNextRunningDate` behaviour) is invalid.
- Day offsets (day count) make run-clock times monotone. Occurrence numbers follow page order.
- A delay > 24 h is kept and flagged `delay_gt_24h`.
- Non-halt rows with a time become `PassingStation` (km from the response).
- `exceptions` is always empty: no MMT exception was ever observed (G3), so its fields are unmapped. If `cancelledFromStation` or `cancelledToStation` is non-null, keep the page and add flag `exception_unparsed` (`# ponytail:` unmapped; upgrade: map to the Task 21 `detail` form once a real cancelled body is captured). A reschedule shows only as a late actual origin departure (user, 2026-10-08).

- [x] Tests (one per fixture, comparing against its golden file), plus:
  - `test_status_for_other_start_date_is_invalid` (RF1);
  - `test_multiday_offsets_monotone` (RF2);
  - `test_repeated_station_occ` (RF3);
  - `test_delay_over_24h_kept_flagged` (RF5);
  - `test_unflagged_time_is_eta_never_actual`;
  - `test_empty_body_invalid`.
- [x] Implement; commit `task 10: mmt parser and validation`.

### Task 11: G2 pilot runner and analysis → 🛑 CP-2

**Files:** create `msynd/collect/pilot.py`, `tests/collect/test_pilot.py`, `docs/gates/G2.md` (written by the analysis).

**Interfaces:**
- `python -m msynd.collect.pilot run --days 7 --rate 0.2` stores everything in `obs.fetch` with bodies kept (no 24 h retention during the pilot). The scheduler:
  - discovers candidates from our own inventory (runs whose scheduled window covers now) — MMT has no boards;
  - keeps a rotating simultaneous cohort `n_sim = floor(0.7·rate·120/h)` (16 at h = 1);
  - polls each cohort run every 120 s from 20 min before to 60 min after each expected event;
  - targets ≥ 40 distinct runs and 50 trains for passing-station prevalence;
  - samples the active-run count each hour.
- `python -m msynd.collect.pilot analyse --out docs/gates/G2.md --params data/params.json` measures:
  - h and the empty/invalid/challenge rates;
  - S for `mmt_status`: lag = first `received_at` showing the actual − the actual time, interval-censored by the previous poll (midpoint, `# ponytail:` → Turnbull), an exponential tail fitted above P90, ε with a Wilson CI, unfinished runs right-censored;
    Censoring (monitor ruling, 2026-10-08): an event whose actual was seen gives an interval [previous poll without it, first poll with it] − actual. An event never seen as actual is known to have happened only if a later event of the same run has an actual at t_L. If the run terminated, it is an ε counterexample (never published). If the run is unfinished at pilot end, it is right-censored at lag ≥ last poll − t_L, because actual ≤ t_L. With no later actual, it is excluded (not known to have happened). ETAs are never truth. Retention: nothing schedules `prune_bodies` during the pilot, so `msynd/ops/nightly.py` is unchanged and the monitor runs no retention job on the pilot DB.
  - pending-after-actual counts;
  - ETA revision rate;
  - passing-station prevalence;
  - the active-run curve.

  It writes `data/params.json` as `{"S": {"mmt_status": {...}}, "minute": "trunc", "provisional": false}`. Before G2, the file is committed with `provisional: true`: lag P50 10 min, exponential, ε = 0.05.

- [x] Tests:
  - `test_cohort_bound` (h = 1, rate 0.2 → 16);
  - `test_s_fit_recovers_synthetic` (synthetic lags with a known exponential tail and ε = 0.03, polled every 120 s → the fitted median is within 1 min and the ε CI contains 0.03);
  - `test_right_censored_not_counted_unpublished`;
  - `test_never_hard_bound` (zero counterexamples → the output still has ε > 0 and a soft tail).
- [x] Implement; commit `task 11: g2 pilot`.
- [x ] **🛑 CP-2 (user):** choose the pilot host (PC or VM; it must stay up for 7 days). The monitor then starts it with `MSYND_LIVE=1`, checks it daily (fetch-log rate ≤ 0.2/s, no challenge) and runs `analyse` at the end. Done 2026-10-10: PC host, ended at 23.3 h by user decision after the VM move failed (spec §18).

---

## M2 — Correctness core (runs while the pilot runs)

### Task 12: Clock-contract labels (spec §3.2, A1)

**Files:** create `msynd/labels.py`, `tests/test_labels.py`.

**Interfaces:** `run_label(dep_delay_a, arr_delay_b) -> float` (Δrun); `dwell(sched_dwell, arr_delay, dep_delay) -> float`; `origin_residual(sched_dep, resched_min, actual_dep) -> float`; `composite(dwell_delta, run_delta) -> float`; `build_live_labels(conn, matured_before) -> pyarrow.Table` (runs completed or expired, events ≥ 6 h old, current revisions only, never `obs.history_daily`).

- [x] Tests:
```python
def test_clock_counterexample():                       # spec §3.2 regression
    assert run_label(dep_delay_a=15, arr_delay_b=20) == 5
    assert dwell(sched_dwell=10, arr_delay=0, dep_delay=15) == 25
    assert composite(dwell_delta=15, run_delta=5) == 20
def test_resched_plus20():  assert origin_residual(100, 20, 130) == 10
def test_resched_negative(): assert origin_residual(100, 60, 150) == -10      # A1 / R3-2
def test_history_never_label(db): ...   # rows only in obs.history_daily -> build_live_labels is empty
def test_labels_only_matured(db): ...   # event 5 h old excluded; 7 h old on completed run included
```
- [x] Implement; commit `task 12: clock contract labels`.

### Task 13: Counter RNG (spec §9.3)

**Files:** create `msynd/rng.py`, `tests/test_rng.py`.

**Interfaces (all `@njit`-compatible):**
```python
RESAMPLE = 0xFFFFFFFF
def splitmix64(x: uint64) -> uint64
def run_key(v_hash: uint64, run_id: int, replicate: int = 0) -> uint64   # splitmix64(splitmix64(v_hash ^ run_id) ^ replicate)
def uniform(key: uint64, particle: int, event_idx: int, attempt: int) -> float64
    # pack = (event_idx << 40) | (attempt << 32) | particle ; u = ((splitmix64(key ^ pack) >> 11) + 0.5) * 2**-53  ∈ (0,1)
def v_hash(versions: dict) -> uint64   # blake2b-8 of canonical JSON (sorted keys)
```
- [x] Tests:
  - known-answer `splitmix64(0) == 0xE220A8397B1DCDAF`;
  - `uniform` is never 0 or 1;
  - same inputs give the same output;
  - batch independence (computing a 1000-particle array in one call or in two halves gives identical values);
  - a χ² uniformity smoke test over 1e5 draws (p > 1e-4).
- [x] Implement; commit `task 13: counter rng`.

### Task 14: Distribution format and effective distribution (spec §3.4, A1, A2)

**Files:** create `msynd/dist.py`, `tests/test_dist.py`.

**Interfaces (array functions are `@njit`; masses are float64):**
```python
SIGNED_LO, SIGNED_HI, SIGNED_NB = -60.0, 240.0, 302    # [under, 300 unit, over]
NONNEG_LO, NONNEG_HI, NONNEG_NB = 0.0, 180.0, 181      # [180 unit, over]
LEVELS = np.array([-30,-10,0,5,10,15,20,30,45,60,90,120,180,240,360,600], float)
QUANTILES = np.array([.02,.05,.1,.25,.5,.75,.9,.95,.98])
def level_weight(d) -> tuple[int, float]                 # (j, w); outside range -> nearest level, w=0
def from_quantiles(q: np.ndarray, signed: bool) -> tuple[np.ndarray, float, float]   # masses, tau_lo, tau_hi (sorted q, A1 tails)
# Effective distribution of T = b + a*max(f, c + s*Qd(u)), Qd from masses[j], masses[j+1] with weight w
@dataclass  # plain tuple in njit code
class Eff: masses: np.ndarray; signed: bool; tau_lo: float; tau_hi: float; j: int; w: float
           b: float; a: float; f: float; c: float; s: float
def eff_quantile(e, u) -> float
def eff_cdf_left(e, x) -> tuple[float, float]           # (P(T<x), P(T>=x)) each computed in its stable coordinate
def eff_window_mass(e, lo, hi) -> float
def eff_sample_window(e, lo, hi, u) -> float            # Q_eff(G(lo) + u*W), clamped to < hi; caller handles W == 0
```
Implementation notes:
- Precompute both cumulative-from-below and cumulative-from-above per level, so CDF and survival are each a sum of non-negative terms.
- Inside a bin, interpolate linearly.
- Tails use the closed form; in the joint overflow tail `Qd(u) = A + B·(−ln(1−u))` with `B = (1−w)τ_j + wτ_{j+1}`.
- `Qd⁻¹` uses 60-step bisection in the CDF coordinate below `Qd(0.5)` and in the survival coordinate above it.

- [x] Tests (write first; all must fail before implementation):
```python
def test_rare_window_conditional_uniform():                       # R3-1 witness
    m = np.zeros(SIGNED_NB); m[1 + 60] = 5e-7; m[1 + 110] = 1 - 5e-7   # bin [0,1) and bin [50,51)
    e = eff(m, signed=True, f=-np.inf)                             # b=0,a=1,c=0,s=1, single level
    assert abs(eff_window_mass(e, 0, 1) / 5e-7 - 1) < 1e-6
    xs = np.array([eff_sample_window(e, 0, 1, u) for u in uniforms(100_000)])
    assert xs.min() >= 0 and xs.max() < 1 and abs(xs.mean() - 0.5) < 0.005

def test_floor_atom_left_limit():                                  # D~U[0,10), floor 5
    m = np.zeros(NONNEG_NB); m[:10] = 0.1
    e = eff(m, signed=False, f=5.0)
    assert isclose(eff_window_mass(e, 5, 6), 0.6) and eff_window_mass(e, 4, 5) == 0 and isclose(eff_window_mass(e, 6, 7), 0.1)
    xs = [eff_sample_window(e, 5, 6, u) for u in uniforms(60_000)]
    assert abs(np.mean(np.array(xs) == 5.0) - 5/6) < 0.01

def test_cdf_quantile_consistency():          # 50 random Eff (random masses, j,w, b,a,f,c,s): empirical P(Q(U)<x) vs eff_cdf_left within 4 se on 20 x's
def test_overflow_tail_mean_tau():            # all mass overflow, tau_hi 7 -> mean 247 ± 0.1, all finite
def test_underflow_tail_signed():             # all mass underflow, tau_lo 4 -> mean -64 ± 0.1
def test_lgbm_roundtrip_half_bin():           # crossing q input sorted; quantile(p_i) within 0.5 of sorted q_i
def test_negative_origin_residual_supported(): # O masses centred at -10, b=100,c=60 -> window [150,151) mass > 0; median init ≈ 150
def test_level_interpolation_same_u():        # Qd(u) == (1-w)Qj(u) + wQj+1(u); d=700 uses level 600 only
def test_extreme_level_and_tails_finite():    # RF5: d=5000, d=-500, mass 1e-12 far tail -> finite, rel err < 1e-6 vs closed form
```
- [x] Implement; tests pass. Commit `task 14: distribution format and effective distribution`.

### Task 15: `pos()` and its shared fixture (spec §9.2, §11 parity)

**Files:** create `msynd/pos.py`, `tests/fixtures/pos_cases.json` (≥ 10 cases), `tests/test_pos.py`.

**Interface:** `pos(t, t_dep_a, anchor_t, anchor_km, t_arr_b, km_a, km_b, pattern: list[tuple[float,float]] | None) -> float` (km along the route). `pattern` holds the scheduled (time fraction, km fraction) points from the anchor to B; `None` means straight in distance.

- [x] Fixture cases:
  - before departure → `km_a`;
  - exactly at `t_dep_a`;
  - midpoint with a straight pattern;
  - a non-linear pattern;
  - after a pass anchor;
  - `t == t_arr_b` → `km_b`;
  - `t > t_arr_b` clamps;
  - multi-day t > 2880;
  - a zero-length segment;
  - `anchor_t == t_arr_b`, which must not divide by zero.
- [x] Implement; commit `task 15: pos and fixture`.

### Task 16: Evidence assembly (spec §3.3, §9.4 step 1, A6)

**Files:** create `msynd/engine/evidence.py`, `tests/engine/test_evidence.py`.

**Interfaces:**
```python
@dataclass(frozen=True)
class RouteEvent: idx: int; station: str; occ: int; kind: str; route_idx: int; halt_idx: int | None
                  sched_off: float; km: float; seg: int
@dataclass(frozen=True)
class Evidence: event: int; actual: bool; minute: float; source: str   # actual: minute m (display); pending: r (run clock)
@dataclass(frozen=True)
class RunEvidence: run_id: int; versions: dict; events: tuple[RouteEvent, ...]; evidence: tuple[Evidence, ...]
                   resched_min: int; stop_event: int | None    # diverted / part-cancelled stop (A11)
def load_run_evidence(conn, run_id: int, asof: datetime | None = None) -> RunEvidence
def canonical_events(route_stops) -> tuple[RouteEvent, ...]   # origin dep only, dest arr only, pass at passing stns
```
- `asof=None` means current truth (`obs.event`). Otherwise use `obs.event_asof(asof)`, pending and exceptions with `received_at ≤ asof`, and `resched_min` from the latest exception revision received before the actual origin departure was received (spec §3.2).
- Apply A6. Order by `route_idx`, then kind.

- [x] Tests:
  - `test_actual_overrides_pending`;
  - `test_latest_pending_only_across_sources` (markers at r=2 and r=3 → one marker, r=3);
  - `test_asof_excludes_later_received` (event at 10:00 received 10:20 is absent at 10:15);
  - `test_resched_asof_receipt`;
  - `test_late_event_equals_fresh_assembly` (RF4: an earlier-route event inserted last gives a result identical to a fresh load);
  - `test_route_rebind_keeps_keys` (a new route version with a passing station leaves the halt event keys unchanged);
  - `test_part_cancelled_tail_sets_stop_event`.
- [x] Implement; commit `task 16: evidence assembly`.

### Task 17: Replay kernel, block-wise SIR (spec §4.1, §9.2–9.4, A2, A6, A7, A11)

**Files:** create `msynd/engine/kernel.py` (njit), `msynd/engine/replay.py`, `tests/engine/test_replay.py`.

**Interfaces:**
```python
@dataclass(frozen=True)
class ModelTables:            # built by cache (Task 27) or by tests
    r: np.ndarray             # [n_seg, 16, 302] masses; r_tau: [n_seg, 2]; bias: [n_seg]
    w: np.ndarray             # [n_halt, 16, 181]; w_tau: [n_halt]
    o: np.ndarray             # [302]; o_tau: [2]
    cold_start: bool
@dataclass(frozen=True)
class Params: n: int = 10_000; k: int = 32; v_max: float = 160.0; ess_min: float = 0.05
              s_tables: dict   # per source: lag grid survival + tail scale + eps (from data/params.json)
@dataclass
class ReplayOut: halt_q: np.ndarray        # [n_halt, 2(arr,dep), 7] run clock, NaN for past/absent
                 next_q: np.ndarray        # [7]
                 paths: np.ndarray         # [K, E] float32 event times (strided particles j*N/K)
                 min_ess: float; surprise: bool; flags: tuple[str, ...]; pits: tuple[float, ...]
def replay(ev: RunEvidence, models: ModelTables, params: Params, key: int) -> ReplayOut
```
Algorithm, which must match exactly:
```
T = full((N, E), nan, f32); nxt = 0
for blk, ev_e in enumerate(evidence in route order):
    for attempt in (0, 1):
        n = N if attempt == 0 else 4*N ; src = arange(n) % N ; Tb = T[src]
        for e in nxt .. ev_e-1:  Tb[:, e] = eff_quantile(eff_for(e, Tb[i]), uniform(key, i, e, attempt))
        if actual: lo, hi = m, m+1                       # 'trunc' minute semantics from params
            W_i = eff_window_mass(...); pit = mean(P(T<lo) + 0.5*W_i)  (record before weighting)
            Tb[:, ev_e] = eff_sample_window(..., uniform(key, i, ev_e, attempt)) where W_i > 0 ; logw = log(W_i)
        else:  t = eff_quantile(...); w = 1 if t >= r else S_src(r - t); logw = log(w)
        ess = (Σw)²/Σw² over n ; if ess/N >= ess_min or attempt == 1: break
    if all W == 0: Tb[:, ev_e] = m + 0.5 ; flags += 'data_invariant' ; T = Tb[:N]
    else: T = Tb[systematic(normalised w, uniform(key, RESAMPLE, ev_e, attempt))]   # n -> N
    nxt = ev_e + 1
forward-draw nxt .. (stop_event or last) without weights; outputs from all N rows
```
`eff_for(e, row)` builds the A2 parameters:
- origin `dep`: b = sched, c = resched, O.
- intermediate `dep`: b = T[arr], W at level = arr delay.
- `arr` at B: b = T[dep A], or the pass rule (A7) if passes exist; R at level = dep delay at A, or the delay at the pass; f = 60·km/v_max; c = sched_run + bias.
- `pass`: A7 with a = the time fraction.
- Chained remainder (ruling, G7-accepted): anchor Pᵢ (P₀ = dep A); `sched_rem = max(0.1, sched(B) − sched(Pᵢ))`, `r = sched_rem/sched_run`, `f_rem = 60·(km_B − km_Pᵢ)/v_max`, R at the delay at Pᵢ; `b = T_Pᵢ`, `c = sched_rem + r·bias`, `s = r`; `a = 1` for B, `a = clip((sched(Pᵢ₊₁) − sched(Pᵢ))/sched_rem, 0, 1)` for the next pass.

Log-weights use log-sum-exp. Display paths use strided rows `j*N//K`.

- [x] Tests (small N = 2,000 for speed; synthetic `ModelTables`):
  - `test_invariants` (monotone along route; dwell ≥ 0; running ≥ floor wherever not an observed override);
  - `test_actual_window_all_particles_inside` (every row has event e in [m, m+1));
  - `test_infeasible_evidence_flagged_no_nan` (an actual faster than V_MAX → `data_invariant`, all finite);
  - `test_deterministic_and_order_free` (same input twice gives identical bytes; evidence tuple shuffled before assembly gives identical output);
  - `test_ess_fallback_triggers` (a narrow window under a wide prior → attempt 1 used and `min_ess` recorded);
  - `test_outputs_shapes_and_strided_paths`;
  - `test_extreme_delay_replay_no_nan` (RF5: an origin delay of 900 min);
  - `test_surprise_flag` (an actual at PIT < 0.001 → `surprise` is True).
- [x] Implement; commit `task 17: replay kernel`.

### Task 18: G7 conditioning suite → 🛑 G7 gate (spec §15 G7, A2, A6–A8)

**Files:** create `tests/g7/reference.py`, `tests/g7/test_g7.py` (marked `slow`), and `docs/gates/G7.md`, written by `python -m tests.g7.report`.

**Interfaces:** `enumerate_posterior(case) -> dict[str, float]` for bin-aligned cases; `lw_reference(case, n=2_000_000, seed) -> (estimates, se)`, which is iid prior sampling with weights equal to the window or survival likelihood, without resampling; `engine_replicates(case, n, reps=20) -> (mean, se_rep)` by salting the replicate into `run_key`.

Cases, each a 2–4 segment model with 2–5 outcome bins per draw:
- (a) the not-reported 2/3 example, exact;
- (b) an actual with an unreported halt before it (bridge);
- (c) a mid-segment pass → A7 criterion, reported and checked;
- (d) a soft lag with a long tail and ε > 0, plus the A4 sensitivity variants;
- (e) evidence infeasible under the floor → flagged, no NaNs;
- (f) actuals and pending markers mixed across 3 blocks;
- (g) degenerate weights, where the ESS fallback triggers and the result stays within tolerance;
- (h) a rare but feasible window of mass 5e-7;
- (i) a floor atom at an interval boundary;
- (j) mixed feasible and infeasible ancestors;
- (k) a repeated pending marker → matches 0.731059 within tolerance and rejects 0.880797;
- (l) a negative O residual;
- (m) the +20 rescheduling case → initial median ≈ 130.

Every case runs at N = 10k and 100k, with the A8 tolerance.

- [x] Write the reference and the tests. Run `uv run pytest -q -m slow tests/g7`.
- [x] Commit `task 18: g7 suite`.
- [x] **🛑 G7 gate (monitor):** all cases pass; `docs/gates/G7.md` records the per-case numbers and the A7 pass-approximation verdict; the result goes into spec §18. If anything fails, fix the engine (Task 14 or 17 follow-up) before Task 31.

### Task 19: Engine bench (G8 prototype, spec §9.4 cost, §9.7)

**Files:** create `msynd/engine/bench.py`.

**Interface:** `python -m msynd.engine.bench --runs 2500 --workers 4 --n 10000` builds synthetic runs (40 segments, LightGBM-shaped masses, evidence on the first half) and reports per-run replay p50/p95/p99, the full rebuild wall time, and msgpack frame build size and time. It writes `docs/gates/G8-local.md`.

- [x] Smoke test: `--runs 20 --n 1000` finishes, and the JSON output has every field.
- [x] Commit `task 19: engine bench`. The monitor runs it locally now and on the VM at CP-3.

---

## M3 — Collector service

### Task 20: Evidence store, one transaction per fetch (spec §3.3, §7.3, §7.4, A3)

**Files:** create `msynd/collect/store.py`, `tests/collect/test_store.py`.

**Interfaces:** `store_status(conn, run_id, fetch: FetchResult, page: ParsedStatus, source='mmt_status') -> StoreResult(changed: bool)`. In one transaction it:
- inserts `event_rev` only when a key is new or its `actual_ts` changed;
- writes `pending_rev` for every pending key;
- writes `ntes_eta_rev` (the name stays; it holds MMT ETAs) for **every** `page.etas` entry (A3);
- passes each `page.exceptions` entry to Task 21 `apply_exception`;
- turns new passing stations into a new route version and rebinds runs with identical halts (dirty);
- applies lifecycle transitions (running / completed);
- calls `mark_dirty` if anything changed.

Migration `0004_mmt_source.sql` adds `'mmt_status'` to `obs.source`. `settings.live_source = 'mmt_status'` replaces the hard-coded `'ntes_status'` defaults in `msynd/engine/__main__.py`, `priority.py`, `evidence.py`'s tie-break, `api/app.py` and `ops/nightly.py`. There is no board, exception-list or history store: MMT has no such pages, and RSTGCN history is Task 42.

- [x] Tests, including `msynd/eval/baselines.py:b2_eta(conn, run_id, station, occ, kind, t)` (create it here):
```python
def test_eta_confirmation_per_key(db):    # A3 / R3-3
    # 09:00 ok page: ETA B=10:30, C=11:00 ; 10:00 ok page: C=11:00 only ; 10:20 ok page: B=10:30 unchanged
    assert b2_eta(db, r, 'B', 1, 'arr', ist('10:10')) is None             # missing at 10:00 → last 09:00, 70 min
    assert b2_eta(db, r, 'C', 1, 'arr', ist('10:10')) == ist('11:00')
    assert b2_eta(db, r, 'B', 1, 'arr', ist('10:25')) == ist('10:30')     # unchanged-present re-confirms
    # 10:40 page: B=10:45 (changed)
    assert b2_eta(db, r, 'B', 1, 'arr', ist('10:39')) == ist('10:30')     # post-cutoff confirmation unused
    assert b2_eta(db, r, 'B', 1, 'arr', ist('10:41')) == ist('10:45')
def test_non_status_fetch_never_confirms(db): ... # autosuggest/capture fetch rows → zero eta rows
def test_unchanged_actual_not_reinserted(db): ...
def test_expected_never_stored_as_actual(db): ...
def test_dirty_in_same_txn(db): ...               # rollback of store → no dirty row
def test_repeated_station_keys(db): ...           # RF3: occ 1 and 2 stored separately
def test_enrichment_never_rekeys(db): ...         # passing station → new route version, event keys identical
```
- [x] Implement; commit `task 20: evidence store`.

### Task 21: Run lifecycle and exceptions (spec §7.3, A11)

**Files:** create `msynd/collect/lifecycle.py`, `tests/collect/test_lifecycle.py`.

**Interfaces:**
- `create_runs(conn, now) -> int` covers the next 24 h of origin dates (IST), using versions valid on the date with a matching `days_of_run` bit, minus cancellations, bound to the latest route version.
- `expire_runs(conn, now) -> int` applies: no actual for 12 h **and** now > scheduled destination arrival + max(12 h, 2 × last observed delay).
- `apply_exception(conn, row, fetch) -> None` (rows come from `ParsedStatus.exceptions`; MMT has no network-wide list) handles cancelled, `resched_min`, `diverted_from` and `part_cancelled`, then calls `mark_dirty`.
- `obs.exception_rev.detail` is `{"kind": "cancelled"|"part_cancelled"|"rescheduled"|"diverted"|"none", ...}` (Task 10 `ExceptionRow.kind`; `none` withdraws an exception). A (train, start_date) is fully cancelled as of t when its latest revision with `received_at ≤ t` has `kind = 'cancelled'`; any later revision of another kind clears it (monitor, 2026-10-08). Each `obs.exception_rev.detail` it writes carries the run's **full cumulative** exception state (resched, diversion, part_cancelled), because evidence assembly (Task 16) reads the latest detail as complete state.

- [x] Tests:
  - `test_creates_only_matching_weekdays`;
  - `test_ist_start_date_at_utc_evening` (RF2: at 2026-10-07 19:00 UTC the IST date is 10-08);
  - `test_cancelled_not_created`;
  - `test_expiry_rule_both_conditions`;
  - `test_expired_never_completed`;
  - `test_exception_marks_dirty`.
- [x] Implement; commit `task 21: run lifecycle`.

### Task 22: Admission and the collector main loop (spec §7.2, §7.5)

**Files:** create `msynd/collect/admission.py`, `msynd/collect/__main__.py`, `tests/collect/test_admission.py`.

**Interfaces:**
- `next_job(state, now) -> Job`. Order: (1) nothing while the host is `stopped`; (2) overdue runs (not fetched within `T_max = 1.25·R·h/rate`); (3) targeted polls by `obs.poll_queue.priority`, falling back to age when no row exists. MMT has no boards, exception list or backfill.
- `python -m msynd.collect` runs the MMT host task, plus hourly `create_runs`, `expire_runs`, body retention (Task 23) and the disk check.

- [x] Tests:
  - `test_no_run_starves_beyond_tmax` (simulate 300 runs with skewed priorities for 6 simulated hours: every run is fetched within T_max);
  - `test_job_order`;
  - `test_stopped_host_issues_nothing`.
- [x] Implement; commit `task 22: admission and collector loop`.

### Task 23: Retention, disk guard, backup (spec §12, §13, A3, A5)

**Files:** create `msynd/ops/nightly.py` (retention and backup parts), `tests/ops/test_retention.py`.

**Interfaces:**
- `prune_bodies(conn, now, ok_hours)` runs hourly from the collector: `ok` bodies after `ok_hours` (24, or 2 when disk ≥ 80 %); non-`ok` bodies and a 1 % `ok` sample after 14 days.
- `prune_rows(conn, now)` runs nightly: `obs.fetch` after 180 days, `pending_rev` after 180 days, `ntes_eta_rev` after 35 days, `forecast_issue` after 120 days.
- `disk_check(path) -> (pct, ok_hours)` writes the metric and raises an alarm at 70 %.
- `backup()` runs `pg_dump -Fc` (ref, obs, ml, `sim.forecast_issue`), keeps 2 copies in `/data/backups`, uploads with `oci os object put` (7 daily / 4 weekly), and alarms if the off-site set is over 15 GB. **Off-site excludes `obs`** (spec §18, 2026-10-08): the local dumps hold ref, obs (no fetch bodies), ml and `sim.forecast_issue`; the uploaded dump holds only ref, ml and `sim.forecast_issue`, until G1 allows raw evidence off the machine. The local dump includes `obs.fetch` with its bodies (pg_dump cannot drop one column; retention already bounds bodies to ~24 h plus the 1 % sample), `# ponytail:` sanitized fetch copy from the same snapshot if dump size matters (monitor, 2026-10-08).

- [x] Tests:
  - `test_old_referenced_fetch_pruned_event_kept` (A5);
  - `test_eta_35_days`;
  - `test_body_retention_drops_to_2h_at_80pct` (inject the disk percentage);
  - `test_sample_kept_14_days`.
- [x] Implement; commit `task 23: retention and backup`.

### Task 24: Image, compose, Caddy (spec §5, §13)

**Files:** create `Dockerfile` (python:3.12-slim, multi-arch, `uv sync --frozen`), and modify `compose.yaml` to add the services `collector`, `engine`, `api`, `caddy` and an `ops` profile (`restart: unless-stopped`, health checks, json-file logging 10m×5, volumes `/data/{pg,models,backups,tiles}`). Create `Caddyfile` (DuckDNS host, basic auth on everything, `/api/*` → api:8000 with WebSocket, static `web/dist`, `/tiles/*`), extend `.env.example`, and write `docs/deploy.md` (≤ 60 lines: VM firewall 80/443 open, 22 restricted to the user's IP, crontab lines for nightly 21:00 UTC and weekly Sat 20:30 UTC).

- [x] Verify: `docker buildx build --platform linux/arm64,linux/amd64 .` succeeds, and `docker compose --profile ops run --rm ops python -c "import lightgbm, onnxruntime, numba"` passes.
- [x] Commit `task 24: image and compose`.

### Task 25: Gate and exit report CLI → 🛑 CP-3

**Files:** create `msynd/ops/report.py`, `tests/ops/test_report.py`.

**Interface:** `python -m msynd.ops.report {a1,a2,a5,a8,invalid,phase2} --since --until --out docs/gates/<name>.md`. It computes:
- A1: aggregates the engine's per-minute `sim.metric` rows (written by Task 32; spec §18, 2026-10-08) over `--since..--until`. It reports mean and minimum of `a1_discovered/a1_active` and `a1_fresh/a1_active`, and the target ratio `a1_discovered_1h/a1_active_1h` against ≥ 95 %, plus the share of minutes with `a1_cold > 0` (counted on cold-start priors). It never recomputes the denominator from schedules;
- A2: p50/p90 of (now − last `received_at`) by class and zone, where a run's zone is that of its last event's station (origin zone before any event; spec §18, 2026-10-08);
- A5: the maximum number of starts in any 60 s window per host, from `obs.fetch` with a SQL window function;
- A8: missing-event rate by class;
- the `invalid` share.

- [x] Tests: `test_a5_window_counts` (61 starts within 60 s are detected); `test_a1_aggregates_metric_rows` (inserted `a1_*` rows → expected mean, minimum and target verdict).
- [x] Commit `task 25: reports`.
- [ ] **🛑 CP-3 (user and monitor):**
  1. The user reviews the G2 results and decides on h, S, board share, T_max and F. If 1 req/s freshness is unacceptable, the user chooses between accepting it, changing the rate, changing the source, or reducing scope. Record this in §18.
  2. **G5:** the user gets an Oracle A1 allocation. The monitor deploys per `docs/deploy.md`, runs the image smoke test, measures storage from the G2 bodies, runs a restore drill into a fresh container (timed), and runs `engine.bench` on the VM. Results go to `docs/gates/G5.md` and `G8-vm.md`.
  3. Start the live collector. After 7 continuous days, run `report phase2` with the Phase 2 exit criteria: A5 holds, the evidence-age distribution and A8 are reported, `invalid` < 1 %, and the restore drill passed.

---

## M4 — Models and evaluation (Phase 3; data accrues meanwhile)

### Task 26: Features (spec §8.2, §8.3)

**Files:** create `msynd/features/__init__.py` (`FEATURE_VER = 1`), `msynd/features/build.py`, `tests/features/test_features.py`.

**Interfaces:**
- `segment_queries(conn, run_id, asof) -> pyarrow.Table` holds the shared per-segment features (§8.2 table minus *d*).
- `training_frame(conn, fold: Fold) -> (X, y)` takes `Fold(train_weeks, valid_weeks)` (rolling-origin by week). Each row's as-of cutoff is the `received_at` of the segment's departure event at A.
- FEATURE_VER 1 omits "count of surprises so far" (spec §18, 2026-10-08).
- `FoldEncoders.fit(train_rows)` covers target encoding (smoothed), slack P05 per halt pair and history stats (only if G4 passed). It is fitted on training rows only.

- [x] Tests:
  - `test_leakage_received_after_asof_absent` (an event at 10:00 received at 10:20 is absent at 10:15);
  - `test_encoders_never_see_validation_rows`;
  - `test_splits_rolling_by_week`;
  - `test_calendar_features_use_sched_dep_plus_d`.
- [x] Implement; commit `task 26: features`.

### Task 27: Empirical and provisional models, registry, CDF cache (spec §3.2 cold start, §8.4, §9.6)

**Files:** create `msynd/models/empirical.py`, `msynd/models/registry.py`, `msynd/engine/cache.py`, `tests/models/test_empirical.py`, `tests/engine/test_cache.py`.

**Interfaces:**
- `fit_empirical(labels, role) -> ModelArtifact` builds per-class histograms; W also has a scheduled-dwell bucket. Buckets: quintile edges of scheduled dwell over the training rows (duplicates merged), stored in the artifact; an empty or unseen bucket falls back to the class-pooled histogram, then the provisional prior (spec §18, 2026-10-08).
- `provisional_models() -> dict[role, ModelArtifact]` gives day-1 broad priors, all marked `cold_start`:
  - R: Δrun ~ Laplace(0, 5);
  - W: sched dwell + Exp(2);
  - O: Exp(10) − 2 (signed).
- `registry.active(conn) -> dict[role, (model_id, artifact)]`; `registry.promote(conn, model_id)` swaps `active` in one transaction and sends `NOTIFY model_changed`.
- `cache.tables_for(conn, ev: RunEvidence, active, bias_gen) -> ModelTables` batches every remaining segment × 16 levels × {R, W}, plus O. The cache key is (V, shared features). Bias shifts R. LightGBM per-level tail scales collapse to the max over the 16 levels per segment (R, lo and hi separately) and per halt (W), `# ponytail:` per-level τ in the kernel (spec §18, 2026-10-08).

- [x] Tests:
  - `test_empirical_masses_sum_to_one_and_signed_o`;
  - `test_provisional_flags_cold_start`;
  - `test_promote_single_active_and_notifies`;
  - `test_cache_hit_on_same_v`, and a miss when the bias generation changes.
- [x] Implement; commit `task 27: empirical models and cache`.

### Task 28: LightGBM quantile models (spec §3.4 LightGBM→CDF, §8.4)

**Files:** create `msynd/models/lgbm.py`, `tests/models/test_lgbm.py`.

**Interfaces:** `train_lgbm(X, y, role) -> ModelArtifact` (9 quantile boosters plus τ per class fitted on training labels); `predict_masses(artifact, Xq: shared features × 16 levels) -> np.ndarray[n, 16, B]` via `dist.from_quantiles`; save and load through `ml.model.path`.

- [x] Tests:
  - `test_quantile_reproduction_half_bin` (synthetic data);
  - `test_signed_o_negative_labels_roundtrip`;
  - `test_noncrossing_after_sort`.
- [x] Implement; commit `task 28: lightgbm models`.

### Task 29: Evaluation harness, baselines, promotion, N-curve (spec §8.6, §4.2, §3.4)

**Files:** create `msynd/eval/harness.py`, `msynd/eval/metrics.py`, extend `msynd/eval/baselines.py`, `tests/eval/test_*.py`.

**Interfaces:**
- `asof_cases(conn, start, end) -> Iterator[Case]` makes cut points every 30 min while running, with the horizon `h = sched_arr + latest observed delay as of t − t` bucketed into 30 m, 2 h, 6 h, 12 h ± 25 % (at most one bucket).
- `score(cases, forecaster) -> Scores`: CRPS, pinball, coverage with width, P50 MAE, PIT, and breakdowns by class, zone, delay band and disruption day.
- `block_bootstrap(diff_by_date, reps=2000) -> CI`.
- Baselines B0 schedule, B1 persistence, B2 (`b2_eta`, A3), B3 active model.
- `promotion_ok(candidate, active) -> Decision` implements the §8.6 rule.
- `n_curve(cases, Ns=(1k,5k,10k,50k,100k)) -> chosen_N` picks the smallest N with MC error ≤ 0.25 min at P05/P95 for 6 h.
- `interp_error(...)` measures delay-level interpolation against direct model calls.

- [x] Tests:
  - `test_horizon_bucket_independent_of_outcome`;
  - `test_identical_eligible_cases_across_baselines`;
  - `test_b2_ineligible_when_stale`;
  - `test_bootstrap_ci_excludes_zero_on_clear_gain`;
  - `test_promotion_rule_rejects_class_regression`.
- [x] Implement; commit `task 29: evaluation harness`.
- [ ] **Phase 3 exit (monitor, after ≥ 4 weeks of live data):** R/W beat B1 on CRPS at 2 h (CI > 0), with coverage within ±5 points. Write `docs/gates/phase3.md`, set N from the N-curve, and record both in §18.

### Task 30: G4 etrain meaning (gated: ≥ 2 weeks of live NTES data, plus G1 allowing etrain)

**Files:** create `msynd/ops/g4.py`, `docs/gates/G4.md`.

**Interface:** `python -m msynd.ops.g4 --trains 20` compares `obs.history_daily` against NTES actuals where arrival and departure delays differ by ≥ 5 min. It covers origin, long intermediate dwell, destination, overnight, multi-day, early running, non-daily and repeated stations, and decides whether `run_date` is the origin date.

- [ ] Test `test_discriminating_cases_only` (cases where the delays are equal are excluded).
- [ ] Commit `task 30: g4`.
- [ ] **🛑** Record the verdict in §18. **Pass** → history features, plus composite model C as cold start until R/W/O have 4 weeks of labels (a follow-up task added then). **Fail** → history stays quarantined; the cold start is the live empirical models from the first 2 weeks.

---

## M5 — Engine service (Phase 4)

### Task 31: Engine main loop and frames (spec §9.5, A9, A10)

**Files:** create `msynd/engine/__main__.py`, `msynd/engine/frames.py`, `tests/engine/test_loop.py`.

**Interfaces:**
- `encode_run(run_id, out: ReplayOut, ev, base_ts) -> dict` returns msgpack-ready data: state, headline, `next_q`, flags, `min_ess`, `halt_q`, and paths as `uint16[K, E, 2]` bytes.
- `write_frame(conn, is_full, versions, runs: dict, tombstones: list) -> gen`.
- The loop:
  - startup: LISTEN, commit, rebuild all active runs, write a full frame;
  - on `dirty` or every 1 s: A9 consume, replay across a `multiprocessing` pool of 4, then **one** transaction: delta frame, matched-ver delete, `NOTIFY sim_frame`;
  - a full frame every 5 min; prune `sim.frame` older than 15 min;
  - on `model_changed` or a new bias generation: `mark_dirty(all active)`.

- [x] Tests:
```python
def test_dirty_race_never_loses_rows(db_url):      # spec §14 Dirty set; A9
    # thread A: 300× (insert evidence for random run; mark_dirty; commit) at random intervals
    # engine: run consume passes with a fake replay that records evidence count seen per run
    # after A stops: run passes until obs.dirty_run is empty
    # assert for every run: last frame payload evidence_count == committed evidence count
def test_one_batch_vs_two_identical_payloads(db_url): ...    # determinism
def test_restart_identical_full_frame(db_url): ...
def test_frame_latency_idle_under_2s(db_url): ...             # A10 normal path
def test_notify_loss_bounded(db_url): ...                     # notifications suppressed -> visible ≤ 1 s + replay
def test_replay_exception_acked_with_error_state(db_url): ...
def test_tombstone_for_retired_run(db_url): ...
def test_late_event_frame_equals_fresh_replay(db_url): ...    # RF4
```
- [x] Implement; commit `task 31: engine loop and frames`.

### Task 32: Poll priorities and issued forecasts (spec §7.2 priority, §9.5, A3)

**Files:** create `msynd/engine/priority.py`, modify `msynd/engine/__main__.py`, create `tests/engine/test_priority.py`.

**Interfaces:**
- `priority(run_out, S, now, age, T_max) = P(next halt event published by now) × width(P10..P90) + age/T_max`. The probability is the particle mean of `1 − S(now − t)`. Upsert to `obs.poll_queue` every 30 s.
- Hourly: insert `sim.forecast_issue` for running runs with `q` (7 per halt) and `b2` (A3 eligibility at issue time, NaN otherwise).
- Each minute: write the A1 triple to `sim.metric` from the replay outputs already held (spec §18, 2026-10-08). A run is **active at t** when its simulated origin departure P50 ≤ t (the actual once observed), its simulated destination arrival P50 > t, and it is not cancelled as of t (Task 21 rule). Metrics: `a1_active`, `a1_discovered` (≥ 1 source event), `a1_fresh` (last event received ≤ F), the same first two as `a1_active_1h`/`a1_discovered_1h` restricted to origin departure P50 ≤ t − 1 h, and `a1_cold` (active runs replayed on cold-start tables).

- [x] Tests:
  - `test_priority_prefers_published_and_wide`;
  - `test_age_term_prevents_starvation`;
  - `test_forecast_issue_b2_nan_when_stale`.
  - `test_a1_active_uses_simulated_arrival` (a late observed run stays active past scheduled arrival; an unseen run leaves at its prior P50).
- Defaults until CP-3 decides them (monitor, 2026-10-08), each marked `# ponytail:`:
  - S for priority comes from `Params.s_tables[source]` with `source='ntes_status'` as a loop argument; without that table, poll-queue upserts are skipped.
  - `T_max` defaults to 30 min and `F` (A1 fresh) to 30 min, both as loop arguments.
  - Poll age = now − the run's latest `obs.fetch.started_at` (no fetch → age = T_max).
  - Forecast issue rows cover remaining halts with finite arrival quantiles; `b2` follows A3 directly on `obs.ntes_eta_rev` (latest row ≤ issue time, ≤ 30 min old), NaN otherwise.
  - A1 uses the latest replay output per run: an observed origin departure counts as ≤ t and an observed destination arrival as ≤ t (halt_q is NaN for past events). Cancellation follows the Task 21 `exception_rev.detail` rule. `a1_cold` counts runs whose model tables had `cold_start`.
- [x] Implement; commit `task 32: priorities and issued forecasts`.

### Task 33: Replay-day end-to-end → 🛑 G8 and Phase 4 exit (spec §14 Replay day)

**Files:** create `tests/e2e/test_replay_day.py` (marked `slow`; skipped unless `MSYND_REPLAY_DAY=<dir>`), and `msynd/ops/replay_day.py`, which exports one pilot or live day of `ok` bodies with their receipt times into that directory.

> **Amendment 2026-10-10 (MMT source, F1 bridge, G8-local throughput).** This amendment overrides the box below wherever they differ.
>
> **Files:**
> - create `msynd/ops/replay_day.py`, `tests/ops/test_replay_day.py` (`db`) and `tests/e2e/test_replay_day.py` (`slow`);
> - modify `msynd/ops/pilot_bridge.py`: move the per-row apply out of `bridge_once` into `apply_status(dst, row) -> 'applied' | 'invalid' | 'no_version'`, with no behaviour change. Its row is `(host, purpose, key, started, received, outcome, status, compressed)`. Both callers use it.
>
> **Export:** `python -m msynd.ops.replay_day export --source DSN --day YYYY-MM-DD --out DIR`. It opens the source with `pilot_bridge.open_source` (read-only) and writes:
> - `fetches.jsonl`: the source's `purpose='status'`, `outcome='ok'` rows whose `received_at` falls on that IST date, ordered by `(received_at, id)`. Each line holds the bridge row fields, with `body_zstd` as base64 of the stored bytes.
> - `ref/<table>.csv`: one file per `ref` table, in FK order (station, train, schedule_version, schedule_stop, route_version, route_stop, segment, route_segment, calendar), via `COPY … TO STDOUT (FORMAT csv, HEADER)`.
> - `manifest.json`: day, fetch count and fetch id range, rows per ref table.
>
> **Replay:** `replay(dir, dsn, *, speed=60, params=None, workers=4) -> dict`. The target is an empty migrated database.
> - Load the ref CSVs with `COPY … FROM STDIN`, using the header's column list. Then `setval` every owned sequence to its column's max.
> - Simulated clock: `sim = first received_at + speed × wall elapsed`.
> - Each fetch is applied with `apply_status` once `sim ≥ received_at`. It is ok for evidence revisions to carry their historical `received_at`.
> - Between fetches, step the engine with the existing functions, not `run_loop`: `consume` until nothing is dirty or a fetch is due, and `write_periodic(now=sim)` with metrics and polls at each simulated minute and issue at each simulated hour. Models and params are as in `engine.main` (registry/provisional, `settings` N and K, `load_s_tables`); there is a spawn pool of `workers`, or serial when `workers=0`.
> - After the last fetch, drain until nothing is dirty.
> - Runs come only from fetch keys (`ensure_run`), never from `create_runs`. `# ponytail:` replay covers only the polled cohort; at 1.8 s per run (G8-local), a whole-network replay at 60× cannot keep up on the PC; add hourly `create_runs` for a network-wide day on the VM.
>
> **Result:**
> - `{applied, invalid, no_version, invalid_share, a1: report.a1(conn, day start, day end), a6: {count, p50, p95, max}, replay_errors, runs_discovered, runs_framed, forecast_issues}`, written to `DIR/report.json`.
> - A6 is wall seconds from the commit of a changed `store_status` to the commit of the first later frame containing that run. It covers the engine part of A10 only; API delivery is pinned by Task 34.
> - `invalid_share = invalid / fetches`.
>
> **Tests:**
> - `tests/ops/test_replay_day.py`, on synthetic sources built like `tests/ops/test_pilot_bridge.py`:
>   - export order and IST-day filtering, with the source read-only;
>   - the ref round trip, including a later insert after load that does not collide with a sequence;
>   - replay produces evidence and a frame for every discovered run, counts an invalid body and a no-version key, writes A1 metrics at simulated timestamps, records A6 samples, and writes `report.json`;
>   - the existing `test_pilot_bridge.py` still passes.
> - `tests/e2e/test_replay_day.py` is skipped unless `MSYND_REPLAY_DAY=<dir>`. It creates and drops its own database, replays at 60× and asserts:
>   - `invalid_share < 0.01`;
>   - `replay_errors == 0`;
>   - every discovered run framed;
>   - `a1.minutes > 0`;
>   - a finite `a6.p95`.

- [x] The test feeds the bodies in `received_at` order at 60× through parsers → store → engine on a test database. It asserts: A1 reported, A6 p95 measured, `invalid` < 1 %, no unhandled exceptions, and frames are produced for every discovered run.
- [x] Commit `task 33: replay-day e2e`.
- [ ] **🛑 Monitor:** run it on a recorded day; run G8 on the VM with all services up (apply §9.7 only to a measured bottleneck); run 7 days live and record A1, A2 and A6, and that `sim.forecast_issue` is populating, in `docs/gates/phase4.md`.

---

## M6 — API and dashboard (Phase 5)

### Task 34: API (spec §10, A10)

**Files:** create `msynd/api/app.py`, `tests/api/test_api.py`.

**Interfaces:** the endpoints exactly as in spec §10.
- `/api/live`: the client sends `{"gen": n}`. If there is a gap, the server sends the latest full frame plus all later deltas; otherwise only new deltas. More than 2 frames queued for one client → drop the queue and send a full frame.
- The API process wakes on `sim_frame` and also polls every 5 s.
- `/api/static/network` serves the export with its ETag.
- `/api/health` reports dispatcher state per host (read from `sim.metric` rows the collector writes), active models, dirty backlog, and the last frame's `gen` and age.

- [x] Tests (FastAPI TestClient):
  - `test_gap_triggers_full_resync`;
  - `test_slow_client_coalesces_to_full`;
  - `test_tombstone_forwarded`;
  - `test_train_number_ambiguous_lists_runs`;
  - `test_run_endpoint_all_remaining_halts_and_eta_age`;
  - `test_replay_window_max_24h`;
  - `test_static_etag_304`.
- [x] Implement; commit `task 34: api`.

### Task 35: Web scaffold, frame client, `position.ts` parity (spec §11)

**Files:** create `web/package.json` (vite, typescript, vitest, maplibre-gl, @deck.gl/core, @deck.gl/layers, @deck.gl/geo-layers, @deck.gl/mapbox, pmtiles, @msgpack/msgpack), `web/index.html`, `web/src/position.ts`, `web/src/frames.ts`, `web/src/*.test.ts`.

**Interfaces:** `pos(...)`, a port of `msynd/pos.py`; `FrameStore.apply(frame) -> 'ok' | 'gap'`, which applies full frames, deltas and tombstones and reports a gap when `gen` is not last + 1; `connectLive(url, store)`, which resyncs on a gap with exponential reconnect.

- [x] Tests:
  - `position.test.ts` reads `../tests/fixtures/pos_cases.json` and expects every case to equal the Python result within 1e-6;
  - `frames.test.ts` covers gap detection, tombstones, and a full frame replacing everything.
- [x] Verify `npm --prefix web test`; commit `task 35: web scaffold and frame client`.

### Task F1 (amendment 2026-10-09): pilot bridge to a view DB

User-approved (decisions A–D in `reports/frontend-live-plan.md`): a dashboard preview on the G2 pilot cohort, with no new MMT traffic and no writes to the pilot DB.

**Files:** create `msynd/ops/pilot_bridge.py`, `tests/ops/test_pilot_bridge.py`; modify `msynd/engine/__main__.py` (`main()` only).

**Interfaces:** `python -m msynd.ops.pilot_bridge --source DSN --target DSN [--follow]`; `bridge_once(source, target) -> int` (rows applied).
- The source connection is read-only (`conn.read_only = True`). It reads `obs.fetch` rows with `purpose='status'`, `outcome='ok'` and `id >` the cursor, in `id` order.
- Per row, in one target transaction: copy the fetch row (without the body), find or create the target `obs.run` for the key's train and start date (reusing lifecycle's version selection; rows without a valid schedule version are skipped and counted), `parse_mmt.parse_status` → `store.store_status`, then advance the cursor (kept in the target DB).
- `--follow` repeats every 60 s. There is no HTTP path.
- Engine `main()`: the pool size comes from `ENGINE_WORKERS` (default 4), and `Params` uses `settings.n_particles` / `settings.k_paths`.

- [x] Tests: synthetic MMT bodies are stored as evidence; a rerun is idempotent; the source is opened read-only; an unknown train is skipped; the engine's `main()` honours `ENGINE_WORKERS` and `N_PARTICLES`.
- [x] Commit `task F1: pilot bridge`.

### Task 36: Map, layers, timeline, panels → 🛑 A7 and Phase 5 exit (spec §11)

**Files:** create `web/src/main.ts`, `web/src/layers.ts`, `web/src/timeline.ts`, `web/src/panel.ts`, `web/src/status.ts`, `web/src/timeline.test.ts`; modify the Caddyfile (tiles).

**Part 1 (amendment 2026-10-09, task F2):** a minimal map ships first, with no basemap: a dark background, the network `PathLayer`, P50 heads as a `ScatterplotLayer`, K streaks as a `TripsLayer`, a click → `/api/run/{id}` panel, a vite proxy to 127.0.0.1:8000, and the `timeline.test.ts` LIVE offset and zone tests. Commit `task 36 part 1: minimal map`. The boxes below stay open until the full task.

Contents (spec §11):
- MapLibre with the Protomaps dark PMTiles (`/tiles/india.pmtiles`).
- `PathLayer` for the network, dashed where confidence < 0.9.
- `TripsLayer` with K streaks: `trailLength` = 10 simulated minutes, colour by median delay (teal → amber → red), opacity ≈ boost/K.
- `ScatterplotLayer` heads with the evidence-state ring style and a pulse on surprise.
- Timeline: pause, 0.25×, 1×, 10×, 60×, 300×, 1000×; range −24 h…+6 h; LIVE = now − 15 s; zone labels per the §11 table; a 30-min frame buffer with a replay fallback.
- Train panel: quantile ribbon from `/api/run/{id}`, schedule, NTES ETA with its age, revision timeline.
- Status bar: A1 triple, evidence-age percentiles, NTES requests per minute, dispatcher state, models with their age, last frame age.
- Search.
- Drop to K = 8 when fps < 30 for 3 s.

- [ ] Tests (`timeline.test.ts`): zone selection for a given simTime; the LIVE offset; the K fallback trigger logic.
- [ ] Commit `task 36: dashboard`.
- [ ] **🛑 Monitor and user:** the monitor generates the tiles once with `pmtiles extract` (India bbox `68,6,98,37`) into `/data/tiles`, and runs a Playwright smoke test (load, WebSocket connects, a train is clickable, reload, network loss and resync, a +6 h jump). The user names their laptop; A7 is measured there (≥ 60 fps at K = 32). Record it in `docs/gates/phase5.md`.

### Task 37: OSM geometry for drawing (spec §7.6; deliberately late)

> Superseded by "Task 37 (amendment 2026-10-12)" before the spec coverage check.

**Files:** create `msynd/static/osm.py`, `tests/static/test_osm.py`.

**Interface:** `python -m msynd.static.osm --pbf india-latest.osm.pbf` uses osmium to extract `railway=rail`, then:
- maps each station to an OSM node by its `ref=` tag, otherwise the nearest node within 1.5 km with a fuzzy name match;
- finds the networkx shortest path between consecutive route stations;
- scores confidence: 1.0 when both stations matched by `ref` and the path is within ±10 % of timetable km; 0.5 when name-matched within tolerance; 0 otherwise (straight line);
- updates `ref.segment` and re-exports the network.

- [ ] Tests on a tiny synthetic graph:
  - `test_confidence_rules`;
  - `test_path_length_tolerance`;
  - `test_override_wins`.
- [ ] Commit `task 37: osm geometry`.

---

## M7 — Ops, GPU, acceptance

### Task 38a (amendment 2026-10-09): first live empirical fit, unpromoted

User-approved (spec §18 "Training data"). It proves the real-data path on runs that are already bridged and sends no MMT requests.

**Files:** create `msynd/ops/weekly.py` and `tests/ops/test_weekly.py`.

**Interfaces:** `python -m msynd.ops.weekly fit-empirical --models-dir DIR`, and `fit_empirical_live(conn, models_dir, now) -> dict[role, model_id]`.
- Labels come from `build_live_labels(conn, now - 6 h)`. They are enriched per run with `ref.train.class`, and for W with the scheduled dwell `dep_off - arr_off` from `ref.schedule_stop` of the bound version.
- For each role in run/dwell/origin, it calls `fit_empirical` with `cold_start = True`, saves to `DIR/live-<role>-<uuid>.json`, and inserts an `ml.model` row (kind `empirical`, train_window = min/max start_date). The metrics hold support counts: rows, distinct runs, distinct trains and service dates, per class.
- **Never `registry.promote`.** A role with zero labels is skipped and reported. As in `rstgcn.fit`, saved files are deleted if the transaction fails.

- [x] Tests:
  - the labels of a synthetic completed run give three unpromoted models with the correct support counts;
  - the active models are unchanged;
  - a role with no labels is skipped;
  - a failure removes the written files.
- [x] Commit `task 38a: live empirical fit`.
- [x] Monitor: run it against `msynd_view` (never `msynd`) and report the support counts. 2026-10-09, rolled back: 309 R / 298 W / 17 O rows; 17 runs, 17 trains, 3 service dates; classes EXP 10 runs, DRNT 3, SF 2, MEX 1, RAJ 1. Far below the support floors.

### Task 38: Nightly leg bias, weekly VM retrain and promotion (spec §12, §8.4–8.6)

> Split 2026-10-10: the bias part is Task 38b; the weekly part is Task 50 (holdout fit and evaluation, never promotes). Promotion stays manual (spec §18).

**Files:** modify `msynd/ops/nightly.py`; create `msynd/ops/weekly.py` and `tests/ops/test_bias.py`.

- **Bias:** over the last 7 days of matured segments, compute the median of (Δrun − active P50) per (halt pair, 3-hour bucket), shrunk by `n/(n+20)`, written as a new `gen` keyed by model (the previous generation is kept).
- **Weekly:** retrain LightGBM R/W/O on the VM, validate on the most recent 2 full weeks with `promotion_ok`, and promote or keep the active model, logging the decision to `sim.metric` and `docs/gates/weekly-<date>.md`. Any failure keeps the active model.

- [ ] Tests:
  - `test_bias_shrinkage`;
  - `test_new_model_starts_zero_bias`;
  - `test_failed_training_keeps_active`.
- [ ] Commit `task 38: nightly bias and weekly retrain`.

### Task 38b (amendment 2026-10-10): nightly leg bias (Task 38 bias part)

Spec §4.3 and §12.1. It reads only the evidence DB and sends no HTTP.

**Files:** modify `msynd/ops/nightly.py`; create `tests/ops/test_bias.py`.

**Interfaces:**
- `leg_bias(conn, now, models=None) -> int | None`. `models` defaults to `registry.active(conn)`; tests pass stubs.
  - The run model is `models["run"]`. If its id is `None` (provisional, unsaved), write nothing and return `None`.
  - **Segments:** the `role == "run"` labels of `build_live_labels(conn, now - 6 h)` (finished runs, matured events, current truth) whose actual departure at the from-halt (`obs.event`, kind `dep`, halt `idx`) is at or after `now - 7 days`. The departure delay is `d` = actual departure (run clock) − `dep_off`.
  - **P50:** per run, `segment_queries(conn, run_id, now)`, then `msynd.engine.cache._predict(conn, artifact, queries)`. These are the same masses that `tables_for` builds. P50 = `eff_quantile(Eff(masses[i], True, lo[i].max(), hi[i].max(), j, w, 0, 1, -inf, 0, 1), 0.5)` with `(j, w) = level_weight(d)`. Residual = label − P50.
  - **Key:** `(from_stn, to_stn, int(sched_dep % 1440 // 180))` from the query row. This is exactly the key `cache.tables_for` reads.
  - **Value:** `bias_min = median(residuals) * n/(n+20)`, where `n` is the number of residuals.
  - **Write, in one transaction:** `gen = coalesce(max(gen) for this model_id, 0) + 1`. Insert one row per key. Then delete this model's rows with `gen < new gen - 1` ("keeping the previous one"). Other models' rows are never touched.
  - With no residuals, write nothing and return `None`.
- `main()` runs `leg_bias` first, in its own transaction. An exception there is logged with `log.exception`, and retention and backup still run.

- [x] Tests:
  - `test_bias_shrinkage`: a stub run model whose P50 differs by delay level; three segments of one pair and bucket, with known labels and departure delays between two levels. Expect `bias_min == median(label - interpolated P50) * 3/23` and `n == 3`.
  - `test_bucket_matches_engine`: a segment with `sched_dep >= 1440` (multi-day) is written under `int(sched_dep % 1440 // 180)`, and `cache.tables_for` at the new gen returns that bias for the segment.
  - `test_window_and_maturity`: a departure older than 7 days, an arrival newer than 6 h, and an unfinished run are all excluded.
  - `test_new_model_starts_zero_bias`: an old active model has gen 3 rows. After `registry.promote` of a new model, the engine's bias generation (`msynd.engine.__main__._versions`) is 0 and `tables_for` gives zero bias. `leg_bias` then writes gen 1 for the new model and leaves the old model's rows untouched.
  - `test_keeps_previous_generation`: after three calls, only gens 2 and 3 remain.
  - `test_provisional_run_model_writes_nothing`.
  - `test_bias_failure_still_runs_retention_and_backup` (`main`'s collaborators monkeypatched).
- [x] Commit `task 38b: nightly leg bias`.

### Task 39: Kaggle pipeline (gated: G1 allows export **and** 🛑 G6 passes)

**Files:** create `msynd/ops/export.py` (month-partitioned Parquet plus a zip of `msynd`; verify privacy through the API after each version), `kaggle/kernel.py` (feature build, ~12-config sweep, neural candidate: GRU over ≤ 8 observed deltas plus §8.2 features, a 302- or 181-way head, CE on a Gaussian-smoothed target σ = 1.5 bins + 0.5·CRPS, AdamW with a cosine schedule and AMP, ONNX export, `metrics.json`), and extend `msynd/ops/weekly.py` (dataset version → kernel push → poll every 5 min, giving up after 12 h → output → `promotion_ok` against LightGBM on identical cases).

- [ ] **🛑 G6 (user and monitor):** private dataset with privacy verified; CLI push, status and output work; GPU confirmed; torch, onnx and lightgbm import offline. If this fails, train on the VM only (Task 38 already covers it).
- [ ] Tests:
  - `test_export_partitions_and_excludes_raw_bodies`;
  - `test_onnx_head_dims` (R 302, W 181, O 302);
  - `test_weekly_timeout_keeps_active`.
- [ ] Commit `task 39: kaggle pipeline`. Phase 6 exit: two unattended weekly cycles, with their decisions logged.

### Task 40: Prospective acceptance and the C1 ablation hook (spec §1 A3/A4, §4.4, Phase 7)

**Files:** create `msynd/ops/acceptance.py`; modify `msynd/features/build.py` (C1 feature behind `FEATURE_VER = 2`, off by default).

- `python -m msynd.ops.acceptance --model-frozen-since <date>` uses only `sim.forecast_issue` rows of the frozen model (4 weeks, a period never used for selection). It reports:
  - A3: P50 MAE at 2 h and 6 h against B1 and B2 (the stored `b2`) on identical eligible cases, with a run-day block-bootstrap CI;
  - A4: coverage, width and CRPS, overall and on disruption days (top-decile network surprise counts).
- C1 (recent observed Δrun residuals of other runs on the same halt pair, as of `received_at`) is enabled only if an ablation shows a CRPS gain with CI > 0 and A4-band calibration. C2 gets its own plan when reached.
- [ ] Tests:
  - `test_acceptance_uses_only_frozen_issue_rows`;
  - `test_c1_feature_asof`.
- [ ] Commit `task 40: acceptance and c1 hook`.

---

## Follow-ups (2026-10-08)

### Task 41: Smooth fitted empirical histograms (Task 27 follow-up)

An unsmoothed histogram gives zero mass to unseen bins. An observed actual there gives every particle W = 0 and forces the `data_invariant` override. The RSTGCN prototype hit this 92 times in 13,510 origin cases.

**Files:** modify `msynd/models/empirical.py` and `tests/models/test_empirical.py`.

**Interfaces:** every histogram `fit_empirical` stores (class-pooled and W dwell buckets) becomes `0.99·fitted + 0.01·prior`, where the prior is the matching `_prior(role, dwell)` masses (the bucket's dwell for W). Tails take `tau = max(fitted, prior)` per side. `provisional_models()` is unchanged. Mark the 1 % with `# ponytail:` (fixed mix weight; upgrade: fit it on validation log-score).

- [x] Tests:
  - `test_unseen_bin_has_nonzero_mass` (labels all 0 → the mass of the bin for +100 is > 0 for run and origin, and for a W bucket);
  - `test_smoothed_masses_sum_to_one`;
  - the existing empirical and cache tests still pass.
- [x] Implement; commit `task 41: smooth empirical histograms`.

### Task 42: RSTGCN history import, station zones, cold-start O and R

RSTGCN (github `KoyenaChowdhury/RSTGCN`, no licence) holds Sept 2024: 1.28 M station rows, 57,485 runs. Use is personal only; the raw data never leaves the PC or VM. Meaning (spec §18, 2026-10-08): one arrival-delay value per station (departure is derived), early running clipped to 0. G4 does not apply.

**Files:** create `msynd/db/migrations/0003_history_seq.sql` (`obs.history_daily` gains nullable `seq smallint`, the per-run file order; user ruling 2026-10-08), `msynd/ops/rstgcn.py`, `tests/ops/test_rstgcn.py`, `tests/fixtures/rstgcn_mini/` (a hand-written CSV with 3 runs, a zone JSON with 3 stations); modify `data/static_sources.json` (`"rstgcn": {"repo": "KoyenaChowdhury/RSTGCN", "sha": "160da04cd3b5dd72588b6cc72b08e84e0b7210b4", "path": "Indian-Railway-Network-and-Delays.zip", "sha256": "f8a6fe9e70972acb8ad98bbe5b7eb594c5b6b3475e21efd6d3b9e19049bdaf91"}`).

**Input format (monitor, measured 2026-10-08 on the pinned zip):**
- Zip members sit under `Indian-Railway-Network-and-Delays/`; read `train_routes_delays_Sep2024.csv` and `stations_zones_mapping.json` from there with `zipfile` (no extraction).
- CSV, UTF-8, header `train,date,station,sch_arr,act_arr,arr_delay,sch_dep,act_dep,dep_delay`. Example row: `12303,2024-09-02,BWN,09:05 AM,09:20 AM,15.0,09:08 AM,09:23 AM,15.0`. `train` is a train-number string, `date` is ISO `YYYY-MM-DD` (the run's origin date), `station` is a station code, and `arr_delay` is a float string of whole minutes (0.0…4336.0; no empty cells). A run is `(train, date)`; its rows are not guaranteed contiguous, so `seq` counts per run in file order. Only `train`, `date`, `station` and `arr_delay` are read.
- Zone JSON: one flat object `{"<station code>": "<zone code>"}`, 4,735 string values, e.g. `{"AADR": "NR", "AAG": "CR"}`.

**Interfaces:**
- `python -m msynd.ops.rstgcn import --zip PATH` verifies the sha256, then:
  - loads `train_routes_delays_Sep2024.csv` into `obs.history_daily` (`value_min = arr_delay`, `seq` = 1-based file order within a run, `occ` by repeats in file order within a run, `fetch_id` NULL; re-import is idempotent);
  - fills `ref.station.zone` where it is NULL from `stations_zones_mapping.json`, never overwriting a non-NULL value.
- `python -m msynd.ops.rstgcn fit --models-dir DIR` builds labels from `obs.history_daily` for runs dated 2024-09:
  - O = the value at the lowest `seq` of each run (rows with NULL `seq` are skipped);
  - R ≈ composite = value(B) − value(A) for consecutive `seq`. `# ponytail:` the composite includes dwell, so R is over-dispersed while W stays provisional; live R replaces it after 4 weeks (Task 38);
  - class from `ref.train` (UNKNOWN otherwise); split train = days 1–23, valid = 24–30;
  - `fit_empirical(labels, role)` for `origin` and `run`, with `cold_start=True`;
  - inserts `ml.model` rows (`kind='empirical'`, `train_window`, `metrics` = mean validation log-score of the candidate and of the provisional prior);
  - promotes through `registry.promote` only when the candidate's validation log-score beats the provisional prior's.
- `build_live_labels` (Task 12) still never reads `obs.history_daily`.

- [x] Tests:
  - `test_import_seq_occ_and_idempotent`;
  - `test_zone_fill_never_overwrites`;
  - `test_bad_sha_refuses`;
  - `test_labels_origin_and_composite`;
  - `test_fit_marks_cold_start_and_promotes_only_on_gain`.
- [x] Implement; commit `task 42: rstgcn history and cold-start models`.
- [x] **Monitor:** run `import` with the downloaded zip and `fit` against the dev DB. Record the counts and log-scores in `docs/gates/rstgcn.md`.

### Task 43: Dwell prior below the scheduled dwell (Task 27 follow-up)

`_prior('dwell', d)` is `d + Exp(2)`, so it gives zero mass to every dwell shorter than the scheduled one. A real short stop then gives every particle W = 0 and forces the `data_invariant` override. Task 41's 1 % smoothing does not help, because the prior it mixes in has the same gap. Decision (user, 2026-10-08): start the dwell prior at half the scheduled dwell.

**Files:** modify `msynd/models/empirical.py` and `tests/models/test_empirical.py`.

**Interfaces:** `_prior('dwell', d)` becomes `d/2 + Exp(2)` on the `NONNEG` bins (`# ponytail:` fixed half-dwell edge; upgrade: fit the edge from live W labels). The provisional model and the Task 41 smoothing prior both call it, so both pick up the change; nothing else changes.

- [x] Tests:
  - `test_dwell_prior_mass_below_scheduled` (d = 10 → bins 5…9 carry mass > 0; bins 0…4 carry 0; masses sum to 1);
  - `test_dwell_prior_zero_dwell` (d = 0 → identical to before);
  - the existing empirical, cache and replay tests still pass.
- [x] Implement; commit `task 43: dwell prior below scheduled dwell`.


### Task 44 (amendment 2026-10-09): `ensure_run` for historical runs (Task 21 follow-up)

User-approved (spec §18 "Training data"). Harvested finished runs can be days old, and `create_runs` refuses them (three-day recovery window, destination + 12 h filter). The bridge already has a private `_run` that does the right thing; this moves it to the lifecycle owner so the harvester (a later task) and the bridge share it.

**Files:** modify `msynd/collect/lifecycle.py`, `msynd/ops/pilot_bridge.py`, `tests/collect/test_lifecycle.py`, `tests/ops/test_pilot_bridge.py`.

**Interfaces:** `lifecycle.ensure_run(conn, train: str, start: date) -> int | None`.
- An existing `(train, start_date)` run is returned unchanged; its schedule and route binding are never rewritten.
- Otherwise it inserts one run bound by `SCHEDULES_SQL` (the version valid on `start` with the weekday bit) and `LATEST_ROUTE_SQL`. No time-window filter. `ON CONFLICT` returns the existing id.
- No valid version → `None`, nothing inserted.
- The bridge calls `ensure_run` (delete `_run`) and keeps its `no_version` count. It also prints the skipped `train/start` keys, deduplicated and capped at 20 (`skipped=...`).
- Exceptions are not applied here (`resched_min` stays 0; `# ponytail:` upgrade: apply the latest exception revision when the harvester stores exceptions).

- [x] Tests:
  - `test_ensure_run_creates_old_run` (start 30 days before now → created; `create_runs` at now does not create it);
  - `test_ensure_run_keeps_existing_binding` (existing run with an older route version → same id, binding unchanged after a newer route version exists);
  - `test_ensure_run_selects_date_valid_version` (two versions with disjoint validity → the one valid on `start`);
  - `test_ensure_run_no_version_returns_none` (wrong weekday → `None`, no row);
  - the bridge tests still pass, and one asserts the skipped key appears in the summary.
- [x] Commit `task 44: ensure_run for historical runs`.

### Task 45 (amendment 2026-10-09): durable fetch provenance and label eligibility (Task 20 follow-up)

User-approved (spec §18 "Training data"). Harvested and polled fetches stay source `mmt_status`; the collection mode and the parser flags are kept on the fetch row (180-day retention), because bodies are pruned after 24 h.

**Files:** create `msynd/db/migrations/0006_fetch_provenance.sql`; modify `msynd/collect/store.py`, `msynd/labels.py`; tests in `tests/db/test_schema.py`, `tests/collect/test_store.py`, `tests/ops/test_pilot_bridge.py`, `tests/test_labels.py`.

**Interfaces:**
- Migration 0006: `obs.fetch.mode text NOT NULL DEFAULT 'poll' CHECK (mode IN ('poll','harvest','probe','backfill'))` and `obs.fetch.flags text[] NOT NULL DEFAULT '{}'`. Nothing else. (The dispatcher and admission set `mode` in a later task; until then every row is `poll`.) Never apply it to `msynd` while the pilot runs.
- `store_status`: inside its transaction, `UPDATE obs.fetch SET flags = <sorted page.flags>, run_id = COALESCE(run_id, <run_id>)` when `page.flags` is non-empty. Completion stays "bound destination arrival actual present" (already so).
- Bridge: no code change; the pilot only polls, so copied rows take the default `'poll'`, and flags come from `store_status`.
- `build_live_labels`: a run is ineligible (no labels at all) when any of its fetches has flag `exception_unparsed`. `delay_gt_24h` stays eligible (spec: kept, flagged). Maturity unchanged (actual time ≤ cutoff); receipts are never invented.

- [x] Tests:
  - `test_fetch_mode_check` (mode 'weird' → CheckViolation; default 'poll'; flags default empty);
  - `test_store_records_parser_flags` (page with `delay_gt_24h` → fetch row flags `{delay_gt_24h}`, run_id set);
  - `test_unparsed_exception_run_has_no_labels` (completed matured run with a flagged fetch → no labels; same run without the flag → labels);
  - `test_delay_gt_24h_run_keeps_labels`;
  - the bridge tests still pass, and a copied row has `mode='poll'`.
- [x] Commit `task 45: fetch provenance and label eligibility`.

### Task 46 (amendment 2026-10-09): per-attempt daily budgets by collection mode (Task 3/22 follow-up)

User-approved (spec §18 "Training data": ≤ 13,200 attempts/day at 0.2 req/s, one owner, ramp ceilings). Every HTTP attempt, including retries and failures, counts against its mode's IST-day budget. Counting reads `obs.fetch`, so a restart cannot reset it.

**Files:** modify `msynd/collect/dispatcher.py`, `msynd/collect/mmt.py`, `msynd/collect/__main__.py`; tests in `tests/collect/test_dispatcher.py`, `tests/collect/test_collector.py` (or the existing collector test file).

**Interfaces:**
- `class BudgetExhausted(Exception)` in `dispatcher.py`, carrying the mode.
- `Dispatcher(..., budgets: dict[str, int] | None = None)`. `None` keeps today's behaviour (no limit; tests and capture). Otherwise a mode missing from the dict has budget 0.
- `Dispatcher.request(..., mode='poll', max_attempts=5)`. `mode` is written on the `obs.fetch` insert (column from Task 45). Inside the host lock, **after** the spacing wait and **before** inserting the row, count `obs.fetch` rows for this host and mode with `started_at` on the current IST date (`clock.now()`); if the count ≥ the budget, raise `BudgetExhausted(mode)` with no row inserted. `max_attempts` (1…5) caps the existing retry loop.
- `MmtClient.train_running(..., mode='poll', max_attempts=5)` passes both through.
- Collector CLI: repeatable `--budget MODE=N` (modes `poll|harvest|probe|backfill`, N ≥ 0); default `poll=5000` only (the approved 16 Oct polling ceiling). The sum of budgets must be ≤ 13,200, else exit with an error. `run_loop` catches `BudgetExhausted` for a poll job, logs it once per IST day, and sleeps 60 s before the next admission.

- [x] Tests:
  - `test_budget_counts_every_attempt_including_retries` (budget 3, transport errors → exactly 3 rows, then `BudgetExhausted`);
  - `test_budget_survives_restart` (rows from an earlier dispatcher on the same IST day count);
  - `test_budget_resets_at_ist_midnight` (rows at 23:59 IST do not count at 00:01 IST next day; use a UTC instant in 18:30–24:00 UTC);
  - `test_budget_per_mode` (harvest exhausted, poll still admitted);
  - `test_missing_mode_budget_is_zero` and `test_no_budgets_unlimited`;
  - `test_max_attempts_caps_retries`;
  - `test_cli_rejects_budget_sum_over_13200`;
  - existing dispatcher and collector tests still pass.
- [x] Commit `task 46: per-attempt daily budgets`.

### Task 47 (amendment 2026-10-09): scoped polling slots and finished-run harvest admission (Task 22 follow-up)

User-approved (spec §18 "Training data"). Scope: route ≥ 150 timetable km, any class. Polling keeps a small cohort with full receipt histories; harvest fetches each finished run once in the remaining capacity. Probe and backfill are not part of this task.

**Files:** modify `msynd/collect/admission.py`, `msynd/collect/__main__.py`; tests in `tests/collect/test_admission.py` and the collector test file.

**Interfaces:**
- `SCOPE_MIN_KM = 150.0` in `admission.py`. A run is in scope when the largest `ref.route_stop.dist_km` of its bound route version is ≥ 150. Both the polling and harvest queries apply it.
- **Polling slots** (`--poll-slots`, default 6; `--poll-interval` seconds, default 120). Reuse the pilot's cohort rule (`msynd/collect/pilot.py` `run_pilot`): a run in the cohort stays until it is no longer running/departed-scheduled; vacancies are filled from eligible in-scope runs preferring trains with no status fetch in the last 7 days, then fewest status fetches, then lowest id. A cohort run is due when its last status attempt is ≥ `poll_interval` old. The cohort is derived from `obs.fetch` (runs with a `mode='poll'` attempt in the last 30 min that are still eligible), so a restart keeps it. Due poll jobs use `mode='poll'`, `max_attempts=5`.
- **Harvest candidates:** in-scope runs with lifecycle in (`scheduled`, `running`, `expired_unknown`) whose scheduled destination arrival (IST start date + last `arr_off` + `resched_min`) + 12 h ≤ now < + 48 h, and:
  - no `mode='harvest'` `ok` fetch yet; or
  - exactly one `ok` harvest fetch, no destination actual, and destination + 24 h ≤ now (the one completion deferral);
  - and fewer than 4 harvest attempts for the run in total (all outcomes).
  Oldest destination time first, then id. Harvest jobs use `mode='harvest'`, `max_attempts=1`; a failed attempt is retried by admission no sooner than 10 min after it.
- **Order:** recovery/breaker handling as today → due poll job → harvest job, only when no cohort run becomes due within the next `1/rate` seconds → otherwise sleep 1 s. `BudgetExhausted` for harvest skips harvest until the next IST day; for poll, as Task 46.
- `run_loop` stores harvest results with `store_status` exactly like polls.
- Older runs (destination + 48 h passed) are backfill, which needs its own user-approved task; never admitted here.

- [x] Tests:
  - `test_scope_excludes_short_routes` (149 km route → never polled or harvested; 150 km → eligible);
  - `test_cohort_bounded_and_kept_across_restart`;
  - `test_vacancy_prefers_unseen_train`;
  - `test_poll_due_after_interval_only`;
  - `test_harvest_window_12h_to_48h` and `test_harvest_one_deferral_to_24h`;
  - `test_harvest_attempt_cap_and_retry_delay`;
  - `test_harvest_never_delays_due_poll` (a cohort run due within 1/rate → poll job, not harvest);
  - `test_harvest_budget_exhausted_skips_until_next_ist_day`;
  - `test_harvest_completes_run` (MockTransport page with the destination actual → run `completed`, fetch `mode='harvest'`);
  - existing admission and collector tests still pass (update them only where polling now uses slots).
- [x] Commit `task 47: scoped polling slots and harvest admission`.

### Task 48 (amendment 2026-10-09): frozen fit boundary and week-block cross-fitted encoders (Task 26 follow-up)

From the training-data review (spec §18 "Training data"). Two leaks in `training_frame`:
- a training row's own target enters its `train_encoded`, and its own observed running time enters its pair `slack` P05;
- labels and departures are read as current truth at `CURRENT_TIMESTAMP`, so a later fit sees revisions received after the fit date.

**Files:** modify `msynd/features/build.py`, `msynd/labels.py`; tests in `tests/features/test_features.py`, `tests/test_labels.py`.

**Interfaces:**
- `build_live_labels(conn, matured_before, asof=None)`. `asof=None` keeps today's behaviour. Otherwise every event read is `obs.event_asof(asof)` instead of `obs.event`, the lifecycle check uses only evidence present at `asof` (completed = the bound destination arrival actual exists in `event_asof(asof)`; `expired_unknown` runs are included only if `updated_at ≤ asof`), the `exception_unparsed` exclusion counts only fetches with `received_at ≤ asof`, and `resched_min` already uses receipt order.
- `training_frame(conn, fold, asof)`: `asof` is required; labels come from `build_live_labels(conn, asof - 6 h, asof)`; departures use `obs.event_asof(asof)`. Remove `CURRENT_TIMESTAMP`.
- `FoldEncoders.fit(train_rows)` is unchanged (it is the encoder used for validation and serving rows: all training rows).
- New `FoldEncoders.cross_fit(train_rows) -> pa.Table`: transforms each **training** row with encoders fit on the training rows of the **other** origin weeks only (the global mean too). A row with no other-week training rows gets `train_encoded = None` and `slack = None` (LightGBM treats them as missing). `training_frame` uses `cross_fit` for `split == 'train'` rows and `fit(...).transform` for `split == 'valid'` rows, preserving row order.

- [x] Tests:
  - `test_training_row_target_never_in_own_encoding` (two training weeks; changing one row's value changes only the other week's `train_encoded`/`slack`, never its own);
  - `test_single_training_week_encodes_missing`;
  - `test_validation_rows_use_all_training_rows` (existing behaviour);
  - `test_frozen_labels_ignore_later_revisions` (a destination actual received after `asof` → run absent at `asof`, present later; a corrected actual received after `asof` → old value);
  - `test_training_frame_asof_is_reproducible` (same `asof` before and after inserting later revisions → identical X and y);
  - existing features, labels and weekly tests still pass.
- [x] Commit `task 48: frozen fit boundary and cross-fitted encoders`.

### Task 49 (amendment 2026-10-09): persisted encoders, explicit model inputs, issue-time model tables (Task 27/28/31 follow-up)

From the training-data review (spec §18 "Training data"). Three serving gaps:
- `segment_queries` leaves `train_encoded` and `slack` empty, and no artifact stores the fold encoders, so a LightGBM model trained on encoded rows is served with them missing;
- `train_lgbm` uses every column it is given, so `training_frame`'s identifiers (`run_id`, `start_date`, `split`, …) would become features;
- `cache.tables_for` always reads features at `statement_timestamp()`, and `eval.baselines.b3_active` duplicates its table-building code to get issue-time features.

**Files:** modify `msynd/features/build.py`, `msynd/models/lgbm.py`, `msynd/engine/cache.py`, `msynd/eval/baselines.py`; tests in `tests/features/test_features.py`, `tests/models/test_lgbm.py`, `tests/engine/test_cache.py`, `tests/eval/test_baselines.py`.

**Interfaces:**
- `MODEL_FEATURES` in `msynd/features/build.py`: the R model inputs, in this order: `class, zone, segments_remaining, is_last, km, sched_dep, sched_run, sched_dwell, last_delta_run, segments_since_observed, hour_sin, hour_cos, weekday, month, fog_north, holiday, train_encoded, slack, d`. Identifiers (`run_id, start_date, train, idx, from_stn, to_stn`) and metadata (`split`) are never model inputs; the encoders carry train and halt-pair information.
- `FoldEncoders.to_json() -> dict` / `FoldEncoders.from_json(dict)`; pair keys are serialised as `"FROM|TO"`. Round trip is exact.
- `train_lgbm(X, y, role, encoders=None)`: when `encoders` is given it is stored on the artifact and saved/loaded with it (key `"encoders"`, absent for none). `X` must already contain only the model inputs; `train_lgbm` raises `ValueError` if any of `run_id, start_date, train, idx, from_stn, to_stn, split` is a column.
- `cache._predict`: if the artifact has encoders, apply `encoders.transform` to the expanded queries before selecting `feature_names`. Artifacts without encoders behave as today.
- `cache.tables_for(conn, ev, active, bias_gen, asof=None)`: `asof=None` keeps `statement_timestamp()`; otherwise `segment_queries` reads at `asof`. `bias_gen=None` means zero bias (no `ml.leg_bias` read). The cache key already includes the feature rows, so different `asof` values never share an entry.
- `b3_active` builds its tables with `tables_for(conn, ev, models, None, asof=case.issued_at)` and deletes its duplicated table code. Its frozen-version, replay and return behaviour is unchanged.

- [x] Tests:
  - `test_encoders_json_roundtrip`;
  - `test_lgbm_artifact_saves_and_applies_encoders` (train with encoders, save, load; serving a query whose `train_encoded`/`slack` are empty gives the same masses as serving the encoder-transformed query to the in-memory model);
  - `test_train_lgbm_rejects_identifier_columns`;
  - `test_tables_for_asof_ignores_later_events` (an event received after `asof` changes `tables_for` output at `now` but not at `asof`);
  - `test_tables_for_no_bias_reads_none` (`bias_gen=None` → zero bias even when `ml.leg_bias` has rows);
  - existing cache, lgbm, baselines and engine tests still pass.
- [x] Commit `task 49: persisted encoders and issue-time model tables`.

### Task 50 (amendment 2026-10-09): weekly holdout fit and evaluation (Task 29/38 follow-up)

From the training-data review (spec §18 "Training data"; Phase 3 gate unchanged; personal local training only). `fit_empirical_live` fits on every matured label, so no candidate is ever scored on data it did not see. This task adds a frozen, holdout-only weekly cycle. It never promotes on its own.

**Files:** modify `msynd/ops/weekly.py`; tests in `tests/ops/test_weekly.py`.

**Interfaces:**
- `python -m msynd.ops.weekly evaluate --models-dir DIR [--now ISO]` calls `evaluate_week(conn, models_dir, now) -> dict` and prints it as JSON. Defaults: `now` = current UTC.
  - **Boundary:** `asof = now`. The validation weeks are the two most recent complete IST Monday–Sunday weeks before `now` (the same window as `promotion_ok`). The training weeks are every earlier week with labels. With no training week, or fewer than 1,000 distinct mature training runs across ≥ 7 service dates, report `{"status": "insufficient", ...support}` and fit nothing.
  - **Candidates, fitted on training-week rows only, labels frozen at `asof`** (`build_live_labels(conn, asof - 6 h, asof)`):
    - R: `train_lgbm(X[MODEL_FEATURES], y, 'run', encoders)` with `X, y` = the `split == 'train'` rows of `training_frame(conn, fold, asof)` and `encoders = FoldEncoders.fit(train rows)`;
    - W and O: `fit_empirical` on the training-week labels (as `fit_empirical_live` builds them).
  - **Support floors** (debate, approved): R ≥ 100 distinct runs per class; W ≥ 100 runs per class and ≥ 30 runs per class/dwell bucket; O ≥ 200 origins per class. The report lists every class with its count and `below_floor: true|false`. A role whose every class is below its floor is not fitted (active model kept for that role in the candidate set).
  - **Scoring:** `asof_cases(conn, start of validation weeks, min(end of validation weeks, asof))` scored with `score({"candidate": b3_active(conn, models=candidate_models), "active": b3_active(conn, models=active(conn))})`, so both see identical eligible cases. Then `promotion_ok(result["candidate"], result["active"], now=now)`.
  - **Recording:** each fitted artifact is saved under `models_dir` and inserted into `ml.model` with `active = false`, `train_window` = the training weeks, and `metrics` = support, floors, overall scores and the decision (accepted + reasons). One `sim.metric` row `weekly_candidate_accepted` (1/0) at `now`. Any exception deletes the saved files and inserts nothing (as `fit_empirical_live`).
  - Promotion stays manual (`registry.promote`), because the Phase 3 gate is unchanged.
- `fit-empirical` is kept unchanged.

- [x] Tests (small synthetic data; lower the floors through a module constant `FLOORS` monkeypatched in tests):
  - `test_insufficient_support_fits_nothing`;
  - `test_candidates_never_see_validation_rows` (a label in a validation week changes nothing in the saved candidate artifacts);
  - `test_scores_on_identical_cases` (candidate and active records have the same keys);
  - `test_below_floor_class_reported`;
  - `test_never_promotes` (no `ml.model` row becomes active);
  - `test_failure_leaves_no_rows_or_files`.
- [x] Commit `task 50: weekly holdout fit and evaluation`.

### Task 51 (amendment 2026-10-10): G7(d) at the measured publication lag (G2 follow-up)

From the G2 decision (spec §18). `data/params.json` now holds the measured `S.mmt_status` (`provisional: false`). Spec §15 G2 requires rerunning G7(d) at the ε CI upper bound and at twice the tail scale. The existing `d_epsilon_upper` and `d_tail_x2` cases use a synthetic table (ε = 0.05, τ = 80) and stay as they are.

**Files:** modify `tests/g7/reference.py`, `tests/collect/test_pilot.py`.

**Interfaces:**
- Two new G7 case names after `d_tail_x2`:
  - `d_measured_eps_upper`: the `d_soft_lag` case with its S table read from `data/params.json` `S.mmt_status` (keys `lag`, `survival`, `tau`, `eps`), with `eps` replaced by `eps_ci[1]`;
  - `d_measured_tail_x2`: the same measured table with `tau` doubled (ε unchanged).
- Same evidence, grid, N (10k and 100k) and A8 tolerance as the other `d_` cases. They are parametrised through `CASE_NAMES` like every other case.
- `test_provisional_params_are_usable` becomes `test_measured_params_are_usable`:
  - `provisional is False` and `minute == "trunc"`;
  - `_lag_table(S.mmt_status)` accepts the table;
  - `0 < eps <= eps_ci[1]` and `tau > 0`.

- [x] Tests:
  - the renamed params test;
  - the new cases pass under `uv run pytest -q -m slow tests/g7 -k measured`.
- [x] Commit `task 51: g7(d) at measured publication lag`.
- [x] Monitor: append the two cases' numbers to `docs/gates/G7.md`.


### Task 52 (amendment 2026-10-11): residential relay egress for the VM collector (CP-3 follow-up)

**Historical design:** Task 53 replaced its CONNECT transport, and Task 78 removed the
laptop dependency. The IP-rejection explanation below was the inference at the time;
use the current direct-cloud deployment instructions in `docs/deploy.md`.

At the time, MMT requests from the VM client failed while the user's home connection worked through the pilot at the same rate. This was initially attributed to the VM's IP, so the collector's source connection was sent through the user's PC: a reverse SSH tunnel to an allowlisted CONNECT relay. The dispatcher on the VM still owned spacing, budgets and logging. The relay forwarded only to MMT, without fingerprint spoofing, challenge solving or a third-party proxy.

**Files:** modify `msynd/collect/dispatcher.py`, `compose.yaml`, `docs/deploy.md`, `tests/collect/test_dispatcher.py`, `tests/test_deploy.py`; create `msynd/collect/relay.py`, `compose.relay.yaml`, `tests/collect/test_relay.py`.

**Interfaces:**
- `real_transport()` keeps the `MSYND_LIVE` guard and passes `proxy=MSYND_PROXY` to `httpx.AsyncHTTPTransport` when that env var is non-empty (an `http://` URL), and no proxy otherwise.
- `python -m msynd.collect.relay [--port 18080] [--ssh USER@HOST] [--identity PATH]` runs on the PC:
  - an asyncio CONNECT proxy bound to 127.0.0.1 only;
  - only `CONNECT rails-ris.makemytrip.com:443` is tunnelled; any other target or method gets 403 and is closed; the request head is capped at 8 KiB; the upstream connect times out after 30 s (502);
  - with `--ssh`, it supervises `ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:PORT:127.0.0.1:PORT [-i PATH] USER@HOST`, restarting it 10 s after any exit;
  - on Windows, it asks the OS to stay awake while it runs (`SetThreadExecutionState`), which is released on exit.
- `compose.yaml` collector command adds `--budget poll=${MMT_POLL_BUDGET:-5000} --budget harvest=${MMT_HARVEST_BUDGET:-0}`, so a ramp step edits only `.env`.
- `compose.relay.yaml` is opt-in via `COMPOSE_FILE=compose.yaml:compose.relay.yaml`: the collector uses `network_mode: host`, `DATABASE_URL` points at `127.0.0.1:55432` (same `POSTGRES_PASSWORD` default), and `MSYND_PROXY=http://127.0.0.1:18080`.
- `docs/deploy.md` gains a short "Residential relay" section: the PC command, the VM sshd keep-alive drop-in (`ClientAliveInterval 30`, `ClientAliveCountMax 3`, so a dead session frees the forwarded port), the `.env` lines, and that the PC must stay powered with the lid open.

- [x] Tests:
  - `test_real_transport_uses_proxy_from_env` (with and without `MSYND_PROXY`; still raises without `MSYND_LIVE`);
  - relay: an allowlisted CONNECT tunnels bytes both ways to a local echo server (allowlist patched); other host, other port and non-CONNECT get 403; an oversized head is rejected; the supervisor restarts a fake ssh command;
  - deploy: the budget interpolation, and the relay override's network mode, DSN and proxy.
- [x] Commit `task 52: residential relay egress`.
- [x] Monitor: VM sshd drop-in and the tunnel work, and the relay reaches MMT from the PC (200, 0.5 s). From the VM, the image's client (Debian 13, OpenSSL 3.5) completes TLS through the tunnel, but MMT never answers (30 s read timeout, 2 KB up against the PC's 1 KB). Superseded by Task 53.

---

### Task 53 (amendment 2026-10-11): the PC's client makes the MMT request (supersedes Task 52's CONNECT path)

**Historical default, now fallback:** Task 78 moved collection directly onto Oracle.
This client relay remains available for an explicit rollback only.

Through Task 52's tunnel, Akamai accepts TLS from the VM image's client (OpenSSL 3.5) but never answers the request, while the PC's client (the one the pilot used, OpenSSL 3.0) gets 200 at once. The VM's TLS is not tuned to imitate anything. Instead, the PC's own httpx client makes each HTTPS request, so source traffic is exactly what the approved PC pilot sent. The VM dispatcher still owns spacing, budgets, retries and logging, and the relay forwards one request per incoming request, never retrying.

**Files:** modify `msynd/collect/relay.py`, `msynd/collect/dispatcher.py`, `compose.relay.yaml`, `docs/deploy.md`, `tests/collect/test_relay.py`, `tests/collect/test_dispatcher.py`, `tests/test_deploy.py`.

**Interfaces:**
- `relay.handle` replaces CONNECT tunnelling with plain HTTP/1.1 forwarding:
  - it reads one request (head ≤ 8 KiB, `Content-Length` body ≤ 64 KiB, no chunked);
  - the `Host` header must equal the allowlisted host and the target must be origin-form (`/path?query`), else 403;
  - it sends the request to `https://HOST/path` through a shared `httpx.AsyncClient` (no redirects, 25 s timeout, no retries), forwarding the method, body and the end-to-end request headers (hop-by-hop headers, `Host`, `Content-Length` and `Accept-Encoding` are dropped);
  - it returns the upstream status with `Content-Type` and `Retry-After` when present, the decoded body, `Content-Length` and `Connection: close`;
  - an upstream httpx error gives 502 and a log line without payloads.
- `real_transport()` keeps the `MSYND_LIVE` guard. When `MSYND_RELAY` (an `http://host:port` URL) is set, it returns a transport that rewrites each request's scheme, host and port to the relay while keeping the original `Host` header. `MSYND_PROXY` is removed.
- `compose.relay.yaml` sets `MSYND_RELAY=http://127.0.0.1:18080` in place of `MSYND_PROXY`.
- `docs/deploy.md`: the relay section says the PC makes the HTTPS request.

- [x] Tests:
  - relay: a forwarded POST reaches a local stub server (allowlist and upstream base patched to loopback http) with method, path, query, body and headers intact, and the status, content type, Retry-After and body come back; a wrong Host, an absolute-form target, CONNECT, an oversized head and an oversized body get 403 with no upstream call; an upstream failure gives 502;
  - dispatcher: with `MSYND_RELAY`, a request to `https://rails-ris.makemytrip.com/x?y=1` reaches the relay URL with `Host: rails-ris.makemytrip.com`;
  - deploy: the override sets `MSYND_RELAY`.
- [x] Commit `task 53: pc client relay`.
- [x] Monitor: relay restarted on the PC, one MMT request from the VM through it, then the collector with `MSYND_LIVE=1`.

---

### Task 54 (amendment 2026-10-11): a "not running on given date" answer cancels the run

Live on the VM, MMT answers HTTP 400 `{"success":false,"error":{"errorCode":1001|1002,"errorDescription":"Train is not running on given date"}}` for inventory runs that do not operate (mostly `00xxx` specials). The outcome stays `invalid`, but nothing records the answer, so poll and harvest admission keep choosing the same runs and the invalid share trips the dispatcher breaker every few minutes. The answer is the source's word on that run, so the run becomes `cancelled`, which both admissions already skip and `create_runs` never recreates.

**Files:** modify `msynd/collect/mmt.py`, `msynd/collect/lifecycle.py`, `msynd/collect/__main__.py`, `tests/collect/test_admission.py` (or the existing collector-loop test module), `tests/collect/test_lifecycle.py`.

**Interfaces:**
- `mmt.not_running(body: bytes) -> bool`: true only for a JSON object with `success` false and `error.errorCode` in {1001, 1002}; anything else (bad JSON, other codes) is false.
- `lifecycle.cancel_not_running(conn, run_id, now) -> bool`: inside the caller's evidence transaction, sets `lifecycle='cancelled'`, `updated_at=now` only when the run is still `scheduled` or `expired_unknown`, calls `mark_dirty` when a row changed, and returns whether it did. A `running` or `completed` run is never changed. `# ponytail:` one answer is final; add a confirmation fetch if MMT shows false negatives.
- `run_loop`: on an `invalid` status result whose HTTP status is 400 and whose body is `not_running`, call `cancel_not_running` in one transaction.
- The breaker still counts the answer as non-`ok`; each such run costs at most one fetch per admission path.

- [x] Tests:
  - `not_running` accepts both codes and rejects other codes, `success` true, non-JSON and non-object bodies;
  - `cancel_not_running` cancels a `scheduled` run and marks it dirty; leaves `running` unchanged;
  - collector loop: with a MockTransport answering the 1001 body, the run becomes `cancelled` and `load_state` no longer admits it for poll or harvest; a 400 with errorCode 9999 leaves it `scheduled`.
- [x] Commit `task 54: cancel runs mmt says are not running`.
- [x] Monitor: deploy to the VM, restart the collector, and confirm the breaker stops tripping.

---

### Task 55 (amendment 2026-10-11): the breaker ignores "not running" answers (user decision)

After Task 54, harvest still reaches many inventory runs that MMT says did not run, about half of those it tries, so the breaker keeps pausing for 10 min. A "not running on given date" answer is a healthy, definitive reply, not source trouble. The user chose (2026-10-11) to exempt it from the breaker. Request spacing, budgets, logging, retries and the challenge stop are unchanged, and the fetch is still logged as `invalid`.

**Files:** modify `msynd/collect/dispatcher.py`, `msynd/collect/mmt.py`, `tests/collect/test_dispatcher.py`, `tests/collect/test_mmt.py`.

**Interfaces:**
- `Dispatcher(..., breaker_exempt=None)`: an optional `Callable[[httpx.Response], bool]`. When an attempt's outcome is `invalid`, a response exists and `breaker_exempt(response)` is true, that outcome is not added to the breaker window. Every other outcome is counted as before.
- `MmtClient.__init__` sets `d.breaker_exempt` to a function that is true only for HTTP 400 whose body is `not_running` (Task 54), in the same way it pins `d.classify`.

- [x] Tests:
  - dispatcher: 25 exempt `invalid` answers in a row leave `state == 'ok'` and add no pause; 25 non-exempt `invalid` answers still trip the breaker; spacing between starts is unchanged;
  - mmt: an `MmtClient` dispatcher fed 25 HTTP 400 1001 answers stays `ok`; 25 HTTP 400 9999 answers trip the breaker.
- [x] Commit `task 55: breaker ignores not-running answers`.
- [x] Monitor: deploy and confirm per-minute starts stay near 12 without 10-minute gaps. Done 2026-10-11 04:05Z: no breaker gaps; poll runs at its 6-per-120 s cohort pace; harvest spent its 512 budget for IST 10-11 (358 of 512 were "not running" answers on stale inventory runs).

---

### Task 56 (amendment 2026-10-11): bounded engine set and a responsive map (user request)

First VM deploy of engine + API + map (2026-10-11 05:30Z). Two bottlenecks:
- **Engine:** `ACTIVE` is every `scheduled`/`running` run, ~13,000 runs, 12,989 of them scheduled with no evidence. `rebuild` prepares all jobs (model tables included) before replaying anything. Memory grew ~1.7 GB/min, and the projected total was ~20 GB on a 23 GB VM, so the monitor stopped the engine. At 0.86 s per replay that is also ~47 min of CPU, and the full frame would be ~200 MB.
- **Map:** `draw()` rebuilds the network layer every 100 ms. Each rebuild does a linear `stations.find` per segment end (15,949 segments × 8,550 stations), plus `routes.find`/`stops.find` per run and a fresh trips array. The page is unresponsive with zero trains.

**Files:** modify `msynd/engine/__main__.py`, `tests/engine/test_loop.py`, `web/src/main.ts`, `web/src/layers.ts`; add web tests as needed.

**Engine:**
- Active set is `running`, plus `scheduled` runs whose scheduled origin departure (`dep_off + resched_min`, IST start date) is within ±1 h of now. That is ~560 runs on 2026-10-11. Mark it `# ponytail:` (window constant; widen after G8 and frame-size measurement).
  - One SQL predicate serves `rebuild`, the version-change mark-all-dirty, and the window refresh.
- `rebuild` prepares and replays in chunks of 200 run IDs, so peak memory is bounded by the chunk, not the active set.
- `consume` replays a dirty run only if it is in the active set. A dirty `scheduled` run outside the window is acknowledged and tombstoned (if previously framed) without replay. Lifecycle handling for other states is unchanged.
- Every 5 min (with the full frame), recompute the set: runs that entered are `mark_dirty`; runs that left (still `scheduled`) get a tombstone in a delta and leave `runs` and the cache.

**Map:**
- Index the network once on load: station code → coordinate, route id → route, and per route stop idx → stop.
- Build the network layer once and reuse it.
- Decode runs (trips, stop times) only when `store.gen` changes. Each 100 ms tick only recomputes heads and `currentTime`.
- Trips data keeps its reference between generations. The hourly time base is passed via `updateTriggers`.
- Rendering output is unchanged.

- [x] Tests:
  - engine: a scheduled run 3 h before departure is not replayed by rebuild or consume, and its dirty row is acknowledged; a scheduled run 30 min before departure and a running run are replayed; a run leaving the window gets a tombstone; rebuild with more than 200 active runs produces the same frame as one batch.
  - web: indexed lookups match the old `find` results; `displayRun` decodes the same trips and head as before on the existing fixtures; the network layer is built once per network.
- [x] Commit `task 56: bounded engine set and responsive map`.
- [x] Monitor: deploy, start the engine, and record cold rebuild time, peak memory and full-frame size. Confirm the map renders trains and stays responsive.

### Task 57 (amendment 2026-10-11): per-run frame time base (blast radius of Task 56)

After the Task 56 deploy, engine memory peaked at ~1.3 GB, but every rebuild failed in `write_frame` with `display paths outside uint16 frame range` and the engine crash-looped. `write_frame` (and the API's `Frames.full`) re-base every run's uint16 ticks (0.1 min) to the earliest `base_ts` in the frame, so one frame can span only ~109 h. Running runs from 10-09 plus scheduled runs ending ~10-14 exceed that. The web already decodes each run from its own `run.base_ts` (`web/src/layers.ts`), so the shared base buys nothing. Contract change: each run's ticks count from that run's `base_ts`. The frame-level `base_ts` stays as the minimum, for information only.

**Files:** modify `msynd/engine/frames.py`, `msynd/api/app.py`, `tests/engine/test_loop.py`, `tests/api/test_api.py`.

- `write_frame` and `Frames.full` keep each run's `paths` and `base_ts` unchanged and drop the frame-wide re-base and its ValueError. The per-run range check stays in `encode_run`; `_job` already maps it to run state `error`.
- [x] Tests: two runs whose bases are 6 days apart give one frame (engine `write_frame`, and an API full resync) with both runs' bytes and `base_ts` unchanged. Existing decode tests still pass.
- [x] Commit `task 57: per-run frame time base`.
- [x] Monitor: deploy, and confirm a full frame is written, the engine stays up and the map shows trains.

### Task 58 (amendment 2026-10-11): runs leave the frame when they leave the engine set (blast radius of Task 56)

After Task 57 the engine published frames, but the map stayed on "Waiting for live frames". Draining the dirty backlog published 6,786 `{'state': 'expired_unknown'|'cancelled'}` entries with no `run_id` (`_prepare`'s non-active branch). The client's `validate` rejects a run whose `run_id` does not match its key, closes the socket and reconnects, silently and forever. These entries also pile up in every full frame until a restart, while `rebuild` never emits them.

**Files:** modify `msynd/engine/__main__.py`, `tests/engine/test_loop.py`, `web/src/frames.ts`.

- `_prepare`: a run that is missing or not `in_engine_set` (completed, cancelled, expired_unknown, or scheduled outside the window) becomes a tombstone. Every published run entry carries `run_id` equal to its key. `rebuild`, `consume` and `_refresh_window` then agree: the frame holds exactly the engine set.
- `connectLive`: log the rejection reason with `console.error` before closing. Validation itself is unchanged.
- [x] Tests: dirty `cancelled` and `expired_unknown` runs produce tombstones and no run entries, and their dirty rows are acknowledged. Every run entry in frames from `rebuild` and `consume` has `run_id == key`.
- [x] Commit `task 58: runs leave the frame with the engine set`.
- [x] Monitor: deploy, restart the engine and API, and confirm the map receives frames and draws trains.

Monitor note 2026-10-11 (Tasks 56–58, VM at bac9940):

- Engine set ≈ 555–700 runs; cold rebuild ≈ 65 s; engine peak ≈ 1.35 GB (5.7 GB and still rising before Task 56); full frame 4.2–5.2 MB.
- The 16k dirty backlog drained in ≈ 80 s.
- After Task 58: full frame gen 1744 had 700 runs, every entry with `run_id == key` and none state-only.
- The web `connectLive`/`FrameStore`, run under Node over a tunnel to the VM API, accepted gen 1749 (700 runs, all with paths, 0 rejections).

### Task 59 (amendment 2026-10-11): relay watchdog (user request)

**Fallback only after Task 78:** normal cloud collection does not run this Windows
watchdog; the `msynd-relay` task is disabled and retained for rollback.

On 2026-10-11 at 13:49Z the PC relay exited. Its log ends with `^C`; the cause is unknown, and there was no low-memory event. Two things then went wrong:

- Nothing restarted the relay.
- Its `ssh -R` child survived as an orphan and kept holding VM `127.0.0.1:18080`. A fresh relay's ssh therefore could not bind (`ExitOnForwardFailure`) until the orphan was killed by hand.

The collector got `transport_error`, its breaker paused 10 min, and collection stopped for 11 min.

**Files:** modify `msynd/collect/relay.py`, `tests/collect/test_relay.py`, `docs/deploy.md`; create `relay-task.ps1`.

- **`relay.py`:** on Windows, `main()` first puts the relay process in a job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (ctypes, stdlib), and keeps the handle open for the life of the process. Children (ssh) inherit the job, so however the relay dies, its ssh dies with it and the VM port is freed. Elsewhere this is a no-op.
- **`relay-task.ps1`:** registers the scheduled task `msynd-relay` for the current user.
  - S4U logon, so there is no console and no logon is needed.
  - Triggers at startup and every 5 min. `MultipleInstances IgnoreNew` means a trigger only starts a relay that is not running.
  - No execution time limit; it starts and keeps running on battery.
  - The action runs `uv run python -m msynd.collect.relay --ssh <target> --identity <key>` from the repo with Windows OpenSSH on PATH, and appends stdout and stderr to a log file.
  - Parameters: `-SshTarget` (required, never hard-coded), `-Identity`, `-Log`. `-Unregister` removes the task.
  - `# ponytail:` a 5-min poll gives a worst-case gap of 5 min plus the 10-min breaker; use a Windows service if that matters.
- **`docs/deploy.md`:** the relay section points to the script instead of the manual command; the section stays short.
- [x] Tests, Windows-only:
  - a child started after the job setup dies within 5 s when its parent is terminated (TerminateProcess), and outlives a parent that skipped the setup;
  - `relay-task.ps1` parses without errors.
- [x] Commit `task 59: relay watchdog`.
- [x] Monitor:
  1. Stop the CIM-launched relay tree.
  2. Register and start the task, then confirm the VM gets `ok` fetches.
  3. Kill the relay's Python process; confirm its ssh exits and the task restarts the relay within 5 min, and that fetches resume after the breaker pause.

**Monitor result (2026-10-11):**
- **Cutover:** at 15:47Z, with no gap in `ok` fetches.
- **Kill test:**
  - killed the relay Python at 15:49:02Z;
  - its ssh exited within 5 s and VM :18080 was freed;
  - the task restarted the relay at 15:52:19Z;
  - after 8 `transport_error`s, `ok` fetches resumed at 16:00Z when the breaker pause ended. The 11-min gap is within the ponytail ceiling.
- **Elevation:** the S4U tree runs in session 0, so killing it needs elevation.

### Task 60 (amendment 2026-10-11): neon trails (user request)

The user asked for a cyberpunk look, with heads that are "practically non-existent", a gradient trail that tapers off, and much smoother motion. Measured on the PC before the change (689 runs):

- Idle rendering runs at 135 fps.
- Every frame arrival (about every 20 s) blocks the main thread for about 300 ms. Each one re-decodes every run and rebuilds about 22k streak paths.
- Heads redraw at 10 Hz because of a 100 ms throttle.
- At zoom 4 the 10-minute trail is about 1 px, so the map shows only 5 px dots.

**Files:** modify `web/src/layers.ts`, `web/src/main.ts`, `web/index.html`, `web/src/layers.test.ts`, `web/src/main.test.ts`.

- **Decode cache:**
  - Keep the `displayRun` result in a `WeakMap` per `RunFrame` object. `FrameStore` keeps unchanged runs as the same object, so a new generation decodes only the runs that changed.
  - The trail data arrays are rebuilt only when a generation changes.
- **Every animation frame:**
  - Drop the 100 ms throttle. The status text updates at most once per second.
  - Each drawn head eases toward its computed head: `head += (target − head)·(1 − exp(−dt/250 ms))`. A run seen for the first time snaps to its target.
- **Comet trail:**
  - Each run's median track (`points`, with `times` made monotone by a running max) is drawn by three stacked `TripsLayer`s with `fadeTrail`:
    - widths 4 / 2 / 1 px;
    - trail lengths L/4 / L/2 / L;
    - alpha rising toward the thin core, so the tail tapers and fades.
  - `L = clamp(4 h · 2^(4 − zoom), 10 min, 6 h)`, so tails keep a roughly constant length on screen.
- **Uncertainty:** a strided subset of 8 particle streaks per run, 1 px and faint. `# ponytail:` all K = 32 streaks cost about 4× in attribute rebuilds; raise the subset when rebuilds become incremental.
- **Colour:**
  - Delay at the next stop when decoded: `(times[e] − origin_ts)/60 − sched_off`; 0 if unknown.
  - Neon stops, linear in between and clamped at the ends: ≤ 0 min cyan `[0,240,255]`, 15 min yellow `[252,238,10]`, ≥ 60 min hot magenta `[255,42,109]`.
  - Trail layers use additive blending.
- **Heads:** 1.5 px, the delay colour mixed 60 % toward white. The overlay gets `pickingRadius: 10`, so tiny heads stay clickable.
- **Look:**
  - background `#05070d`;
  - network 1 px `[0,200,255,28]`;
  - panels are translucent near-black with a thin cyan border and glow, and a monospace font stack;
  - `LIVE` in neon magenta with a text-shadow glow.
- [x] Tests:
  - an unchanged run is not decoded again on a new generation;
  - the three comet tiers have widths 4 / 2 / 1 with lengths increasing in that order;
  - median timestamps are monotone;
  - the colour stops and clamps are correct;
  - the zoom-to-L clamp is correct;
  - easing moves part of the way, converges, and snaps on first sight;
  - there are at most 8 strided streaks per run.
  - `npm --prefix web test` and `npm --prefix web run build` both pass.
- [x] Commit `task 60: neon trails`.
- [x] Monitor:
  1. Take a Playwright screenshot through the tunnel.
  2. Check that a frame arrival blocks the main thread for < 100 ms on the PC, and that idle fps has not dropped.
  3. Copy `web/dist` to the VM.
- **Monitor result (2026-10-11):**
  - Redrawing deck on every frame cost 14–20 ms per frame on the PC's Intel iGPU: the 16k-segment network was about 7 ms and the streaks about 7 ms. That cut idle fps to 49.
  - Live heads move far less than a pixel per frame, so the monitor changed `main.ts`: it redraws at about 70 Hz while any head is still easing (more than 0.05 px from its target, after which the head snaps) and at 10 Hz otherwise. The `parameters` object was also moved to a module constant.
  - Measured afterwards over 150 s on the PC (635 trains): idle 144 fps (p50 6.9 ms, max 69 ms); frame arrivals took 52–70 ms of main-thread time; a pan ran at about 100 fps.
  - Deployed at a43d235.

### Task 61 (amendment 2026-10-11): station-chain geometry (user request: "why are there so many tracks webbed together")

Today every segment is drawn as a straight line between consecutive route stops, and so are the train paths. Express trains make long hops between stops: 1,882 segments are longer than 50 km, 727 are longer than 100 km, and the longest is 1,781 km. Their straight chords cross open country and form a web.

Local trains stop every ~10 km (median segment), so the network already contains the real corridor as a chain of short segments. This task routes each long hop along that chain. It is an interim step; Task 37's OSM geometry later replaces these chains wherever it has a confident match.

**Files:** modify `msynd/static/export.py`, `tests/static/test_segments.py` (or a new `tests/static/test_export.py`), `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/main.ts`.

- **Polyline contract (it fills the unspecified `ref.segment.polyline` text):** JSON `[[lon, lat], …]` running from `stn_lo` to `stn_hi`; `NULL` means a straight line. Overrides and Task 37 use the same format.
- **Export:**
  - **Train identity on routes:** each route gains `train`, `name` and `class` (from `ref.schedule_version` → `ref.train`).
  - **Which segments get chains:** a segment has no stored polyline, both stations have coordinates, and the great-circle chord is longer than 20 km.
  - **Finding the chain:** Dijkstra over the undirected station graph of all segments, with the chord edge itself removed.
    - Edge weight is the squared great-circle km, so many short hops beat one long chord. Stop the search once the distance passes chord².
  - **Accepting the chain:** only if it has ≥ 2 edges and its summed great-circle km is ≤ 1.3 × the chord. The exported segment then gets `polyline` (station coordinates, `stn_lo` → `stn_hi`) and `derived: true`. Every other segment exports `derived: false`.
  - **Storage and contract:** the DB is not written, and the etag still comes from the content.
  - `# ponytail:` a Dijkstra per long segment (~2k, a few minutes at most); precompute in `build_segments` if export time matters.
- **Web:**
  - **Paths:** `displayRun` expands each consecutive stop pair `(i, i+1)` through the route segment with `idx == i`.
    - Use the polyline when present (reversed when `forward` is false). Otherwise use the straight pair.
    - Vertex timestamps are interpolated linearly by cumulative polyline length between the two event times.
    - `headAt` walks the same polyline by the `pos()` km fraction.
    - Both the comets and the streaks use the expanded paths.
  - **Network layer:** draws stored and straight segments only, and skips `derived` ones because their constituent segments already draw them.
    - It also skips straight segments longer than 150 km. `# ponytail:` hidden long chords remain only as train trails; Task 37 replaces them.
  - **`Network` type:** routes gain `train`, `name` and `class`; segments gain `derived`.
- [x] Tests:
  - On a synthetic DB network where a chord A–D runs alongside a chain A–B–C–D, the chord exports with the chain's polyline and `derived: true`. A chain with a 1.5× detour is rejected, and a segment with a stored polyline is untouched.
  - Routes carry the train number, name and class.
  - A run's path through a chained segment passes through the intermediate station coordinates, with monotone interpolated timestamps.
  - `headAt` at the midpoint time lies on the polyline. A reversed segment runs from its own start.
  - The network layer omits `derived` segments and straight segments over 150 km.
  - `uv run pytest -q`, `npm --prefix web test` and `npm --prefix web run build` all pass.
- [x] Commit `task 61: station-chain geometry`.
- [x] Monitor:
  1. Re-run the export on the VM with the new code and reload the API's static file.
  2. Report how many segments got chains, and the export time.
  3. Take a Playwright screenshot to confirm the web is gone.
  - **Result 2026-10-11:** 4,230 chains derived. Export took 8.9 s on the PC and 8.3 s on the VM, producing `network-87e15c3030de.json.gz` (3.3 MB). Deployed at aa1dad5. The screenshot shows rail corridors with no chord web. Found: MapLibre 6's worker is not emitted by Vite ("Worker failed to load"); fixed in Task 62.

### Task 62 (amendment 2026-10-11): modern map UI (user request: "critically improve the overall UX and the UI")

The user reported three problems:
- the map has no basemap: India floats in black;
- the map zooms out to two world copies;
- the train panel is a raw P10/P50/P90 table with `(1)` occurrence labels.

Hovering a train shows nothing, and there is no search.

**Files:** modify `web/index.html`, `web/src/main.ts`, `web/src/layers.ts`, `web/src/panel.ts`, `web/src/*.test.ts`; create `web/src/basemap.ts`, `web/src/search.ts`, `web/src/style.css`.

No new dependency: `pmtiles` and `maplibre-gl` are already installed. The UI font is the system stack `Inter, 'Segoe UI Variable', system-ui, sans-serif`, with `font-variant-numeric: tabular-nums` for times.

- **Map limits:**
  - `renderWorldCopies: false`, `minZoom 3.5`, `maxZoom 13`, `maxBounds [[60, 0], [106, 40]]`.
  - The first view fits `[[68, 6.5], [97.5, 36]]` with padding, so it fits any screen.
- **Basemap (`basemap.ts`):**
  - **Source:** register the `pmtiles` protocol and add vector source `pmtiles:///tiles/india.pmtiles` (Protomaps v4 schema).
  - **Style (hand-written, no symbol layers, so no glyphs are needed):**

    | Layer | Style |
    |---|---|
    | `earth` | fill `#0b1119` |
    | `water` | fill `#03050a` |
    | `boundaries` country | `#2b3b4f`, 1 px |
    | `boundaries` region | `#18222f`, 0.6 px dashed |
    | `roads` (kind `highway` / `major_road`, from z6) | `#111923` |
    | `transit` rail (from z8) | `#0e2630` |

    Each layer is drawn below the deck overlay.
  - **Missing tiles:** a missing or failing tile source is ignored, so the background-only map still works in dev.
  - **Attribution:** compact, `© OpenStreetMap · Protomaps`.
- **Station labels:**
  - A deck `TextLayer` of station names in title case, 11 px, `[140,160,185,200]`.
  - A station is labelled when its route count (from the network) is at least the threshold for the current zoom: < 5 → 400, < 6 → 150, < 7 → 60, < 9 → 20, else 1. Thresholds are a named constant.
- **Selection:**
  - Clicking a head, or picking a search result, selects the run.
  - Its expanded path draws as a 2 px bright line `[0,240,255,190]`, other trains dim to 30 % alpha, and the map eases to the head (`easeTo`, keeping the zoom if it is ≥ 6, else zoom 6).
  - Esc or the panel's close button clears the selection.
- **Hover card:** next to the cursor over a head:
  - `12951 · Mumbai Rajdhani`;
  - origin → destination station names;
  - a delay chip (`On time` within ±2 min, `+12 min`, `−3 min`) coloured by `delayColor`;
  - the evidence state in words: `observed` → `Live · updated N min ago` (from `headline.last_received`), `stale` → `Stale · N min`, `schedule_only` → `Schedule only`.
- **HUD (top left, glass card):**
  - **Card style:** `backdrop-filter: blur(12px)`, background `rgba(10,14,22,.72)`, a 1 px `rgba(120,200,255,.14)` border, radius 14 px.
  - **Contents:**
    - a pulsing dot and `LIVE`;
    - the IST clock `HH:MM:SS`;
    - `N trains`;
    - frame freshness `updated N s ago`, green < 60 s, amber < 180 s, red otherwise.
  - **Search input:** `Train number or name`.
    - It debounces 250 ms against `/api/search`, shows up to 8 trains (`number · name`), and supports ↑ ↓ Enter Esc.
    - Choosing a train calls `/api/train/{number}` and selects a returned run that is in the live store, preferring the latest `start_date`. If none is live, the row says `Not running now`.
- **Legend (bottom left, glass):** a gradient bar cyan → yellow → magenta labelled `On time`, `15 min`, `60+ min late`.
- **Train panel (`panel.ts`):**
  - **Placement:** a right-hand drawer, 380 px wide and full height minus 16 px margins, glass styled. Below 640 px wide it becomes a bottom sheet at most 60 vh tall.
  - **Header:**
    - train number, name and run date;
    - origin → destination names;
    - the delay chip and evidence text;
    - a close `×` button with an `aria-label`.
  - **Next stop card:**
    - station name, predicted P50 time (IST) and the range `P10–P90`;
    - the scheduled time struck through when it differs by ≥ 2 min.
  - **Stop timeline:**
    - one row per halt with arrival and departure merged. Show arrival, but departure at the origin.
    - A vertical rail links the row dots: past dots are filled and dim, the next one glows, future ones are hollow.
    - Each row shows the station name with its code muted and the scheduled time.
      - Past halts also show the actual time when `revisions` has one.
      - Future halts also show the P50 coloured by delay, with a small range.
    - `2nd visit` appears only when `occ > 1`.
  - **Data:** `/api/run/{id}` `schedule`, `halts` and `revisions`, and names from the network. Times use `Intl` in the Asia/Kolkata time zone.
  - **Errors:** a failed load shows a short inline error with a Retry button.
- **Motion and accessibility:**
  - The pulse stops under `prefers-reduced-motion`.
  - Focus rings are visible.
  - The panel is an `aside` with an `aria-label`, and the hover card has `role=tooltip`.
  - No horizontal scroll at 360 px wide.
- Task 60's redraw gating and Task 61's paths stay as they are.
- [x] Tests:
  - the basemap style has the pmtiles source and no symbol layers;
  - the map options carry the limits above;
  - each label threshold returns the right station subset;
  - a selection dims the other trains and highlights the route;
  - panel rows merge arrival and departure, show `2nd visit` only for `occ > 1`, and give the delay chip text at −3, 0, 1 and 12 min;
  - search picks the live run with the latest `start_date`, or reports `Not running now`;
  - the hover text for each evidence state;
  - `npm --prefix web test` and `npm --prefix web run build` both pass.
- [x] Commit `task 62: modern map ui`.
- [x] Monitor:
  1. Generate `/data/tiles/india.pmtiles` on the VM: `pmtiles extract` of a recent Protomaps daily build, bbox `68,6,98,37`, maxzoom 10, as Task 36 planned.
  2. Copy `web/dist`.
  3. Take Playwright screenshots at zoom 4.5 and 8, plus one of the panel. Check that the console has no worker or tile errors and that idle fps is still ≥ 100 on the PC.
  - **Follow-up (user request 2026-10-11: "look for the more prominent stations first… thicken the tracks and the trains… increase the trails size"):**
    - Labels rank by prominence = calls by routes ≥ 300 km + 2 × their terminals, replacing route count. Thresholds: < 5 → 400, < 6 → 250, < 7 → 150, < 8 → 80, < 9 → 40, < 10 → 20, else all.
    - A greedy Mercator declutter runs per half-zoom bucket, and labels rebuild only when the bucket changes.
    - Labels use the SDF font with a dark halo; the non-SDF atlas clipped "j".
    - Widths: network 1.5 px, comets [6, 3, 1.5], streaks 1.5 px, heads 2 px, selection 3 px. Trails are 1.5× longer.
    - The attribution is styled to match the glass HUD, and a data favicon stops the 404.
  - **Result 2026-10-11:** tiles are a wide extract (bbox `60,0,106,40`, z0–10, ~208 MB, from the 20261011 daily build), so `maxBounds` never shows a void. Deployed at e17b71e (web only, no image rebuild). Playwright checks:
    - the overview shows only major stations, and zoom 8 is decluttered;
    - the hover card, the panel with P10–P90 and the stops, and search ("Not running now") all work;
    - no worker or tile errors in the console;
    - ~83 fps in headless Playwright.

### Task 63 (amendment 2026-10-12): map geometry fixes (user reports: "weird geometric [shapes] consistently over zooming"; "why tf is the thing a straight line and not following the rails")

**Root causes (monitor, 2026-10-12):**
- **Basemap stripes.** Protomaps' `water` source layer mixes polygons with river and canal `LineString`s. Our `water` fill has no geometry filter, so MapLibre closes each river line with a straight chord and fills it, which draws large straight-edged dark shapes at every zoom. Proof: hiding `water`, or filtering it to polygons, removes them, and the Kangsabati then draws correctly.
- **Straight trains.** 327 segments over 20 km still draw as chords (76 over 100 km), and 323 trains have a hop over 100 km, including 11 Rajdhanis, 14 Durontos and 15 Vande Bharats. The worst case is `00441` CP→PTNR (1,312 km).
  - **Missed paths (69 of the 142 over 50 km):** a short-hop path within 1.3× already exists. Task 61's squared-km Dijkstra prefers meandering tiny hops, so the path it finds fails the ratio check.
  - **Isolated endpoints:** e.g. CP's only segment is CP–PTNR.
  - **Chord ratio misreads real detours:** the 1.3× chord ratio rejects genuine detours, e.g. JSM–UDZ, whose rail distance is 1.9× its chord.
  - **Monitor simulation of the rule below:** 0 straight segments over 150 km remain, 2 between 100 and 150 km, and 8 between 50 and 100 km. It runs in under 1 s.

**Files:** `web/src/basemap.ts`, `web/src/basemap.test.ts`, `msynd/static/export.py`, `tests/static/test_export.py`.

- **Basemap:** every `fill` layer (`earth`, `water`) gets `filter: ['==', '$type', 'Polygon']`. The legacy `$type` form matches MultiPolygon too.
- **Station chains:** this replaces Task 61's search in `segment_geometry`.
  - **Track km per segment:** the median `|dist_km(i+1) − dist_km(i)|` over every route stop pair that uses the segment.
  - **Edge weight:** `max(track km, great-circle km)`.
  - **Search graph:** only segments with a great-circle chord ≤ 80 km.
  - **Snapping:** a station with no such edge gets snap edges, weighted by great-circle km, to every station within 15 km that has one.
  - **Search:** a plain-km Dijkstra from `stn_lo` to `stn_hi`.
    - It never uses the segment itself, and it stops once the distance exceeds 1.2 × the target, where the target is `max(segment track km, chord)`.
  - **Acceptance:** a path is accepted when it has ≥ 2 edges. The 1.2 × target search bound is the only length limit, so a chain shorter than the target still beats the chord. (A ±20 % lower bound was tried and dropped in review: it left BSB–MFP, BSB–SEE and RC–SC straight.)
  - **Unchanged from Task 61:** the eligibility rules (no stored polyline, both coordinates, chord > 20 km), the polyline format, `derived`, no DB writes, and the content etag.
  - `# ponytail:` station chains are straight between stations; Task 37's OSM geometry replaces them with real track curves.
- [x] Tests:
  - every basemap fill layer is polygon-filtered;
  - a 2-stop route whose origin has no other segment (the CP→PTNR shape) chains through a station within 15 km;
  - a chain whose track km match but whose great-circle km detours 1.5× is still rejected (keep the existing case);
  - a real detour whose route track km is 1.9 × the chord, and whose chain matches that, is accepted;
  - a chain shorter than the target track km is still drawn;
  - a chain that the old squared-km search missed is now found. The case: a meandering path of tiny hops beside a direct path of medium hops, where only the direct one is within tolerance;
  - `uv run pytest -q tests/static` passes, as do `npm --prefix web test` and `npm --prefix web run build`.
- [x] Commit `task 63: map geometry fixes`.
- [x] Monitor (2026-10-12, VM at ede9c2e, network-0120115ba2b6 exported in 7.6 s): straight segments over 20/50/100/150 km went from 327/142/76/– to 179/11/2/0; 4,436 segments derived; CP–PTNR (00441) now draws through 49 stations, BSB–MFP 12, JSM–UDZ 29. Basemap fills are polygon-only and the screenshots show no stripes.
  1. Rebuild the VM image and re-export the network.
  2. Recount straight segments over 20, 50, 100 and 150 km.
  3. Deploy `web/dist`.
  4. Take Playwright screenshots of Jhargram at z10.6, and of `00441` selected.

### Task 64 (amendment 2026-10-12): timetable trains (user choice: "Show timetable trains")

**Why (monitor, 2026-10-12):** search for a Rajdhani says "Not running now" while it is on the rails. The engine set holds `running` runs plus `scheduled` runs that depart within ±1 h, so a run the collector has not yet observed drops out one hour after departure. At 19:10Z on 10-11, 1,826 `scheduled` runs were between departure and arrival, but only 106 were in the set, and 5 were `running`.

**Files:** `msynd/engine/__main__.py`, `msynd/engine/frames.py`, `tests/engine/test_loop.py` (it also tests `encode_run`), `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/main.ts`, `web/src/main.test.ts`.

- **Engine set:** the `scheduled` branch of `ENGINE_SET` becomes "now lies between the origin departure − `WINDOW` and the destination arrival + `WINDOW`".
  - Both times include `resched_min`. The origin departure is the `dep_off` of the first `ref.schedule_stop` by `idx`; the destination arrival is the `arr_off` of the last one.
  - `running` stays in the set unchanged. `cancelled`, `completed` and `expired_unknown` stay out.
  - `_engine_ids`, `_prepare` and `_refresh_window` keep sharing the one `ENGINE_SET`, so entering runs are replayed and leaving `scheduled` runs are tombstoned exactly as today.
  - Update the `# ponytail:` comment: the window is the whole timetable journey ± 1 h; narrow it per class if G8 or frame size says so.
- **Frames:** `encode_run` ships at most 8 display paths for a `schedule_only` run, the evenly strided rows `j·K/8` (all rows when K ≤ 8). Observed and stale runs keep all K. The web already draws only 8 streaks per run, and `displayRun` reads `path_shape`.
  - `# ponytail:` schedule-only runs ship 8 of K paths to bound frame size; send all K if the panel ever needs them.
- **Web:**
  - `trainAlpha` takes the run's state. A `schedule_only` run draws at 0.45 × its alpha (heads, comets and streaks) unless it is the selected run, which stays at 1. With another run selected it draws at 0.3 × 0.45.
  - The status count reads `"<n> live · <m> by timetable"`, where m counts `schedule_only` displays and n the rest, both formatted with `toLocaleString('en-IN')`. With no timetable runs it reads `"<n> live"`.
  - Search needs no change: `pickLiveRun` finds any run in the frame.
- [x] Tests:
  - engine window boundaries, using the fixture's 880-minute journey: departing in 61 min → out; departing in 60 min → in; departed 61 min ago and arriving later → in; arrived 60 min ago → in; arrived 61 min ago → out (tombstoned);
  - the refresh test still tombstones a `scheduled` run once its arrival + 1 h passes, and replays one that enters;
  - a `running` run that is past its timetable arrival stays in the set;
  - `encode_run` on a `schedule_only` run with K = 32 ships `path_shape` (8, E, 2), with rows 0, 4, …, 28; an observed run ships (32, E, 2);
  - `trainAlpha`: schedule-only unselected → 0.45, selected → 1, another selected → 0.135; observed → 1, 1 and 0.3;
  - the count text for 5 observed plus 1,826 schedule-only displays is `"5 live · 1,826 by timetable"`;
  - `uv run pytest -q tests/engine` passes, as do `npm --prefix web test` and `npm --prefix web run build`.
- [x] Commit `task 64: timetable trains`.
- [x] Monitor:
  1. Deploy to the VM: rebuild the image, restart the engine, deploy `web/dist`.
  2. Measure the set size, the cold rebuild time, the peak engine memory and the full-frame size. Stop and report if peak memory exceeds 4 GB or a full frame exceeds 16 MB.
  3. Check that searching a Rajdhani that is between departure and arrival selects it.
  4. Take a Playwright screenshot.
- **Monitor result (2026-10-12 01:25 IST):** b47933e deployed. Mutations caught: dropping the arrival + 1 h bound, the 8-row stride, and the 0.45 factor. The full frame holds 1,684 runs (1,678 `schedule_only`, 6 observed); before this task there were 106. The full frame is 9.9 MB raw and 5.8 MB deflated (it was 0.54 MB). The cold rebuild took about 4.5 min from restart to first full frame, and peak engine memory was 1.77 GB. Searching 12302 HWH Rajdhani selects it as schedule-only, with its next stop at DDU. Open issue: at national zoom the additive trails of about 1,700 trains saturate the corridors to white, and the delay colours on schedule-only trains are model priors.

---

### Task 65 (amendment 2026-10-12): lean, brighter train rendering (user requests: "the deployed page is consuming about 1 GB of memory"; "these lines and their trails are very faint! increase the luminosity and the trail size"; "yes do task 65")

**Why (monitor, 2026-10-12, measured on the deployed data with 3,261 drawn runs, 3,260 of them `schedule_only`):**
- Playwright Chrome used 761 MB private in the renderer and 593 MB in the GPU process. The JS heap after GC was only 184 MB, so the rest is deck.gl attribute buffers and allocator churn. rAF ran at 19 fps.
- Each run draws 8 streaks, giving 26,088 streak paths and 1.61 M streak vertices. On top of that, 3 comet layers of 202 k vertices each.
- Every delta frame (every 3–15 s) rebuilds `trailData` for all runs. The new arrays force deck.gl to regenerate every trail attribute, which shows up as ~300 ms long tasks.
- Every redraw (10–70 Hz) spreads all 3,261 displays into new head objects, so the heads layer recomputes all of its attributes.
- The API forwards the engine's 5-minute full frame (16.05 MB raw, at the Task 64 STOP threshold) to every connected client. On a client that is already in sync this is 16 MB of mostly unchanged runs, and `displayRun` re-runs for all of them (478 ms).
- Schedule-only trails draw at 0.45 alpha with zoomed-in trails of only 10 min, so they read as faint.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/main.ts`, `web/src/main.test.ts`, `msynd/api/app.py`, `tests/api/test_api.py`.

- **Timetable style:**
  - A `schedule_only` run draws only a head plus one core trail, both in a neutral colour `TIMETABLE = [200, 214, 235]`. It has no glow comets and no streaks.
  - Observed and stale runs keep their delay colours, glow comets and 8 streaks.
  - `trainAlpha` drops the 0.45 schedule factor and keeps only the selection rule: selected or nothing selected → 1, another run selected → 0.3.
  - The hover and panel delay text stays as is.
- **Brightness and length:**
  - `trailLength(zoom) = max(2700, min(32400, 32400·2^(4−zoom)))`.
  - Live comets: widths [8, 4, 2] with alphas [64, 150, 255]; trail lengths keep today's divisors: length/6, length/3 and length/1.5. Live streaks: width 1.5, alpha 24.
  - Timetable core: width 2, alpha 230, trail length/1.5.
  - Heads: radius 3 for live runs and 2.5 for timetable runs; fill = the trail colour lightened 60 % toward white.
- **Render budget:**
  - Live and timetable trails go in separate layers, so a live-run delta never rebuilds the timetable layer.
  - `trailData` returns the identical trail objects for an unchanged `Display` and the same hour `base`, cached in a WeakMap.
  - Every trail layer sets `dataComparator`, a shallow element-identity comparison. A delta that leaves a layer's runs unchanged therefore regenerates none of its attributes.
  - The heads layer keeps `data` = the displays array (replaced only per generation) and reads positions from the eased-head map, with `updateTriggers.getPosition` bumped per redraw. A redraw recomputes only positions and allocates no per-display objects.
  - `# ponytail:` there is no viewport culling or LOD; add culling if the timetable layer's vertex count passes ~1 M.
- **API:** when `Frames.ingest` receives an engine full frame whose gen is `self.gen + 1`, each connected client instead gets a delta with the same gen, ts and versions. It holds only the runs whose decoded payload differs from `self.runs`, plus tombstones for runs that are gone. History still keeps the full frame as the resync baseline, so new or gapped clients get full state exactly as today.
- [x] Tests:
  - a `schedule_only` display yields no glow or streak data and one timetable core with the neutral colour; an observed display keeps 3 comet layers and 8 streaks;
  - `trailLength` at zooms 4, 6, 7 and 10 → 32400, 8100, 4050 and 2700;
  - the comet, core and head widths, alphas and radii above;
  - `trainAlpha` for every state: nothing selected → 1, selected → 1, another selected → 0.3;
  - `trailData` called twice on the same displays and hour returns identical (`toBe`) trail objects; a new `base` hour gives new ones;
  - `dataComparator` is true for arrays with identical elements and false when any element differs;
  - main: a delta that changes only the observed run leaves the timetable layer's `data` elements identical; a redraw without a new generation does not change the heads layer's `data` reference;
  - API: a client in sync receives an engine full frame as a delta with only the changed run and a tombstone for a removed run, while a fresh subscriber at gen 0 receives the full frame;
  - `npm --prefix web test`, `npm --prefix web run build` and `uv run pytest -q tests/api` pass.
- [x] Commit `task 65: lean, brighter train rendering`.
- [x] Monitor: deploy the image (api) and `web/dist`. Remeasure the renderer and GPU memory, the JS heap, rAF and long tasks against the numbers above, and take a Playwright screenshot zoomed into Bilaspur.
- **Monitor result (2026-10-12 09:40 IST):** 3e6b66c deployed (api + dist). 10 of 11 mutants caught; the survivor (tombstone union in the full→delta pack) is equivalent because a client at gen-1 holds exactly `self.runs`. Playwright Chrome, 3,195 displays (3,194 `schedule_only`): renderer 761 → 456 MB, GPU process 593 → 366 MB, rAF 19 → 112 fps, long tasks 302/277 ms → max 122 ms. JS heap after GC 184 → 267 MB; the sampled live heap is ~132 MB, of which ~59 MB is `displayRun` expanded arrays (8 unused expanded trips per timetable run ≈ 30 MB), ~24 MB decoded frames, ~21 MB network index. Open: the user reported "the colours disappeared" (flat TIMETABLE colour) → follow-up 65b.

---

**Follow-up 65b (2026-10-12, user: "the colours disappeared"):** timetable trails and heads keep the run's delay colour (`d.color`) instead of the flat `TIMETABLE` grey-blue; delete `TIMETABLE`. Heads lighten 25 % toward white (not 60 %). Keep the timetable layer's width 2, alpha 230, no comets or streaks. Commit `task 65: restore delay colours`.
- [x] Tests updated (timetable trail colour == `delayColor(delay)`; head fill is the 25 % lightened delay colour).
- [x] Monitor: deploy `web/dist`, screenshot.
- **Monitor result (2026-10-12):** 5c8ca2c deployed (dist). Mutants caught: head lightening 0.25→0.6 and a flat core colour. `reports/t65b-colours*.png` show teal, amber and red trails and heads again.

**Follow-up 65c (2026-10-12, user: "lines a bit too thick... aesthetically pleasing while maintaining the visibility"; "i lwk liked the aesthetics where the blobs' colours were merging"; "the memory usage is up again"):** thin additive glow instead of thick opaque strokes. `GlowTrips` appends a Gaussian cross-section (white core 0.15·e^(−14x²), alpha ·e^(−3.5x²)·(1−age), halo narrowing down the tail) to TripsLayer's own fade inject; `GlowDots` heads use a soft blob. Both are skipped while picking. Low zoom dims via k = clamp((zoom−4)/4): timetable width 6·(0.5+0.5k), opacity 0.35+0.65k, alpha 190; heads radius 4.5/5.5·(0.6+0.4k), opacity 0.45+0.55k, alpha 230; the three comet tiers become one `comets` layer (7 px, alpha 255). Memory: `schedule_only` displays return `trips: []`; trails keep only the vertices bracketing [base − 32400, base + 3600]. Commit `task 65: glow trails`.
- **Monitor result (2026-10-12):** 50dc0de deployed (dist). 7/7 mutants caught after the monitor added one tight-bracket test. Same API, OSM network: JS heap after GC 207.5 → 146 MB, 80 → 107 fps, renderer 469 → 301 MB, GPU 434 → 373 MB (pre-OSM 148 / 285 / 352). `reports/t65c-final-{wide,mid}.png`.

### Task 37 (amendment 2026-10-12): OSM rail geometry (user requests: "why tf is the thing a straight line and not following the rails?? there's no straight rail"; "yes do task 65, then task 37")

**Why (monitor, 2026-10-12):**
- The VM has 16,400 segments and none has a polyline, so every trail is a station-to-station chord or a Task 61 station chain. Image #11 shows a straight trail from Gatora to Champa.
- `build_segments` resets every `polyline` and `confidence` before it applies the overrides, and `export` calls it on every run. Geometry written straight into `ref.segment` would therefore be wiped by the next export.
- Size: the segments total 456,668 track-km, about 6.7× India's route-km, because halt-to-halt segments overlap; the median is 28 km and p90 is 62 km. Each run's trail also expands over its whole route, so polylines must be simplified.
- All 10,741 route versions have `dist_src='gtfs'`, so segment km is real track km.

**Files:**
- create `msynd/db/migrations/0007_segment_osm.sql`, `msynd/static/osm.py` and `tests/static/test_osm.py`;
- modify `msynd/static/segments.py`, `msynd/static/export.py`, `tests/static/test_segments.py` and `docs/deploy.md`.

No new dependency. The routing uses stdlib `heapq`, not networkx, and the `osmium` binary runs only in a throwaway container that the monitor drives.

- **Storage (0007):**
  - `ref.segment_osm(stn_lo text, stn_hi text, polyline text not null, confidence real not null, primary key (stn_lo, stn_hi))` with `check (stn_lo <= stn_hi)` and `check (confidence > 0 and confidence <= 1)`.
  - `build_segments` does its reset, then copies `ref.segment_osm` onto the matching `ref.segment` rows, then applies the overrides CSV as today. Overrides always win. Removing an override row brings back the OSM geometry.
  - OSM rows with no matching segment are ignored, because the network can change.
- **Input:** `python -m msynd.static.osm --opl PATH`.
  - `PATH` is the output of `osmium cat -f opl,add_metadata=false,locations_on_ways=true` on a file filtered with `osmium tags-filter … nw/railway=rail,narrow_gauge n/railway=station,halt` and then `osmium add-locations-to-ways`. Way lines therefore carry `n<id>x<lon>y<lat>` node refs.
  - The OPL is untrusted:
    - parse it with the stdlib only and decode `%xxxx%` escapes;
    - skip and count malformed lines, non-finite coordinates and points outside lon 68–98 / lat 6–37;
    - never eval anything.
  - Rail ways are those whose `railway` tag is `rail` or `narrow_gauge`; station points are nodes tagged `railway=station|halt`.
- **Rail graph:**
  - Nodes are the OSM node ids; an edge joins each pair of consecutive way nodes, weighted by haversine km.
  - After anchoring, contract degree-2 chains. Keep junctions, way ends and anchor nodes; each contracted edge keeps its coordinate list and its km.
  - Budget: under 30 min and 4 GB RSS on the VM for India.
- **Station match:**
  - Only stations with coordinates take part. Each one takes the first tier that applies:
    - `ref`: an OSM station whose `railway:ref` or `ref` (split on `;`, case-insensitive) equals our code, within 5 km of our coordinate;
    - `name`: otherwise, the nearest OSM station within 1.5 km whose normalised name has a `difflib` ratio ≥ 0.8 with ours;
    - `coord`: otherwise, our own coordinate.
  - Name normalisation: lowercase, letters only, with the words junction, jn, halt and station dropped.
  - Anchor: the nearest rail node within 1 km of the matched point (the OSM station for `ref` and `name`, our coordinate for `coord`). Use a 0.01° grid index. No anchor means none of that station's segments are routed.
- **Routing:**
  - Run A* (haversine heuristic) between the two anchors of each segment. Prune any state whose g + h exceeds 1.3 × the target km.
  - The target km is the median |Δ`dist_km`| over the segment's route stops, as in `segment_geometry`.
  - Accept the path only if its km, including the station-to-anchor stubs, is within ±10 % of the target.
- **Confidence:**
  - 1.0 when both ends are `ref` matches;
  - 0.5 when the worse end is a `name` match;
  - 0.3 when either end is a `coord` match. The original rule's "0 = straight line" assumed no third tier; this one keeps rail geometry wherever the length check passes.
  - Rejected segments get no row, so they keep today's chain or straight fallback.
- **Geometry:**
  - `[station lo, path…, station hi]` uses our station coordinates at both ends, so trails meet the station dots. It runs from `stn_lo` to `stn_hi`.
  - Simplify with Douglas–Peucker at 75 m (local equirectangular metres), always keeping the endpoints. Round to 5 decimals and store as compact JSON.
  - `# ponytail:` one simplification tolerance for all zooms; add a coarse low-zoom LOD if the network or trail vertex counts hurt.
- **Write and report:**
  - Replace all of `ref.segment_osm` in one transaction.
  - Print JSON with:
    - stations per tier (`ref`, `name`, `coord`, `no_anchor`);
    - segments per confidence, and failures by reason (`no_anchor`, `no_path`, `length`);
    - total polyline vertices;
    - runtime.
  - The monitor re-exports through the existing export command.
- **Chains (`export.segment_geometry`):** when a Task 61 derived chain is rebuilt, each constituent segment contributes its own polyline, oriented along the chain, if it has one; otherwise its station chord. Proximity links (`edge_id` None) stay straight.
- **`docs/deploy.md`:** add the OSM refresh: the three osmium steps in a throwaway container, `migrate`, `osm --opl` and the re-export command. Keep the file under 60 lines.
- [x] Tests (tiny synthetic OPL written in `tmp_path`, plus the test DB):
  - `test_parse_opl` covers escaped tags, located way nodes, and skipped malformed or out-of-bbox lines;
  - `test_path_follows_rails_not_chord`: on an L-shaped track, the polyline passes the corner vertex;
  - `test_confidence_rules` covers ref+ref → 1.0, ref+name → 0.5, name+coord → 0.3, and a far `ref` (> 5 km) falling through to the next tier;
  - `test_path_length_tolerance`: a path at 1.15 × target km gets no row, and one at 1.05 × gets a row;
  - `test_geometry_endpoints_orientation_simplified`: the polyline starts at `stn_lo`'s coordinate and ends at `stn_hi`'s, and a dense straight run collapses to its endpoints;
  - `test_override_wins` and `test_osm_survives_rebuild_and_unoverride` (in `test_segments.py`);
  - `test_chain_reuses_constituent_polylines`;
  - `uv run pytest -q tests/static tests/db` passes.
- [x] Commit `task 37: osm rail geometry` (5fa612e; 170+583 passed, 1 skipped; web 118; 16/16 mutations caught).
- [x] Monitor, on the VM (2026-10-12):
  1. Download `india-latest.osm.pbf` and its `.md5` from download.geofabrik.de into its own `/data/osm/src`, and verify the md5.
  2. Run the three osmium steps in a throwaway `debian:bookworm-slim` container.
  3. Rebuild the image; run `migrate`, then `python -I -m msynd.static.osm --opl /data/osm/rail.opl` in the ops container. Record the report, peak RSS and runtime.
  4. Re-export the network, and record the gz size and polyline vertex total.
  5. In the browser, remeasure memory against Task 65's numbers and screenshot Bilaspur–Champa (the trails must follow the rails).
  6. OSM data stays under ODbL; the map already credits © OpenStreetMap. The derived table lives only in the VM DB and never in the public repo.
  - Results: stations ref 7681 / name 103 / coord 691 / no_anchor 75; segments 1.0: 8679, 0.5: 97, 0.3: 655; failures no_anchor 502, no_path 2069, length 4409; 158,567 vertices; 28.9 s, peak RSS 587 MB. Export `network-af87af0ce6e9` 4.27 MB gz (was 3.13 MB), 178,283 network vertices. Bilaspur–Champa trails follow the rails (`reports/t37-osm-bilaspur.png`). Browser with the old dist: heap 148 → 211 MB, 131 → 75 fps; follow-up 65c windows trail vertices to recover it.

### Task 66 (amendment 2026-10-12): float32 frame quantiles (user request: "always use float32 for shrkingking the frame size further... if its viable")

**Why (monitor, 2026-10-12, latest VM full frame, gen 5470):**
- The frame is 15,859,848 bytes for 3,171 runs, 3,165 of them `schedule_only`.
- By field: `halt_q` 9.68 MB (1,088,570 msgpack float64 values, 45,136 of them None); `paths` 4.81 MB (already uint16); `versions` 0.47 MB; `next_q` 0.20 MB; `headline` 0.08 MB.
- As raw little-endian float32 bytes, `halt_q` would be ~4.35 MB, so a full frame drops to ~10.5 MB.
- Quantiles are run-clock minutes below 10,000, where float32 resolves better than 0.06 s.
- Epoch fields (`ts`, `base_ts`, `origin_ts`, `headline.last_received`) stay float64, because float32 resolves only 128 s at epoch magnitude.

**Files:** `msynd/engine/frames.py`, `msynd/engine/bench.py`, `msynd/api/app.py`, `tests/engine/test_loop.py`, `tests/api/test_api.py`, `web/src/frames.ts`, `web/src/frames.test.ts`, `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/timeline.test.ts`.

- **Format:**
  - `halt_q` = `np.asarray(out.halt_q, '<f4').tobytes()` plus `halt_shape = [n_halt, 2, 7]`.
  - `next_q` = 28 bytes of `<f4`.
  - NaN means absent and replaces None. Delete `_quantiles`.
  - This mirrors `paths`/`path_shape`. Never use msgpack's global `use_single_float`.
- **API `/api/run/{id}`:**
  - Decode with `np.frombuffer(…, '<f4').reshape(halt_shape)`.
  - The JSON response keeps its shape: lists of floats (float32-rounded) with None for NaN, and the same `any(q is not None …)` halt filter.
  - `Frames.ingest` and resync pass the bytes through unchanged.
- **Web:**
  - `validate` checks, when present, that `halt_q` is a `Uint8Array` with `halt_shape` = 3 ids `[n, 2, 7]` and `byteLength = n·2·7·4`, and that `next_q` is a 28-byte `Uint8Array`.
  - `layers.ts` reads the P50 with `DataView.getFloat32(((h·2 + kind)·7 + 3)·4, true)` over the run's bytes. Subarrays may be unaligned, as with `paths`. A NaN keeps today's display-median fallback.
- `bench.py` encodes the same way, so its frame size matches production.
- There is no shim for list-form quantiles. Frames are pruned after 15 min, and the monitor deploys engine + api together, waits for the first new full frame, then deploys `web/dist`.
- [x] Tests:
  - `encode_run`'s `halt_q` bytes decode to the replay array as float32, with NaN where the replay had NaN;
  - `next_q` is 28 bytes;
  - a fractional-second `origin_ts`/`base_ts` survives msgpack exactly;
  - the loop tests compare decoded arrays instead of `[None]*7` lists;
  - the API run endpoint returns the same halts JSON from a bytes frame (None for NaN, past halts omitted);
  - web: `validate` rejects a wrong `halt_q` byteLength or shape and a non-`Uint8Array`;
  - web: the delay uses the P50 from an unaligned `halt_q` subarray, and a NaN P50 falls back to display medians;
  - `uv run pytest -q tests/engine tests/api` and the web checks pass.
- [x] Commit `task 66: float32 frame quantiles` (109d2a3; engine/api 114, offline 371 + 1 skipped; web 137).
- [x] Monitor: rebuild the image and restart engine + api. After the new full frame, deploy dist and record the frame's raw and deflated size, the client heap and the fps.
  - Results: full frame gen 5636 is 10.26 MB raw (was 15.86 MB) and ~7.2 MB deflated.

### Task 67 (amendment 2026-10-12): clean basemap tiles (user request: "these weird small dots feel like geometric artifact... and maybe also consume rendering memory for such a small overlooked but messy detail so fix that")

**Why (monitor, decoded VM tiles 5/22/13, 6/45/27, 7/90/54, 7/91/54, 8/181/109, 9/365/219 and 10/727/438):**
- **The specks are water rings.** Protomaps merges each water `kind` into one multipolygon per tile, broken into tiny rings.
  - Tile 6/45/27 has ~200 inland water rings, ~150 of them under 4 px² (256-px tile basis). `water/river` alone has 93 rings, 76 of them under 4 px².
  - Earth there is one 5-vertex square, so these are not earth holes.
  - Water features carry only `kind`, `kind_detail` and `sort_rank` (no `min_zoom`), so no style filter can drop small rings.
- **Most of each tile is never drawn.**
  - The style draws only `earth`, `water` polygons, `boundaries`, `roads` and `transit` (`web/src/basemap.ts`); station labels come from the network.
  - Yet `landuse` (13–58 KB raw per tile), `places` (2–25 KB), `landcover` (≤ 18 KB) and `pois` make up 55–75 % of each tile. For example, 7/91/54 is 109 KB raw, of which landuse is 58 KB and places 21 KB.
  - The archive is ~208 MB.

**Files:** create `msynd/static/tiles.py` and `tests/static/test_tiles.py`; modify `docs/deploy.md`.

**Command:** `python -I -m msynd.static.tiles IN.pmtiles OUT.mbtiles`. Stdlib only (`mmap`, `gzip`, `sqlite3`, `json`).

- **Read (untrusted input), PMTiles v3:**
  - Require the `PMTiles` magic, version 3, gzip internal and tile compression, and the MVT tile type; otherwise exit with a clear error.
  - Directories are varints: `n`, delta tile ids, run lengths, lengths, then offsets, where 0 means previous offset + length. Run length 0 marks a leaf directory; depth ≤ 4.
  - Check that every offset and length lies inside its section. Cap every gunzip at 16 MB and reject malformed varints.
  - Tile ids map to z/x/y by the PMTiles Hilbert scheme.
- **Rewrite** each distinct tile body once (cache by offset), then write it for every tile id in its run:
  - **Layers:** keep only `STYLE_LAYERS = ('earth', 'water', 'boundaries', 'roads', 'transit')`, with a comment that it mirrors `web/src/basemap.ts`. Copy kept layers' bytes verbatim, except `water`.
  - **Water features:** in `water`, keep only polygon features. Decode their geometry (MoveTo, LineTo and ClosePath, zigzag deltas) into rings. A ring with positive surveyor area (tile coordinates, y down) starts a polygon; the negative rings after it are its holes.
  - **Ring threshold:** drop every ring whose |area| < `MIN_RING_PX2 = 16` px² of a 256-px tile, i.e. `16·(extent/256)²` extent units. A dropped exterior takes its holes.
  - **Re-encode:** write the surviving rings with fresh cursor deltas, and drop features with no rings left. Keep each feature's id, tags and type unchanged.
  - **Ceiling:** add the comment `# ponytail: one area threshold at every zoom; z10 also feeds overzoom to z13, where a dropped pond is ~32 px wide; add a per-zoom threshold if missed`.
- **Write** an mbtiles file:
  - tables `metadata` and `tiles`, TMS y = 2^z − 1 − y, gzip bodies;
  - metadata `name`, `format=pbf`, `minzoom`, `maxzoom`, `bounds`, `center`, `attribution`, and `json` holding the source's `vector_layers` filtered to `STYLE_LAYERS`;
  - print JSON: tiles in and out, bytes per layer in and out, water rings and vertices in and out, runtime.
- The monitor converts the file with the existing go-pmtiles binary (`pmtiles convert`), verifies it, keeps `india.pmtiles.orig` for rollback, and swaps it in.
- **`docs/deploy.md`:** add the tile clean-up steps, keeping the file ≤ 60 lines.
- [x] Tests (synthetic, built inside the test):
  - Hilbert tile id ↔ z/x/y for z0–z2 against the spec values, and a roundtrip on random z10 ids;
  - a root → leaf directory with run length 2 and a 0 offset decodes to the right entries;
  - bad magic, an out-of-range offset, a truncated varint and an over-cap gunzip each raise a clear error;
  - water filter: given a big square, a tiny square and a big square with a tiny hole, the tiny square and the tiny hole go, and both big rings stay with identical coordinates;
  - a water LineString is dropped; `earth` and `roads` bytes pass through verbatim; `landuse` and `places` are dropped;
  - end to end: a tiny pmtiles written in the test becomes mbtiles rows with TMS y, gzip bodies and filtered `vector_layers`;
  - `uv run pytest -q tests/static` passes.
- [x] Commit `task 67: clean basemap tiles` (5be6927; static 98, offline 412 + 1 skipped).
- [x] Monitor: run the tool on the VM, convert and swap. Record the archive size, the water rings left in 6/45/27, and the browser's renderer and GPU memory, and screenshot Jaipur–Jhansi at the zoom of image #14.
  - Results: archive ~208 MB → 70,482,146 B, `/data/tiles/india.pmtiles.orig` kept; vector layers now boundaries, earth, roads, water. No specks at ~z8 (`reports/t68-zoom-specks.png`).

### Task 68 (amendment 2026-10-12): glass dashboard (user request: "let's just always displlay the total train count, available, always ... Choose a glass morphism UI for all the elements that's supposed to be on screen other than the map, with better fonts, alignment and sizing... AND allowing the option to hide and unhide them")

**Files:** modify `web/index.html`, `web/src/style.css`, `web/src/main.ts`, `web/src/main.test.ts`, `web/package.json` and `web/package-lock.json`; create `web/src/hud.ts` and `web/src/hud.test.ts`.

- **Count:**
  - `#train-count` always shows the total of drawn trains: `3,171 trains`, or `1 train`, grouped in `en-IN`.
  - The live/timetable breakdown appears only in its `title`.
  - Logic: `trainCount(displays) → { text, title }` in `hud.ts`.
- **Hide and unhide:**
  - **Always visible:** a compact top-left bar holding the LIVE dot, the total count and a glass toggle `#hud-toggle`. The toggle has an eye icon, `aria-pressed`, the fixed `aria-label="Hide overlays"` and a `title` that names the `H` shortcut.
  - **Hidden mode** hides the clock, freshness, search, `#legend`, `#panel`, `#hover` and the zoom control.
  - The selection and its map highlight stay. Unhiding shows the panel again if a train is selected.
  - The compact attribution stays visible in both states (OSM ODbL and Protomaps credit).
  - `H` toggles, except while focus is in an input or textarea or a modifier key is held.
  - **State** lives in `document.body.dataset.hud` (`shown` or `hidden`), and CSS does the hiding. It persists in `localStorage['msynd.hud']` inside try/catch; if storage is blocked, the default is shown. `initHud(doc, storage)` in `hud.ts` owns all of this.
- **Glass:** one set of tokens on `:root`, used by every overlay: `#live`, `#legend`, `#panel`, `#hover`, `#hud-toggle`, the MapLibre control group and the attribution.
  - Background: a translucent gradient, alpha ~0.42–0.55.
  - Blur: `backdrop-filter: blur(18px) saturate(160%)`.
  - Edges: a 1 px rgba(255,255,255,~.1) border, an inset top highlight and a soft shadow.
  - Radius tokens: 16, 10 and 8.
  - Fallback: under `@supports not (backdrop-filter: blur(1px))`, alpha ~0.85.
  - Keep the delay colours, the freshness colours and the reduced-motion rule.
- **Type:**
  - Self-host Inter Variable: `npm install --save-exact @fontsource-variable/inter` (OFL-1.1, no CDN). Vite bundles the woff2, and unicode-range limits downloads to the latin subset. This amendment admits that one web dependency.
  - Stack: `'Inter Variable', 'Segoe UI Variable', system-ui, sans-serif`, with `tnum` and `cv11`.
  - One type scale (11/12/13/15/22/28 px), weights 450/500/600, and tracking .08em on small uppercase labels.
- **Alignment and sizing:**
  - One spacing scale: 4/8/12/16/24.
  - Every overlay uses the same outer gutter: 16 px, or 12 px under 640 px.
  - `#live` and `#legend` share a width and form the left column.
  - The panel is inset by the gutter at top, right and bottom.
  - Zoom buttons are 36 px.
  - The mobile layout keeps working.
- [x] Tests:
  - `trainCount` for 0, 1 and 3,171 trains, plus the breakdown title;
  - `initHud`: the initial state comes from storage; a click and `H` both toggle, persist the state and flip `aria-pressed`;
  - `initHud`: `H` is ignored in the search input and with Ctrl, Meta or Alt held; storage that throws on read and on write still toggles;
  - main: the count element shows the total;
  - web test, build and `tsc` pass.
- [x] Commit `task 68: glass dashboard` (e7fbbbe; web 152, tsc and build pass).
- [x] Monitor: deploy the dist. Take screenshots in both states, at desktop width and at 390 px. Measure fps with the panel open against 65c's 107; if the blur costs more than 15 %, lower `#panel`'s blur.
  - Results: two monitor fixes. Lightning CSS kept only the hand-written `-webkit-backdrop-filter` and dropped the unprefixed one, so there was no blur; the `-webkit-` line was deleted (never hand-write it), and the build now emits both. Autofill keydowns can lack `key`, so the check uses `key.key?.`. At 1440×900: heap 129 MB (was 146), fps 97 with the panel closed and 105 with it open (65c: 107), so the blur is free. Screenshots `reports/t68-*.png`.

### Task 69 (amendment 2026-10-12): HUD polish (user request: "/loop until the UI looks production grade")

**Why (monitor, 2026-10-12, screenshots `reports/t68-*.png`):**
- At 390 × 844 the expanded attribution covers the legend's "60+ min late" label.
- Before the first frame the bar reads "0 trains" next to a red "Waiting for live frames", so a normal load looks broken.
- The eye toggle shows the same open eye in both states.
- `index.html` has an empty favicon and no theme colour.
- The `rail` basemap layer reads source-layer `transit`, which the cleaned tiles no longer contain (their vector layers are boundaries, earth, roads and water).

**Files:** `web/index.html`, `web/src/style.css`, `web/src/main.ts`, `web/src/main.test.ts`, `web/src/hud.ts`, `web/src/hud.test.ts`, `web/src/basemap.ts`, `web/src/basemap.test.ts`, `msynd/static/tiles.py`.

- **Mobile stack (≤ 639 px):** while `#legend` is visible, the bottom-right control stack (zoom and attribution) sits above the legend with a `--space-2` gap, so nothing overlaps. When the legend is hidden (hidden mode, or the panel open), the stack keeps its current placement. Desktop is unchanged. CSS only.
- **Loading state:**
  - `liveStatus(age: number | null) → { text, freshness }` in `hud.ts` owns the status text. `null` gives `Connecting…` with freshness `pending`; otherwise `updated N s ago` with fresh < 60 ≤ aging < 180 ≤ old, as today.
  - `#connection[data-freshness="pending"]` is muted (`#8fa3b8`), never the red `old` colour.
  - Until the store holds a frame, `#train-count` reads `Loading…` with no title; this is also the initial HTML text. After the first frame it shows `trainCount` as today.
  - The `Network unavailable: …` path is unchanged.
- **Eye toggle:** the button holds an eye icon and an eye-off (slashed) icon, and CSS shows only the eye-off one when `aria-pressed="true"`. The fixed `aria-label` and the `title` stay.
- **Head:** an inline SVG data-URI favicon (a cyan train-head dot on `#05070d`), `<meta name="theme-color" content="#05070d">`, `<meta name="color-scheme" content="dark">` and a one-line `<meta name="description">`.
- **Dead layer:** delete the `rail` layer from `basemap.ts`, and drop `'transit'` from `STYLE_LAYERS` in `msynd/static/tiles.py`, so the mirror comment stays true.
- [x] Tests:
  - `liveStatus`: null → `Connecting…`/`pending`; 0 and 59 → fresh; 60 → aging; 180 → old;
  - main: before any frame the count reads `Loading…` and `#connection` has freshness `pending`; after a frame the count shows the total;
  - basemap: the layer ids no longer include `rail`, and no layer uses source-layer `transit`;
  - web test, build and `tsc` pass; `uv run pytest -q tests/static` passes.
- [x] Commit `task 69: hud polish`. (ab4b8ab; 158 web tests, `tsc`, build and 98 `tests/static` pass; monitor mutations `pending`→`old` and `if (store.latest)`→`if (true)` each fail tests.)
- [x] Monitor: deploy the dist, then screenshot 390 × 844 (legend fully visible, no overlap; hidden mode with the eye-off icon), the loading state and the desktop. Done 2026-10-12: VM at ab4b8ab; `reports/t69-mobile-{loading,shown,hidden}.png` and `reports/t69-desktop-{shown,panel,search}.png` show no overlap, the eye-off icon, muted `Connecting…` and `Loading…`.

### Task 70 (amendment 2026-10-12): legible station labels (user request: "/loop until the UI looks production grade")

**Why (monitor, 2026-10-12, `reports/t69-label-crop.png`):** station labels such as "Gaya Jn" or "Barauni Jn" are faint grey smudges on the bright trails. Root cause: deck.gl 9.4 measures `outlineWidth` in SDF atlas pixels (64 px glyphs) and divides it by `fontSettings.radius` (default 12). The halo extends `0.75 × outlineWidth` atlas pixels past the glyph edge and is clipped by `fontSettings.buffer` (default 4). So the current `outlineWidth: 0.18` draws a 0.135 atlas-px halo, about 0.03 screen px: no halo at all.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`.

- `stationLabelLayer`:
  - `fontSettings: { sdf: true, buffer: 10, radius: 16 }`;
  - `outlineWidth: 12`, which gives a 9 atlas-px halo, about 1.7 px at 12 px text;
  - `outlineColor: [5, 7, 13, 255]`;
  - `fontWeight: 600`;
  - `getColor: [214, 224, 238, 255]`;
  - `getSize`: 12 below zoom 7, otherwise 13.
- Add a comment stating the halo relation (`0.75 × outlineWidth` atlas px, which must be ≤ `buffer` and < `radius`).
- In `labelStations`, the collision box estimate scales with the new size: use 7 px per character instead of 6.5 and keep the `ponytail:` comment. Nothing else changes.
- [x] Tests (replace the label-style assertions in `layers.test.ts`):
  - halo in screen px, `0.75 × outlineWidth × size / 64`, is ≥ 1.5 at both sizes;
  - `0.75 × outlineWidth ≤ fontSettings.buffer` and `outlineWidth < fontSettings.radius`;
  - outline alpha is 255; `fontWeight` is 600; sizes are 12 below zoom 7 and 13 at zoom 7 and above;
  - the existing collision tests still pass after the width change (update their expectations only if the 7 px estimate changes which labels are kept, and say so).
  - Web test, build and `tsc` pass.
- [x] Commit `task 70: legible station labels`. (8fbb491; 163 web tests, `tsc` and build pass; `main.test.ts` size expectation 12 → 13 at zoom ≥ 7.)
- [x] Monitor: deploy the dist; crop the label region at 1440 × 900 (India zoom and the Bihar zoom used for `t69-label-crop.png`); check that labels are readable over trunk trails and that fps with the panel closed is ≥ 60. Done 2026-10-12: VM at 8fbb491 (web only); `reports/t70-india{,-crop}.png` and `t70-label-crop.png` show crisp labels with a dark halo; fps 114.6 with 2,807 trains. Found a pre-existing defect: lowercase `j` is misplaced ("Prayagra; Jn", "Vi ayawada Jn" in `reports/t70-j-{prayag,vijay}.png`), fixed by Task 71.

### Task 71 (amendment 2026-10-12): correct glyph placement in station labels (user request: "/loop until the UI looks production grade")

**Why (monitor, 2026-10-12; debug-page comparisons in `reports/t70-jfix-{13,26}-cmp.png`):** deck.gl 9.4's SDF atlas misplaces glyphs that have a negative left bearing. In `font-atlas-manager.js`:
- `measureText` sets the cell width to `actualBoundingBoxRight − actualBoundingBoxLeft`, but the ink width is right + left. For Inter 600 at 64 px, `j` has left 6 and right 14, so the cell is 8 + 2b wide while TinySDF's image is 20 + 2b. The next glyph's image (often the space) then overwrites j's right columns.
- The SDF draw ignores TinySDF's `glyphLeft`, and layout puts the image's left edge on the pen position, so the ink always starts at pen + buffer. A glyph with a negative bearing (`j`: −6) shifts right into the previous letter. `y` (left 1) also overlaps by 2 px.

**Plan change (dependencies):** `@mapbox/tiny-sdf` becomes a direct dependency of `web/` (`^2.2.0`). It is the library deck.gl already uses for this atlas (`@deck.gl/layers` requires `^2.0.5`; 2.2.0 is installed and already bundled), so no new code ships.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`, `web/package.json`, `web/package-lock.json`.

- In `layers.ts`, a module-level `stationFontRenderer(settings)` passed as the TextLayer prop `_getFontRenderer`:
  - It must be a single module-level function: TextLayer compares the prop by identity and rebuilds the atlas when it changes.
  - Build `new TinySDF({ fontSize, buffer, radius, cutoff, fontFamily, fontWeight: String(fontWeight) })` from the settings.
  - Per glyph, `t = sdf.draw(ch)` is cached in a `Map`, because deck calls `measure(ch)` and then `draw(ch)` for each new character.
  - `shift = max(0, buffer + t.glyphLeft)`.
  - `measure(ch)` returns `{ advance: t.glyphAdvance, width: t.glyphWidth + shift, ascent: t.glyphTop, descent: t.glyphHeight − t.glyphTop }`.
  - `draw(ch)` returns `{ data, left: buffer, top: buffer }`, where `data` is an ImageData `(t.width + shift) × t.height` holding TinySDF's alpha in the A channel at column offset `shift`, with RGB at 0. So the image exactly fills its cell (`width + 2·buffer`), and the ink starts at pen + `2·buffer + glyphLeft`, a uniform shift for every glyph with `glyphLeft ≥ −buffer`.
  - `measure()` with no character returns `{ advance: 0, width: 0, ascent, descent }`, with `ceil(fontBoundingBoxAscent/Descent)` of `measureText('A')` in the same font. This is deck's own rule.
  - Put the arithmetic in pure exported helpers (`glyphShift`, `padGlyphAlpha(alpha, width, height, shift) → Uint8ClampedArray` RGBA, `glyphMetrics(t, buffer)`), so vitest can test it without a DOM.
- **Compensation:** every glyph's ink sits `2·buffer` atlas px right of its true position, so `getPixelOffset` becomes `[−2·buffer·size/64, 10]` (about −4 px at 13 px).
- **Comment `ponytail:`** deck's atlas cache key ignores `_getFontRenderer`, so another TextLayer with identical font settings would share this atlas. Upgrade path: drop the renderer once deck.gl measures the ink width and honours `glyphLeft`.
- `package.json`: add `"@mapbox/tiny-sdf": "^2.2.0"` to `dependencies`. In `package-lock.json`, add the same entry to the root `packages[""].dependencies`. The `node_modules/@mapbox/tiny-sdf` entry already exists, so nothing else changes and no install is needed.
- [x] Tests (in `layers.test.ts`, no DOM):
  - with a synthetic glyph (buffer 10, `glyphLeft` −6, `glyphWidth` 20, image width 40), `shift` is 4, the measured width is 24, and the cell `24 + 2·10` equals the padded image width 44; the same holds for `glyphLeft` 0 and 3, and for −12 (shift clamps to 0);
  - `padGlyphAlpha` puts source column 0 at padded column `shift`, alpha only, with RGB 0;
  - the ink offset from the pen (`shift + buffer`) equals `2·buffer + glyphLeft` for each `glyphLeft ≥ −buffer`;
  - `stationLabelLayer` passes the same `_getFontRenderer` function on two calls, and the pixel offset x is `−2·buffer·size/64` at both sizes;
  - web test, build and `tsc` pass.
- [x] Commit `task 71: correct glyph placement in station labels`. (5d331d8; 178 web tests, `tsc` and build pass; monitor mutations `shift = buffer` and pixel offset `[0, 10]` fail 6 and 4 tests.)
- [x] Monitor: deploy the dist. At India zoom, crop "Prayagraj Jn" and "Vijayawada Jn" against `reports/t70-j-*.png`: `j` must be whole and in place. Labels must stay centred under their station dot. fps with the panel closed must be ≥ 60. Done 2026-10-12: VM at 5d331d8 (web only); `reports/t71-j-{prayag,vijay}{,-dot}.png` read "Prayagraj Jn" and "Vijayawada Jn" with whole, correctly placed `j`; fps 102.5 with the panel closed (2,755 trains).

---

### Task 72 (amendment 2026-10-12): slimmer engine frames (user request: "Do the smaller Live updates, but critically analyse the blast radius first")

**Why (monitor, 2026-10-12; measured read-only on the VM's latest full frame):**
- The frame holds 2,711 runs. 2,705 of them are `schedule_only` and ship 8 of K = 32 paths; 6 are `observed` and ship all 32.
- Size is 9.11 MB raw and 6.49 MB deflated. `paths` is 4.26 MB raw / 3.26 MB deflated; `halt_q` is 3.85 MB raw / 3.41 MB deflated.
- Per-run `versions` is 428 KB raw but only about 20 KB deflated, so slimming it is housekeeping, not the win.
- A `schedule_only` run never draws streaks (`displayRun` returns `trips = []` for it). Its paths only place the head and fill past or pass anchors, and for that the web already takes the per-event median of the shipped paths.
- So one median path per `schedule_only` run carries the same display information. Measured on the real frame: 5.10 MB raw / 3.73 MB deflated (−43 %).

**Plan change (spec §10 size estimate):** the frame carries K paths only for runs with evidence; a timetable-only run carries one per-event median path. This replaces the 8-path subsample.

**Files:** `msynd/engine/frames.py`, `tests/engine/test_loop.py`.

- In `encode_run`, settle `state` before packing. For a `schedule_only` run, replace `paths` with `np.median(paths, axis=0, keepdims=True)` before the tick conversion, so the uint16 range check covers the shipped row. The per-event median of paths that are each non-decreasing is itself non-decreasing (order statistics are monotone, and so is the mean of the two middle ones), so the web's monotonic check still holds. `path_shape` becomes `(1, E, 2)`.
- `# ponytail:` comment: timetable-only runs ship one median path; ship K once the UI draws timetable streaks.
- Per-run `versions` in `encode_run` becomes `{'route': evidence.versions['route']}`. That is the only field the web reads (`displayRun`, `hoverText`).
  - Unchanged: the frame-level `versions` in `write_frame` (spec §10: "Frames carry gen and versions"), the error payload in `worker.py` (keeps the full versions for diagnosis; `tests/engine/test_errors.py` pins it), and every key that is derived from `ev.versions` (the RNG `v_hash`, the CDF cache, `sim.forecast_issue.versions`).
- [x] Tests (in `tests/engine/test_loop.py`):
  - Replace the `test_encode_run_strides_only_schedule_paths` cases. A `schedule_only` run with K = 1, 8, 10 or 32 (paths `[j, j+1]`, plus one case whose rows are unsorted per event) ships `path_shape == (1, 2, 2)`, route indices `[3, 9]`, and ticks equal to `rint` of the per-event median. `observed` and `stale` still ship all K rows unchanged.
  - Encoded per-run `versions` is exactly `{'route': <route id>}`.
  - `test_model_change_and_bias_generation_dirty_all_active` no longer reads per-run `versions['models']`. Keep its intent by asserting that every active run was replayed with the new model: wrap `fake_replay` to record `ev.versions['models']['run']` per run id, and assert the recorded value for each of `runs[1:]` once the frame-level versions show the new model and bias generation.
  - `uv run pytest -q tests/engine` passes.
- [x] Commit `task 72: slimmer engine frames`.

### Task 73 (amendment 2026-10-12): live frames carry halt P50 only (user request: "Do the smaller Live updates…")

**Why (monitor, 2026-10-12):**
- After Task 72, `halt_q` (all 7 quantiles per halt) is most of what remains: 3.41 MB of the 3.73 MB deflated.
- The map reads only P50 from it (`displayRun` `eventTimes`, index `((h*2+kind)*7+3)`).
- The train panel's P10/P50/P90 rows come from `/api/run/{id}`, which decodes the full `halt_q` from `sim.frame` in the DB, not from the live socket.
- With the WebSocket carrying P50 only, the full frame measures 1.82 MB raw and 1.02 MB deflated: −84 % against today's 6.49 MB.
- Rejected: `uint16` quantiles. The observed range reaches 6,139.6 run-clock minutes against a 6,553.4-minute ceiling at 0.1-min ticks, so delays over a day would overflow.

**Plan change (spec §10 `WS /api/live`):** `sim.frame` stays the durable record with all quantiles. The API projects each frame once, as it ingests it, to the wire form: per run, `halt_q` + `halt_shape` become `halt_p50`, a little-endian float32 `[H, 2]` (arr, dep) byte string with NaN kept. Every WebSocket send (live, resync, full and slow-client fallback) carries the projected form.

**Files:** `msynd/api/app.py`, `tests/api/test_api.py`, `web/src/frames.ts`, `web/src/frames.test.ts`, `web/src/layers.ts`, `web/src/layers.test.ts`.

- `app.py`:
  - Add a small module-level `wire(frame)` that returns the frame with each run projected. A run without `halt_q` (error states, tombstones) passes through unchanged.
  - `Frames.ingest` projects right after `unpack`. `history` and `runs` hold projected frames, so `full()`, `resync()` and the full→delta diff all work on the wire form.
  - `outgoing` is always the msgpack of the projected frame (or of the delta built from it). It is never the original payload bytes.
  - `/api/run/{id}` is unchanged: it still reads and decodes the full `halt_q` from `sim.frame`.
- `frames.ts`:
  - `RunFrame` gains `halt_p50?: Uint8Array`, and `halt_q`/`halt_shape` are removed from the type and the validator.
  - `validate` accepts `halt_p50` only as a `Uint8Array` whose byte length is a multiple of 8, and rejects anything else.
- `layers.ts` `eventTimes` reads P50 as `getFloat32((h*2+kind)*4)` from `halt_p50` when `h < byteLength/8`. Otherwise it falls back to the path median as now; that fallback also covers a frame from an API that hasn't been redeployed yet. Update the existing `ponytail:` comment.
- [x] Tests:
  - API: a full frame whose run has a known `halt_q` reaches the socket with `halt_p50` equal to `halt_q[:, :, 3]`, as float32 bytes with NaN kept, and with no `halt_q`/`halt_shape`. The same holds on the delta, resync and slow-client full paths.
  - An error run passes through unchanged.
  - `/api/run` still returns all 7 quantiles. Adjust any existing test that compared forwarded bytes with the stored payload.
  - Web: `frames.test.ts` validator cases for `halt_p50` (valid `[2,2]`, empty, length 12, non-`Uint8Array`); `layers.test.ts` `eventTimes` uses `halt_p50` P50 when finite and the path median when NaN or absent.
  - `uv run pytest -q tests/api` and the web test, build and `tsc` pass.
- [x] Commit `task 73: live frames carry halt p50 only`.
- [x] Monitor (after Tasks 72 and 73): rebuild the image and restart the engine and the API. Then measure the WebSocket full frame on the VM, raw and deflated. Expect about 1.8 MB raw / 1.0 MB deflated. Check that the map shows trains after a hard refresh, a train panel shows its P10/P50/P90 rows, and fps with the panel closed is ≥ 60.
  - Done 2026-10-12 (VM at 763a6cf, API 172.18.0.3): DB full frame 4.70 MB raw / 3.46 MB deflated; WS full frame 1.67 MB raw / 0.95 MB deflated (2,426 runs; 2,420 schedule_only with K = 1, 6 observed with K = 32; per-run versions route only). The browser received a 1,668,018-byte first frame and drew 2,420 trains; panel P10–P90 rows present; fps 104.2 with the panel closed.

### Task 74 (amendment 2026-10-12): calmer glass with refraction (user request: "the glass blur is too crazy, go a little less crazy, and add refractions/wavy curving")

**Why:** `blur(18px) saturate(160%)` turns the map behind every HUD surface into flat haze. A lighter blur keeps the map readable through the glass. A gentle displacement adds the refraction the user asked for, and a curved specular sheen reads as a glass edge.

**Files:** `web/index.html`, `web/src/style.css`, `web/src/main.ts`, `web/src/main.test.ts`.

- `style.css` tokens:
  - `--glass-filter: blur(9px) saturate(135%)`.
  - `--glass-background` adds a curved top sheen over the existing tint: `radial-gradient(140% 70% at 20% -20%, rgba(255,255,255,.10), transparent 55%)`, then the current linear gradient.
  - `--glass-shadow`, curved edge highlights: `inset 0 1px 0 rgba(255,255,255,.16), inset 0 -1px 0 rgba(255,255,255,.05), inset 1px 0 0 rgba(255,255,255,.06), inset -1px 0 0 rgba(255,255,255,.03), inset 0 0 18px rgba(140,200,255,.05), 0 8px 24px rgba(0,0,0,.28)`.
  - Keep the `@supports not (backdrop-filter: blur(1px))` fallback.
- Refraction, as progressive enhancement:
  - Add a hidden inline SVG at the top of `<body>` (`width="0" height="0"`, `aria-hidden="true"`, absolutely positioned) with `<filter id="glass-refract" color-interpolation-filters="sRGB">`. The filter is `feTurbulence type="fractalNoise" baseFrequency="0.006 0.009" numOctaves="2" seed="3"`, then `feGaussianBlur stdDeviation="3"`, then `feDisplacementMap in="SourceGraphic" scale="12" xChannelSelector="R" yChannelSelector="G"`.
  - `body[data-refract]` sets `--glass-filter: url(#glass-refract) blur(7px) saturate(135%)`.
  - `main.ts` sets `document.body.dataset.refract = ''` only when `'userAgentData' in navigator`. That covers Chromium, the only engine that renders SVG filters in `backdrop-filter`. Firefox and Safari keep the plain blur, because an unsupported `url()` there would void the whole declaration.
  - Comment `ponytail:` naming that UA-capability gate, with the upgrade path: drop it once `@supports` can detect SVG backdrop filters.
- Never hand-write `-webkit-backdrop-filter` (Lightning CSS then drops the unprefixed line).
- [x] Tests (in `main.test.ts`, with the existing stubs): `refract` is set when the navigator stub has `userAgentData` and absent without it. Web test, build and `tsc` pass. The built CSS keeps `url(#glass-refract)` verbatim.
- [x] Commit `task 74: calmer glass with refraction`.
- [x] Monitor: deploy the dist. Compare screenshots of the HUD and the open panel against `reports/` from Task 69. The map must show through the glass with visible gentle waves, with no dark fringe wider than about 2 px at the edges. Text contrast is unchanged. fps with the panel closed must be ≥ 60 and within 15 % of Task 71's 102.5. If the refraction costs more than that, cut `numOctaves` to 1 and then `scale`; drop the `url()` before accepting a slower map.
  - Done 2026-10-12: Chromium computes `url("#glass-refract") blur(7px) saturate(1.35)`; map lines bend gently through the HUD and panel, with no edge fringe (reports/t74-panel.png, t74-panel-crop.png vs t74-panel-crop-norefract.png); fps 104.2 (Task 71: 102.5). No tuning needed.


### Task 75 (amendment 2026-10-13): per-run cooldown after repeated MMT 5xx (user request: "plan and delegate", approving the council's recommendation)

**Why (monitor, VM DB 2026-10-12):** run 11632 got MMT HTTP 500 375 times between 05:45 and 14:19Z. Each poll job retries up to 5 times, so one broken run fills the 5-minute window with non-`ok` outcomes. The host breaker then pauses everything for 10 minutes roughly every 13.5 minutes, which cost about 75 % of polling. The fix is to isolate the run, not to loosen the breaker: every real 5xx still counts toward the host breaker.

**Plan change (spec §7.1 retries, §7.2 admission):** a run whose 3 most recent status attempts all got HTTP ≥ 500 is held out of polling, harvest and breaker recovery for 1 hour after the latest one. After that it gets one probe attempt per hour until it answers with anything other than 5xx. There is no doubling.

**Files:** `msynd/collect/dispatcher.py`, `msynd/collect/mmt.py`, `msynd/collect/admission.py`, `msynd/collect/__main__.py`, `tests/collect/test_dispatcher.py`, `tests/collect/test_admission.py`.

- `admission.py`: constants `COOLDOWN_5XX = 3` and `COOLDOWN = timedelta(hours=1)`. `PollRun` gains `streak_5xx: int = 0`. `Job` gains `stop_after_5xx: int`, which `next_job` sets to `max(1, COOLDOWN_5XX - run.streak_5xx)` for every job it builds (poll, overdue, recovery, harvest).
- `dispatcher.py`: `request(..., stop_after_5xx: int | None = None)`. When set, the retry loop also returns once this burst has seen that many responses with HTTP ≥ 500. Timeouts, 429 and transport failures without a status don't count toward it. `None` keeps today's behaviour. Breaker accounting is unchanged.
- `mmt.py`: `train_running` passes `stop_after_5xx` through.
- `__main__.py` `load_state`:
  - for each candidate run, `streak_5xx` = the number of its most recent completed MMT status attempts (any mode; ignore `inflight`) with `http_status >= 500` and no other completed attempt after them;
  - exclude from both the poll cohort and harvest any run with `streak_5xx >= COOLDOWN_5XX` whose latest such attempt started less than `COOLDOWN` before `now`;
  - the state comes from `obs.fetch` only, so it survives restarts.
  - `run_loop` passes `job.stop_after_5xx`.
- `# ponytail:` comment on the constants: a fixed 1 h cooldown with no doubling; add backoff only if the recovery-yield report shows hourly probes wasting budget.
- [x] Tests:
  - Dispatcher (fake clock): with `stop_after_5xx=3`, the script 503,503,503,200 logs 3 rows and returns `transport_error`; 503, timeout, 503, 503 logs 4 rows; `None` keeps 5 attempts; widespread 5xx across different runs still trips the host breaker.
  - `load_state` on the test DB: 3 consecutive 5xx with the latest 30 min ago → absent from cohort and harvest; latest 61 min ago → present, and its job has `stop_after_5xx == 1`; a 200 or 400 after the 5xx resets the streak; a streak of 2 gives `stop_after_5xx == 1`; a streak of 1 gives 2; an `inflight` row is ignored.
  - A healthy run beside a cooling run is still polled within `T_max` (extend the starvation test or add a case).
  - `uv run pytest -q tests/collect` passes.
- [x] Commit `task 75: per-run cooldown after repeated mmt 5xx`.
- [x] Monitor: deploy the collector. After 24 h, no run should have more than 3 consecutive 5xx within any hour, and breaker pauses per day should drop.
  - **Monitor result (2026-10-14 ~03:00Z, ~17 h after deploy):** no run had more than 3 consecutive 5xx, and 5xx is near zero. However, the breaker still tripped about every 25 min on 10-13 from 18Z to 23Z (72 breaker rows), and ok/h fell from ~180 to 57–85.
    - The cause is now HTTP 200 pages that fail `parse_mmt.validate` with "non-monotone event times". These are genuine MMT data errors: for example run 3371 (18477) has a VKR dep at 23:35 followed by a JTI arr at 23:30.
    - Such runs store nothing, stay in the cohort and are re-polled every 120 s. Run 3371 alone produced 122 invalid answers between 21:03Z and 02:58Z.
    - Follow-up options are listed in "Monitor status (2026-10-16)" below.

### Task 76 (amendment 2026-10-13): truthful source freshness (user request: "plan and delegate")

**Why (monitor, 2026-10-12/13):** with the PC relay off overnight, the HUD still said "updated N s ago", because the engine writes a full frame every 5 minutes whether or not new evidence arrived. `/api/health` reads `dispatcher_state:*` metrics, but nothing writes them, so `dispatchers` is always `{}`.

**Plan change (spec §10 `/api/health`, §11 status bar):** health reports source freshness from `obs.fetch` (last `ok` poll and last `ok` harvest, separately), and the collector writes its dispatcher state. The HUD's main text reflects live-data age, not frame age.

**Files:** `msynd/collect/__main__.py`, `msynd/api/app.py`, `tests/api/test_api.py`, `tests/collect/test_admission.py`, `web/src/hud.ts`, `web/src/hud.test.ts`, `web/src/main.ts`, `web/src/main.test.ts` (only if needed).

- Collector: in `run_loop`, insert `sim.metric (clock_timestamp(), 'dispatcher_state:<host>', 0|1|2)` (ok/breaker/stopped, matching `app.py`) at start, whenever `d.state` changes, and at least every 5 minutes.
- API `/api/health` gains `source: {"poll_ok_at": ts|null, "harvest_ok_at": ts|null}`: the max `received_at` of `ok` MMT status fetches per mode, restricted to `started_at` within the last 2 days so the `(host, started_at)` index applies. Keep the existing fields.
- Web:
  - `main.ts` fetches `/api/health` on load and every 60 s, ignoring failures.
  - `hud.ts` gains a pure `sourceStatus(pollAgeS | null, harvestAgeS | null, frameAgeS | null, dispatcher?: string)`, which returns `{text, freshness, title}`:
    - text `live data N min ago` (seconds under 2 min);
    - freshness `fresh` under 5 min, `aging` under 15 min, `old` beyond;
    - `unknown` with text `live data unknown` when health is unavailable or has no poll time;
    - when the dispatcher is `breaker` or `stopped`, the text adds ` · paused` or ` · stopped`;
    - the title carries the frame age and harvest age.
  - Keep `liveStatus` only if something else still uses it; otherwise replace it.
- [x] Tests:
  - API: health returns `source.poll_ok_at` and `source.harvest_ok_at` from seeded fetch rows (an `ok` poll, a newer failed poll, an `ok` harvest), with nulls when there are none.
  - Collector: the metric row is written at start and on a state change.
  - Web: the `sourceStatus` cases above, including fresh frames with old polling showing `old`.
  - `uv run pytest -q tests/api tests/collect`, plus the web test, build and `tsc`, pass.
- [x] Commit `task 76: truthful source freshness`.
- [x] Monitor: deploy. The HUD shows live-data age, and `/api/health` lists the MMT dispatcher.


### Task 77 (amendment 2026-10-13): local-only backup and VM cron (user decisions 2026-10-13: leg bias on now; local-only backup; weekly evaluate Mon 02:00 IST)

**Why:** the VM has no OCI credentials (`/data/oci` is empty), so `backup()` would fail at the upload, and the nightly job would exit non-zero every night. The user chose local-only backups for now. The weekly cron line in `docs/deploy.md` lacks the `evaluate --models-dir /data/models` subcommand.

**Plan change (spec §12 backup):** an empty `OCI_BUCKET_NAME` means local-only. The local dump (2 copies) still runs. The off-site dump and upload are skipped with one warning log line. A set bucket keeps today's behaviour.

**Files:** `msynd/ops/nightly.py`, `tests/ops/test_retention.py` (or wherever `backup` is tested), `docs/deploy.md`.

- `backup()`: when the bucket is empty, write only the local dump (with the same 2-copy pruning), log `off-site backup disabled: OCI_BUCKET_NAME empty`, and return. No `ValueError`. No offsite file is created.
- `# ponytail:` comment: local-only until the user approves OCI credentials, which re-enables off-site by setting the bucket.
- `docs/deploy.md` cron block:
  - weekly becomes `30 20 * * 0 … python -m msynd.ops.weekly evaluate --models-dir /data/models` (Sun 20:30 UTC = Mon 02:00 IST);
  - fix the line's comment;
  - note that an empty `OCI_BUCKET_NAME` means local-only backups.
- [x] Tests: an empty bucket runs one `pg_dump` (with `--schema=obs`) and no `oci` calls, keeps 2 local copies and raises nothing; a set bucket is unchanged (existing tests).
- [x] Commit `task 77: local-only backup and cron line`.
- [x] Monitor (VM): set `OCI_BUCKET_NAME=` in `/opt/msynd/.env` (back it up first) and install the crontab from `docs/deploy.md` (nightly 21:00 UTC with leg bias, weekly Sun 20:30 UTC). Run nightly once by hand and check: the bias rows are written, the engine replays, and a local dump appears in `/data/backups`.
  - **Monitor result (2026-10-14):** the first scheduled nightly ran at 21:00:01Z on 10-13 and finished in ~30 s. It wrote leg_bias gen 2 (13,442 rows) and `local-2026-10-14.dump` (28 MB), and kept 2 copies.
    - Gap: the syslog shows `No MTA installed, discarding output`, so cron output (including tracebacks) is lost. A log redirect is pending (see status below).
    - The first weekly run is Sun 2026-10-18 at 20:30Z. It cannot pass `MIN_RUNS` before Mon 2026-11-02.

### Monitor status (2026-10-16): open decisions for the next session

**Ramp gate (spec §18 "Training data"): batch 10-15 IST FAILS, so harvest is held at 512/day.**
- The measurement runs ad hoc SQL over `obs.fetch mode='harvest'`, with the IST date taken from `started_at`. "Usable" means an `ok` answer plus an actual destination arrival. R pairs are consecutive halts with dep A and arr B present; W pairs are intermediate halts with both arr and dep present.
- 10-15: 490 due keys, 512 attempts, 0 challenges.
  - Usable **53.9 %** (needs ≥ 80 %); R 98.9 %, W 99.2 %.
  - Classes below 70 % with n ≥ 30: PASS **0.8 %** (n=124), EXP 69.8 % (n=225).
  - 10-13 was 60.9 % usable and 10-14 was 70.5 %.
- Cause:
  - 195 of the 490 keys got MMT 400 "not running on date" and were marked `cancelled`. That includes 121 of the 124 PASS runs, which means the GTFS inventory lists runs that don't operate.
  - Another 28 keys were invalid 200 pages (non-monotone).
  - If the 400s are excluded from the denominator: 295 keys (< 300), 89.5 % usable, EXP 87 %.
- The current setting is unchanged: VM `.env` has MMT_RATE=0.3, MMT_POLL_BUDGET=7500, MMT_HARVEST_BUDGET=512.

**Decisions needed (the user's), with the monitor's recommendation:**
1. Gate denominator: exclude MMT 400 "not running" keys (not operated, so not a collection failure). *Recommend yes.*
2. Harvest scope: drop the PASS class (0.8 % yield, ~25 % of the harvest budget). This reverses "any class" for PASS only. *Recommend yes.*
3. Ramp: keep 512 until a batch passes under rules 1 and 2, then go to 1,500. Each step re-checks the gate.
4. Invalid-page breaker trips (Task 75 follow-up), choose one:
   - **A (recommended):** the per-run streak also counts consecutive `invalid` answers, giving 3 → 1 h cooldown, with the breaker unchanged.
   - **B:** store the page and drop or flag inconsistent events. This changes the validation rule.
   - **C:** the breaker stops counting invalid answers.
5. Cron logs: add `>> /data/logs/cron-<job>.log 2>&1` to both crontab lines in `docs/deploy.md` and on the VM, plus a logrotate entry. *Recommend yes.*
6. Earlier options, still unchosen: a recovery-yield report (offline); off-site backup with a bucket-scoped OCI key (OCI Create needs the user's OK). The always-on relay device option is superseded by Task 78's direct Oracle collection.

**Model training timeline:**
- Nightly leg bias already retrains every night.
- The first weekly LightGBM `evaluate` that can clear `MIN_RUNS=1000` is Mon 2026-11-02. Phase 3 promotion comes from about 11-08.
- The ramp adds labels but does not move these dates.

### Task 78 (2026-10-16): laptop-independent cloud collection

User authorized replacing the laptop relay, superseding the earlier deployment restriction.

- [x] Re-test the actual VM instead of assuming all cloud egress is rejected. Six requests
  through a temporary VM-host OpenSSL 3.0 relay returned HTTP 200 and parsed successfully.
  The existing Trixie/OpenSSL 3.5.7 container's direct control request timed out (fetch 32075).
- [x] Build the collector separately on `python:3.12-slim-bookworm` using the existing
  Dockerfile and lockfile. The image uses OpenSSL 3.0.22; direct fetches 32094–32096 all
  returned 200 and parsed. Other services retain their image and PostgreSQL dump compatibility.
- [x] Add the database restart policy. Apply it with `docker update` without recreating
  the production database; Docker is enabled at boot.
- [x] Deploy at 17:52:59 UTC with `COMPOSE_FILE=compose.yaml`, preserving the live flag,
  rate and budgets. Disable the Windows relay task and close its residual SSH tunnel.
  No third-party relay, cloud account or paid resource was needed.
- [x] Preserve the previous image and VM configuration, with rollback script, under
  `/opt/msynd/deploy-backups/20261016-direct-collector/`.
- [x] Monitor follow-up (2026-10-17 13:08 UTC): healthy for about 19 hours with no
  relay configured. Since the laptop relay stopped: 3,125 `ok`, 562 `invalid`,
  3 `transport_error`, zero challenges. Latest 30 minutes: 90 `ok`; actual-event
  revisions continue arriving. Invalid source pages remain subject to existing validation.
- [x] Review: no runtime/configuration blockers. Clarify the runbook to invoke the
  retained rollback script, which restores the old Compose file and image as well as `.env`.
- [x] Verification: Compose regression tests fail before the change and pass afterwards;
  image imports and direct source probes pass. Full Python suite: 869 passed, 1 skipped
  (2026-10-17; `MSYND_TEST_DSN` uses `127.0.0.1` because Windows `localhost` stalled).
  Web: 186 passed. A full-day soak and VM reboot are not claimed.
- [x] Monitor follow-up (2026-10-17): the PC relay's still-running session-0 tree (uv.exe 8400, started 10-15)
  was killed elevated at 13:38Z; VM :18080 is free and the task stays Disabled. VM reboot test at
  14:00:25Z: SSH back in 35 s, all five services healthy without intervention, iptables and crontab
  intact, MMT fetches resumed 14:01Z with OpenSSL 3.0.22. Direct-path ok latency p50 ≈ 0.85 s
  (relay ≈ 1.15 s), p95 3–5 s; transport errors 0–3/h; zero challenges.

### Monitor decisions (2026-10-17, user)

The user chose on the "Monitor status (2026-10-16)" list: **1c 2a 3a 4A 5a 6a 7a 8a 9 build**, with D and E
left to the monitor.
1. Ramp-gate denominator excludes keys whose answer was MMT 400 "not running" (1001/1002). Follow-up
   Task 79 also stops spending attempts on runs MMT has recently said do not operate on that weekday.
2. PASS leaves harvest (Task 79). Polling scope is unchanged.
3. Harvest stays at 512 until one IST batch passes the gate under rules 1–2, then 1,500, then 3,700, each
   step re-checked by the Task 81 report.
4. Invalid answers count toward the per-run cooldown (Task 80); the host breaker is unchanged.
5. The Bookworm/OpenSSL 3.0 collector is kept. Ceiling: a source-side TLS policy change or the Bookworm
   base ageing out sends collection back to the relay rollback.
6. Reboot test done (above).
7. Cron output goes to `/data/logs/cron-{nightly,weekly}.log` with logrotate (monitor ops, 10-17).
8. Off-site backup: pending the user's choice between the OCI bucket path (8a) and Google Drive.
9. Ramp-gate / recovery-yield report CLI: Task 81.
- D (monitor's lean): CP-3 waits for the first batch that passes under rules 1–2. The Phase 4 seven-day
  live window counts from the laptop-independent switch (2026-10-16 17:53Z) and is reviewed on 10-24.
  A7 follows Phase 4.
- E: no action; the first trainable weekly evaluate is still Mon 2026-11-02.

### Task 79 (amendment 2026-10-17): admission skips PASS harvest and known non-operating weekdays (user decisions 1c, 2a)

**Why (monitor, 10-15 batch):** 195 of 490 harvest keys got MMT 400 "not running", including 121 of 124
PASS runs; the GTFS inventory lists runs that do not operate. Harvesting PASS yields 0.8 % usable data
for about 25 % of the harvest budget.

**Plan change (spec §7.2 admission, §18 "Training data"):**
- Harvest admits only runs whose `ref.train.class` is not `PASS`. Polling scope is unchanged.
- Both the poll cohort and harvest skip run (T, D) when, among runs of train T with start dates
  D−7, D−14, D−21 and D−28, the most recent one that got an MMT answer (an `ok` status fetch, or an
  HTTP 400 status fetch on a run whose lifecycle is `cancelled`) got the 400 "not running" answer.
  With no such answer in that window the run is admitted as today, so a skipped weekday is re-probed
  at least every fifth week.

**Files:** `msynd/collect/__main__.py` (`load_state` only), `tests/collect/test_admission.py` (or the
existing `load_state` test module).

- Evidence comes from `obs.run` and `obs.fetch` only (status bodies may be pruned, so do not parse bodies).
- `# ponytail:` comment on the skip: weekly-pattern inference only; skipped runs still appear as timetable
  runs in the engine; upgrade to a running-days table if recovery-yield shows misses.
- [x] Tests on the test DB:
  - a PASS run is absent from harvest and still present in the poll cohort; a non-PASS run is in harvest;
  - prior same-weekday run at D−7 cancelled with a 400 fetch → D is absent from cohort and harvest;
  - the same, but a newer same-weekday answer (D−7 ok, D−14 cancelled) → D is admitted;
  - the only 400 answer at D−35 → admitted; a 400 at D−6 (other weekday) → admitted;
  - a cancelled run without any 400 fetch (exception cancellation) does not trigger the skip.
- [x] `uv run pytest -q` green. Commit `task 79: skip pass harvest and known non-operating weekdays`. (dabd856; 883 passed, 1 skipped. Monitor perf fix: the first draft used a correlated subquery, 96.5 s per load_state on production; the set-based anti-join takes 1.06 s vs 0.90 s before, harvest candidates 4,266 → 3,360.)
- [x] Monitor: rebuild `msynd-collector:local`, restart the collector, and compare the next IST batch's
  400 share and harvest attempts. (Deployed 2026-10-17 ~15:45Z at dabd856; rollback image `msynd-collector:pre-t79`. Baseline IST 10-12…10-17: harvest 400 share 49.8, 28.6, 20.5, 39.8, 43.5, 21.4 %; PASS keys 114, 83, 54, 124, 124, 56 of 416–490. First batch under Task 79: IST 10-18.)
  IST 10-18: 400 share 8.0 % (34 of 427 keys), PASS keys 0, 512 attempts, 0 challenges.

### Task 80 (amendment 2026-10-17): invalid answers count toward the per-run cooldown (user decision 4A)

**Why (monitor):** a run whose MMT page fails validation (e.g. non-monotone event times) stays in the
cohort and is re-polled every 120 s: about 30 invalid answers an hour on 10-17 00–10Z, and the 10-13
18–23Z breaker storm.

**Plan change (spec §7.1 retries, §7.2 admission; extends Task 75):** the per-run streak counts the most
recent consecutive completed MMT status attempts that are HTTP ≥ 500 **or** outcome `invalid`. A streak
of 3 holds the run out of polling, harvest and recovery for 1 h after the latest such attempt, then one
probe per hour, exactly as Task 75. The host breaker and the dispatcher are unchanged.

**Files:** `msynd/collect/admission.py`, `msynd/collect/__main__.py`, their tests.

- Rename to match the meaning: `streak_5xx` → `bad_streak`, `COOLDOWN_5XX` → `COOLDOWN_STREAK`.
  `Job.stop_after_5xx` keeps its name and is still `max(1, COOLDOWN_STREAK - run.bad_streak)`.
- A 400 "not running" answer cancels the run (Task 54), so it never matters to the streak; do not
  special-case it.
- [x] Tests (`load_state`, test DB): 3 consecutive `invalid` (HTTP 200) with the latest 30 min ago → absent
  from cohort and harvest; latest 61 min ago → present with `stop_after_5xx == 1`; a mix 503, invalid,
  invalid → absent; an `ok` after them resets; a `transport_error` without status keeps Task 75's rule for
  status-less failures; existing Task 75 tests still pass after the rename.
- [x] `uv run pytest -q` green. Commit `task 80: invalid answers count toward the run cooldown`. (71f4b58; 899 passed, 1 skipped; production load_state 1.05 s, unchanged.)
- [x] Monitor: deploy; after 24 h, no run has more than 3 invalid answers in any hour, and invalid/h drops. (Deployed 2026-10-17 16:32Z; rollback image `msynd-collector:pre-t80`. Baseline: ~30 invalid/h on 10-17 00–10Z, 0 since 12Z.)
  24 h check, 10-18 16:48Z: the worst run had 3 invalid answers in an hour and none had more; poll invalid answers fell to
  15 in 24 h (0–7/h); the IST 10-18 harvest batch's 162 invalid answers were capped at 3 per run.

### Task 81 (amendment 2026-10-17): ramp-gate report (user decisions 1, 9)

**Why:** the ramp gate (spec §18 "Training data") has been measured with ad hoc SQL. The user approved a
read-only report that also records recovery yield.

**Files:** `msynd/ops/report.py` (new function and subcommand), `tests/ops/test_report.py`.

- `python -m msynd.ops.report ramp --day YYYY-MM-DD` (IST date of harvest `started_at`) prints JSON:
  - `keys`: distinct runs with ≥ 1 harvest attempt that day; `not_running`: those whose harvest answer
    was HTTP 400 on a run now `cancelled`; `due` = keys − not_running;
  - `usable`: due keys with an `ok` harvest answer and an actual arrival event (`obs.event_rev`, kind `arr`)
    at the destination halt; `usable_pct`;
  - `r_pairs_pct` (consecutive halts with dep at A and arr at B present) and `w_pairs_pct`
    (intermediate halts with arr and dep present), over usable keys;
  - `by_class`: `{class: {n, usable_pct}}` from `ref.train.class`;
  - `attempts`, `challenges` (harvest fetches with outcome `challenge`);
  - `passes`: due ≥ 300, usable ≥ 80 %, both pair shares ≥ 90 %, no class with n ≥ 30 below 70 %,
    zero challenges; plus `failing` (the rules that failed).
- Read-only: one connection, no writes, no HTTP.
- [x] Tests on the test DB with a synthetic day: the 400 keys leave the denominator; the class threshold only
  applies at n ≥ 30; one challenge fails the gate; R/W pair counting on a 3-halt route; the IST day boundary.
- [x] `uv run pytest -q` green. Commit `task 81: ramp gate report`. (906519a; 906 passed, 1 skipped. Codex was
  blocked by stale ACLs on `msynd/ops` and `tests/ops` (missing the newer sandbox SIDs; `icacls /reset /T`
  needs elevation), so the monitor implemented it.)
- [x] Monitor: run it on the VM for 10-13…10-17 and record the results here. (Deployed 2026-10-17 17:00Z,
  rollback image `msynd:pre-t81`. Every day: 512 attempts, 0 challenges.)

  | IST day | keys | not running | due | usable % | R % | W % | failing |
  |---|---|---|---|---|---|---|---|
  | 10-13 | 448 | 128 | 320 | 85.3 | 98.2 | 98.4 | — |
  | 10-14 | 440 | 90 | 350 | 88.6 | 98.2 | 98.5 | — |
  | 10-15 | 490 | 195 | 295 | 89.5 | 98.9 | 99.2 | due |
  | 10-16 | 460 | 200 | 260 | 84.6 | 98.3 | 98.5 | due |
  | 10-17 | 416 | 89 | 327 | 87.8 | 98.8 | 98.9 | — |
  | 10-18 | 427 | 34 | 393 | 87.5 | 98.3 | 98.6 | — |
  | 10-19 | 1229 | 162 | 1067 | 88.2 | 99.3 | 99.5 | — |
  | 10-20 | 3004 | 415 | 2589 | 89.3 | 99.3 | 99.5 | — |

  Classes with n ≥ 30: EXP 83.7–87.2 %, SF 88.9–97.0 %. These days predate Task 79, so the ramp (decision 3a)
  waits for IST 10-18, the first batch with PASS out of the harvest.
  IST 10-18 passes (EXP 85.4 %, SF 92.5 %, UNKNOWN 86.7 %, 0 challenges), so harvest goes to 1,500 from IST 10-19: a one-off
  VM timer `msynd-ramp-1500` sets `MMT_HARVEST_BUDGET=1500` and recreates the collector at 18:30:30Z 10-18 (log
  `/data/logs/ramp-1500.log`, `.env` backup `.env.bak-20261019-ramp`). 3,700 waits for IST 10-19 to pass.
  IST 10-19 ran 1,500 attempts on 1,229 keys in 18:30–21:24Z (~520/h; polling held 176–184/h, 0 breaker) and passes
  (EXP 86.1 %, SF 89.7 %, UNKNOWN 93.6 %, 0 challenges). Timer `msynd-ramp-3700` sets 3,700 at 18:30:30Z 10-19 (log
  `/data/logs/ramp-3700.log`, backup `.env.bak-20261020-ramp`); poll stays 7,500 (total 11,200 ≤ 13,200).
  IST 10-20 at 3,700 covers 3,004 keys, the full daily scope, and passes (EXP 86.8 %, SF 90.2 %, UNKNOWN 94.4 %,
  0 challenges). **The ramp is complete.** Polling used 4,036–4,358 of its 7,500 a day on 10-18…10-20, so the planned step
  to 9,500 would change nothing and is not taken (spec §18 ramp status).

### Task 82 (amendment 2026-10-17): red selected route with a travelling pulse (user request)

**Why:** the selected route is drawn cyan, the same hue as the network and on-time trains, so it disappears.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/main.ts`, `web/src/main.test.ts`.

- `selectionLayers(display, tick)` returns the route in red (`selection`) and an additive glow pulse
  (`selection-pulse`) that runs from origin to destination every `PULSE_MS`, its tail draining into the
  destination before it restarts. Route data is cached per display, so per-frame layers re-upload nothing.
- `main.ts` keeps the selected display and rebuilds these layers each redraw; a selection redraws at ~30 Hz.
- [x] Tests: red colour; pulse timestamps rise from 0 along the route; `currentTime` at 0, half and a full
  period; the same data object across ticks; main renders both layers and advances the pulse.
- [x] `npm --prefix web test` green. Commit `task 82: red selected route with a travelling pulse`. (1985690;
  186 passed. The pulse path drops repeated halt points and rounds joints: zero-length segments splayed
  the 16 px glow into rays.)
- [x] Monitor: deploy `web/dist`; check it in the browser. (Deployed 2026-10-17 17:00Z, previous build in
  `web/dist.old`. Checked on 11401 against the production API: red route, pulse runs Pune → Supaul in 3 s.)

### Task 83 (amendment 2026-10-17): the pulse brightens once it passes the train (user request)

**Why:** the pulse should read as where the train is going; the part already travelled stays quiet.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/main.ts`.

- `selectionLayers(display, tick, simTime)` splits the pulse path at the head's distance along the route
  (interpolated over `display.times`, which matches `headAt`): dim behind (alpha 80), bright ahead (255),
  one continuous timeline, so the pulse still starts at the origin. Split data is cached by that distance.
- [x] Tests: split point equals `headAt`; the halves join with continuous rising timestamps; behind is
  dimmer; before departure everything is ahead; after arrival everything is behind.
- [x] `npm --prefix web test` green. Commit `task 83: selected-route pulse dims behind the train and
  brightens ahead`. (7df94f6; 186 passed.)
- [x] Monitor: deploy `web/dist`; check it in the browser. (Deployed 2026-10-17 17:20Z, bundle
  `index-DPwUWbkJ.js`. On 11401 the pulse is faint on Pune → Daund and bright from the train northward.)

### Task 84 (amendment 2026-10-17): grid label declutter cached per zoom bucket (navigation fix #1)

**Why:** the user asked for smoother navigation. The perf council (Claude and Codex, measured in the browser) found that crossing a
half-zoom bucket at z9 and above froze the page for 400–550 ms, twice per zoom click, because each new label was checked against
every accepted label.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`.

- `labelStations` checks a candidate only against accepted boxes that share one of its 128 px grid cells. It caches the result
  per network and bucket. The output is identical to the full check.
- [x] Test: 1,500 seeded stations at z10, z11.5 and z13 match a brute-force oracle. A second zoom in the same bucket returns the same array.
- [x] `npm --prefix web test` green. Commit `task 84: grid label declutter cached per zoom bucket`. (c8a5eab; 187 passed.)
  Zoom steps now show 60–120 ms long tasks. What remains is deck's TextLayer layout for a new label set and the head
  loop in `draw` (Task 87).

### Task 85 (amendment 2026-10-17): hover and click pick train heads on the CPU (navigation fix #2)

**Why:** hovering ran deck's GPU picking. Each move stalled on `readPixels`: 41 long tasks, 2.4 s in about 2 s of hovering. It also
set `settling`, which forced redraws at about 70 Hz while the pointer moved.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/main.ts`, `web/src/main.test.ts`.

- `nearestRun(displays, heads, project, x, y, radius = 10)` returns the run whose drawn head (or target head, if it has not been
  drawn yet) projects nearest to the pointer within `radius` px. On a tie the first one wins.
- The heads layer is not pickable and `trainLayers` drops its select and hover callbacks. The overlay drops `pickingRadius`, so deck does no picking.
- `main.ts` handles map events itself. `mousemove` is ignored while `map.isMoving()`. `mouseout` hides the tooltip. `click`
  selects whatever `nearestRun` finds through `map.project` over `drawnHeads`.
- `hoverRun` no longer wakes the redraw loop. Tooltip text and size are rewritten only when the hovered run changes, on a new
  frame and on the 1 s tick. A pointer move only repositions the tooltip.
- [x] Tests (monitor wrote these first; they fail before the change): nearest head, radius and tie order. Heads are not pickable.
  In `main.test.ts`, hover and click go through map events, the tooltip is clamped to the window, a miss, mouseout or moving map
  hides it, and hovering leaves the idle overlay asleep.
- [x] `npm --prefix web test` green. Commit `task 85: hover and click pick train heads on the CPU`. (92670a7; 188 passed.
  Codex implemented it in `.worktrees/task-85`: `web/` is owned by an old sandbox account and lacks the current sandbox SIDs.)
- [x] Monitor: re-measure hover long tasks in the browser. (2 s sweep over Mumbai/Pune, 134 moves: 0 long tasks, down from
  41 / 2.4 s; frame p50 7 ms. Hover shows 12162 and a click opens its panel.)

### Task 86 (amendment 2026-10-17): Gaussian glow for heads, trails and the pulse (user request)

**Why:** the user asked: "give the glow head (not only the pulse, the entire trains and the pulse) gaussian blur, because
currently it looks like a cropped image clip of a glow, and has boundaries, instead of one unifying blob that can extend
seamlessly outwards ... It should also reduce the computation costs". Each glow stops at its geometry with about 3 % alpha left
(`exp(-3.5)`), so dots and strokes show a rim, and the pulse front is a hard cut.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`.

- Heads (radial), comets, timetable trails and the pulse (across the width) all use one falloff, `exp(-0.5 (d/σ)²)`, with the
  geometry edge at 3.5σ, so the edge is below 1/255 of the peak. Overlapping glows add into one blob, and a brighter glow reaches
  further. Radii and widths grow by √(6.125/3.5) ≈ 1.32 so the 10 % contour stays where it is today. The white core stays.
- The pulse front fades over a Gaussian lead ahead of `currentTime` (0.1 × trailLength) instead of stopping at a hard cut.
  Trains keep the cut, which sits under the head blob, so their glow never runs ahead of a train.
- Cost: in the vertex shader, the glowing trips and the streaks collapse every path segment that lies wholly outside
  [currentTime − trailLength, currentTime + lead]. Only the visible trail is rasterized; today the whole 10 h trip window is rasterized
  and most of it discarded per fragment. Not a full-screen blur: that adds passes and would blur the map and labels too.
- [x] Tests first: the edge alpha is below 1/255; the shader injects contain the Gaussian, the lead and the segment cull, and keep the
  picking guards; radii and widths are scaled.
- [x] `npm --prefix web test` green (189). Commit `task 86: gaussian glow for heads, trails and the pulse`.
- [x] Monitor: before and after screenshots; zoom GPU frame p50 (55 ms before); deploy.
  (2026-10-18, d199a5d. Codex ran in `.worktrees/task-86` (the `web/src` ACL workaround). Review fix: the first pass dropped GLOW's
  `(1.0 - age)` tail factor, so the India view ran hot; it is restored (the plan's "linear fade" wording was wrong). Heads show no rim.
  rAF frames, zoom clicks plus a drag pan, ~2,000 trains: India p50/p90/p99 20.8/69.4/94 ms (max 97-111, was 181), Mumbai
  14.5/45/66 ms (max 63-77, was 139); a wheel-zoom sweep at Mumbai p50 24.9 ms.)

### Task 87 (amendment 2026-10-17): heads on the sim clock, not the camera clock (navigation fix #3)

**Why:** after Task 84 the CPU profile of a zoom shows the head loop in `draw` (`headAt` and `easeHead` for ~2,600 runs) at 1.76 s
self time in about 7 s of zooming. `map.on('move')` sets `settling`, so every camera frame recomputes every head, and the heads
layer's `getPosition` trigger is the frame tick, so every push re-uploads every position.

**Files:** `web/src/main.ts`, `web/src/main.test.ts`.

- Heads are recomputed on their own clock: every 100 ms, or every frame while a head is still easing. Each recompute bumps a heads
  version, which becomes the heads layer's `getPosition` trigger.
- Other pushes (the 30 Hz pulse, a zoom change, a new frame, a selection) reuse the current heads: no head loop, no position upload.
- `map.on('move')` only hides the tooltip. MapboxOverlay already redraws deck on each map render. The draw loop pushes props
  once per animation frame when the zoom changed since the last push, because trail length, width and opacity depend on zoom.
- [x] Tests first in `main.test.ts`: a move without a zoom change pushes nothing; a zoom change pushes with the same heads version;
  idle heads recompute at 10 Hz and easing heads every frame until they settle; the pulse pushes at 30 Hz with the same version.
- [x] `npm --prefix web test` green (189). Commit `task 87: heads on the sim clock`.
- [x] Monitor: frame intervals for zoom clicks and a drag pan against the Task 84 baseline (Mumbai p50/p90/p99 20.9/48.7/76.4 ms,
  India 27.7/70.1/97.3 ms).
  (2026-10-18, ~2,000 trains, two runs each: Mumbai 13.9/34.8/41.7 ms, max 42-48 (Task 86: 14.5/45/66, max 77); India
  20.7-27.3/56-62/70 ms, max 71-76 (Task 86: 20.8/69.4/94, max 111). The selection pulse still animates.)

### Task 88 (amendment 2026-10-17): deck canvas at one device pixel, no MSAA (navigation fix #4)

**Why:** zoom is GPU-bound (p50 frame 55 ms). On the user's 1.25 DPR screen deck fills 1.56 × the CSS pixels, with multisampling,
and the glows are soft anyway.

**Files:** `web/src/main.ts`, `web/src/main.test.ts`.

- The overlay keeps `interleaved: false` and adds `useDevicePixels: 1` and a WebGL context without antialiasing or a preserved
  drawing buffer (deck 9.4 `deviceProps`). The basemap keeps its own pixel ratio.
- Skipped: the council's zoom-banded trail window. Task 86's segment cull already limits rasterising to the visible trail; add
  it only if the re-measure shows trail fill still dominates.
- [x] Test first: the overlay is constructed with these props. `npm --prefix web test` green (189). Commit `task 88: deck canvas at one
  device pixel without msaa`.
- [x] Monitor: zoom frame p50 and screenshots at z5 and z10 (line softness). Done 2026-10-17 (headless DPR 1, so only the MSAA
  drop is measured): India p50/p90/p99/max 13.9/34.9/48.7/56 and 13.8/34.9/48.8/55 ms (Task 87: 20.7/62.2/69.7/76 and
  27.3/55.8/69.4/71); Mumbai 13.8/34.7/48.6/49 and 13.8/34.6/48.6/77 (Task 87: 13.9/34.8/41.7/48). Deck canvas 1400x900,
  antialias and preserveDrawingBuffer false. No visible softening at either zoom; the Gaussian glow hides the aliasing.

### Task 89 (amendment 2026-10-17): delta frames re-upload only the trails that changed (navigation fix #5)

**Why:** each generation `trailData` builds new trail objects for every display, so `dataComparator` fails and deck re-tessellates
and re-uploads all ~2,600 timetable trails, even when a delta frame changed a few runs. That long task can land mid-gesture.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`.

- `trailData` reuses a display's trail objects while the display and the hour base are unchanged (cache keyed by display).
- The timetable trails split into 16 layers by `run_id % 16`, each with its own `dataComparator`, so a delta re-uploads only the
  chunks it touches. Selection and picking behave as before.
- [x] Tests first: an unchanged display yields the identical trail object; changing one run changes exactly one timetable chunk's data.
- [x] `npm --prefix web test` green (193). Commit `task 89: delta frames re-upload only changed trails`. (main.test.ts's two
  'timetable' lookups now read 'timetable-10'; assertions unchanged.)
- [x] Monitor: the long task when a delta frame lands, before and after. Done 2026-10-17 ~19:20Z, at night: deltas are 0-3 KB
  (1,987 of 1,993 runs schedule_only), so the gain cannot show yet. Before: 0 long tasks over 45-60 s. After: 3 deltas in 3 min,
  one 80 ms long task (max frame 76), others none (max frame 40, 14); idle p50/p90/p99 7.0/14.0/23.6. 16 chunk layers did not
  regress zoom/drag: India 13.8/35.0/55.4/56 and 13.9/34.8/41.9/43, Mumbai 13.7/34.6/48.9/62 and 7.4/34.6/35.0/36. Recheck at
  daytime with large deltas.

### Task 90 (amendment 2026-10-17): selection without a colour re-upload, and a bounded run lookup (navigation fix #6)

**Why:** selecting or clearing a train changes `updateTriggers.getColor` on the comets, streaks, timetable and heads, so deck
recomputes every per-vertex colour. Opening the panel calls `/api/run/{id}`, whose prediction lookup (`msynd/api/app.py:255`)
reads `sim.frame` newest first with no bound, so the client cursor pulls every stored frame (up to 15 min of ~7 MB fulls).

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`, `msynd/api/app.py`, the `/api/run` tests.

- With a train selected, the four train layers dim through layer `opacity` (× 0.3), with constant colours. The selected train
  draws on top in its own small layers (trail and head) at full strength.
- The lookup reads only `gen >= COALESCE((SELECT max(gen) FROM sim.frame WHERE is_full), 0)`, newest first. Results are unchanged.
- [x] Tests first: no colour trigger depends on `selected`; the selected layers hold only the selected run; `/api/run` finds a run
  in a delta after the latest full and ignores frames older than it.
- [x] `npm --prefix web test` and `uv run pytest -q` green (DB tests included). Commit `task 90: selection dims by opacity; bounded
  run lookup`.
- [x] Monitor: selection long task before and after; deploy the api (rebuild `msynd:local`, restart api).
  2026-10-17: 190 web tests, 907 pytest passed (1 skipped). Selecting 12951 / 12002 / 11401: long tasks before [72,70], [], []
  with max frames 90/21/69 ms, after [65,62], [], [] with 69/14/70; clearing max frames 42/14/28 before, 28/14/15 after. The
  first select is mostly one-time panel, fetch and selection build. Other trains dim, the selected route draws on top. Deployed
  19:42Z (VM d00e085, bundle index-CPiHTK8n.js); `up -d api` also recreated `db` (compose drift since Task 78, pgdata kept)
  and restarted collector and engine. `/api/run` for five running trains: 200 in 85-180 ms with halts.

### Task 91 (amendment 2026-10-17): selecting a train fits its whole route (user pick 2a)

**Why:** the user asked that selecting a train "pan out (or in) to fit the entire screen e2e with the path" and picked a fit of
the whole route, origin to destination, capped so that short routes do not zoom in too far.

**Files:** `web/src/main.ts`, `web/src/main.test.ts`.

- `selectRun` replaces `easeTo` with `fitBounds` over the bounding box of `display.points`, padded clear of the panel and the
  HUD, `maxZoom: 10`, `linear: true`, `duration: 600`. Only a user selection (click or search) moves the camera; new frames never refit.
  A run without geometry leaves the camera where it is.
- [x] Tests first: bounds and padding for a wide and a phone viewport, the zoom cap, no refit on a new generation.
- [x] `npm --prefix web test` green. Commit `task 91: selecting a train fits its whole route`.
- [x] Monitor: select a long route (12951) and a short one; frame intervals during the fit.
  2026-10-18 (1400x900, headless): 12951 fits Mumbai Central to New Delhi between the HUD, panel and legend; frames during
  the fit p50/p90/max 13.9/41.7/55 and 13.9/41.7/56 (easeTo before: 14.0/41.7/70, 13.8/34.7/69). Short 68404 Puri to
  Khurda Road (44 km) stops at the zoom-10 cap; the fly from India level to z10 loads new basemap tiles, max 195 and 166
  (long tasks [113,81,54,66,197], [83,57,99,54,166]) against 91 and 132 for the z6 easeTo. A run with no display leaves the
  camera. Review fix: on a phone the fit ran before /api/run filled the panel, which then grew to 60vh over the route, so
  the bottom padding now reserves the full 60vh box (the band left at 390x844 is about 124 px).

### Task 92 (amendment 2026-10-17): selected route coloured by speed (user pick 3a)

**Why:** the user asked for "the colour representing how fast or slow the train is going" on the selected route and picked
fixed km/h bands from deep red through amber to white-hot, avoiding the network's cyan, with the pulse on top.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`.

- Each gap between consecutive stops gets its average speed from `dist_km` and `eventTimes`, then a band colour: under 30 km/h
  deep red, 30–60 red, 60–90 amber, 90–110 gold, 110 and over white-hot. A gap with no positive duration keeps today's red.
- The selected route line in `selectionLayers` uses these colours, cached per display. The pulse layers stay above it, unchanged.
- [x] Tests first: band edges, a zero-duration gap, the colours cached per display, no cyan-like colour, the pulse layer last.
- [x] `npm --prefix web test` green (201). Commit `task 92: selected route coloured by speed` (2a1acea).
- [x] Monitor: 12951 shows deep-red/red/amber/gold/white-hot gaps (Ratlam–Nagda red, Nagda–Kota gold); selecting 12951 at
  1400x900 p50 13.8–20.7 / p90 41.7–48.7 / max 56–83 ms over three runs (Task 91: 13.9 / 41.7 / 55–56); deployed.

### Task 93 (amendment 2026-10-18): a page that fails validation does not pause the host (user pick: breaker option a)

**Why (monitor):** on 10-17 18–20Z, 27–31 % of harvest answers were HTTP 200 JSON that `parse_status` rejects (non-monotone
event times; all runs from 10-12…15). Each one counted toward the host breaker, which paused MMT for 10 min at a time
(19:33–19:42Z, again at 19:43Z) and live polling with it: 66 polls in the 19Z hour against ~190/h before. The user chose to
exempt these answers. Data quality is unchanged: they stay `invalid` and never become labels.

**Plan change (spec §7.1 circuit breaker):** an MMT answer that is `invalid` only because `parse_status` rejected a 2xx JSON
body is left out of the host breaker window, like the 400 "not running" answer (Task 54). It is still logged `invalid` with its
body and still counts toward the per-run cooldown (Task 80). Challenges, 429, 5xx, transport errors, empty bodies, non-JSON 2xx
and other 4xx still count.

**Files:** `msynd/collect/mmt.py`, `tests/collect/test_mmt.py` (`tests/collect/test_dispatcher.py` only if needed).

- `breaker_exempt(response)` also returns true for a 2xx response that `classify('status', response)` accepts. The dispatcher
  consults it only for `invalid` outcomes, so this is exactly the parse-rejected case. Mark with `# ponytail:` that a site-wide
  MMT format change no longer pauses the host; the ramp report's usable share is where it would show.
- [x] Tests first (test DB): with the collector's `_classify_status`, 25 HTTP 200 JSON answers that fail `parse_status` keep
  the dispatcher `ok` and are logged `invalid`; 200 HTML or captcha still stops the host; 200 non-JSON, 500 and a 400 that is not
  "not running" still trip the breaker; the existing "not running" exemption tests pass.
  - Two older tests encoded the old rule and were updated: the admission state test trips the breaker with a 200 non-JSON
    body; the pilot breaker test is parametrized (rejected JSON does not pause, non-JSON still does).
- [x] `uv run pytest -q` green (926 passed, 1 skipped). Commit `task 93: pages that fail validation do not pause the host`.
- [x] Monitor: deploy the collector; breaker trips from invalid answers stop and the poll rate recovers toward ~190/h.
  (Deployed 2026-10-17 20:55Z; rollback image `msynd-collector:pre-t93`. Breaker heartbeats/h: 4, 12, 14 at 18-20Z, then 0
  in every hour 21Z-02Z 10-18; the last one was 20:54:06Z. MMT poll ok/h: 139, 65, 56, then 174-181. 14 rejected harvest
  answers at 21Z did not trip it. 0 restarts overnight; no transport errors or challenges after the deploy.)

### Task 94 (amendment 2026-10-18): hide the train panel on every screen size; the route refits (user request)

**Why:** the user asked: "need to fix phone, show an option to hide the panel across all devices to improve responsiveness (obv
when the panel hides, u pan in more)". On a 390×844 phone the expanded panel leaves about 124 px of map for the fitted route.

**Files:** `web/index.html`, `web/src/panel.ts`, `web/src/main.ts`, `web/src/style.css`, `web/src/panel.test.ts`,
`web/src/main.test.ts`.

- A hide/show button beside the close button (`aria-expanded`; labels "Hide train details" / "Show train details"). Hidden: the
  panel collapses to a fixed-height bar with the train number and date and the name on one line (ellipsis); the next stop and the
  stop list are not shown. Close still clears the selection.
- The choice stays for later selections until toggled back (page memory only).
- The bar keeps the panel's place: the top-right card on desktop, the bottom of the screen on phone (< 640 px). The map controls
  go where they are with no panel open, clear of the bar.
- Toggling refits the selected route with Task 91's `fitBounds` options. While collapsed the padding reserves the bar's own rect:
  phone bottom = innerHeight − bar.top + 16; desktop top = max(live.bottom, bar.bottom) + 16 and no right reservation. Expanded
  keeps Task 91's padding, including the phone 60vh reservation. No fit without a display.
- [x] Tests first: the button toggles the state, label and `aria-expanded`; the state survives close and a new selection; the
  collapsed padding on desktop and phone; a toggle calls `fitBounds` again with the new padding; the expanded cases are unchanged.
- [x] `npm --prefix web test` green (202) and the build passes. Commit `task 94: hide the train panel; the route refits`.
- [x] Monitor: screenshots at desktop and 390×844 in both states; deploy.
  - 2026-10-18 (12951/12952): desktop 1400×900 bar 380×89 at the top right, controls back at the right edge, route recentred
    (a tall N–S route stays height-bound). Phone 390×844: expanded panel top 326 px (map band ~124 px); collapsed bar 89 px at
    the bottom, controls just above it, the route fills the screen. Collapsed state kept across close and a new selection.
    Deployed 2026-10-17 ~20:40Z (VM b2a95d3, index-DB85gftl.js); site 401 behind auth.

### Task 95 (amendment 2026-10-18): off-site backup to Google Drive with rclone (user decision 8: "Just the google drive")

**Why:** off-site backup has been local-only since Task 77; the OCI credentials were never approved, and the image never had the
OCI CLI. The user chose Google Drive only. The off-site dump (ref, ml and sim without frame and metric; no obs) is a few MB
today; the local dumps with obs are about 33 MB.

**Plan change (spec §12 nightly backup, §18 backup):** the off-site copy goes to a Google Drive folder through rclone instead of
Oracle Object Storage. Retention (7 daily, 4 weekly), the 15 GB alarm and the `obs` exclusion are unchanged.

**Files:** `Dockerfile`, `compose.yaml`, `.env.example`, `msynd/ops/nightly.py`, `docs/deploy.md`, `tests/ops/test_retention.py`,
`tests/test_deploy.py`.

- The image installs Debian's `rclone`. `BACKUP_REMOTE` (for example `gdrive:msynd-backups`) replaces `OCI_BUCKET_NAME`; empty
  means local-only as today, and the warning names the new variable. The ops service mounts `${RCLONE_CONFIG_DIR:-/data/rclone}`
  read-write at `/root/.config/rclone` (rclone rewrites a refreshed OAuth token) and drops the `/root/.oci` mount and `OCI_*`.
- Upload with `rclone copyto <offsite> <remote>/<cadence>/<day>.dump`, list with `rclone lsjson <remote> --recursive --files-only`,
  prune with `rclone deletefile --drive-use-trash=false <remote>/<path>` so pruned copies leave the Drive quota. Names, retention,
  size sum and alarm work as before.
- `docs/deploy.md`: one-time auth on a machine with a browser, `rclone config create gdrive drive scope=drive.file` (rclone sees
  only the files it creates); copy `rclone.conf` to `/data/rclone/rclone.conf` (dir 700, file 600, root) and set `BACKUP_REMOTE`.
- [x] Tests first: the rclone command lines for a weekday and a Sunday (weekly), retention deletes the oldest beyond 7 and 4 without
  trash, the size alarm, a failure keeps two valid local copies, an empty `BACKUP_REMOTE` stays local-only with the new warning;
  compose mounts the rclone dir and not the OCI dir; the Dockerfile installs rclone.
- [x] `uv run pytest -q` green (926 passed, 1 skipped). Commit `task 95: off-site backup to Google Drive`.
- [x] Monitor: rebuild `msynd:local` on the VM; once the user has authorized Drive, a manual backup uploads and `pg_restore --list`
  reads the downloaded copy.
  (2026-10-18: `msynd:local` rebuilt with rclone v1.60.1; rollback image `msynd:pre-t95`. The user authorized Drive with
  `scope=drive.file`; the config is root-only (600) at `/data/rclone/rclone.conf`, and `.env` sets
  `BACKUP_REMOTE=gdrive:msynd-backups`. A manual backup at 03:35Z took 48 s and uploaded `daily/` and `weekly/2026-10-18.dump`
  (8.6 MB each; 10-18 is a Sunday). The downloaded copy lists 79 TOC entries, with data for 10 ref, 2 ml and 1 sim tables.)

### Task 96 (amendment 2026-10-19): stations sit on the track; no spurs, fewer chords, smoother curves (user report: "weird impossible railway tracks… whay these jagged lines? … Why can't they be smooth?")

**Why (monitor, 2026-10-19, live `network-af87af0ce6e9`):**
- 6,540 of 15,949 segments (41 %) have no polyline and draw as straight chords (median 6.4 km, p95 15.9 km); 90 % of the
  10,743 routes touch at least one. Failures in the Task 37 refresh: `length` 4,409, `no_path` 2,069, `no_anchor` 502.
- 87 % of routes have at least one hairpin (> 150° turn). Of the unique OSM-polyline hairpins, 4,916 sit at the first or last
  track vertex and only 305 inside. The polyline runs `[our station, path…, our station]` from the nearest track *vertex*, so
  when that vertex lies behind the station, or our coordinate lies off the track, the trail spikes out and back (Tharban, Tripal
  Halt). Our station coordinate is a median ~0.3 km off the track (> 1 km for 16 %), which makes the V notches at stations (Gaya).
- `length`: targets are whole km (98 %) and the km includes both station stubs, so ±10 % rejects most short hops.
- Jagged curves: Douglas–Peucker at 75 m is ~4 px at z13, and each stop pattern is simplified on its own, so overlapping
  segments fray into parallel strands.
- Not the cause: Task 88's DPR 1 and no MSAA (deck's path shader still anti-aliases its edges). Left as is; measure separately.

**Plan change (spec §7.6, drawing only):** each anchored station gets a point on the track, the foot of its matched point on
the nearest rail edge. Polylines run foot to foot, and the exported station dot moves to the foot. Prediction keeps the timetable
km and `ref.station` coordinates.

**Files:** create `msynd/db/migrations/0008_station_osm.sql`; modify `msynd/static/osm.py`, `msynd/static/export.py`,
`tests/static/test_osm.py`, `tests/static/test_export.py`, `tests/db/test_schema.py`. No web change.

- **Storage (0008):** `ref.station_osm(code text primary key, lon double precision not null, lat double precision not null)`
  with checks lon 68–98 and lat 6–37; no foreign key, like `ref.segment_osm`. `refresh` replaces it in the same transaction as
  `ref.segment_osm` (lock both, delete both, insert both).
- **Snap (`anchor_stations`):** tiers and the matched point are unchanged. The anchor is the nearest point on a raw rail edge
  within 1 km of the matched point (local equirectangular metres around the matched point), not the nearest vertex.
  - Candidates are the edges incident to rail nodes within 2 km. `# ponytail:` an edge with both ends > 2 km away is missed;
    add an edge grid if `no_anchor` shows it.
  - A foot inside an edge splits it: a new negative node id joins both ends in both directions with haversine km, and goes
    into the points and the grid, so a later station on the same edge sees the pieces. A foot at an end reuses that node.
  - Ties go to the smaller distance, then the smaller (node, node) pair, so runs are deterministic.
- **Routing:** A* between the two snapped nodes. The km is the on-track path only (no stubs). Accept when
  |km − target| ≤ max(0.1 × target, 1 km); prune at g + h > max(1.3 × target, target + 1 km). Both stations on the same node
  count as `no_path`.
- **Geometry:** the polyline is `simplify(path)`; its ends are the two feet. Douglas–Peucker at 15 m (~0.8 px at z13) as a module
  constant; keep the `# ponytail:` note (one tolerance for all zooms; a low-zoom LOD if vertex counts hurt). Rounding stays 5 dp.
- **Station points:** a station that ends at least one accepted segment gets its foot, rounded like the polyline, in
  `ref.station_osm`. The report adds `snapped` (that count) and the p50/p90 move in metres from our coordinate.
- **Export:** `stations` read `coalesce(station_osm, station)` lon/lat, so chords, Task 61 chains and the 15 km isolated-origin
  snap all use the track point, and every polyline starts exactly on its station dot.
- [x] Tests first (synthetic OPL in `tmp_path`, test DB):
  - a station beside the middle of a long edge: the polyline starts at the foot, not at a vertex or our coordinate, and has no
    turn back (replaces `test_geometry_uses_our_station_coordinates`);
  - a station whose nearest vertex lies 0.5 km behind it along the track gets no spur;
  - two stations on one edge both split it, and the segment between them routes;
  - whole-km floor: target 3 accepts 3.9 km and rejects 4.1 km; large targets keep ±10 % (rework `test_path_length_tolerance`;
    `test_length_includes_station_anchor_stubs` becomes "stubs are not counted");
  - `simplify` keeps a 30 m bump, drops a 10 m bump, and a dense straight run still collapses to its endpoints;
  - `ref.station_osm` rows are written only for stations with an accepted segment, replaced on refresh, rolled back with
    `ref.segment_osm` on a write failure, and checked by the bbox constraints;
  - export: the station list and the chord and chain endpoints use the snapped point when present, our coordinate otherwise;
  - the migration list in `test_migrations_are_idempotent` includes `0008_station_osm.sql`.
- [x] `uv run pytest -q` green (942 passed, 1 skipped). Commit `task 96: stations on the track, foot-to-foot polylines`.
- [x] Monitor, on the VM: rebuild the image (rollback `msynd:pre-t96`), migrate, OSM refresh and re-export; compare the report,
  vertices, runtime and RSS with Task 37, and the gz size. Rerun the blast counts (chord share, hairpins, spurs). Browser heap
  and fps against Tasks 88/89; screenshots at z11–12 at Tharban, Tripal Halt and Jawalamukhi Road, Sirpur Town and Gaya.
  - Budget: if the network gz passes ~10 MB or the India p50 frame time regresses > 20 %, raise the tolerance to 30 m and rerun.
  - Rollback: delete the new network file; the API serves the newest by mtime.
  - Watch for snaps onto dead-end sidings or yards (residual hairpins); excluding `service=siding|yard|spur` from snap
    candidates is the follow-up if they show.
  (2026-10-19, after the 75 m follow-up: `msynd:local` rebuilt (rollback `msynd:pre-t96`, DB dump
  `/data/backups/pre-t96-segment_osm.dump`), 0008 migrated, live `network-809975913a4b.json.gz`, 5,362,214 bytes. Stations:
  ref 7,699, name 103, coord 704, no_anchor 44; snapped 7,813, move p50 78 m, p90 1,306 m. Segments at 1.0 / 0.5 / 0.3:
  11,964 / 147 / 945; failures no_anchor 1,348, no_path 2,329, length 1,059. Vertices 168,445 (was 158,567), 33.7 s, max RSS
  595 MB. Blast, old → new: spurs > 200 m 2,513 → 597, > 1 km 453 → 76; routes with a hairpin 87 % → 60 %; reversals
  126,360 → 28,779; chord route-km 8.4 % → 5.2 %; unique OSM hairpins at an anchor 4,916 → 587, interior 305 → 217.
  Screenshots `reports/t96-{tharban,tripal,sirpur,gaya,lunsu}.png`: the Tripal Halt fan and the Lunsu spur are gone, Gaya and
  Dehri follow the track. Chords left: Bhokar–Hadgaon Road (and a parallel pair toward Himayatnagar), Sirpur Town
  MKDI–SRUR and SRUR–SKZR, Kangra Mandir–Samloti. Siding snaps not yet checked.)
- [x] Budget follow-up (monitor, 2026-10-19): India zoom p50 frame time, same network re-simplified: old 34.7 ms, 75 m 34.7,
  30 m 41.8–48.9 (+20–41 %), 15 m 55.5–56 (+60 %). The cost is vertices, not the 578 extra routes, and the 30 m fallback still
  breaks the budget, so `SIMPLIFY_M` goes back to 75 m. The snap fixes stay. The `# ponytail:` note gains the measurement:
  smoother curves need a zoom LOD (a fine polyline only past ~z10). Tests: `simplify` keeps a 100 m bump and drops a 50 m bump.
  Commit `task 96: simplify at 75 m (frame budget)`; rerun refresh and export on the VM.
  (2026-10-19: 01e2d05, 114 targeted tests with the DB. Full suite at a43fcd3: 940 passed, 1 skipped, 2 failed, both
  timing flakes that pass on rerun (`tests/api` and `test_relay.py` in full: 48 passed): `test_health_metrics_and_search`
  saw `last_frame.age` −0.54 s, though both ends read the test DB's `clock_timestamp()`, so its clock stepped back; and
  relay `absolute` under the 11-minute run. Only `osm.py` changed since the 942-pass run. India p50 34.7 ms, as before.)

---

### Task 97 (amendment 2026-10-19): a selected train outshines the rest (user request)

**Why:** the user asked, with a screenshot of 17631 selected at India zoom: "when i select a train, i need all other glows (of
other trains) to dim down, and increase the glow of the train selected". Since Task 90 the other trains dim to × 0.3, but ~2,500
additive trails still sum to near white in dense corridors, and the selected trail and head copy their source layer's
zoom-scaled width and opacity (a timetable trail at z4: width × 0.5, opacity 0.35), so the selected train is no brighter than
any other.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`.

- With a train selected, the comets, streaks, timetable chunks and heads dim through layer `opacity` × `DIM` = 0.1 (was 0.3);
  colours, data and triggers unchanged (Task 90).
- `selected-trail` draws at opacity 1 and `widthScale` `SELECTED_GLOW` = 1.6 at every zoom; `selected-head` at opacity 1 and
  `radiusScale` 1.6. Everything else is cloned from the source layer as now. Both are module constants.
- [x] Tests first: rework "selection changes only opacity" to × `DIM`; the selected trail (comet and timetable) and head have
  opacity 1 and scale 1.6 at zoom 4, 6 and 8, other props as the source layer.
- [x] `npm --prefix web test` green (202). Commit `task 97: a selected train outshines the rest`.
- [x] Monitor: screenshots of a selected train at India zoom and z8 before and after; frame times with a selection; deploy
  web/dist (keep `web/dist.old`, restart caddy).
  (2026-10-19, after all three commits: `reports/t97-{before,after}-{india,z8}.png` with 17631 selected; before, the network
  and the other trails form a bright web, after, 17631 stands clear over a faint one. rAF ms with 17631 selected, India zoom
  4.42 ↔ 6, two interleaved runs each, before (01e2d05) p50 48.7 / 48.7, p90 62.5 / 55.7, p99 76.5 / 69.4; after (a43fcd3)
  48.6 / 48.7, 56.2 / 55.7, 70.2 / 62.9, so no cost. Deployed `index-ISQSpKxx.js`; `web/dist.old` is the 10-17 build; caddy
  healthy, 401 unauthenticated. VM repo at a43fcd3; no image rebuild (web only).)
- [x] Follow-up (monitor, 2026-10-19): after 91fd712 the India view still looked bright. Isolating layers showed the trains do
  dim; the cyan web is the static `network` layer (15,895 paths, alpha 36), which ignores the selection. With a train selected
  the network also draws at opacity × `DIM`. Its data and the cached per-network instance are unchanged (a dimmed clone, no
  attribute rebuild). Tests first in `web/src/layers.test.ts` (and `main.test.ts` if the wiring is tested there). Commit
  `task 97: the network dims with the other trains`. (f86d477, 205 web tests.)
- [x] DIM 0.01 (monitor, 2026-10-19): after f86d477 the India view was still a bright web. deck.gl gamma-corrects layer
  `opacity` (the shader gets opacity^(1/2.2); measured, one network path at alpha 36 writes 36 at opacity 1 and 13 at 0.1), so
  DIM 0.1 kept ~35 % alpha and overlapping additive trails still summed to white. Screenshots at 0.1, 0.03 and 0.01 (India and
  z8): 0.01 (~12 % alpha) lets the selected train stand clear while the rest stays a faint context web. DIM = 0.01 with a
  one-line comment on the gamma; tests follow the constant. Commit `task 97: dim to 0.01 (deck.gl opacity gamma)`.
  (a43fcd3, 205 web tests, build passes.)

---

### Task 98 (amendment 2026-10-19): fine geometry past z10 (zoom LOD, user request)

**Why:** the user asked for smoother curves ("why these jagged lines?"). Task 96 measured that a whole-network 15 m polyline
costs +60 % India-zoom frame time, and the monitor's high-zoom A/B (z11 ↔ z12.5 at Delhi, Mumbai and Kolkata, rAF p50 75 m
41.6–48.4 ms, 15 m 62.5–69.4 ms) shows the cost is total trail vertices whatever the viewport. So swapping the whole network
at z10 would regress too; the fine geometry has to be culled to the area in view.

**Files:** `msynd/static/osm.py`, `tests/static/test_osm.py`, `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/main.ts`,
`web/src/main.test.ts`.

- Server: `SIMPLIFY_M = 15`. The `# ponytail:` note says the web derives the 75 m overview and swaps in this geometry past z10
  (Task 98). Test: `simplify` keeps a 20 m bump and drops a 10 m bump.
- Web, `layers.ts`: `simplify(points, metres)` is a Douglas–Peucker that mirrors `osm.simplify` (local equirectangular about the
  first point, iterative, endpoints kept). `overview(network)` returns the network with each polyline simplified at
  `OVERVIEW_M` = 75; segments without a polyline and all other fields are unchanged.
- Web, `main.ts`: the fetched network is `fine`; `network` is `overview(fine)` and serves the panel, hover and the India view as
  now. Detail mode starts at zoom ≥ `DETAIL_ZOOM` = 10 and ends below 9.5.
  - In detail mode, `region` is the viewport bounds grown by one viewport on each side (3 × 3). When the viewport leaves it, the
    region is recomputed and the displays rebuilt; leaving detail mode drops the region and rebuilds.
  - On a rebuild, every run is decoded against `network` as now. A run whose coarse path has a consecutive-point bbox meeting
    the region is decoded against `fine` instead (kept coarse if that fails). The decode cache is per network.
  - Trails are built from the region's runs plus the selected run; off-region trails are culled (they cannot be in view).
    Heads, the count, picking, hover and `fitSelection` keep all displays.
  - The background is `networkLayer(fine)` in detail mode, `networkLayer(network)` otherwise; the selection highlight uses the
    current display.
  - `trainLayers` skips `selected-trail` when the selected run has no trail (today `find(...)!`).
  - `# ponytail:` on the region: a whole-route bbox test, a spatial index if rebuilds show in profiles.
- [x] Tests first: `simplify` (the Python fixtures at 75 m; a straight line keeps its ends); `overview`; `trainLayers` with a
  selected run that has no trail; in `main.test.ts` a curved polyline draws simplified at z4, fine at z11 (network and trail),
  still fine at z9.7, coarse at z9.4; a run outside the region keeps its head but has no trail; leaving the region rebuilds.
- [x] `uv run pytest -q` (osm tests) and `npm --prefix web test` green. Commit `task 98: fine geometry past z10`.
- [x] Follow-up (monitor, 2026-10-19): the first build's z9 → z11 crossing peaked at 645 / 382 / 389 ms (Delhi / Mumbai /
  Kolkata; old code 90–187). Timers in `draw` put 65–220 ms in the region test (every point pair of every run, on every
  rebuild) and ~120 ms in re-tessellating the shared `network` id on each background swap. Fixes: a per-display bbox in a
  WeakMap (a miss skips the pair test); both backgrounds mounted with stable ids `network` / `network-fine`, toggled by
  `visible`; entering detail mode and recomputing the region only while `!map.isMoving()` (leaving below 9.5 stays immediate).
  (210 web tests, build passes; the T98 commit includes it.)
- [x] Monitor: deploy web/dist first (it works with the 75 m file), then rebuild the image (rollback `msynd:pre-t98`) and rerun
  the OSM refresh and export on the VM (rollback: delete the new network file). Measure gz size, heap, India-zoom frames
  (budget: within 10 % of 34.7 ms p50), z11 ↔ z12.5 frames against the A/B above, the hitch at the z10 crossing, and
  screenshots at z11–13 at Tharban, Gaya and Tripal Halt.
  (2026-10-19: 9197b72; deployed with Task 99 in one image and one refresh, see there; web `index-Blk5cjAR.js`, `web/dist.old`
  the Task 97 build. The 15 m file is 6,754,645 bytes gz (75 m was 5,362,214; it also carries Task 99's 1,949 new segments).
  India rAF p50 / p90 / p99: 15 m file 34.6 / 35.1 / 55.6, live 34.7 / 41.7 / 62.7, within budget. Crossing z9 → z11 worst
  frame 375 / 229 / 194 ms (15 m file), live 381 / 291 / 244, a one-time pause when the zoom settles (fine decode of the
  regional runs); zoom-out 97–132 ms. Heap after GC: India 284 MB, after detail regions 352–366 MB. z11 ↔ z12.5, same
  machine and minute (rAF on a 144 Hz display, 6.9 ms steps): 75 m everywhere p50 27.8–28, the LOD 42.1–48.5, the LOD with
  the coarse background swapped in 34.2–34.7, so the whole-India fine background costs 1–2 vsyncs: Task 100. Screenshots
  `reports/t99-after-{bhokar,tripal,gaya,gayajn}.png` (Tharban is in the Bhokar view): smooth curves, no 75 m corners.
  Selection (Task 97) still dims the fine background: `reports/t98-select-{overview,z8,z11}.png` with 12301.)

---

### Task 99 (amendment 2026-10-19): link parallel tracks, keep stations off sidings, accept short direct paths

**Why:** after Task 96 the user asked to fix what was left: straight chords (Bhokar–Hadgaon Road, Sirpur Town, Kangra
Mandir–Samloti), ~2.2k hairpins where segments join at stations, and station snaps onto sidings. The monitor reproduced the
VM refresh locally (Oct 12 OPL, live stations and targets) and traced three causes:
- Double lines mapped as two ways with no crossover in OSM: each station snaps to a different track and the path detours to
  a far junction or fails (SKZR–SRUR: shortest path 94.8 km for 18 km).
- ~1.7k stations (~21 %) snap onto `service=siding|yard|spur|crossover` edges. A dead-end siding makes an in-and-out spur.
- Timetable km overstates track km on short hops: 630 of 1,064 length failures have a path shorter than the target, most
  of them direct (BOKR–HDGR target 22.5 km, track 18.0, straight 14.5).

Measured on that copy (today's rules → all three): ok 13,055 → 14,362, plus 645 short direct; no_path 2,353 → 1,103;
length 1,064 → 362; artefact station-join hairpins 2,162 → 1,334; join excursions > 300 m 1,704 → 950; chord route-km
7.8 % → 2.9 %; runtime 35 → 70 s. Linking at 8 / 15 / 25 m gives 157k / 315k / 575k links and 1,415 / 1,334 / 1,214
hairpins; 15 m takes most of the gain. Siding preference alone moves 1,690 feet, nearly all < 50 m, and loses 30 segments
(WL moves onto an unconnected through track); with linking it loses 4 and wins 20. Most excursions left over 1 km are real
reversals at terminals (VSKP, HWH).

**Files:** `msynd/static/osm.py`, `tests/static/test_osm.py`.

- `parse_opl` also returns the set of raw edges from ways tagged `service=*`, as sorted node pairs.
- `anchor_stations` prefers a non-service edge within 1 km and falls back to a service edge only when none is in range.
  Splitting a service edge keeps both halves service.
- `link_parallel(points, graph)` runs after anchoring and before contraction. It joins two nodes within `LINK_M` = 15 m whose
  local bearings differ by ≤ 20°, unless the track already joins them within max(3 × their distance, 50 m). The link weight is
  the straight distance. Use a grid with cells of about `LINK_M` (not the 1 km `grid_index`). The report gains `links`.
- `refresh` also accepts a path shorter than the target when it is ≤ 1.25 × the straight distance between the feet. Longer
  paths and winding short ones stay length failures.
- `# ponytail:` on `link_parallel`: 315k links double refresh runtime; thin to one link per node if that matters.
- [x] Tests first: `parse_opl` returns service edges. A station 20 m from a siding and 60 m from the main line snaps to the
  main line; with only a siding in range it snaps to the siding. Two parallel ways 6 m apart get links, a way crossing at
  90° gets none, and consecutive nodes of one way get none. With the DB, stations on different tracks of an unconnected
  double line get a polyline. A short direct path is accepted; a short winding one is still a length failure.
- [x] `uv run pytest -q` green. Commit `task 99: link parallel tracks, keep stations off sidings, accept short direct paths`.
  (2026-10-19: 4867a90 on top of 9197b72; full suite 956 passed, 1 skipped; `tests/static` 128 on impl/v1. The local copy
  gives 13,469 / 180 / 1,358 segments, exactly the prototype's 15,007, no_path 1,105, length 362, 314,979 links, 70 s.)
- [x] Monitor: with Task 98, rebuild the image (rollback `msynd:pre-t98`), rerun the OSM refresh and export on the VM
  (rollback: delete the new network file). Compare the report with the numbers above. Rerun the blast counts. Screenshots at
  Bhokar–Hadgaon Road, Sirpur Town, Kangra Mandir, Tharban and Gaya.
  (2026-10-19: `msynd:local` 815c215ae72d, DB dump `/data/backups/pre-t99-segment_osm.dump`, live
  `network-9ef62a1ed75d.json.gz`, 6,754,645 bytes (previous `809975913a4b`). Stations ref 7,699, name 103, coord 704,
  no_anchor 44; snapped 8,388, move p50 76 m, p90 1,310 m. Segments 1.0 / 0.5 / 0.3: 13,467 / 180 / 1,358 (15,005, was
  13,056); failures no_anchor 1,384, no_path 1,109, length 362. Vertices 364,905 (15 m), links 314,979, 79.5 s, max RSS
  863 MB. Blast on the live files, old → new: artefact station-join hairpins 2,163 → 1,342; join excursions > 100 m
  2,153 → 1,329, > 300 m 1,705 → 959, > 1 km 339 → 319 (largest PRR–TATA–CKP 6.36 km, a terminal reversal); chord
  route-km 6.8 % → 2.9 % (local copy). Screenshots `reports/t99-{before,after}-{bhokar,sirpur,gaya,gayajn,tripal,kangra}.png`:
  Bhokar–Hadgaon Road (via Tharban), SRUR–SKZR and Kangra Mandir–Samloti follow the track; Dehri, Son Nagar and Gaya Jn are
  smooth single lines. MKDI–SRUR stays a chord: its timetable km (11.2) is shorter than the straight line (17.3 km), so a
  station coordinate or the timetable is wrong.)

---

### Task 100 (amendment 2026-10-19): fine background only in the region

**Why:** Task 98's high-zoom check (z11 ↔ z12.5, same machine and minute) gave p50 42.1–48.5 ms with the LOD against 27.8–28
with 75 m everywhere. Swapping the coarse background in brings it to 34.2–34.7. In detail mode the background draws every
fine path in India (~370k vertices), though only the region (3 × 3 viewports) can be in view. The step that remains is the
fine regional trails, the smoothness itself.

**Files:** `web/src/layers.ts`, `web/src/layers.test.ts`, `web/src/main.ts`, `web/src/main.test.ts`.

- `networkLayer(network, selected, region?)`: with a region, the data is only the paths whose bbox meets the region. Each
  path's bbox is computed once per network. Layers are cached per network and region (nested WeakMaps), so one region keeps
  one data reference across draws, with the full and dimmed variants as now. The id is `network-fine`. Without a region,
  behaviour is unchanged.
- `main.ts`: in detail mode the background is `networkLayer(fine, selected, region)`, and the overview stays mounted and
  hidden as now. Outside detail mode there is no `network-fine` layer. A new region re-tessellates only its subset.
- `# ponytail:` on the filter: a per-path bbox test on each region change; a spatial index if it shows in profiles.
- [x] Tests first: with a region, a fine path outside it is not in the background's data and one crossing it is. The data is
  the same reference across draws in one region and changes with the region. A selection dims the regional layer. At z4
  there is no `network-fine` layer (adjust Task 98's "both layers present" test to this).
- [x] `npm --prefix web test` green (212), build passes. Commit `task 100: fine background only in the region` (e15201f).
- [x] Monitor: z11 ↔ z12.5 frames (target: p50 at the coarse-background figure, ~35 ms), the crossing hitch, India frames,
  a z12 screenshot, then deploy web/dist (keep `web/dist.old`, restart caddy).

**Result (2026-10-19, 15 m file, 144 Hz display):** z11 ↔ z12.5 p50 27.7–27.9 ms (was 42.1–48.5), p99 90–118 (was
118–138), past the target: the regional background is cheaper than the India-wide coarse one (34.6–41.6 in the same run).
India p50/p90 34.7/41.7–41.8, unchanged. Crossing worst frame over two runs: Delhi 410–472 (first entry, includes the
one-time per-path bbox pass), Mumbai 250–313, Kolkata 271–299 (was 244–381). Screenshots `reports/t100-{sirpur,gayajn,
kangra}.png`: background fills each new region. Deployed: bundle index-DVnxGxbf.js, `web/dist.old` = Tasks 98–99 build
(index-Blk5cjAR.js); VM repo e15201f3.

---

### Task 101 (amendment 2026-10-21): drop the unused extras

**Why:** nothing imports `kaggle`, `oci-cli` or `networkx`. Kaggle serves only Phase 6, closed under the §18 no-export
rule; off-site backups use rclone (Task 95); the OSM routing uses stdlib `heapq` (Task 37 amendment). Yet the image installs
the `ops` extra, so every container carries both CLIs.

**Files:** `pyproject.toml`, `uv.lock`, `Dockerfile`, `docs/deploy.md`.

- Remove the `ops` and `import` extras. `uv lock` drops only their packages and transitive deps nothing else needs; no
  other version moves.
- Dockerfile: `uv sync --frozen --no-dev --no-install-project`. The compose `ops` profile is unrelated and stays.
- deploy.md: the image line no longer mentions ops extras.
- Re-add `kaggle` with Phase 6 if the no-export rule is ever lifted.
- [x] `uv lock --check` and `uv run pytest -q` green. Commit `task 101: drop unused extras`. (bdb7f71; 956 passed,
  1 skipped. `uv.lock` lost 54 packages and nothing else; Codex reported BLOCKED only because its sandbox cannot reach the
  test Postgres, so the monitor ran the full suite.)
- [x] Monitor: throwaway arm64 build on the VM (`msynd:t101-test`, since removed): 1.31 GB vs 1.86 GB for `msynd:local`;
  all 49 `msynd` modules import with no network; `pg_dump` 17.11 and rclone present; kaggle, oci, networkx, requests and
  yaml absent. The running services keep their image until the next deploy that needs a rebuild.

### Task 102 (amendment 2026-10-21): start collector-data training today (user request)

**Why:** the first weekly fit that clears the §8.6 support gate is the Mon 11-02 cron, but a read-only count on 10-21 found
7,984 eligible matured runs over 13 service dates. Fit one inactive shadow candidate on everything eligible now and freeze it;
later collected histories score it (user's pick: replay future histories, no live forecast recorder). Support thresholds,
the weekly schedule and the promotion gates stay as they are. Spec §8.6 "Early shadow candidate", §18.

**Files:** `msynd/labels.py`, `msynd/features/build.py`, `msynd/models/registry.py`, `msynd/ops/shadow.py` (new),
`tests/test_labels.py`, `tests/features/test_features.py`, `tests/models/test_registry.py` (new),
`tests/ops/test_shadow.py` (new).

**1. Practical extraction, same meaning**
- `build_live_labels`: replace the correlated destination-completion, `resched_min` and `exception_unparsed` checks with
  set-based joins (or CTEs). Preserve the receipt cutoffs, six-hour maturity, `event_asof` revisions, destination-arrival
  completion, the `expired_unknown` `updated_at` rule, `resched_min` received before the departure receipt, and `(station,
  occ)` identity for repeated stations. Output rows and order are unchanged.
- Keyword-only `run_ids=None` on `build_live_labels` and `training_frame`: `None` selects every eligible run; an empty
  collection selects none (and issues no per-run queries).
- `Fold` accepts empty `valid_weeks` (training-only). `train_weeks` stays nonempty, distinct Mondays; the ordering check
  applies only when validation weeks exist.
- `training_frame`: one `segment_queries` read and one batched `with_delay` per distinct `(run_id, departure received_at)`.
  Different receipt times of one run stay separate reads. Output order, week cross-fitted training encodings and the
  validation (serving) encoders are unchanged.

**2. `python -m msynd.ops.shadow`** (`MSYND_LIVE` is never read or set)
- `preflight --models-dir DIR`: eligible support at one receipt boundary (labels matured 6 h before it), reported with the
  weekly helpers (`MIN_RUNS`, `MIN_SERVICE_DATES`, `FLOORS`, `_support`, `_role_support`; import them, do not copy).
  Timed phases: the complete label extraction and a deterministic 100-run feature probe (eligible run ids sorted, evenly
  spaced) must each finish in ≤ 60 s. Collection provenance per eligible run, from the `obs.fetch.mode` of its event
  revisions: `poll_only`, `harvest_only`, `mixed` (≥ 2 known modes), `unknown` (no linked fetch, or only probe/backfill).
  Prints one JSON report and exits nonzero on any failed threshold or timing. No `ml.model` row, file or directory is written.
- `fit --models-dir DIR`: holds one session advisory lock (`pg_try_advisory_lock`; a held lock exits nonzero). Opens a
  repeatable-read, read-only extraction transaction, captures the receipt boundary once, runs the preflight checks in it
  (stops on failure) and reuses that label extraction. Training-only fold over every training week, partial weeks included.
  The same snapshot reads the active baselines and the replay parameters. The transaction ends before CPU training.
  Fits R with the existing `train_lgbm` (single thread, nine quantiles, encoders saved) and W/O with `fit_empirical`; a role
  whose classes are all below floor is skipped and reported, as in the weekly job.
- `score --model-id ID [--until YYYY-MM-DD]`: see 4.

**3. Freeze and register**
- New directory `DIR/shadow-<uuid4 hex>/`: candidate artifacts plus copies of all three baselines (the active artifact's file
  bytes, or a saved `provisional_models()` artifact for a role with no active row). Reload each, check serving parity through
  the engine's `_predict` on probe queries (exact equality with the in-memory model), and record sha256 hashes, all before
  registration. `registry` gains one loader with the kind/role checks; `active()` and shadow both use it.
- One short transaction inserts the candidates with `active=false`. Each candidate row's `metrics` = its role support plus
  `shadow`: candidate ids/paths/hashes/kinds per role, baseline source ids (null for provisional), paths and hashes,
  `FEATURE_VER`, `MODEL_FEATURES`, replay parameters (`n`, `k`, `v_max`, `ess_min`, S tables, `params_hash`, `bias_gen` 0),
  support and floors, skipped roles, data cutoff, matured-before, the actual freeze time (after verification), the last
  training start date, the validation window and the phase timings.
- Success is printed only after commit. A failure before commit removes the new directory. If commit itself fails, check on a
  fresh connection; remove the files only when that confirms no row references them, otherwise keep them and say so.
- The provisional priors are code (`_prior`), so the image that fitted the candidate is kept for scoring (monitor).

**4. Future validation (`score`)**
- Window: 14 IST service dates starting the day after the freeze date (a 10-21 freeze scores 10-22…11-04), stored in the manifest.
- Loads the exact candidate and copied baseline files; rejects a model id not in its manifest, a missing role, a hash
  mismatch, or a `FEATURE_VER`/feature-column mismatch. The current active rows are never read.
- `--until` is exclusive and defaults to the earlier of today (IST) and the window end; later explicit dates are rejected.
  Issue times and service dates both lie in `[window start, until)`, so no case touches a training run.
- Frozen replay parameters, zero leg bias. Candidate, frozen baseline, B1 and B0 on identical eligible cases. B2 (fresh
  provider ETA, `b2_eta`'s 30 min rule) is compared on its own subset, reusing the primary records, so a missing ETA never
  drops a primary case.
- Reports CRPS, P50 error, coverage, width, PIT, breakdowns by horizon, class and zone, and support (cases, runs, dates).
  Block-bootstrap CIs of the paired CRPS gain only with ≥ 7 represented service dates.
- Status `awaiting_data` with no scored case; `partial` until `--until` is the window end and every window run is terminal
  with `updated_at` ≥ 6 h old; else `complete`. Never refits, tunes or promotes.
- Held-out screening only: §8.6 promotion and the prospective A3/A4 acceptance are separate.

**Tests first** (see each fail first):
- [x] Labels: eligibility (unfinished, immature, unparsed exception, expiry receipt, rescheduling, repeated station) and
  equality with the pre-change SQL (kept in the test as an oracle) for current and frozen boundaries; `run_ids` None/empty/subset.
- [x] Features: training-only fold accepted, empty train weeks rejected; one read per identical `(run, receipt)`; different
  receipts of one run stay isolated; output and encoders equal the ungrouped path.
- [x] Shadow: thresholds unchanged (same constants objects), below-floor roles skipped, preflight writes no row or file and
  fails on a breached threshold or timing; advisory lock refuses a second job.
- [x] Freeze: artifact and encoder round trips, baseline copies match the source (hash and predictions), provisional copy
  for a missing role, rows inactive with the manifest, active ids unchanged, failure before commit leaves no row or file.
- [x] Score: future-only cases, frozen baselines survive a later promotion, stale ETA excluded only from the B2 subset,
  empty window `awaiting_data`, partial marking, CI only at ≥ 7 dates, tampered hash / foreign id / feature mismatch rejected.
- [x] Focused tests, `uv run pytest -q` and `npm --prefix web test` green. Commit `task 102: start collector-data training today`.
- [x] Monitor: pinned image from the task commit, `preflight` on the VM, then `fit` (`MSYND_LIVE=0`); record phase timings,
  candidate ids and the image tag; check the active `ml.model` ids are unchanged; keep the image until scoring is done.

**Results (2026-10-21):** commit ba77e91. Review dropped Codex's evidence-loader switch (b3_active already overwrites the
loaded versions) and a calendar stub (parity uses the lock session); the full suite has 998 passed and 1 skipped, and 212 web tests pass.
VM image `msynd:t102-ba77e91` (01620019949a, kept), run by `/opt/msynd/deploy-backups/shadow.sh`. Preflight `ready`:
8,007 runs, 2,919 trains, 13 dates, 348,372 rows; provenance harvest-only 7,451, poll-only 556; labels 11.1 s, probe 1.27 s.
Fit at 09:32–09:34Z: cutoff 09:32:13Z; labels 10.8 s, probe 1.25 s, extraction 55.0 s, training 23.7 s, freeze 0.66 s.
Candidates **3** (run, lgbm_quantile), **4** (dwell), **5** (origin), all inactive, with no role skipped, in
`/data/models/shadow-d6b51c03a2a34cc388b7117f372fb6a1/`; baselines are copies of 2 (run), 1 (origin) and the provisional dwell. Train
window `[10-08,10-21)`, validation `[10-22,11-05)`, N 10,000 / K 32. Active ids are still 1 and 2. `score --model-id 3`
returns `awaiting_data` with matching hashes; a future `--until` is rejected.

### Monitor status (2026-10-21)

- Spec synced to the code: §7.6 and §11 (b2e64e7); §8.4–8.6, §13 and §15 Phases 3 and 6 (14f6a11); §11 Timeline,
  Train panel, Status bar and Performance re-audited and rewritten to what ships (this commit).
- **Training timeline confirmed** (read-only label count, 2026-10-19 ~13:55Z): 3,328 matured runs with start_date < 10-19
  over 11 dates (10-08…10-18). The Mon 10-26 evaluate would train on start_date < 10-12 only: 1,004 runs over 4 dates, so
  it reports `insufficient`. The first weekly candidate fit is the **Mon 2026-11-02 02:00 IST** cron. Backfilling
  10-05…10-07 is the only way to pull it to 10-26 (MMT lookback for those dates unverified; the user's call).
- **Ramp complete:** harvest 3,700/day live since 2026-10-19 18:30:40Z (timer `msynd-ramp-3700`). The IST 10-20 gate
  passes on 3,004 keys, the full daily scope (Task 81 table). Polling stays at 7,500: it uses ~4.3k a day, so the 9,500
  step is not taken (spec §18).
- Unused `ops`/`import` extras removed in Task 101; the image shrinks by ~0.55 GB at its next rebuild.
- **Early shadow candidate** (Task 102): ids 3/4/5 frozen 2026-10-21 09:33Z on 8,007 runs, all inactive. Score after the
  window: `shadow.sh score --model-id 3` (partial runs allowed earlier). The 11-02 weekly fit and the §8.6 gates are unchanged.

## Spec coverage check

| Spec section | Task(s) |
|---|---|
| §1 G1–G6, A1–A8 | 25 (A1, A2, A5, A8), 31 and 33 (A6), 36 (A7), 40 (A3, A4) |
| §2 Source facts, §15 G1–G8 | 6 (G1), 11 (G2), 7 and 9 (G3), 30 (G4; n/a to RSTGCN, 42), CP-3 (G5), 39 (G6), 18 (G7), 19 and 33 (G8) |
| §3.1 Identity | 2, 4, 5, 16, 20 |
| §3.2 Clock | 12, 17 |
| §3.3 Availability | 2, 16, 20 (A3) |
| §3.4 Format | 14, 28 (A1, A2) |
| §4.1–4.2 Conditioning, N | 17, 18, 29 |
| §4.3 Bias | 38 |
| §4.4 Coupling | 40 (C1 hook); C2 deferred |
| §5 Architecture | 24 |
| §6 Data model | 2 |
| §7.1 Dispatcher | 3, 8, 75, 80, 93 |
| §7.2 Admission | 22, 32, 79 |
| §7.3 Lifecycle | 20, 21 |
| §7.4 Parsing | 8, 10 |
| §7.5 Backfill | none approved; RSTGCN one-off import 42 |
| §7.6 Static | 4, 5, 37, 61, 63, 96, 99 |
| §8 Learning | 26–30, 38, 39, 41–43, 50, 102 |
| §9 Engine | 13–17, 19, 27, 31, 32 |
| §10 API | 34 |
| §11 Dashboard | 35, 36, 56–58, 60, 62, 64, 65, 67–71, 74, 82–92, 94, 97, 98, 100 |
| §12 Nightly | 23, 38, 77, 95 |
| §13 Ops | 23, 24, CP-3, 77, 78, 95 |
| §18 Training data | 50, 79–81 |
| §14 Testing | Each row is pinned in its owning task: Clock (12), Identity (16, 20), Leakage (16, 26), Late data (16, 31), Dirty set (31), Determinism (17, 31), G7 (18), As-of (2), NTES baseline (20), Rescheduling (12, 18), Frame latency (31), Format (14), Dispatcher (3), Validation (10), WS (34), pos parity (15, 35), Restore (CP-3), Replay day (33), Model gate (29, 38) |

**Skipped on purpose:** the separate Phase 4 "debug map", because the Task 36 dashboard covers it. Add one only if the dashboard lags behind engine work.
