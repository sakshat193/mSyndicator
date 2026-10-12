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

- Python 3.12, with dependencies locked by `uv`. Runtime dependencies: `httpx selectolax zstandard psycopg[binary,pool] numpy numba lightgbm onnxruntime pyarrow fastapi uvicorn websockets msgpack`. Ops extras: `kaggle oci-cli`. Import extras: `networkx` plus the `osmium-tool` binary. Dev: `pytest`. **No new runtime dependency without a plan change.**
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

Akamai rejects MMT requests from the VM's IP, while the user's home connection worked through the pilot at the same rate. The collector therefore runs on the VM, and its only source connection goes through the user's PC: a reverse SSH tunnel to an allowlisted CONNECT relay. The dispatcher on the VM still owns spacing, budgets and logging. The relay never forwards to any host except MMT. There is no fingerprint spoofing, no challenge solving and no third-party proxy.

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

---

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
| §7.1 Dispatcher | 3, 8 |
| §7.2 Admission | 22, 32 |
| §7.3 Lifecycle | 20, 21 |
| §7.4 Parsing | 8, 10 |
| §7.5 Backfill | none approved; RSTGCN one-off import 42 |
| §7.6 Static | 4, 5, 37 |
| §8 Learning | 26–30, 38, 39, 41–43 |
| §9 Engine | 13–17, 19, 27, 31, 32 |
| §10 API | 34 |
| §11 Dashboard | 35, 36 |
| §12 Nightly | 23, 38 |
| §13 Ops | 23, 24, CP-3 |
| §14 Testing | Each row is pinned in its owning task: Clock (12), Identity (16, 20), Leakage (16, 26), Late data (16, 31), Dirty set (31), Determinism (17, 31), G7 (18), As-of (2), NTES baseline (20), Rescheduling (12, 18), Frame latency (31), Format (14), Dispatcher (3), Validation (10), WS (34), pos parity (15, 35), Restore (CP-3), Replay day (33), Model gate (29, 38) |

**Skipped on purpose:** the separate Phase 4 "debug map", because the Task 36 dashboard covers it. Add one only if the dashboard lags behind engine work.
