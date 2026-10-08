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
Never edit the spec or the plan. Never send HTTP to enquiry.indianrail.gov.in or etrain.info.
If a step needs a decision the plan does not make, stop and start your final message with "BLOCKED:".
```

**🛑 Checkpoints** are steps that need the user (accounts, legal or data-use decisions, starting live traffic, hardware measurements) or wall-clock time. At each one the monitor stops and asks the user. Codex never runs a 🛑 step or anything that needs `MSYND_LIVE=1`.

---

## Global Constraints

These are copied from the spec; every task implicitly includes them.

- Python 3.12, with dependencies locked by `uv`. Runtime dependencies: `httpx selectolax zstandard psycopg[binary,pool] numpy numba lightgbm onnxruntime pyarrow fastapi uvicorn websockets msgpack`. Ops extras: `kaggle oci-cli`. Import extras: `networkx` plus the `osmium-tool` binary. Dev: `pytest`. **No new runtime dependency without a plan change.**
- Postgres 16. Timestamps are UTC `timestamptz` and are displayed in IST. Offsets are run-clock minutes (float), counted from 00:00 IST on the origin start date. Migrations are numbered SQL files applied by `msynd/db/migrate.py`.
- Deliberately absent: PostGIS, Redis, Kafka, persisted particle state, Rust.
- HTTP: one Dispatcher per host. Request **starts** are spaced ≥ 1/rate apart, with no burst and one request in flight. Tokens, session pages and retries all count. `NTES_RATE=1.0`, `ETRAIN_RATE=0.2`, pilot ≤ 0.3 req/s. Changing a rate is the user's decision. The user agent is `m-syndicator/<ver> (personal research)`.
- `V_MAX = 160` km/h. Every prediction distance is timetable km. The physical floor is `60 × km / V_MAX` minutes.
- Event keys are `(run_id, station, occ, kind)`, never a sequence number. Forecasts address halts by `schedule_stop.idx`. The canonical order at one station is `arr < pass < dep`.
- As-of rule: anything computed as of *t* uses only revisions with `received_at ≤ t`. An expected time never becomes an actual.
- Delay levels are `[-30,-10,0,5,10,15,20,30,45,60,90,120,180,240,360,600]`. Quantiles are P5, P10, P25, P50, P75, P90, P95. Defaults: N = 10,000; K = 32 display paths. Surprise means PIT < 0.001 or > 0.999.
- The engine runs Numba `@njit` kernels first. Any native port must wait for a measured G8 bottleneck.
- Frames are msgpack plus permessage-deflate. Each display-path event is `uint16 route_idx` + `uint16` time in 0.1 min from the frame base.
- Dashboard: Vite + TypeScript, MapLibre, deck.gl `PathLayer`/`TripsLayer`/`ScatterplotLayer`, and a self-hosted Protomaps PMTiles extract of India in a dark style.
- Personal use only. Raw source data never leaves the VM or the user's PC unless G1 allows it.
- **Live-traffic guard:** the real httpx transport for NTES and etrain is built only when `MSYND_LIVE=1`. Tests use `httpx.MockTransport`.

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
| M1 Source access | 6–11 | 🛑 CP-1 (G1 and live approval) → fixtures (G3) → parsers → 🛑 CP-2 (7-day G2 pilot starts) |
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
- [ ] **🛑 CP-1 (user):** decide G1 per use and record it in spec §18. Explicitly approve:
  - (a) protocol discovery and fixture capture, ≈ 150 requests at ≤ 0.3 req/s;
  - (b) the 7-day G2 pilot at ≤ 0.3 req/s;
  - (c) whether etrain backfill and Kaggle export are allowed.

  Without (a) and (b), M1 stops and M2 continues alone.

### Task 7: Capture CLI and protocol discovery

**Files:** create `msynd/collect/capture.py`, `tests/collect/test_capture.py`.

**Interfaces:** `python -m msynd.collect.capture plan.json --out tests/fixtures/<host>/`. `plan.json` is a list of `{name, purpose, method, url, form?}`. Each item goes through the `Dispatcher` (rate from `--rate`, default 0.3). It writes `<name>.body` plus a `manifest.json` with status, sha256, received_at and outcome; cookies and token values are redacted.

- [ ] Test with MockTransport: the files and manifest are written, and the redaction removes `Set-Cookie` and any `csrf`-like values.
- [ ] Implement; commit `task 07: capture cli`.
- [ ] **Monitor (live, after CP-1a):** `MSYND_LIVE=1 python -m msynd.collect.capture discovery.json`. This fetches the NTES landing page, `mntes.js`, `mntes_utility.js`, `LibraryFunctions.js`, one `GetCSRFToken`, the etrain search suggestion for 12951 and the etrain JS. Commit the fixtures.

### Task 8: NTES and etrain clients (spec §2, §7.1)

**Files:** create `msynd/collect/clients.py`, `tests/collect/test_clients.py`.

**Interfaces:**
```python
class NtesClient:
    def __init__(self, d: Dispatcher): ...
    async def ensure_session(self) -> None            # landing page sets cookie (purpose 'session')
    async def submit(self, purpose: str, path: str, form: dict, *, key=None, run_id=None) -> FetchResult
        # fresh GetCSRFToken (purpose 'token') then the form request: h = 2, never reuses a token
    async def train_running(self, train: str, start: date, run_id=None) -> FetchResult  # ShowRunCStn&refDate
    async def live_station(self, stn: str, hours: int) -> FetchResult
    async def exceptions(self) -> FetchResult
    async def schedule(self, train: str) -> FetchResult
class EtrainClient:
    async def history(self, train: str, period: str = '1y') -> FetchResult   # suggestion -> slug -> history (2 requests)
```
The token placement (header or form field) and the form field names come from the discovered JS fixtures.

- [ ] Tests with MockTransport and the discovered fixtures:
  - every `submit` issues exactly 2 requests (token, then form) and never reuses a token;
  - `empty` triggers `ensure_session` once;
  - `EtrainClient.history` makes 2 requests and parses the slug from the suggestion fixture.
- [ ] Implement; commit `task 08: ntes and etrain clients`.

### Task 9: G3 fixture capture (monitor, live)

- [ ] Monitor builds `fixtures_plan.json`:
  - NTES running status for: normal running; not started; completed; cancelled, diverted and rescheduled (trains taken from a fresh exceptions page); a multi-day train and a repeated-station train (both from GTFS); one with passing stations;
  - Live Station boards: NDLS 2 h, a quiet station, and one capture near 23:45 IST;
  - exceptions; one schedule; one map spot (to document that it is not GPS);
  - etrain history `1y` and `1m` for 12951.
- [ ] Run it live at 0.3 req/s and commit to `tests/fixtures/ntes|etrain/`. Record in `docs/gates/G3.md` which edge cases were captured and which could not be found.

### Task 10: Parsers and semantic validation (spec §7.1 Validation, §7.4)

**Files:** create `msynd/collect/parse_ntes.py`, `msynd/collect/parse_etrain.py`, `msynd/collect/validate.py`, `tests/collect/test_parse_*.py`, `tests/fixtures/**/<name>.expected.json` (golden files; the monitor hand-checks the first version of each against the page).

**Interfaces:**
```python
@dataclass(frozen=True)
class ParsedEvent: station: str; occ: int; kind: str; ts: datetime          # actual only (explicit actual marker)
@dataclass(frozen=True)
class ParsedEta: station: str; occ: int; kind: str; ts: datetime
@dataclass(frozen=True)
class StatusPage: train: str; start_date: date; events: list[ParsedEvent]; pending: list[tuple[str,int,str]]
                  etas: list[ParsedEta]; passing: list[tuple[str, float | None]]  # (station, km or None)
                  flags: list[str]                                              # e.g. 'delay_gt_24h'
def parse_status(html: bytes) -> StatusPage
def parse_board(html: bytes) -> BoardPage          # rows: train, start_date?, arr/dep (actual vs '*' expected)
def parse_exceptions(html: bytes) -> list[ExceptionRow]   # kind: cancelled|part_cancelled|rescheduled|diverted
def parse_schedule(html: bytes) -> ScheduleRows
def parse_etrain_search(body: bytes) -> str          # slug
def parse_etrain_history(html: bytes) -> list[HistoryRow]
def validate_status(req_train: str, req_start: date, page: StatusPage) -> str   # 'ok' | 'invalid'
```
Validation rules (spec §7.1): the train and start date must match the request; actual and expected markers must be explicit; times must be monotone along the route after day offsets; no expected time may appear as an actual; a delay > 24 h is kept with `flags += ['delay_gt_24h']`. Occurrences are assigned by counting repeats in page order.

- [ ] Tests (one per fixture, comparing against the golden JSON), plus:
  - `test_status_for_other_start_date_is_invalid` (Review Focus 1);
  - `test_multiday_offsets_monotone` (RF2);
  - `test_repeated_station_occ` (RF3);
  - `test_delay_over_24h_kept_flagged` (RF5);
  - `test_expected_never_actual` (a starred time never appears in `events`);
  - `test_empty_body_raises_empty` (the caller maps it to `empty`);
  - `test_last_report_line_departed_from` (the "Departed from ADAVALI (ADVI) on 07-Oct-2026 14:19" form).
- [ ] Implement; commit `task 10: parsers and validation`.

### Task 11: G2 pilot runner and analysis → 🛑 CP-2

**Files:** create `msynd/collect/pilot.py`, `tests/collect/test_pilot.py`, `docs/gates/G2.md` (written by the analysis).

**Interfaces:**
- `python -m msynd.collect.pilot run --days 7 --rate 0.3` stores everything in `obs.fetch` with bodies kept (no 24 h retention during the pilot). The scheduler:
  - discovers candidates from boards;
  - keeps a rotating simultaneous cohort `n_sim = floor(0.7·rate·120/h)`;
  - polls each cohort run's status every 120 s from 20 min before to 60 min after each expected event;
  - follows ≥ 30 train-station pairs on boards;
  - targets ≥ 40 distinct runs in total and 50 trains for passing-station prevalence;
  - samples the active-run count each hour.
- `python -m msynd.collect.pilot analyse --out docs/gates/G2.md --params data/params.json` measures:
  - `h` (requests per enquiry);
  - token lifetime (probe: reuse after 1, 5, 15, 60 s on ≤ 20 probes);
  - the empty-200 rate;
  - S per source: lag = first `received_at` showing the actual − actual time, interval-censored by the previous poll (midpoint, `# ponytail:` → Turnbull), with an exponential tail fitted above P90, ε with a Wilson CI, and unfinished runs right-censored;
  - pending-after-actual counts;
  - board retention and yield per request;
  - passing-station prevalence;
  - the active-run curve.

  It writes `data/params.json` with `{"S": {"ntes_status": {...}, "ntes_board": {...}}, "minute": "trunc", "provisional": false}`. Before G2, `data/params.json` is committed with `provisional: true`: lag P50 10 min, exponential, ε = 0.05.

- [ ] Tests:
  - `test_cohort_bound` (h = 2, rate 0.3 → 12);
  - `test_s_fit_recovers_synthetic` (synthetic lags with a known exponential tail and ε = 0.03, polled every 120 s → the fitted median is within 1 min and the ε CI contains 0.03);
  - `test_right_censored_not_counted_unpublished`;
  - `test_never_hard_bound` (zero counterexamples → the output still has ε > 0 and a soft tail).
- [ ] Implement; commit `task 11: g2 pilot`.
- [ ] **🛑 CP-2 (user):** choose the pilot host (PC or VM; it must stay up for 7 days). The monitor then starts it with `MSYND_LIVE=1`, checks it daily (fetch-log rate ≤ 0.3/s, no challenge) and runs `analyse` at the end.

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

**Interfaces:** `store_status(conn, run_id, fetch: FetchResult, page: StatusPage) -> StoreResult(changed: bool)`. In one transaction it:
- inserts `event_rev` only when a key is new or its `actual_ts` changed;
- writes `pending_rev` for every pending key;
- writes `ntes_eta_rev` for **every** present ETA (A3);
- turns new passing stations into a new route version and rebinds runs with identical halts (dirty);
- applies lifecycle transitions (running / completed);
- calls `mark_dirty` if anything changed.

It also has `store_board(...)`, `store_exceptions(...)` and `store_history(...)`; history rows go to `obs.history_daily` only.

- [ ] Tests, including `msynd/eval/baselines.py:b2_eta(conn, run_id, station, occ, kind, t)` (create it here):
```python
def test_eta_confirmation_per_key(db):    # A3 / R3-3
    # 09:00 ok page: ETA B=10:30, C=11:00 ; 10:00 ok page: C=11:00 only ; 10:20 ok page: B=10:30 unchanged
    assert b2_eta(db, r, 'B', 1, 'arr', ist('10:10')) is None             # missing at 10:00 → last 09:00, 70 min
    assert b2_eta(db, r, 'C', 1, 'arr', ist('10:10')) == ist('11:00')
    assert b2_eta(db, r, 'B', 1, 'arr', ist('10:25')) == ist('10:30')     # unchanged-present re-confirms
    # 10:40 page: B=10:45 (changed)
    assert b2_eta(db, r, 'B', 1, 'arr', ist('10:39')) == ist('10:30')     # post-cutoff confirmation unused
    assert b2_eta(db, r, 'B', 1, 'arr', ist('10:41')) == ist('10:45')
def test_token_fetch_never_confirms(db): ...      # token/session fetch rows → zero eta rows
def test_unchanged_actual_not_reinserted(db): ...
def test_expected_never_stored_as_actual(db): ...
def test_dirty_in_same_txn(db): ...               # rollback of store → no dirty row
def test_repeated_station_keys(db): ...           # RF3: occ 1 and 2 stored separately
def test_enrichment_never_rekeys(db): ...         # passing station → new route version, event keys identical
```
- [ ] Implement; commit `task 20: evidence store`.

### Task 21: Run lifecycle and exceptions (spec §7.3, A11)

**Files:** create `msynd/collect/lifecycle.py`, `tests/collect/test_lifecycle.py`.

**Interfaces:**
- `create_runs(conn, now) -> int` covers the next 24 h of origin dates (IST), using versions valid on the date with a matching `days_of_run` bit, minus cancellations, bound to the latest route version.
- `expire_runs(conn, now) -> int` applies: no actual for 12 h **and** now > scheduled destination arrival + max(12 h, 2 × last observed delay).
- `apply_exception(conn, row, fetch) -> None` handles cancelled, `resched_min`, `diverted_from` and `part_cancelled`, then calls `mark_dirty`. Each `obs.exception_rev.detail` it writes carries the run's **full cumulative** exception state (resched, diversion, part_cancelled), because evidence assembly (Task 16) reads the latest detail as complete state.

- [ ] Tests:
  - `test_creates_only_matching_weekdays`;
  - `test_ist_start_date_at_utc_evening` (RF2: at 2026-10-07 19:00 UTC the IST date is 10-08);
  - `test_cancelled_not_created`;
  - `test_expiry_rule_both_conditions`;
  - `test_expired_never_completed`;
  - `test_exception_marks_dirty`.
- [ ] Implement; commit `task 21: run lifecycle`.

### Task 22: Admission and the collector main loop (spec §7.2, §7.5)

**Files:** create `msynd/collect/admission.py`, `msynd/collect/__main__.py`, `tests/collect/test_admission.py`.

**Interfaces:**
- `next_job(state, now) -> Job`. Order: (1) challenge/session recovery; (2) overdue runs (not fetched within `T_max = 1.25·R·h/(rate·share)`); (3) exceptions every 30 min; (4) targeted polls by `obs.poll_queue.priority`, falling back to age when no row exists; (5) boards at their `params.board_share` (default 0.10, discovery and pending only); (6) schedule refresh.
- `python -m msynd.collect` runs one asyncio task per host, plus hourly `create_runs`, `expire_runs`, body retention (Task 23) and the disk check.
- etrain backfill runs only if `ETRAIN_BACKFILL=1` (set after G1). It is resumable, and its monthly top-up uses `d=1m`.

- [ ] Tests:
  - `test_no_run_starves_beyond_tmax` (simulate 300 runs with skewed priorities for 6 simulated hours: every run is fetched within T_max);
  - `test_job_order`;
  - `test_board_share_respected`;
  - `test_backfill_disabled_by_default`.
- [ ] Implement; commit `task 22: admission and collector loop`.

### Task 23: Retention, disk guard, backup (spec §12, §13, A3, A5)

**Files:** create `msynd/ops/nightly.py` (retention and backup parts), `tests/ops/test_retention.py`.

**Interfaces:**
- `prune_bodies(conn, now, ok_hours)` runs hourly from the collector: `ok` bodies after `ok_hours` (24, or 2 when disk ≥ 80 %); non-`ok` bodies and a 1 % `ok` sample after 14 days.
- `prune_rows(conn, now)` runs nightly: `obs.fetch` after 180 days, `pending_rev` after 180 days, `ntes_eta_rev` after 35 days, `forecast_issue` after 120 days.
- `disk_check(path) -> (pct, ok_hours)` writes the metric and raises an alarm at 70 %.
- `backup()` runs `pg_dump -Fc` (ref, obs without fetch bodies, ml, `sim.forecast_issue`), keeps 2 copies in `/data/backups`, uploads with `oci os object put` (7 daily / 4 weekly), and alarms if the off-site set is over 15 GB.

- [ ] Tests:
  - `test_old_referenced_fetch_pruned_event_kept` (A5);
  - `test_eta_35_days`;
  - `test_body_retention_drops_to_2h_at_80pct` (inject the disk percentage);
  - `test_sample_kept_14_days`.
- [ ] Implement; commit `task 23: retention and backup`.

### Task 24: Image, compose, Caddy (spec §5, §13)

**Files:** create `Dockerfile` (python:3.12-slim, multi-arch, `uv sync --frozen`), and modify `compose.yaml` to add the services `collector`, `engine`, `api`, `caddy` and an `ops` profile (`restart: unless-stopped`, health checks, json-file logging 10m×5, volumes `/data/{pg,models,backups,tiles}`). Create `Caddyfile` (DuckDNS host, basic auth on everything, `/api/*` → api:8000 with WebSocket, static `web/dist`, `/tiles/*`), extend `.env.example`, and write `docs/deploy.md` (≤ 60 lines: VM firewall 80/443 open, 22 restricted to the user's IP, crontab lines for nightly 21:00 UTC and weekly Sat 20:30 UTC).

- [ ] Verify: `docker buildx build --platform linux/arm64,linux/amd64 .` succeeds, and `docker compose --profile ops run --rm ops python -c "import lightgbm, onnxruntime, numba"` passes.
- [ ] Commit `task 24: image and compose`.

### Task 25: Gate and exit report CLI → 🛑 CP-3

**Files:** create `msynd/ops/report.py`, `tests/ops/test_report.py`.

**Interface:** `python -m msynd.ops.report {a1,a2,a5,a8,invalid,phase2} --since --until --out docs/gates/<name>.md`. It computes:
- A1: the coverage triple — scheduled-active, discovered and freshly-observed runs, where the scheduled-active denominator comes from our own inventory;
- A2: p50/p90 of (now − last `received_at`) by class and zone;
- A5: the maximum number of starts in any 60 s window per host, from `obs.fetch` with a SQL window function;
- A8: missing-event rate by class;
- the `invalid` share.

- [ ] Test `test_a5_window_counts` (61 starts within 60 s are detected).
- [ ] Commit `task 25: reports`.
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

- [ ] Tests:
  - `test_leakage_received_after_asof_absent` (an event at 10:00 received at 10:20 is absent at 10:15);
  - `test_encoders_never_see_validation_rows`;
  - `test_splits_rolling_by_week`;
  - `test_calendar_features_use_sched_dep_plus_d`.
- [ ] Implement; commit `task 26: features`.

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

- [ ] Tests:
  - `test_empirical_masses_sum_to_one_and_signed_o`;
  - `test_provisional_flags_cold_start`;
  - `test_promote_single_active_and_notifies`;
  - `test_cache_hit_on_same_v`, and a miss when the bias generation changes.
- [ ] Implement; commit `task 27: empirical models and cache`.

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

- [ ] Tests:
  - `test_horizon_bucket_independent_of_outcome`;
  - `test_identical_eligible_cases_across_baselines`;
  - `test_b2_ineligible_when_stale`;
  - `test_bootstrap_ci_excludes_zero_on_clear_gain`;
  - `test_promotion_rule_rejects_class_regression`.
- [ ] Implement; commit `task 29: evaluation harness`.
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

- [ ] Tests:
  - `test_priority_prefers_published_and_wide`;
  - `test_age_term_prevents_starvation`;
  - `test_forecast_issue_b2_nan_when_stale`.
- [ ] Implement; commit `task 32: priorities and issued forecasts`.

### Task 33: Replay-day end-to-end → 🛑 G8 and Phase 4 exit (spec §14 Replay day)

**Files:** create `tests/e2e/test_replay_day.py` (marked `slow`; skipped unless `MSYND_REPLAY_DAY=<dir>`), and `msynd/ops/replay_day.py`, which exports one pilot or live day of `ok` bodies with their receipt times into that directory.

- [ ] The test feeds the bodies in `received_at` order at 60× through parsers → store → engine on a test database. It asserts: A1 reported, A6 p95 measured, `invalid` < 1 %, no unhandled exceptions, and frames are produced for every discovered run.
- [ ] Commit `task 33: replay-day e2e`.
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

- [ ] Tests:
  - `position.test.ts` reads `../tests/fixtures/pos_cases.json` and expects every case to equal the Python result within 1e-6;
  - `frames.test.ts` covers gap detection, tombstones, and a full frame replacing everything.
- [ ] Verify `npm --prefix web test`; commit `task 35: web scaffold and frame client`.

### Task 36: Map, layers, timeline, panels → 🛑 A7 and Phase 5 exit (spec §11)

**Files:** create `web/src/main.ts`, `web/src/layers.ts`, `web/src/timeline.ts`, `web/src/panel.ts`, `web/src/status.ts`, `web/src/timeline.test.ts`; modify the Caddyfile (tiles).

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

### Task 38: Nightly leg bias, weekly VM retrain and promotion (spec §12, §8.4–8.6)

**Files:** modify `msynd/ops/nightly.py`; create `msynd/ops/weekly.py` and `tests/ops/test_bias.py`.

- **Bias:** over the last 7 days of matured segments, compute the median of (Δrun − active P50) per (halt pair, 3-hour bucket), shrunk by `n/(n+20)`, written as a new `gen` keyed by model (the previous generation is kept).
- **Weekly:** retrain LightGBM R/W/O on the VM, validate on the most recent 2 full weeks with `promotion_ok`, and promote or keep the active model, logging the decision to `sim.metric` and `docs/gates/weekly-<date>.md`. Any failure keeps the active model.

- [ ] Tests:
  - `test_bias_shrinkage`;
  - `test_new_model_starts_zero_bias`;
  - `test_failed_training_keeps_active`.
- [ ] Commit `task 38: nightly bias and weekly retrain`.

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

## Spec coverage check

| Spec section | Task(s) |
|---|---|
| §1 G1–G6, A1–A8 | 25 (A1, A2, A5, A8), 31 and 33 (A6), 36 (A7), 40 (A3, A4) |
| §2 Source facts, §15 G1–G8 | 6 (G1), 11 (G2), 9 (G3), 30 (G4), CP-3 (G5), 39 (G6), 18 (G7), 19 and 33 (G8) |
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
| §7.4 Parsing | 10 |
| §7.5 Backfill | 22 |
| §7.6 Static | 4, 5, 37 |
| §8 Learning | 26–30, 38, 39 |
| §9 Engine | 13–17, 19, 27, 31, 32 |
| §10 API | 34 |
| §11 Dashboard | 35, 36 |
| §12 Nightly | 23, 38 |
| §13 Ops | 23, 24, CP-3 |
| §14 Testing | Each row is pinned in its owning task: Clock (12), Identity (16, 20), Leakage (16, 26), Late data (16, 31), Dirty set (31), Determinism (17, 31), G7 (18), As-of (2), NTES baseline (20), Rescheduling (12, 18), Frame latency (31), Format (14), Dispatcher (3), Validation (10), WS (34), pos parity (15, 35), Restore (CP-3), Replay day (33), Model gate (29, 38) |

**Skipped on purpose:** the separate Phase 4 "debug map", because the Task 36 dashboard covers it. Add one only if the dashboard lags behind engine work.
