# m-syndicator — Live All-India Train Tracker: Design Spec (v2)

- **Date:** 2026-10-07
- **Status:** Draft v2, awaiting review
- **What changed from v1:** this version answers [the adversarial review](../reviews/2026-10-07-live-train-tracker-adversarial-review.md). §19 maps each review finding to the section that resolves it.
- **Scope:** every long-distance Indian Railways passenger train. Mumbai suburban is covered by a separate, later spec (§17).

---

## 0. Summary

A personal-use system that estimates, for every scheduled long-distance train run in India:
- where the train probably is now, and
- the distribution of its arrival time at every remaining halt.

On a map, each train appears as a cloud of glowing streaks. The timeline covers the past, now and the next 6 hours, with speed controls.

**How it works**

- **Learned delay models.** Separate models predict:
  - running time between consecutive halts,
  - dwell at each halt,
  - departure delay at the origin.

  All three are trained on delays collected live. Retraining is weekly; the GPU part runs on Kaggle only if the data-use gate passes (§15, G1).
- **Monte Carlo.** Each train is simulated as 10,000 independent "ghost trains".
  - Every station report fixes the ghosts' time exactly at that event.
  - "Not yet arrived as of the last check" is enforced by sampling only futures consistent with it.
  - Under the model, this is exact conditioning. It is the particle filter's "kill inconsistent ghosts, resample" step done analytically: there are no weights, no degeneracy and no resets (§4).
- **Collection under a hard budget.** One HTTP dispatcher per source host spaces request *starts* at least 1 s apart (counting tokens, sessions and retries). It decides what to poll from uncertainty and age. NTES tokens cost an extra request per enquiry, so at about 2,500 running trains a full poll cycle takes roughly 1.5 h. The dashboard therefore shows each train's **evidence age** rather than hiding it (§7.2).
- **Interactions between trains.** In v1 each train is simulated on its own. Train-to-train coupling is a gated experiment: it is enabled only after a causal ablation shows it helps (§4.4).

---

## 1. Goals, non-goals, acceptance conditions

### Goals
- **G1 — coverage.** Every scheduled long-distance run gets an estimate. Each run is labelled with its evidence state:
  - `observed`: last source event received ≤ *F* minutes ago
  - `stale`
  - `schedule_only`: no event yet
  - `cancelled`, `expired_unknown`
  - the attributes `diverted` and `part_cancelled`
- **G2 — estimates.** For each run: a position distribution now, and arrival-time quantiles at every remaining halt.
- **G3 — timeline.** One timeline with three zones, each labelled honestly:
  - past = station reports with reconstructed motion
  - recent = what the system showed
  - future = forecast to +6 h
- **G4 — playback controls.** Pause, 0.25× … 1000×, scrubbing, and a LIVE button.
- **G5 — polite load.** For each source host, request starts are spaced ≥ 1/rate apart. NTES defaults to 1.0/s and etrain to 0.2/s. All requests count.
- **G6 — learning.** Weekly retraining on live-collected data. A nightly leg-level bias correction.

### Non-goals (v1)
- Mumbai suburban (§17).
- Freight trains, which can't be observed.
- Bookings, PNR, fares and platforms.
- Multiple users or a public launch.
- Position fixes between stations: no GPS source exists (§2).
- Simulating a diverted train beyond its diversion point.

### Acceptance conditions

Each of these must be demonstrated with measurements; they are not promises. Numeric targets marked *(G2)* are fixed after the Phase 0 capacity gate.

| # | Condition | Measured how | Target |
|---|---|---|---|
| A1 | Coverage, reported as three numbers: **scheduled-active** runs (from schedule versions, run calendar and exceptions), **discovered** (≥ 1 source event), **freshly observed** (last event received ≤ *F*) | Engine metrics each minute. NTES offers no independent running list, so the denominator is our own schedule inventory, and the dashboard says so | discovered ≥ 95 % of scheduled-active runs whose origin departure is > 1 h old; freshly-observed share reported, target *(G2)* |
| A2 | Freshness | p50 / p90 of (now − `received_at` of the run's last event), by class and zone | *(G2)*; initial *F* = 60 min |
| A3 | Accuracy | **Prospective**: forecasts issued hourly by a model frozen for 4 weeks, scored on matured actuals in a period never used for selection. Median absolute error of P50 at the 2 h and 6 h horizon buckets (§8.6), compared with B1 (persistence) and B2 (NTES ETA ≤ 30 min old at issue time) on identical eligible cases | ≥ 10 % better than both, with the 95 % run-day block-bootstrap CI excluding 0 |
| A4 | Calibration | Same forecasts: coverage of P10–P90 and P05–P95, reported with mean interval width and CRPS | 75–85 % / 85–95 % |
| A5 | Load | Verified from the `obs.fetch` log, which records every request including tokens and retries | ≤ 60 request starts in any 60 s window per host at rate 1.0 |
| A6 | Engine latency (Oracle VM, all services running) | Time from evidence commit to the frame containing it | p95 ≤ 5 s; full rebuild after a model change ≤ 3 min |
| A7 | Dashboard | Frame rate on the user's laptop (named in the Phase 5 report) with K = 32 streaks per run, whole network | ≥ 60 fps; automatic fallback to K = 8 below 30 fps |
| A8 | Completeness | Per-run missing-event rate: halts of completed runs with no actual arrival or departure | reported by class; target *(G2)* |

---

## 2. Source facts

Facts are split into what has been observed and what is still unverified.

Observed facts come from the review's spot checks on 2026-10-07; [evidence](../reviews/2026-10-07-live-train-tracker-evidence.json) is recorded with hashes. Unverified items are resolved by Phase 0 gates (§15). Each gate has a pass, fail or unavailable result and a recorded decision.

### NTES (`https://enquiry.indianrail.gov.in/mntes/`)

**Observed**
- Every form submit first calls `GetCSRFToken`. Reusing a token produced **empty HTTP 200** responses.
  - Until G2 measures token lifetime, a logical enquiry costs **h = 2** HTTP requests.
  - An empty 200 response is invalid content, not data.
- Endpoints:
  - `tr?opt=TrainRunning&subOpt=ShowRunCStn` (also takes `&refDate=<start date>`)
  - `q?opt=LiveStation&subOpt=show`, with "Trains in Next 2/4/8 Hours"
  - `q?opt=ExcpTrains&subOpt=show`
  - `q?opt=TrainServiceSchedule&subOpt=show`
  - `TrnMap?opt=map&subOpt=spot`
- Live Station boards mark expected times with `*`. One NDLS sample showed an unstarred (actual) arrival 6 min old.
- Running status lists non-stopping (passing) stations. Its last-report line reads like "Departed from ADAVALI (ADVI) on 07-Oct-2026 14:19".
- **The map is not GPS.**
  - Its marker is `getStnLatLng(cStn[0])`, i.e. the coordinates of the last-reported station.
  - Its text, e.g. "207 Kms. ahead of Madgaon Jn.", is derived from the same report.
  - v1 therefore has **no position observations**.

**Unverified (gate G2)**
- How long boards keep actual times after a departure.
- Publication lag *L*: from an event happening to it becoming visible.
- Token lifetime.
- Whether displayed minutes are truncated or rounded. Default assumption: truncated, so jitter is U(0, 1) min.
- What NTES regards as acceptable automated use (gate G1).

### etrain.info (Tripozo)

**Observed**
- `train-history?train=N` returns a search form, not a redirect. A search suggestion gives the history slug, so one train costs **2 requests**.
- `/train/<slug>/history?d=1y` holds 365 dated rows, one value per halt per day. Verified for 12951: 2025-10-01 to 2026-09-30, 8 halts.

**Unverified (gate G4)**
- Whether a value is the arrival or departure delay, at the origin, intermediate halts and the destination.
- Whether `run_date` is the origin date or the station-local date.
- How repeated stations are handled.
- The terms state "Personal Non-Commercial Use". Whether bulk automated fetching, training and export are covered is part of gate G1.

### Static data
- **GTFS** (`Neo2308/indianrailways-gtfs`) is unofficial, generated around 2025-11-09, and contains about 10.5k routes. It is used only to seed schedule versions; NTES schedules take precedence where fetched.
- **OSM** (Geofabrik India) and **datameet/railways** are used **for drawing only** (§7.6).

### Literature

These papers motivate experiments; they do not validate this design.
- arXiv 1806.02825 studies n-order Markov delay models on Indian trains.
- arXiv 2510.01262 predicts station-period *average* arrival delay, not per-train distributions.
- arXiv 2510.09350 (Dutch) reports that persistence is a strong MAE baseline.

### Infrastructure
- **Oracle Always Free.** A1 ARM, up to 4 OCPU / 24 GB, 200 GB block storage, about 20 GB Object Storage.
  - Instances can be reclaimed as idle under documented CPU, network and memory thresholds over 7 days.
  - A1 capacity can be unavailable in a region.
- **PostgreSQL LISTEN/NOTIFY** wakes only sessions that are listening. The correct pattern is LISTEN, commit, inspect state, then react to notifications.
- **Kaggle.** About 30 GPU-h/week (not published, and it varies); 12 h sessions.

---

## 3. Contracts

Everything downstream depends on four contracts: identity, clock, availability, and distribution format.

### 3.1 Identity
- **Run** = (train number, origin start date in IST). A run never changes identity.
- **Schedule version.** An *immutable* halt sequence with timetable times and **timetable distances**, plus a validity window, days of running and a content hash.
  - A timetable change creates a new version.
  - A run is bound to one schedule version at creation, **forever**.
- **Route version.** An *immutable* ordered station list: the schedule version's halts plus any passing stations learnt from NTES.
  - Enrichment creates a new route version.
  - A run may be **rebound only to a route version with an identical halt sequence**. Rebinding marks the run dirty, so it is fully replayed.
- **Stop occurrence** = (station, occ). `occ` counts repeats of the station within the version (1, 2, …).
  - Events are keyed by `(run_id, station, occ, kind)`, **never by sequence number**, so enrichment and imports can never re-key stored observations.
  - Forecasts address halts by `schedule_stop.idx`, which is immutable for a run.
- **Geometry.** Physical segments are undirected station pairs used **only for drawing** in v1. `ref.route_segment` gives explicit ordered membership. Manual geometry fixes live in git (`data/segment_overrides.csv`), so they are versioned.

### 3.2 Clock and events
- **Run clock.** Minutes since 00:00 IST on the origin start date (float). Multi-day runs need no special case.
- **Event kinds**:
  - `arr` and `dep` at halts
  - the origin has `dep` only; the destination has `arr` only
  - `pass` at passing stations
  - canonical order at one station: `arr` < `pass` < `dep`
- **Delay** of an event = actual − scheduled, from the run's bound schedule version. Passing-station scheduled times come from the source when published; otherwise they are interpolated by distance.
- **Run segment *i*** is halt A_i to halt A_{i+1}.
  - Running target: `Δrun_i = arr_delay(A_{i+1}) − dep_delay(A_i)`.
  - Model **R** predicts it.
- **Dwell at an intermediate halt** = actual minutes `sched_dwell + dep_delay − arr_delay` (≥ 0). Model **W** predicts it.
  - *Change* in dwell is `dep_delay − arr_delay`. That is a delay difference, not a dwell time.
- **Origin departure delay.** Model **O** predicts it.
- **Physical floor.** Sampled running time ≥ `60 × km / V_MAX` minutes.
  - `V_MAX` = 160 km/h, the maximum permitted passenger speed on IR.
  - No per-segment speed limit is invented; the OSM `maxspeed` tag is not used.
  - Distance is always timetable km.
- **Regression case** (from the review): scheduled dwell 10, arr delay at A 0, dep delay at A 15, arr delay at B 20. This gives `Δrun = 5`, actual dwell 25, and arrival-to-arrival composite 20.
- **History.** etrain gives one value per halt-day, so it can only form the composite `Δcomp = Δdwell + Δrun`. It is **never used as an R or W label**. Once G4 passes it may supply:
  1. Fold-local *features*: per (train, halt pair) historical mean and P90 of the composite.
  2. A cold-start composite model C, used until R/W/O have 4 weeks of live labels. In this mode dwell equals the scheduled dwell, and the composite drives the running draw. The approximation is flagged `cold_start` in every frame.

  If G4 fails, history stays quarantined, and the engine starts with a fallback: an empirical per-class R/W/O pooled from the first 2 weeks of live data.

### 3.3 Observation and availability
- **Revisions.**
  - Every parsed actual time, expected time, "pending" marker and exception is an **append-only revision** carrying `received_at` and `fetch_id`.
  - Current truth is the latest revision per key.
  - A revision can change a time but cannot delete an event; retractions are not modelled.
- **As-of rule.** Anything computed "as of *t*" (features, forecasts, baselines, evaluation) uses only revisions with `received_at ≤ t`.
- **Labels** use matured current revisions: the run is completed or expired and ≥ 6 h have passed since the event.
- **History rows** have no availability time. They supply only label statistics and features, are never replayed as a live stream, and any history-only backtest is labelled *oracle-latency* and never used for acceptance.
- **NTES baseline.** The latest NTES ETA revision with `received_at ≤ t` and age ≤ 30 min. Otherwise the case is ineligible for comparison.
- **Expected never becomes actual.** An actual requires the source's actual marker. A passed scheduled time does not count.
- **Pending evidence.** `obs.pending_rev` records "source showed this event as *not yet happened* as of `received_at`". For example:
  - in running status: the next halt after the last report
  - on a board: a starred time at that station

### 3.4 Distribution format (the model contract)

**Bins**
- Each model returns a probability vector over fixed bins.
- **R:** edges at −60, −59, …, 240 min. That gives 300 unit bins, plus underflow (−∞, −60) and overflow [240, ∞): **302 bins**.
- **W and O:** edges 0 … 180, giving 180 unit bins plus overflow [180, ∞): **181 bins**.

**Sampling** with a uniform *u*:
- Find the bin by inverse CDF, then place the value linearly inside the bin.
- Underflow returns the lower edge; the physical floor then applies.
- Overflow returns `edge + τ·(−ln(1 − u'))`, an exponential excess. *τ* is the mean excess beyond the edge, fitted per class on training labels and stored in `ml.model.tail`.

**Truncation to [lo, hi]**
- Map the uniform as `u' = F(lo) + u·(F(hi) − F(lo))`.
- If `F(hi) − F(lo) < 1e-6`, the model says the window is impossible. Sample `lo + τ·Exp(1)`, capped at `hi`.

**LightGBM to CDF**
- Fit quantiles {0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98}, then sort them so they cannot cross.
- The CDF is piecewise linear between quantiles.
- Tails are exponential, anchored on the outer quantiles: the upper scale is `s = (q.98 − q.90)/ln 5`, and the lower tail mirrors it.
- Mass is rasterised into the bins. The excess beyond 240 becomes overflow with τ = s, which is consistent because the exponential is memoryless.

**Delay conditioning**
- Models are evaluated at 16 delay levels: `[-30,-10,0,5,10,15,20,30,45,60,90,120,180,240,360,600]`.
- A particle with delay *d* between levels L_j and L_{j+1} draws `(1−w)·Q_j(u) + w·Q_{j+1}(u)`, with the same *u* and `w = (d − L_j)/(L_{j+1} − L_j)`. Outside the range, the nearest level is used.
- Phase 3 measures the interpolation error against direct model calls.

---

## 4. Modelling approach

### 4.1 Exact Markov conditioning

R, W and O are first-order in the run's own delay. Their other features are known in advance: static data, calendar, and run context that has already been observed.

So an exactly reported event fixes the run's state, and conditioning on it is exact. The particle machinery reduces to:

1. **Halt event observed.** Every particle's time for that event is set to the observed time plus minute jitter. Everything downstream is re-drawn from there.
2. **Pass event observed.** This sets an anchor inside the segment. The remaining running time is re-drawn as `sched_rem + Δ·(sched_rem/sched_run)`, with Δ ~ R at the delay seen at the pass.
   - *ponytail:* linear scaling of Δ by the remaining scheduled fraction. Replace it with a sub-segment model trained on live pass events once enough exist.
3. **Pending ("not yet") evidence.** The draw producing that event is truncated to `> received_at − L`. *L* is the p95 publication lag measured in G2; the default is 10 min.
4. **Unreported events between two observed ones** are drawn truncated to `≤` the next observed time. If a lower bound from pending evidence conflicts with a later actual, the actual wins.
   - *ponytail:* this is not joint bridge sampling, but it keeps all invariants.

Particles are therefore **i.i.d. samples of the future given all evidence**:
- no importance weights, resampling, ESS or degeneracy
- no "reset" heuristic

**Surprise.** Before conditioning on each halt event, its PIT under the cloud is recorded. PIT < 0.001 or > 0.999 counts as a *surprise*. It flags the train "uncertain" for 30 min and feeds calibration monitoring.

If a hidden per-run variable is added later (for example "this train is having a bad day"), importance weights return. That is out of scope for v1.

### 4.2 Why many particles still matter

Every particle shares the observed past, but futures fan out over up to 40 segments. The P05/P95 forecasts and tail risks are Monte Carlo estimates, so more particles mean less Monte Carlo error.

- Default **N = 10,000 per run**.
- Phase 3 plots Monte Carlo error at P05/P50/P95 for N ∈ {1k, 5k, 10k, 50k, 100k}. The chosen N is the smallest whose MC error is ≤ 0.25 min at P05/P95 for the 6 h horizon.
- Particles exist only during a run's replay, about 220 KB per run at N = 10k. **Memory is not the limit; replay CPU is** (A6).

### 4.3 Freight and disruptions

These are invisible. They show up as surprises and as leg-level residuals.

The nightly bias (§12) corrects persistent leg-level error, keyed by (model, halt pair, 3-hour bucket). Because it is keyed by model, a newly promoted model starts at zero bias, so a residual is never counted twice.

### 4.4 Train-to-train coupling (gated experiments, Phase 7)

- **C1.** Feature: recent observed `Δrun` residuals on the same halt pair, from other runs, computed as-of by `received_at`. This is causal and identifiable at leg level.
- **C2.** Probabilistic occupancy:
  - For each canonical segment and time window, take the *fraction of particles* of each other run present. A posterior mean is never used.
  - Only segments with geometry confidence ≥ 0.9 are used.
  - Features include higher-priority trains due and same-direction headway.

**Enable rule.** A coupling experiment is enabled only if a prospective ablation shows:
- CRPS gain with a bootstrap CI > 0, and
- calibration within the A4 bands, overall and on high-disruption days.

**Known v1 limitation.** Separate per-train simulation ignores correlated network disruptions. Calibration is therefore reported separately for disruption days, defined as top-decile network-wide surprise counts.

---

## 5. Architecture

```
                 ┌───────────────────── Oracle VM (Docker Compose, arm64) ───────────────────────┐
 etrain ◀────────┤ collector: one Dispatcher per host (advisory lock, spacing, validation)       │
 NTES   ◀────────┤   status/board/exceptions/schedule/backfill ──revisions + obs.fetch──▶ Postgres│
                 │   lifecycle, pending, dirty_run (same txn) ──NOTIFY dirty──▶ engine            │
                 │                         ◀── obs.poll_queue (priority) ──────────── engine      │
                 │ engine: replay dirty runs → summaries + K display paths → sim.frame (gen log) │
                 │ api (FastAPI): reads sim.frame by gen; WS full/delta + resync; REST            │
                 │ caddy: TLS, basic auth, static web + PMTiles                                   │
                 └────────────────────────────────────────────────────────────────────────────────┘
 Kaggle (gated G1/G6) ◀── weekly export ──▶ candidate models + metrics ──▶ promotion on VM
 Oracle Object Storage ◀── nightly pg_dump + model artifacts (off-site backup)
```

| Service | Entrypoint | Owns |
|---|---|---|
| `db` | `postgres:16` | All persistent state |
| `collector` | `python -m msynd.collect` | **All outbound HTTP**, parsing, validation, run lifecycle, the dirty-run set. Backfill and schedule refresh are tasks *inside* it |
| `engine` | `python -m msynd.engine` | Replay, forecasts, poll priorities, frames, issued forecasts |
| `api` | `uvicorn msynd.api.app:app` | REST and WebSocket |
| `caddy` | `caddy:2` | TLS, auth, static files, tiles |
| cron | `python -m msynd.ops.{nightly,weekly}` | Bias, retention, backups, the weekly training job |

**Deliberately absent**
- PostGIS: positions are 1-D, and geometry is only for drawing.
- Redis and Kafka.
- Persisted particle state: runs are replayed from evidence.
- A Rust port: see §9.7.

---

## 6. Data model (Postgres 16)

**Conventions**
- Timestamps are UTC `timestamptz`, displayed in IST.
- Offsets are run-clock minutes.
- Migrations are numbered SQL files applied by `msynd/db/migrate.py`.

```sql
-- ── ref: static, versioned ──────────────────────────────────────────
CREATE SCHEMA ref;
CREATE TABLE ref.station (
  code text PRIMARY KEY, name text NOT NULL,
  lat double precision, lon double precision, zone text,
  is_junction boolean NOT NULL DEFAULT false);

CREATE TABLE ref.train (
  number text PRIMARY KEY,                 -- keep leading zeros
  name   text NOT NULL,
  class  text NOT NULL DEFAULT 'UNKNOWN'); -- RAJ|SHT|VB|DRNT|SF|MEX|EXP|PASS|SPL|UNKNOWN

CREATE TABLE ref.schedule_version (        -- immutable
  id serial PRIMARY KEY,
  train text NOT NULL REFERENCES ref.train,
  valid_from date NOT NULL, valid_to date, -- NULL = open
  days_of_run smallint NOT NULL,           -- bit0 = Mon … bit6 = Sun, at origin
  source text NOT NULL,                    -- 'gtfs' | 'ntes'
  content_hash bytea NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (train, content_hash));

CREATE TABLE ref.schedule_stop (           -- halts only
  version_id int REFERENCES ref.schedule_version, idx smallint,
  station text NOT NULL REFERENCES ref.station, occ smallint NOT NULL DEFAULT 1,
  arr_off real, dep_off real,              -- NULL arr at origin / dep at destination
  dist_km real NOT NULL,                   -- timetable km from origin: the only distance used for prediction
  PRIMARY KEY (version_id, idx), UNIQUE (version_id, station, occ));

CREATE TABLE ref.route_version (           -- immutable; halts + learnt passing stations
  id serial PRIMARY KEY,
  schedule_version_id int NOT NULL REFERENCES ref.schedule_version,
  content_hash bytea NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (schedule_version_id, content_hash));

CREATE TABLE ref.route_stop (
  route_version_id int REFERENCES ref.route_version, idx smallint,
  station text NOT NULL REFERENCES ref.station, occ smallint NOT NULL DEFAULT 1,
  halt boolean NOT NULL, dist_km real NOT NULL,
  sched_off real,                          -- NULL → interpolate by distance
  PRIMARY KEY (route_version_id, idx), UNIQUE (route_version_id, station, occ));

CREATE TABLE ref.segment (                 -- drawing geometry, undirected
  id serial PRIMARY KEY,
  stn_lo text NOT NULL, stn_hi text NOT NULL,  -- stn_lo < stn_hi
  polyline text,                           -- NULL → straight line
  osm_km real,
  confidence real NOT NULL,                -- §7.6; 0 = straight line
  UNIQUE (stn_lo, stn_hi));

CREATE TABLE ref.route_segment (
  route_version_id int REFERENCES ref.route_version,
  idx smallint,                            -- between route_stop idx and idx+1
  segment_id int NOT NULL REFERENCES ref.segment,
  forward boolean NOT NULL,
  PRIMARY KEY (route_version_id, idx));

CREATE TABLE ref.calendar (day date PRIMARY KEY, holiday text, fog_north boolean NOT NULL DEFAULT false);

-- ── obs: evidence, append-only where it matters ─────────────────────
CREATE SCHEMA obs;
CREATE TYPE obs.lifecycle  AS ENUM ('scheduled','running','completed','cancelled','expired_unknown');
CREATE TYPE obs.event_kind AS ENUM ('arr','pass','dep');   -- enum order = canonical order at a station
CREATE TYPE obs.source     AS ENUM ('ntes_status','ntes_board','etrain_hist');

CREATE TABLE obs.run (
  id bigserial PRIMARY KEY,
  train text NOT NULL REFERENCES ref.train,
  start_date date NOT NULL,                -- origin date, IST
  schedule_version_id int NOT NULL REFERENCES ref.schedule_version,  -- never changes
  route_version_id    int NOT NULL REFERENCES ref.route_version,     -- only to identical-halt versions
  lifecycle obs.lifecycle NOT NULL DEFAULT 'scheduled',
  resched_min integer NOT NULL DEFAULT 0,
  diverted_from text,                      -- simulation stops here
  part_cancelled jsonb,                    -- [{"from":"BRC","to":"NDLS"}]
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (train, start_date));

CREATE TABLE obs.fetch (                   -- one row per HTTP request: the load log
  id bigserial PRIMARY KEY,
  host text NOT NULL,
  purpose text NOT NULL,                   -- status|board|token|session|exceptions|schedule|search|history
  key text,
  started_at timestamptz NOT NULL, received_at timestamptz,
  http_status smallint,
  outcome text NOT NULL,                   -- ok|empty|invalid|transport_error|challenge
  body_zstd bytea);                        -- retention §12
CREATE INDEX ON obs.fetch (host, started_at);

CREATE TABLE obs.event_rev (               -- actual times; insert when new or changed
  id bigserial PRIMARY KEY,
  run_id bigint NOT NULL REFERENCES obs.run,
  station text NOT NULL, occ smallint NOT NULL, kind obs.event_kind NOT NULL,
  actual_ts timestamptz NOT NULL,
  source obs.source NOT NULL,
  fetch_id bigint REFERENCES obs.fetch,
  received_at timestamptz NOT NULL);
CREATE INDEX ON obs.event_rev (run_id, station, occ, kind, received_at);
CREATE INDEX ON obs.event_rev (received_at);

CREATE VIEW obs.event AS                   -- current truth; as-of queries add WHERE received_at <= $t
  SELECT DISTINCT ON (run_id, station, occ, kind) *
  FROM obs.event_rev ORDER BY run_id, station, occ, kind, received_at DESC, id DESC;

CREATE TABLE obs.pending_rev (             -- "not yet happened" as of received_at
  run_id bigint REFERENCES obs.run, station text, occ smallint, kind obs.event_kind,
  fetch_id bigint REFERENCES obs.fetch, received_at timestamptz,
  PRIMARY KEY (run_id, station, occ, kind, received_at));

CREATE TABLE obs.ntes_eta_rev (            -- baseline only, never evidence; insert on change
  run_id bigint REFERENCES obs.run, station text, occ smallint, kind obs.event_kind,
  expected_ts timestamptz NOT NULL,
  fetch_id bigint REFERENCES obs.fetch, received_at timestamptz,
  PRIMARY KEY (run_id, station, occ, kind, received_at));

CREATE TABLE obs.exception_rev (
  train text, start_date date, detail jsonb NOT NULL,
  fetch_id bigint REFERENCES obs.fetch, received_at timestamptz,
  PRIMARY KEY (train, start_date, received_at));

CREATE TABLE obs.history_daily (           -- quarantined until G4 records its meaning in §18
  train text, run_date date, station text, occ smallint,
  value_min smallint,                      -- NULL = cancelled/unknown
  fetch_id bigint REFERENCES obs.fetch,
  PRIMARY KEY (train, run_date, station, occ));

CREATE TABLE obs.dirty_run (               -- written in the same txn as the evidence; consumed by engine
  run_id bigint PRIMARY KEY,
  since timestamptz NOT NULL DEFAULT now());

CREATE TABLE obs.poll_queue (              -- written by engine, read by collector
  run_id bigint PRIMARY KEY REFERENCES obs.run,
  priority real NOT NULL, reason text NOT NULL,
  updated_at timestamptz NOT NULL);

-- ── ml ───────────────────────────────────────────────────────────────
CREATE SCHEMA ml;
CREATE TABLE ml.model (
  id serial PRIMARY KEY,
  role text NOT NULL,                      -- run|dwell|origin|composite
  kind text NOT NULL,                      -- lgbm_quantile|neural|empirical
  feature_ver smallint NOT NULL,
  trained_at timestamptz NOT NULL,
  train_window daterange NOT NULL,
  path text NOT NULL,
  tail jsonb NOT NULL,                     -- τ per class (§3.4)
  metrics jsonb NOT NULL,
  active boolean NOT NULL DEFAULT false);
CREATE UNIQUE INDEX ON ml.model (role) WHERE active;

CREATE TABLE ml.leg_bias (                 -- nightly; keyed by model so a new model starts at 0
  model_id int REFERENCES ml.model, gen int NOT NULL,
  from_stn text, to_stn text,
  hour_bucket smallint,                    -- 3-hour bucket of scheduled departure
  bias_min real NOT NULL, n integer NOT NULL,
  PRIMARY KEY (model_id, gen, from_stn, to_stn, hour_bucket));

-- ── sim ──────────────────────────────────────────────────────────────
CREATE SCHEMA sim;
CREATE TABLE sim.frame (                   -- durable frame log, kept 15 min
  gen bigserial PRIMARY KEY,
  ts timestamptz NOT NULL,
  is_full boolean NOT NULL,
  versions jsonb NOT NULL,
  payload bytea NOT NULL);

CREATE TABLE sim.forecast_issue (          -- prospective evaluation
  run_id bigint, issued_at timestamptz,
  versions jsonb NOT NULL,
  halt_idx smallint[] NOT NULL,            -- schedule_stop.idx
  q real[] NOT NULL,                       -- 7 per halt: P5,P10,P25,P50,P75,P90,P95 arrival, run clock
  PRIMARY KEY (run_id, issued_at));

CREATE TABLE sim.metric (
  ts timestamptz, name text, value double precision,
  PRIMARY KEY (ts, name));
```

**NOTIFY channels**
- `dirty`: wakes the engine.
- `sim_frame`: wakes the API.
- `model_changed`.

None of these is relied on for delivery. Durable state is in `obs.dirty_run` and `sim.frame`, and every listener reconciles at start and every 5 s.

---

## 7. Collector

### 7.1 The Dispatcher: the single owner of HTTP for each host
- **One owner per host.** At start it takes `pg_try_advisory_lock(hashtext('http:' || host))`. A second process for the same host refuses to start. Backfill, schedule refresh and exceptions are tasks inside the collector, not separate processes.
- **Spacing rule.** Request **starts** are ≥ `1/rate` apart, with no burst and one request in flight per host. At 1.0/s that guarantees ≤ 60 starts in any 60 s window (A5).
  - Every request goes through this rule and is logged in `obs.fetch`: pages, tokens, session pages, retries.
  - Rates: NTES 1.0/s, etrain 0.2/s (`NTES_RATE`, `ETRAIN_RATE`).
- **Session.**
  - The landing page sets the cookie.
  - Each submit fetches a fresh CSRF token (h = 2). Token reuse is allowed only if G2 measures a safe lifetime.
- **Validation.** Each response gets one of these outcomes:
  - `transport_error`: timeout, 5xx or 429. Retried with backoff 2 s·2ⁿ (cap 5 min, jitter), honouring `Retry-After`. Retries use the same spacing.
  - `empty`: a 200 with no content. Refresh the session once, then retry once.
  - `invalid`: content that fails *semantic* checks:
    - train number, start date and station identity must match the request
    - actual and expected markers must be explicit
    - times must be monotone along the route after day offsets
    - no expected time may be promoted to an actual
    - extreme delays are allowed; a delay > 24 h is flagged in a metric but kept
  - `challenge`: a CAPTCHA, JS challenge or login wall. **Stop that host** until manual restart, and alert on `/api/health`.
- **Circuit breaker.** If non-`ok` outcomes exceed 20 % of ≥ 20 requests in 5 min, pause the host for 10 min.
- **User agent.** `m-syndicator/<ver> (personal research)`.

### 7.2 Admission: what to fetch next

**Capacity (lower bound, no retries).** One full targeted cycle over *R* running runs takes `R·h / (rate·share)` seconds. With R = 2,500, h = 2, rate 1 and a 90 % targeted share, that is **≈ 93 min**.

This is the honest baseline. Boards (if G2 shows they keep actual times) and uncertainty ranking make it smarter, not faster. Raising the rate is a user decision, never done silently.

**Job types and their order**
1. `challenge`/session recovery.
2. **Overdue runs.** Any running run not fetched within `T_max = 1.25 × R·h/(rate·share)` goes first. This fairness rule means no run starves.
3. **Exceptions** every 30 min.
4. **Targeted polls** by priority. The engine computes it every 30 s as:

   `priority = P(next halt event has happened by now − L) × width(P10..P90 of that event) + age/T_max`

   This favours trains that probably have a new report waiting and whose estimate is wide.
5. **Boards.** The share is set by G2's measured yield of usable actual and pending events per request. If boards keep no actual times, they are used only for discovery and pending evidence, at a 10 % share.
6. **Schedule refresh**, low priority.

### 7.3 Run lifecycle (owned by the collector)
- **Creation.** Runs are created 24 h ahead, hourly, from schedule versions valid on the start date and matching `days_of_run`, minus cancellations. Each run is bound to a schedule version and its latest route version.
- **States**
  - `running`: first actual event.
  - `completed`: actual destination arrival.
  - `cancelled`: from exceptions.
  - `expired_unknown`: no actual received for 12 h **and** now > scheduled destination arrival + max(12 h, 2 × last observed delay). This is never labelled "completed".
- **Attributes.** `resched_min`, `diverted_from` and `part_cancelled` are columns, not states.
- **Dirty set.** Every evidence or lifecycle change upserts `obs.dirty_run` **in the same transaction**, then sends `NOTIFY dirty`.

### 7.4 Parsing
- **Pure functions**, each tested against Phase 0 fixtures:
  - `parse_status(html) → events, pending, etas, passing_stations`
  - `parse_board`
  - `parse_exceptions`
  - `parse_schedule`
  - `parse_etrain_search`
  - `parse_etrain_history`
- **Bodies.** Kept per §12 so pages can be re-parsed after a parser fix.
- **Passing stations.** New passing stations seen in a status page create a new route version, and runs with identical halts are rebound (§3.1).

### 7.5 History backfill (etrain)
- **Gates first.** Runs only after G1 (permitted use) and is useful only after G4 (meaning).
- **Cost.** Search suggestion → slug → `history?d=1y`: 2 requests per train. About 7,800 requests at 0.2/s ≈ **10.8 h** before retries. Resumable.
- **Monthly top-up** with `d=1m`.
- **Completeness** is measured against *expected operated runs*: days of running × schedule validity, minus known cancellations. It is not measured against 365 calendar rows.

### 7.6 Static import
- **Schedule versions.** GTFS is converted into schedule versions, deduplicated by content hash. NTES `TrainServiceSchedule` refreshes, done through the dispatcher at low priority, add newer versions.
- **Station coordinates** come from GTFS, with gaps filled from datameet.
- **OSM geometry**, for drawing only:
  1. Map each station to an OSM node: by `ref=` tag, otherwise the nearest node within 1.5 km with a fuzzy name match.
  2. Take the shortest path between consecutive route stations.
  3. Score confidence:
     - 1.0: both stations matched by `ref` **and** path length within ±10 % of the timetable km
     - 0.5: name-matched within tolerance
     - 0: anything else, drawn as a straight dashed line
- **Not used for prediction in v1:** OSM attributes (track count, electrification, maxspeed).
- **Overrides.** Manual fixes go in `data/segment_overrides.csv`, versioned in git.

---

## 8. Learning

### 8.1 Labels

| Model | Label | Source |
|---|---|---|
| R | `Δrun` per run segment | Live, matured current revisions |
| W | Actual dwell at intermediate halts | Live |
| O | Origin departure delay | Live |
| C (cold start) | Composite `Δcomp` | History, only after G4 |

### 8.2 Features (`msynd/features`, versioned by `FEATURE_VER`)

One query is (run, segment *i*, delay level *d*). Everything except *d* is shared by all particles of the run.

| Group | Features |
|---|---|
| Train | class, train number (target-encoded, fold-local, smoothed), segment index, segments remaining, is-last |
| Segment | timetable km, scheduled running min, scheduled dwell at A, slack (scheduled running − fold-local P05 of observed running on this halt pair), zone at A |
| Calendar | hour of `sched_dep + d` (sin/cos), weekday, month, `fog_north` × northern zone, holiday ±1 day |
| Delay | *d*: departure delay at A for R; arrival delay for W |
| Observed run context | the run's last *observed* `Δrun` and how many segments ago it occurred; count of surprises so far |
| History (only if G4 passes) | fold-local per (train, halt pair) composite mean and P90 |

Not in the v1 baseline: anything about other trains (C1, C2 in §4.4) and OSM attributes.

### 8.3 Causal rules (each enforced by a test)
- Features as of *t* use only revisions received ≤ *t*.
- Target encoding, slack, history statistics, bias and calibration are fitted inside the training fold only.
- Splits are rolling-origin by week; nothing is randomised across time.
- Labels come only from matured revisions.

### 8.4 Models
- **Baseline, on the VM CPU.** LightGBM quantile models for R, W and O (and C if used), converted to CDFs per §3.4. This is always retrained weekly on the VM, so a working model exists without Kaggle.
  - O and W start as *empirical* tables (per class and scheduled-dwell bucket) until there are enough labels for LightGBM.
- **Neural candidate, on Kaggle GPU** (gated by G1/G6):
  - **Inputs:** the §8.2 features plus the run's observed sequence of segment deltas (up to 8, through a small GRU or transformer), plus *d*.
  - **Head:** 302-way (R) or 181-way (W/O) softmax. Loss is cross-entropy on a Gaussian-smoothed target (σ = 1.5 bins) plus 0.5 × CRPS.
  - **Training:** AdamW, cosine schedule, mixed precision.
  - **Export:** ONNX, run on the VM with ONNX Runtime.
  - **Shipped only** if it passes §8.6 against the LightGBM model on identical cases.
- **Distillation.** Only if measured ONNX inference breaks A6.

### 8.5 Kaggle GPU workflow (gated: G1 must allow the export, G6 must pass)
- **Export** (`msynd.ops.export`): month-partitioned Parquet of revisions, runs, versions and models, plus a zip of the `msynd` package so feature code is identical. As-of features are built on Kaggle from the revisions.
  - The dataset is private, and its privacy is verified through the API after each version.
  - **This is not the backup** (§12).
- **Weekly job** (`msynd.ops.weekly`, Sunday 02:00 IST):
  1. `kaggle datasets version`.
  2. `kaggle kernels push`. The kernel does feature build, sweep (about 12 configs), final training and offline evaluation, then outputs ONNX and `metrics.json`.
  3. Check status every 5 min, giving up after 12 h. This is an external job, so polling is the only option.
  4. `kaggle kernels output`.
  5. Promotion check on the VM (§8.6). The VM LightGBM retrain runs regardless, and any failure keeps the active model.
- **Other GPU uses**
  - Neural candidates and sweeps.
  - Phase 7 coupling models.
  - Large backtests. These are CPU Numba inside Kaggle sessions; a CUDA port is added only if backtest wall-time becomes the bottleneck.
- **Budget** of about 30 GPU-h/week: the weekly job is ≤ 11 h, and the rest is for experiments.

### 8.6 Evaluation and promotion (`msynd/eval`)

**Offline as-of replay** over live-collected revisions:
- For each run, take cut points *t* every 30 min while it is running.
- Condition on revisions received ≤ *t* (events and pending).
- Forecast, then score each remaining halt.

**Horizon bucket.** Defined at issue time and independent of the model or the outcome:

`h = (scheduled arrival + latest observed delay as of t) − t`

Buckets are 30 min, 2 h, 6 h and 12 h, each ±25 %. A case falls in at most one bucket.

**Metrics**
- CRPS
- pinball loss at P5 … P95
- coverage of P10–P90 and P05–P95, with mean width
- median absolute error of P50
- PIT histogram
- breakdowns by class, zone, delay band and disruption-day flag
- **95 % CIs by block bootstrap over service dates**

**Baselines.** All are compared on identical eligible cases.

| | Baseline |
|---|---|
| B0 | Schedule |
| B1 | Persistence: latest observed delay carried forward |
| B2 | NTES ETA (≤ 30 min old at *t*) |
| B3 | The active model |

**Promotion** (weekly). Validate on the most recent 2 full weeks. The candidate must show:
- CRPS ≥ 2 % better than the active model, with the CI lower bound > 0
- coverage within ±5 points of nominal
- no class worse by > 5 % CRPS

**Acceptance (A3/A4).** Prospective only, from `sim.forecast_issue` rows of a model frozen for 4 weeks. That period is never used for selection.

---

## 9. Engine

### 9.1 Inputs and versions
- **Version tuple *V*:**
  - schedule version
  - route version
  - R / W / O (or C) model ids
  - bias generation
  - `FEATURE_VER`
  - *N*
- **Input:** a run's evidence, i.e. the current revisions of its events and pending markers.
- **Determinism.** *V* plus the evidence fully determines the run's output.

### 9.2 Per-particle state during replay

Structure-of-arrays, about 22 B per particle:

| Field | Type | Meaning |
|---|---|---|
| `seg` | int16 | current run segment |
| `t_arr_a` | f32 | arrival at A_i (departure for the origin) |
| `t_dep_a` | f32 | departure from A_i |
| `anchor_t` | f32 | last pass anchor time; `t_dep_a` if none |
| `anchor_km` | f32 | last pass anchor distance; `km(A_i)` if none |
| `t_arr_b` | f32 | arrival at A_{i+1} |

**Draws.** Each event time comes from exactly one draw:

| Slot | Produces |
|---|---|
| 0 | origin departure (O) |
| 1 | dwell → departure (W) |
| 2 | running → arrival (R) |
| 3 + k | remainder after pass *k* (R) |

**Invariants**
- Times are monotone along the route.
- Dwell ≥ 0.
- Sampled running time ≥ the physical floor.
- Observed times always win. If an observation breaks an invariant relative to another observation, it is kept and a `data_invariant` metric is raised.

**Position at time *t*** (pure function `pos`, shared with the browser):
- `t < t_dep_a`: at A_i.
- Otherwise: `anchor_km + (km_B − anchor_km)·g((t − anchor_t)/(t_arr_b − anchor_t))`. Here `g` maps time fraction to distance fraction using the scheduled passing pattern from the anchor to B, or straight-line in distance if no times are published.

### 9.3 Random numbers

`u = splitmix64(key(V, run_id) ⊕ pack(particle, segment, slot, k)) / 2⁶⁴`

- The generator is stateless and counter-based.
- Results are independent of batching, arrival order of evidence, and restarts.
- Forecast draws and live draws are the same draws, because conditioning only *replaces* the draws for observed events.

### 9.4 Replaying a run

The function `replay(run) → summaries, display paths`:
1. **Load** *V*, the bound route, the current revisions, and the pending markers. Order the events canonically: route index, then kind.
2. **Initialise** N particles. Origin departure = scheduled + `resched_min` + O-draw.
3. **For each observed event, in order:**
   - Draw any unobserved events before it, truncated to ≤ its time (§4.1).
   - Draw its own prediction and record its PIT (surprise).
   - Set it to the observation plus jitter.
4. **For the remaining events:** draw forward to the destination (or to `diverted_from`). Apply pending truncation `> received_at − L` to the draw producing each pending event. Downstream pending markers truncate only their own draw.
   - *ponytail:* no upstream propagation of downstream pending markers.
5. **Outputs:**
   - Per remaining halt: arrival and departure quantiles (P5, P10, P25, P50, P75, P90, P95) computed from **all N particles**.
   - Per-run headline: next-halt quantiles, evidence state, last-event age, surprise flag.
   - **K = 32 display paths:** particles 0 … 31, which are an unbiased sample because particles are i.i.d. Each path is the full event list to the destination.
6. **Discard** the particle arrays. The engine keeps only the outputs, about 10 KB per run.

**Cost estimate.** About 40 segments × N = 10k ≈ 0.4 M draws ≈ 10–20 ms per run on one core. A full rebuild of 2,500 runs on 4 cores takes about 10–15 s. Both are to be measured in G7.

### 9.5 Main loop

| Trigger | Work |
|---|---|
| Startup | LISTEN, commit, rebuild all active runs from the database, write a full frame |
| `NOTIFY dirty`, or every 5 s | `DELETE FROM obs.dirty_run RETURNING run_id` (atomic consume, commit-safe), then replay each run |
| Every 10 s, if anything changed | Append a **delta frame** with the changed runs' complete per-run payloads and tombstones for retired runs |
| Every 5 min | Append a **full frame** |
| Every 30 s | Recompute poll priorities and upsert `obs.poll_queue` |
| Hourly | Append `sim.forecast_issue` for running runs |
| `model_changed`, or a new bias generation | Mark all active runs dirty |

**Time passing with no new evidence does not change a forecast.** The cloud moves along its precomputed paths, and the poll priority rises.

### 9.6 Model inference and cache
- **Per-run CDF cache**, keyed by (*V*, shared query features, segment, level). It covers every remaining segment × 16 levels × {R, W}, about 1,280 queries per run.
- **Batching.** Queries are batched per replay.
- **Rebuild trigger.** The cache is rebuilt when *V* or the run's observed context features change.
- **Bias.** Applied as a shift of the R CDF.
- **No network context in v1**, so future segments need only known features.

### 9.7 Performance gate (replaces "port to Rust")
- **Numba first.** `@njit` kernels for the draw and replay loops. Numba is already native code.
- **G7 profile.** On the Oracle VM, with all services running, profile each stage separately:
  - draws
  - CDF interpolation
  - LightGBM and ONNX inference
  - per-run replay p50 / p95 / p99
  - full rebuild
  - frame build and serialisation
  - database writes
- **What to optimise.** Only the measured bottleneck, in this order: fewer queries through caching or batching, then smaller N within the §4.2 rule, then a native port of a proven kernel bottleneck with a comparative prototype. **Rust is an option only at that last step.**

---

## 10. API (`msynd/api/app.py`)

| Endpoint | Returns |
|---|---|
| `GET /api/static/network?v=` | Stations, segment polylines and confidence, route versions (stops, km, passing pattern) and route→segment membership. One file per static version, with an ETag |
| `WS /api/live` | Client sends its last `gen`. Server sends the latest full frame plus all later deltas if there's a gap, otherwise only new deltas. Frames carry `gen` and versions; a client that sees a gap asks for a resync. **Slow clients:** if more than 2 frames are queued, the queue is dropped and a full frame is sent. Encoding: msgpack + permessage-deflate |
| `GET /api/replay?from&to` | (≤ 24 h) Current event revisions and run bindings in the window, labelled "station reports; motion reconstructed" |
| `GET /api/run/{run_id}` | Schedule; quantiles for **all** remaining halts; latest NTES ETA with its age; the revision timeline (actual time and `received_at`); lifecycle and attributes; evidence state |
| `GET /api/train/{number}?date=` | Resolves to run(s). Lists them if ambiguous |
| `GET /api/search?q` | Trains and stations |
| `GET /api/health` | Per-host dispatcher state (ok, breaker, or stopped-on-challenge), active models, dirty backlog, last frame `gen` and age |
| `GET /api/metrics` | Recent `sim.metric` rows |

**API process.** Wakes on `sim_frame`, reads `sim.frame WHERE gen > last` in order, and also polls every 5 s.

**Display-path encoding.** Per event: `uint16 route_idx` plus `uint16` time in 0.1 min from the frame base (range 109 h), i.e. 4 B per event.

**Size estimate.** About 2,500 runs × 32 paths × ~20 remaining events × 4 B ≈ 6.4 MB per raw full frame, before deflate. To be measured in Phase 4; if it's too large, cap paths at +8 h and extend them later.

**Security.**
- Caddy: TLS (DuckDNS) and basic auth on everything.
- Postgres is not exposed.
- Firewall: ports 80 and 443 open; port 22 restricted to the user's IP.

---

## 11. Dashboard (`web/`, Vite + TypeScript)

**Map**
- MapLibre with a self-hosted Protomaps PMTiles extract of India in a dark style.

**Layers**
- **Network** (`PathLayer`). Geometry with confidence < 0.9 is drawn dashed.
- **Trains** (`TripsLayer`). K streaks per run, `trailLength` = 10 simulated minutes.
  - Colour = median delay (teal → amber → red).
  - Opacity ≈ boost / K, so a tight cloud reads as solid and a wide one as faint.
- **Train heads** (`ScatterplotLayer`).
  - Ring style shows the evidence state: solid = `observed`, hollow = `stale`, dotted = `schedule_only`.
  - A pulse means *surprise*.

**Timeline.** Pause, 0.25×, 1×, 10×, 60×, 300×, 1000×. The range is −24 h to +6 h, with a LIVE button. In LIVE mode `simTime = now − 15 s`.

| Zone | Source | Label |
|---|---|---|
| Past | `/api/replay` | "Station reports, motion reconstructed": one path per run, gaps shown |
| Recent | In-browser buffer of received frames, up to 30 min. On a fresh load it falls back to replay data | "As shown live" or "reconstructed" |
| Future | Display paths from the latest frames | "Forecast": streaks fan out |

**Train panel**
- Name and number with the run date.
- Evidence state and last-report age.
- Per-halt **quantile ribbon** (P5–P95, P10–P90, P25–P75, P50) next to the schedule and NTES's ETA (with age).
- Revision timeline.
- Rendering uses `/api/run/{id}`, i.e. all N particles, not the display streaks.

**Status bar**
- Coverage triple (A1).
- Evidence-age percentiles.
- NTES requests per minute.
- Dispatcher state.
- Active models and their age.
- Last frame age.

**Performance**
- Typed arrays.
- Attributes are updated only for changed runs.
- Drop to K = 8 if the frame rate is below 30 fps for 3 s.

**Parity.** `web/src/position.ts` is a port of `pos()`, checked against the shared fixture `tests/fixtures/pos_cases.json`.

---

## 12. Nightly job (`msynd.ops.nightly`, 02:30 IST)

1. **Leg bias.** Over the last 7 days of matured live segments, compute the median of (actual `Δrun` − active-model P50) per (halt pair, 3-hour bucket).
   - Shrink by `n/(n+20)`.
   - Write a new `gen`, keeping the previous one.
   - The bias is reported in the evaluation, and is reset by model change because it is keyed by model id.
2. **Retention**

   | Data | Kept |
   |---|---|
   | `obs.fetch.body_zstd` for `ok` responses | 24 h |
   | Bodies for non-`ok` responses, plus a 1 % `ok` sample | 14 days |
   | `obs.fetch` rows | 180 days |
   | `obs.pending_rev`, `obs.ntes_eta_rev` | 180 days |
   | `sim.forecast_issue` | 120 days |
   | `sim.frame` | 15 min (pruned by the engine) |
   | `obs.event_rev`, `obs.run`, `ref.*`, `ml.*` | kept; small, see §13 |

3. **Backup.** `pg_dump -Fc` of `ref`, `obs` (fetch bodies excluded), `ml` and `sim.forecast_issue`, plus active model files.
   - Saved to `/data/backups` (2 kept).
   - Also uploaded to **Oracle Object Storage** (separate from the VM disk) with the OCI CLI: 7 daily and 4 weekly copies.
   - A size alarm fires if the off-site set exceeds 15 GB.
4. **Static export.** Re-export the static network file if a version changed.

---

## 13. Operations

**Compose.** arm64 images, `restart: unless-stopped`, health checks, JSON logs (10 MB × 5). Volumes: `/data/{pg,models,backups,tiles}`.

**Secrets.** Kept in `.env` with mode 600: Kaggle token, basic-auth hash, DuckDNS token, OCI configuration, rates. `.env.example` is committed.

**Dependencies.** Locked with `uv`:
- Python 3.12, `httpx`, `selectolax`, `zstandard`, `psycopg[binary,pool]`, `numpy`, `numba`, `lightgbm`, `onnxruntime`, `pyarrow`, `fastapi`, `uvicorn`, `msgpack`, `kaggle`
- `osmium-tool` and `networkx` for the import
- G5 smoke-tests the **pinned** image on the actual A1 VM.

**Storage budget.** These are estimates; G5 and Phase 2 replace them with measurements. The review measured uncompressed NTES pages at 0.4–1.4 MB.

| Item | Estimate |
|---|---|
| `obs.event_rev` with indexes | ~10 GB per year |
| Fetch bodies: 24 h of all `ok` responses at ~86k requests per day × zstd size | measure (≤ 15 GB assumed) |
| Fetch bodies: 14-day non-`ok` set plus sample | ≤ 5 GB |
| Forecast issues | ~3–4 GB (120 days) |
| Pending and ETA revisions | ~3 GB (180 days) |
| PMTiles India | 1–3 GB |
| Images | ~5 GB |
| Local dumps | ~2 × 3 GB |
| WAL and bloat headroom | 20 % |

- **Alarm** at 70 % disk use.
- **At 80 %**, the 24 h `ok`-body retention is dropped automatically to 2 h.

**Oracle**
- G5 confirms the actual allocation (A1 capacity can be unavailable, in which case retry or try the other India region).
- Reclaim risk is handled by the off-site backup plus a **measured restore drill**. No artificial load is generated.
- A replacement VM is restored by: restore the dump, then the engine rebuilds every run from evidence.

**Recovery objectives**
- Data loss ≤ 24 h of evidence: the nightly dump. Gaps can be partially refilled by the etrain top-up if G1/G4 allow.
- Restore time ≤ 2 h, measured.

---

## 14. Testing

Each item names what must fail if the logic breaks.

| Area | Test |
|---|---|
| Clock | The review's counterexample: `Δrun = 5`, dwell 25, composite 20. History is never used as an R/W label |
| Identity | Adding a passing station (a new route version) does not change any stored event key. Replay output after a static import is identical (golden test) |
| Leakage | An event at 10:00 received at 10:20 is absent from features and forecasts as of 10:15. The NTES baseline never uses a snapshot after *t*. Fold-local encoders don't see validation rows |
| Late data | An event received after events later on the route is incorporated, and the result equals a fresh replay |
| Dirty set | Two connections, with commits racing the consumer: no committed dirty row is ever lost |
| Determinism | The same evidence in one batch or two gives identical outputs. Restart gives identical outputs |
| Conditioning | Observed halt events: every particle within the jitter. Pending: no particle event ≤ `received_at − L`. Invariants: speed ≤ `V_MAX`, dwell ≥ 0, monotone times |
| Distribution format | Overflow draws are finite and follow τ. LightGBM→CDF reproduces its quantiles to within half a bin. The interpolation test |
| Dispatcher | Simulated clock with tokens, retries and session recovery: ≤ 60 starts in any 60 s window. A second process cannot take the host lock |
| Validation | Empty 200 → `empty`. Identity mismatch → `invalid`. An expected time is never stored as an actual |
| WebSocket | A missed `gen` triggers a full resync. Slow-client coalescing. A tombstone removes a run |
| `pos()` parity | Python and TypeScript on the shared fixture |
| Restore | Dump, restore into a fresh container, rebuild; frames equal the original's for the same evidence |
| Replay day (end to end) | A recorded day of fetch bodies goes through parsers and engine at 60×. Checks: A1, A6, `invalid` < 1 %, no unhandled exceptions |
| Model gate | §8.6 promotion is the acceptance test for every model |

---

## 15. Delivery phases

Each gate ends in pass, fail or unavailable, with fixtures or measurements and a decision recorded in §18. **Fixtures alone are not an exit.**

### Phase 0 — Contract and capacity gates (≈ 1–2 weeks, mostly waiting on pilot data)

| Gate | Work | If it fails |
|---|---|---|
| **G1 Data use** | Record terms and licences, with retrieval date, in `docs/sources.md`: NTES (incl. the Disclaimer page), etrain, the GTFS repo, OSM (ODbL), Protomaps. Cover automated collection, model training, export to Kaggle, and portfolio display | Change source or scope. *User decision*: e.g. no Kaggle export (train on the VM only), or no etrain backfill |
| **G2 NTES protocol and capacity** | 7-day pilot at ≤ 0.3 req/s. Measure: *h* (tokens and sessions), token lifetime, empty-200 behaviour, publication lag *L*, board actual-time retention (follow ≥ 30 train-station pairs through arrival, dwell and departure; busy and quiet stations, midnight, late trains), board yield per request, pending semantics, passing-station prevalence (50 trains), and the active run count by hour | Sets *h*, *L*, board share, *T_max* and *F*. If the freshness achievable at 1 req/s is unacceptable, the *user decides*: accept it, change the rate, change the source, or reduce scope |
| **G3 Fixtures** | Committed fixtures for every page type, including edge cases: cancelled, diverted, rescheduled, multi-day, empty | — |
| **G4 etrain meaning** | After G2 has ≥ 2 weeks of live NTES data and etrain covers those dates: for ≥ 20 trains, compare etrain values against NTES actual arrivals and departures where they differ by ≥ 5 min. Cover origin, intermediate long dwell, destination, overnight and multi-day, early running, non-daily, repeated stations. Settle `run_date` semantics | History stays quarantined. Cold start uses the live empirical fallback (§3.2). Not blocking: G4 may finish during Phase 3 |
| **G5 Platform** | Get the Oracle A1 allocation. Run the pinned arm64 image smoke test (imports, a LightGBM and ONNX round-trip). Measure the storage budget from G2 bodies. Run a restore drill into a fresh container | Fix the image, or use the other region. Development continues on the user's PC meanwhile |
| **G6 Kaggle** (only if G1 allows export) | Private dataset and verified privacy. CLI push, status and output. GPU confirmed. `torch`/`onnx`/`lightgbm` import offline (otherwise phone-verify and enable internet, or ship wheels in the dataset) | Train on the VM only; drop the neural candidate |
| **G7 Engine profile** | Prototype §9.4 with LightGBM on synthetic runs, on the VM with all services running: per-run replay p50/p95/p99, full rebuild, frame build | Apply §9.7 to the measured bottleneck |

### Phase 1 — Static network and identity (≈ 1 week)
- Schedule and route versions, stations, OSM geometry with confidence, overrides.
- Backfill, only if G1 passes.
- **Exit:**
  - ≥ 95 % of GTFS trains have a schedule version and drawable geometry at any confidence.
  - Confidence distribution reported.
  - History completeness against expected operated runs reported.

### Phase 2 — Collector (≈ 1–2 weeks)
- Dispatcher, lock, validation, session handling, lifecycle, dirty set, pending markers, admission with fairness, boards (per G2), exceptions, retention, nightly backup to Object Storage.
- Until the engine exists, priority = age only.
- **Exit**, after 7 continuous days covering the weekly pattern:
  - A5 verified from the fetch log.
  - Evidence-age distribution and A8 reported.
  - `invalid` < 1 %.
  - A restore drill passes.

### Phase 3 — Baseline models and evaluation (runs while live data accumulates)
- Features with the causal rules, LightGBM R/W/O, the empirical fallbacks, and C if G4 passes.
- The §8.6 harness with B0–B2. The Monte Carlo error curve picks *N*. Delay-level interpolation error measured.
- **Exit** (needs ≥ 4 weeks of live data): R/W beat B1 on CRPS at the 2 h horizon (CI > 0), with coverage within ±5 points.

### Phase 4 — Engine (≈ 1–2 weeks; can start on the cold-start models)
- Replay, conditioning, pending truncation, dirty set, determinism, cache, summaries, frame log, poll priorities, issued forecasts, and a debug map.
- **Exit:**
  - The replay-day end-to-end test passes.
  - 7 days live with A1, A2 and A6 measured.
  - `sim.forecast_issue` populating.

### Phase 5 — Dashboard (≈ 1–2 weeks)
- API, WebSocket resync, PMTiles, layers, timeline zones and labels, panels, search and filters.
- **Exit:**
  - A7 measured on the named laptop.
  - Reload, network-loss and large-jump tests pass.

### Phase 6 — Kaggle GPU pipeline (≈ 1–2 weeks; gated by G1 and G6)
- Export, neural candidate, sweeps, weekly automation, promotion.
- **Exit:** two unattended weekly cycles, with the promotion decisions logged.

### Phase 7 — Acceptance and coupling experiments (ongoing)
- Prospective A3 and A4 on a model frozen for 4 weeks.
- Nightly bias evaluated.
- C1, then C2, through ablation (§4.4).

---

## 16. Risks

| Risk | Mitigation |
|---|---|
| Capacity at 1 req/s is about 1.5 h per full cycle | Honest evidence states and ages; priority plus fairness; boards if G2 allows. Rate changes are explicitly the user's decision |
| NTES blocks us or changes its format | Spacing, backoff and breaker; stop on challenge; bodies kept for re-parsing; fixtures |
| Data-use terms | G1 before bulk collection or export; keep raw data local; data never published |
| Wrong history meaning | History quarantined until G4; history is never an R/W label |
| Schedule or route change corrupting history | Immutable versions; event keys never depend on sequence numbers |
| Late or corrected reports | Revisions with `received_at`; full replay per dirty run |
| Lost notifications | Durable dirty set and frame log; NOTIFY is only a wake-up |
| Leakage inflating results | As-of rule, fold-local fitting, prospective acceptance |
| Unmodelled disruptions | Surprise flag, truncation, nightly bias; disruption-day calibration reported |
| Geometry errors | Drawing only, with confidence shown; never a prediction input in v1 |
| Kaggle unavailable or quota shrinks | VM LightGBM retrained weekly regardless; Kaggle is never on the recovery path |
| Oracle reclaim or capacity | Off-site backup plus measured restore; no artificial load |
| Storage growth | Bounded retention per table, alarms, automatic body-retention drop |

---

## 17. Mumbai suburban (separate spec, not v1)
- About 3,000 services a day on Central and Western Railway. There is no public live feed: NTES lists suburban timetables only, and etrain excludes suburban trains.
- Model state should be per corridor segment (a line-delay field), not per train.
- Possible sources: approach CR/WR or the Yatri app team for access, or estimate delays from the timetable plus public disruption notices.

---

## 18. Decisions log
- **Purpose:** personal, research and portfolio. **Hosting:** Oracle Always Free A1 in an India region.
- **Scope:** all long-distance trains in India, with honest evidence states. Mumbai locals excluded.
- **Collection:** hybrid targeted polls and boards, through one dispatcher per host. Request starts are spaced (no burst); tokens and retries count.
- **Clock contract:** running (dep → arr) and dwell are separate models. History supplies only composite features or a cold start, after G4.
- **Filtering:** exact Markov conditioning with i.i.d. particles (N = 10k default, set by the Monte Carlo error curve). No weights or resampling in v1.
- **Recovery:** each dirty run is replayed in full. Durable dirty set and frame log. Counter-based random numbers.
- **No position observations:** the NTES map gives station markers, not GPS.
- **Coupling:** a gated experiment (C1, then C2).
- **Performance:** Numba plus a profile gate. Rust only for a measured kernel bottleneck.
- **Models:** LightGBM on the VM always. A neural candidate on Kaggle only if G1/G6 pass and it wins the promotion test.
- **Backup:** off-site to Oracle Object Storage. Kaggle is not a backup.
- **Phase 0 outcomes:** *to be appended here (G1–G7 results and decisions).*

---

## 19. Review resolution map

| Review finding | Resolution |
|---|---|
| Phase 0: board retention | G2 measurement; boards limited to discovery and pending evidence if they fail (§7.2) |
| Phase 0: map GPS | Dropped; no position observations (§2, §18) |
| Phase 0: etrain meaning | G4 with discriminating cases; quarantine; never an R/W label (§3.2) |
| Phase 0: performance / Rust | Profile gate G7; §9.7 |
| Collection arithmetic | Capacity stated (§7.2); *h* measured; fairness via *T_max*; rate change is a user decision |
| 1 Mixed clocks | §3.2 contract plus regression test |
| 2 Availability leakage | `received_at` revisions, as-of rule, NTES baseline age, fold-local fitting (§3.3, §8.3) |
| 3 Mutable sequences | Immutable schedule and route versions; (station, occ) keys; explicit route→segment membership (§3.1, §6) |
| 4 Checkpoint and late data / NOTIFY | Full replay per dirty run; transactional dirty set; LISTEN-then-reconcile (§9.4–9.5) |
| 5 State and likelihood contract | Explicit dwell; event-directed draws; floor in minutes with `V_MAX`; no roughening, weights or underflow reset; no position likelihood (§4.1, §9.2) |
| 6 Determinism | Counter-based RNG keyed by *V* and draw identity; *V* bound to outputs; parity with tolerance (§9.3) |
| 7 Network context | Coupling deferred to gated C1/C2 with probabilistic occupancy and ablation (§4.4) |
| 8 Geometry | Drawing only; timetable km for prediction; confidence; versioned overrides (§7.6) |
| 9 CDF tails / forecast coverage | Bin edges, tail τ, truncation; every remaining segment queried; context-free v1 (§3.4, §9.6) |
| 10 Success criteria | A1–A8 with three-way coverage, evidence age and missing-event rate; lifecycle and attributes split; 2-request backfill (§1, §7.3, §7.5) |
| 11 Rate limiter | Start-spacing rule in a single owner per host, with advisory lock; semantic validation; challenge stop (§7.1) |
| 12 Frame delivery | `gen` frame log, full plus delta, resync, tombstones; runs addressed by run id; all-halt quantiles from all N; encoding specified (§9.5, §10) |
| 13 Replay honesty | Zones labelled "reconstructed"; fresh-load fallback (§11) |
| 14 Promotion vs acceptance | Issued forecasts; as-of horizon buckets; bootstrap CIs; prospective untouched acceptance; literature reframed (§2, §8.6) |
| 15 Storage and recovery | Per-table retention, measured budget, off-site Object Storage backup, restore drill, no PAYG claim (§12, §13) |
| 16 Data use | G1 gate; Kaggle export conditional (§15) |
