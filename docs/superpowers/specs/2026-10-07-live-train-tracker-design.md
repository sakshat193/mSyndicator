# m-syndicator — Live All-India Train Tracker: Design Spec (v1)

- **Date:** 2026-10-07
- **Status:** Draft, awaiting review
- **Scope of this spec:** v1, covering every long-distance Indian Railways passenger train. Mumbai suburban gets its own spec later (see §16).

---

## 0. Summary

A personal-use system that estimates, for every long-distance train in India:
- where it is right now, and
- a probability distribution for its arrival time at each station still ahead.

It shows these on a map. Each train is drawn as a cloud of glowing streaks. You can rewind into the past, watch live, and fast-forward into a simulated future.

**How it works:**
- **Particle filters.** Each running train has its own particle filter: thousands of simulated "ghost trains".
- **Learned delay model.** The ghosts move according to a model of how delay changes over each stretch of track. The model is trained on a year of history and retrained weekly on Kaggle GPUs.
- **Live reports.** A rate-limited collector reads public NTES pages. It mixes station-board sweeps with single-train polls that target the trains the filter is least sure about. When a real report arrives, ghosts that disagree with it are dropped.
- **Coupling between trains.** Trains affect each other through a shared map of which trains occupy which stretches of track. This feeds each train's delay model as context ("factored filter, track-occupancy coupling").

---

## 1. Goals, non-goals, success criteria

### Goals
- **G1:** Show every running long-distance train in India: about 3,900 train numbers, roughly 2,500 running at any moment (estimate).
- **G2:** For each train, give a current position estimate and an arrival-time distribution at every remaining station. Update both continuously.
- **G3:** One timeline covering three time zones:
  - **past:** replay of what actually happened
  - **now:** live estimates
  - **future:** forecast up to 6 h ahead
- **G4:** Speed controls: play, pause, slow-motion, fast-forward, and dragging the timeline.
- **G5:** Light load on public sources: a hard global request limit, 1 request/s to NTES by default.
- **G6:** Learn from a year of history (backfill). Retrain weekly on Kaggle GPUs, and apply small nightly corrections on the server.

### Non-goals (v1)
- **Mumbai suburban locals.** There's no public live data source; they get a separate spec (§16).
- **Freight trains.** Not observable. Their effect is modeled as hidden per-segment congestion.
- **Bookings and seats.** No tickets, PNR, seat availability or fares.
- **Users and public launch.** No user accounts, multi-user support or public launch. This is personal and non-commercial (decision §17).
- **Platform numbers, coach positions, and live-tracking diverted trains past the point of diversion.**

### Success criteria (measured, not judged by eye)

| # | Criterion | Target |
|---|---|---|
| S1 | Coverage: trains NTES shows as running that also have a live estimate | ≥ 95 % at any time |
| S2 | Accuracy: median absolute arrival-time error, 2 h and 6 h ahead, over 1 week of live data | ≥ 10 % better than NTES's own expected times **and** than "carry today's delay forward" |
| S3 | Calibration: share of actual times inside the predicted 80 % / 90 % range | 75–85 % / 85–95 % |
| S4 | Load on NTES: requests per second, averaged over any 60 s | ≤ the limit (default 1.0), never more than 3 in any one second |
| S5 | Freshness: engine update time (all trains) / gap between dashboard and engine | ≤ 1 s / ≤ 15 s |
| S6 | Archive completeness: hours per month with no collector data | < 1 % |
| S7 | Dashboard performance, full network with 32 streaks per train, mid-range laptop | ≥ 60 fps (automatically drops to 8 streaks per train below 30 fps) |

---

## 2. Facts this design rests on

These come from research done on 2026-10-07. Anything still uncertain is resolved in Phase 0 (§14), which has a decision rule for each item.

- **No official public API.** NTES (CRIS) and IRCTC are web and app only. This is a personal, non-commercial project, so we read public pages at polite rates (§15).
- **NTES** (`https://enquiry.indianrail.gov.in/mntes/`):
  - **Access.** No CAPTCHA on enquiry pages; the only CAPTCHA is on the feedback form. It uses a session cookie plus a CSRF token.
  - **Endpoints seen in `javascripts/mntes.js`:**
    - Single-train running status: `tr?opt=TrainRunning&subOpt=ShowRunCStn`
    - Station board: `q?opt=LiveStation&subOpt=show`
    - Cancelled / rescheduled / diverted trains: `q?opt=ExcpTrains&subOpt=show`
    - Schedules: `q?opt=TrainServiceSchedule&subOpt=show`
    - Train map: `TrnMap?opt=map&subOpt=spot`
  - **Where the data comes from.** Live data reaches NTES through the Control Office Application. That gets station-master and controller entries, plus RTIS (GPS on locomotives) automatic arrival, departure and pass-through times.
- **etrain.info** (Tripozo):
  - `/train/<Name-Slug>-<number>/history?d=1y` returns 365 daily rows for every halt station, as a JS array `et.rsStat.tooltipData = [[Date, delayMin@stn1, delayMin@stn2, ...], ...]`.
  - Verified on 12951: rows run from 2025-10-01 to 2026-09-30, about 67 KB per page.
  - Halt stations only, one delay number per station-day. Negative values (early running) occur. Suburban trains are excluded.
  - Their terms limit use to "personal non-commercial use".
- **Unofficial GTFS feed:** `Neo2308/indianrailways-gtfs`, about 10.5k routes. It's the starting schedule source.
- **Geometry sources:**
  - OpenStreetMap (Geofabrik `asia/india-latest.osm.pbf`) for track geometry and attributes (`tracks`, `electrified`, `maxspeed`).
  - `datameet/railways` for station coordinates.
- **Research findings:**
  - Most Indian trains' delays follow a first-order Markov process: the next delay depends mostly on the current delay (arXiv 1806.02825).
  - Delay spreads along the network, and congestion and spacing features help (arXiv 2510.01262, Indian network; arXiv 2510.09350, Dutch network).
- **Oracle Cloud Always Free:**
  - Ampere A1 ARM64, up to 4 OCPU / 24 GB RAM, 200 GB block storage. Mumbai and Hyderabad regions.
  - Oracle reclaims Always Free instances it considers idle (§15).
- **Kaggle:** about 30 GPU-hours/week (the figure isn't published and changes), 12 h sessions, 2× T4 or 1× P100. It can be driven with the `kaggle` CLI (datasets, kernels push/status/output).

---

## 3. Modeling approach: the answer to "per train or per track?"

**State lives per train. Trains interact through the track.**

1. **The basic unit is a leg.** A leg is the stretch between two consecutive *halt* stations of one train run. History has one delay number per halt per day, so that's the level we can learn at. Passing stations inside a leg are used for drawing positions and for live updates (§8.3), but the model predicts per leg.
2. **What the model predicts.** For each leg, the distribution of **delay change** (Δd = delay at the end of the leg minus delay at its start). It's conditioned on:
   - **the train's current delay** (the first-order Markov term)
   - **static leg features:** distance, scheduled time, slack, track count, train class
   - **time:** hour, weekday, month, fog season, festival days
   - **network context:** other trains on the same segments, their estimated delays and priorities, and recent unexplained slowdowns per segment
3. **Why not one joint simulation of all trains.** A particle filter over all trains' combined state would need exponentially many particles.
   - Instead, each train has its own filter (a **factored filter**).
   - Every 2 minutes, the engine builds a **track-occupancy map** from each train's *average* estimate (**mean-field coupling**).
   - That map becomes the network-context input to each train's delay model.
4. **Freight is a hidden variable.** Freight isn't visible, so its effect shows up as recent unexplained slowdowns per segment: actual delay change minus predicted, for the other trains that just passed. The engine computes this live, and the model learns to use it.

**One distribution format everywhere (the model contract).** Every model, whether LightGBM or neural, outputs a **CDF over 1-minute Δd bins**:
- bins run from −60 to +240 min, plus two overflow bins (302 bins in total)
- one CDF per (leg query, delay level)

The engine only reads this format, so models can be swapped freely. Particles sample from it with one uniform random number (inverse-CDF lookup, then a uniform offset within the bin).

**Delay levels (16):** `[-30, -10, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 600]` min. A particle uses the level nearest to its own delay at the start of the leg.
- *Known limit:* nearest-level lookup adds up to about half a level-step of error. If evaluation shows it matters, switch to linear interpolation between levels.

---

## 4. Architecture

```
                        ┌──────────────── Oracle VM (Docker Compose, arm64) ─────────────────┐
 etrain.info ──backfill──▶                                                                    │
 GTFS + OSM ──static────▶  collector ──raw + parsed obs──▶  Postgres  ◀── frames/forecast ── engine
 NTES boards ◀─sweep─────┤  (Python)                        LISTEN/NOTIFY ── obs_new ───────▶ (Python +
 NTES trains ◀─targeted──┤     ▲                                                              Numba kernel)
 NTES exceptions ◀───────┤     └──── poll_queue (ranked by uncertainty) ◀────────────────────┘   │
                         │                                                   api (FastAPI) ◀──────┘
                         │                                        Caddy (TLS, basic auth) ── web (static)
                         └──────────────────────────────────────────────────────│──────────────────┘
                                                                                 ▼
  Kaggle GPUs ◀── weekly: dataset version → train / sweep / distill / GPU replay ──▶ model.onnx + metrics
```

**Services (one Python package `msynd`, one image, different entrypoints):**

| Service | Entrypoint | Job |
|---|---|---|
| `db` | `postgres:16` | Every piece of stored state. Also the message channel between services (LISTEN/NOTIFY) |
| `collector` | `python -m msynd.collect live` | Session handling, global request limit, board sweeps, targeted polls, exceptions, storing raw responses |
| `engine` | `python -m msynd.engine` | Particle filters, coupling, poll ranking, live frames, forecasts |
| `api` | `uvicorn msynd.api.app:app` | REST and WebSocket for the dashboard |
| `caddy` | `caddy:2` | TLS, basic auth, static web files and map tiles, reverse proxy to `api` |
| cron jobs | `python -m msynd.ops.{nightly,weekly}` | Nightly corrections, backups, weekly Kaggle run |

One-off commands: `python -m msynd.collect static-import`, `python -m msynd.collect backfill`.

**Deliberately left out:**
- **No PostGIS.** Positions are one-dimensional distances along routes, and geometry is only used for drawing.
- **No Redis or Kafka.** Postgres LISTEN/NOTIFY plus small tables handle the small volume of messages.
- **No engine checkpoints on disk.** The engine's state can be rebuilt from stored observations with fixed random seeds (§8.8). The checkpoints in §8.3 live in memory only.

---

## 5. Data model (Postgres 16)

All timestamps are `timestamptz` in UTC and shown in IST. Schedule offsets are minutes from **00:00 IST on the run's start date**, so multi-day trains are handled naturally. Migrations are plain numbered SQL files applied by a tiny runner (`msynd/db/migrate.py`); no Alembic.

```sql
-- ── ref: static network ─────────────────────────────────────────────
CREATE SCHEMA ref;

CREATE TABLE ref.station (
  code        text PRIMARY KEY,          -- NTES station code, e.g. 'NDLS'
  name        text NOT NULL,
  lat         double precision,
  lon         double precision,
  zone        text,
  is_junction boolean NOT NULL DEFAULT false
);

CREATE TABLE ref.train (
  number      text PRIMARY KEY,          -- '12951', '01830' (keep leading zero)
  name        text NOT NULL,
  class       text NOT NULL,             -- RAJ|SHT|VB|DRNT|SF|MEX|EXP|PASS|SPL|OTHER
  origin      text REFERENCES ref.station,
  dest        text REFERENCES ref.station,
  days_of_run smallint NOT NULL,         -- bit 0 = Monday … bit 6 = Sunday, at origin
  source      text NOT NULL,             -- 'gtfs' | 'ntes'
  updated_at  timestamptz NOT NULL DEFAULT now()
);

-- Full station sequence incl. passing stations once NTES has shown them.
CREATE TABLE ref.train_stop (
  train       text REFERENCES ref.train,
  seq         smallint,
  station     text REFERENCES ref.station,
  halt        boolean NOT NULL,
  arr_off     integer,                   -- minutes from run-start 00:00 IST (NULL at origin)
  dep_off     integer,                   -- NULL at destination
  dist_km     real NOT NULL,             -- cumulative from origin
  PRIMARY KEY (train, seq)
);

-- Physical segment = consecutive station pair on any fine-grained route.
CREATE TABLE ref.segment (
  id          serial PRIMARY KEY,
  from_stn    text REFERENCES ref.station,
  to_stn      text REFERENCES ref.station,
  length_km   real,
  n_tracks    smallint,                  -- from OSM; NULL = unknown
  electrified boolean,
  max_kmh     smallint,
  polyline    text,                      -- encoded polyline for drawing; NULL → straight line
  UNIQUE (from_stn, to_stn)
);

CREATE TABLE ref.route_override (        -- manual fixes when OSM mapping is wrong
  from_stn text, to_stn text, polyline text NOT NULL,
  PRIMARY KEY (from_stn, to_stn)
);

CREATE TABLE ref.calendar (              -- holidays / festivals / fog-season flags
  day date PRIMARY KEY, holiday text, fog_north boolean NOT NULL DEFAULT false
);

-- ── obs: observations ───────────────────────────────────────────────
CREATE SCHEMA obs;

CREATE TYPE obs.run_status AS ENUM
  ('scheduled','running','completed','cancelled','part_cancelled','diverted','rescheduled');
CREATE TYPE obs.event_kind AS ENUM ('arr','dep','pass');
CREATE TYPE obs.source     AS ENUM ('ntes_train','ntes_board','ntes_spot','etrain_hist');

CREATE TABLE obs.run (
  id          bigserial PRIMARY KEY,
  train       text REFERENCES ref.train,
  start_date  date NOT NULL,             -- date at origin, IST
  status      obs.run_status NOT NULL DEFAULT 'scheduled',
  resched_min integer NOT NULL DEFAULT 0,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (train, start_date)
);

-- Ground truth. One row per (run, stop, kind); upserted, revisions counted.
CREATE TABLE obs.event (
  run_id      bigint REFERENCES obs.run,
  seq         smallint,
  kind        obs.event_kind,
  actual_ts   timestamptz NOT NULL,
  source      obs.source NOT NULL,
  first_seen  timestamptz NOT NULL DEFAULT now(),
  revisions   smallint NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id, seq, kind)
);
CREATE INDEX ON obs.event (actual_ts);

-- NTES's own predictions, kept ONLY as a baseline to beat (never fed to the filter).
CREATE TABLE obs.ntes_eta (
  run_id      bigint REFERENCES obs.run,
  seq         smallint,
  kind        obs.event_kind,
  expected_ts timestamptz NOT NULL,
  fetched_at  timestamptz NOT NULL,
  PRIMARY KEY (run_id, seq, kind, fetched_at)  -- insert only when value changed
);

-- Optional GPS-like position fixes, if Phase 0 shows the NTES map endpoint returns them.
CREATE TABLE obs.position (
  run_id      bigint REFERENCES obs.run,
  ts          timestamptz NOT NULL,
  dist_km     real NOT NULL,             -- projected onto the run's route
  PRIMARY KEY (run_id, ts)
);

-- Backfill from etrain: one delay number per halt per day.
CREATE TABLE obs.history_daily (
  train       text,
  run_date    date,                      -- start date at origin (verified in Phase 0)
  station     text,
  delay_min   smallint,                  -- NULL = cancelled/unknown
  PRIMARY KEY (train, run_date, station)
);

CREATE TABLE obs.exception (
  train       text, start_date date,
  kind        obs.run_status NOT NULL,
  detail      jsonb NOT NULL,
  fetched_at  timestamptz NOT NULL,
  PRIMARY KEY (train, start_date, kind)
);

-- Raw responses, zstd-compressed, kept 14 days for debugging / re-parsing.
CREATE TABLE obs.raw (
  id          bigserial PRIMARY KEY,
  source      obs.source NOT NULL,
  key         text NOT NULL,             -- e.g. 'train:12951:2026-10-07', 'board:NDLS'
  fetched_at  timestamptz NOT NULL,
  http_status smallint NOT NULL,
  parse_ok    boolean NOT NULL,
  body_zstd   bytea NOT NULL
);
CREATE INDEX ON obs.raw (fetched_at);

-- Engine → collector requests. Upsert by key; collector pops highest priority.
CREATE TABLE obs.poll_queue (
  train       text, start_date date,
  priority    real NOT NULL,
  reason      text NOT NULL,
  requested_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (train, start_date)
);

-- ── ml / sim ────────────────────────────────────────────────────────
CREATE SCHEMA ml;
CREATE TABLE ml.model (
  id          serial PRIMARY KEY,
  kind        text NOT NULL,             -- 'lgbm_quantile' | 'neural' | 'neural_distilled'
  feature_ver smallint NOT NULL,
  trained_at  timestamptz NOT NULL,
  data_cutoff date NOT NULL,
  path        text NOT NULL,             -- file under /data/models
  metrics     jsonb NOT NULL,
  active      boolean NOT NULL DEFAULT false
);
CREATE UNIQUE INDEX ON ml.model (active) WHERE active;

CREATE TABLE ml.segment_bias (           -- nightly correction (§11)
  segment_id  integer REFERENCES ref.segment,
  hour_bucket smallint,                  -- 0..7 (3-hour buckets)
  bias_min    real NOT NULL,
  n           integer NOT NULL,
  updated_at  timestamptz NOT NULL,
  PRIMARY KEY (segment_id, hour_bucket)
);

CREATE TABLE ml.filter_params (          -- tuned by GPU replays (§7.4)
  train_class text PRIMARY KEY,
  n_particles integer NOT NULL,
  obs_scale_min real NOT NULL,           -- Laplace scale for observation likelihood
  roughen_min real NOT NULL,
  updated_at  timestamptz NOT NULL
);

CREATE SCHEMA sim;
CREATE TABLE sim.latest (                -- single-row slots: 'frame', 'forecast'
  slot        text PRIMARY KEY,
  ts          timestamptz NOT NULL,
  payload     bytea NOT NULL
);
CREATE TABLE sim.metric (
  ts timestamptz NOT NULL, name text NOT NULL, value double precision NOT NULL,
  PRIMARY KEY (ts, name)
);
```

**Storage estimate (per year):**

| Data | Estimate | Retention |
|---|---|---|
| `obs.event` | about 3,000 runs/day × 60 events ≈ 65M rows ≈ 6 GB | kept forever |
| `obs.raw` | at ~1 request/s, 0.4–1.7 GB/day | 14 days only (about 25 GB at most) |
| `obs.history_daily` | about 10M rows | kept forever |

Everything fits in the 200 GB volume with plenty of room.

**NOTIFY channels:**
- `obs_new`: payload is `run_id`. Sent by the collector after new events or positions are upserted.
- `model_changed`: sent by the weekly job.
- `sim_latest`: payload is the slot name. Sent by the engine.

---

## 6. Collector

### 6.1 Request limits and politeness
- **NTES:** one in-process token bucket for the host. Rate 1.0/s and burst 3 by default (`NTES_RATE`, `NTES_BURST`).
- **etrain:** a separate bucket at 0.2/s.
- **Sharing the NTES budget:**
  - Targeted polls are served first whenever the queue's top priority is above `θ_poll`.
  - Otherwise the board sweep gets the slot.
  - The board sweep is guaranteed at least 30 % of slots, so boards never starve.
- **Retries and the circuit breaker:**
  - Retry 429, 5xx and timeouts with exponential backoff (2 s × 2ⁿ, at most 5 min, with jitter).
  - The circuit breaker trips when an endpoint's error or parse-failure rate goes over 20 % in 5 min. That endpoint then pauses for 10 min, the trip is logged and a metric records it.
- **User-Agent:** honest and configurable (`m-syndicator/<ver> (personal research)`). No personal data in headers.
- **Session:** fetch the landing page to get the cookie and CSRF token. Refresh after a 403/419 response, or every 30 min.

### 6.2 Commands
- **`static-import`** (one-off, then weekly):
  1. GTFS → `ref.train`, `ref.train_stop` (halts), `ref.station`. Station coordinates come from GTFS stops, with `datameet/railways` filling gaps.
  2. NTES schedule (`TrainServiceSchedule`) refresh for trains whose GTFS validity has lapsed, or that NTES lists but GTFS lacks. Low priority in the shared bucket.
  3. OSM:
     - Filter `railway=rail` ways and `railway=station|halt` nodes with `osmium tags-filter` and build a graph.
     - Map station codes to OSM nodes (`ref=` tag, otherwise nearest node within 1.5 km with fuzzy name match).
     - Find the shortest path between each **consecutive pair of stations on a fine-grained route** to get `ref.segment` geometry and attributes.
     - Pairs that don't map get a straight line. This only affects drawing, because prediction works in 1-D distance.
  4. **Fine-grained routes.** The first time a run's NTES running status shows passing stations, they're inserted into `ref.train_stop` with `halt=false`. Segments are created as needed.
- **`backfill`** (one-off, plus a monthly top-up):
  1. For each `ref.train`, `GET /train-history?train=<n>` to get the slug URL (Phase 0 confirms this redirect; otherwise build the slug from the name).
  2. Then `GET /train/<slug>/history?d=1y` and parse `et.rsStat.tooltipData` → `obs.history_daily`.
  3. It can resume: trains already done for the target window are skipped. It takes about 3,900 requests, roughly 5.5 h at 0.2/s.
  4. The monthly top-up uses `d=1m`, to fill holes the live collector missed.
- **`live`** (always running):
  - **Run discovery,** daily at 00:05 IST plus every hour:
    - Create `obs.run` rows for trains whose `days_of_run` matches today.
    - Mark a run `running` when its first event arrives.
    - Close runs once the destination arrival has been seen, or after scheduled end plus 12 h.
  - **Board sweep:**
    - Compute a station set *S* daily with greedy set cover over today's runs. Every run must cross a board station at least every 150 km of its route, preferring junctions.
    - The expected size of *S* is 200–400.
    - Go through *S* round-robin, with priority for boards whose upcoming trains have the widest uncertainty.
    - Parse each board into actual events for trains that have arrived or departed (if Phase 0 confirms boards carry them), plus NTES expected times (stored in `obs.ntes_eta` only).
  - **Targeted polls:**
    - Pop the highest-priority row from `obs.poll_queue`.
    - Fetch the running status for (train, start_date).
    - Upsert all events, store NTES ETAs, add any new passing stations, then `NOTIFY obs_new`.
  - **Exceptions:** poll `ExcpTrains` every 30 min → `obs.exception`, and update `obs.run.status` and `resched_min`.
  - **Positions (conditional):** used only if Phase 0 finds that `TrnMap?opt=map&subOpt=spot` returns a timestamped lat/lon. Positions are then projected onto the route to give `obs.position`, which the filter uses as an extra observation type.
  - **Before the engine exists (Phase 2):** a simple scheduler fills the queue by polling each running train every 40 min, round-robin.

### 6.3 Parsing
- Parsers are pure functions: `parse_running_status`, `parse_board`, `parse_exceptions`, `parse_etrain_history`. Each one has unit tests against fixtures captured in Phase 0 (`tests/fixtures/`).
- If a parser fails, the raw response is kept with `parse_ok=false` and a metric is incremented. That train or board is skipped this cycle, and the loop never crashes.
- **Times:** NTES shows times with minute precision, in IST, with day offsets. Parsers convert them to UTC using the run's start date. Any time more than 36 h away from the scheduled time is rejected as a parse error.

---

## 7. Learning pipeline

### 7.1 Training examples ("legs")
- **From history:**
  - For each (train, run_date) and each consecutive pair of halts *A → B*:
    - `d_A = delay at A`
    - target `Δd = d_B − d_A`
  - About 10M legs a year.
- **From live data:** the same thing, but from `obs.event`, where *A* is a departure and *B* an arrival.
  - Live legs also give dwell time (departure delay minus arrival delay at a halt) and passing-station times. These are kept for the within-leg position profile and for checking calibration.
- **What delay means in history:** Phase 0 checks whether etrain's number is the arrival or the departure delay, by comparing 20 trains against NTES actual times. Until then, the assumption is:
  - departure delay at the origin
  - arrival delay everywhere else

### 7.2 Features (`msynd/features`, versioned by `FEATURE_VER`)

| Group | Features |
|---|---|
| Train | class/priority, train number (target-encoded, smoothed), leg index, legs remaining, is-last-leg |
| Leg static | distance, scheduled time, **slack** (scheduled time minus the leg's historical 5th-percentile actual time), number of intermediate stations, min/mean `n_tracks`, electrified share, zone/division at A |
| Time | scheduled hour at A (sin/cos), weekday, month, `fog_north` on a northern-zone leg, holiday ±1 day |
| State | `d_A`, delay at the previous halt, Δd of the previous leg |
| Network context | trains scheduled on the leg's segments within ±30 min (count and same-direction count); mean and max estimated delay among them; whether a higher-priority train is due on the leg within 20 min (overtake/crossing risk); spacing to the train ahead (estimated minutes) and that train's delay |
| Recent slowdowns | mean leftover error (actual Δd minus predicted median) on the leg's segments over the last 2 h, from other trains (a stand-in for hidden freight and congestion) |

**Leakage rule (enforced by a test):** a feature for a leg starting at time *t* may only use rows whose `actual_ts < t`.
- For history rows, which only have a date, an event's time is taken as its scheduled time plus its delay.

**Splits:** by time only, never random.
- Rolling origin: train on months 1…*m*, validate on *m+1*, test on *m+2*.
- The weekly job uses the most recent full week as its test week.

### 7.3 Models (all produce the §3 CDF format)
- **Baseline (CPU) — LightGBM quantile regression:**
  - Quantiles {0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98}.
  - Predicted quantiles are sorted so they can't cross. The CDF is built by straight-line interpolation onto the 302 bins, with exponential tails.
  - Trains in about 1 h per weekly cycle on the VM.
  - It's also the fallback whenever Kaggle is unavailable.
- **Main (GPU) — neighborhood-attention network:**
  - **Inputs.** One token for the target leg, made from its static, time and state features. Plus up to *K* = 32 **neighbor tokens**: other trains on the leg's segments within ±60 min. Each neighbor token holds relative time and position, direction, priority, estimated delay and recent Δd.
  - **Encoder.** Two blocks: a transformer encoder over [target; neighbors] with width 128 and 4 attention heads. In effect this is a graph-attention network over the changing neighborhood graph in space and time.
  - **Output head.** 302-way softmax over the Δd bins.
  - **Training:**
    - Cross-entropy against a Gaussian-smoothed target (σ = 1.5 bins) plus a CRPS term (weight 0.5).
    - AdamW, cosine schedule, mixed precision on T4/P100.
  - **Export.** ONNX for CPU inference on the VM through ONNX Runtime.
- **Distilled model (CPU):**
  - Trained only if the main model's ONNX inference on the VM goes over the engine's budget (§8.4).
  - A narrower student (width 64, 1 block, *K* = 16) is trained on the teacher's output distributions (KL loss).

### 7.4 Kaggle GPU workflow
- **Repo folder `kaggle/`:**
  - `train_kernel.py` (a script kernel)
  - `replay_kernel.py`
  - `kernel-metadata.json` per kernel: `enable_gpu: true`, `enable_internet: false`, `dataset_sources: ["<user>/m-syndicator-data"]`, private. Phase 0 confirms the exact key for choosing the accelerator.
- **Data export** (`msynd.ops.export`):
  - Parquet files partitioned by month for `ref.*`, `obs.history_daily`, `obs.event` and `obs.run`, plus a **zip of the `msynd` package source**.
  - The kernel adds the package to `sys.path`, so feature code is the same in training and serving.
  - Expected size is about 200–400 MB compressed, uploaded weekly as a new dataset version. It doubles as the off-site backup.
- **Weekly job** (`msynd.ops.weekly`, Sunday 02:00 IST):
  1. Export, then `kaggle datasets version -p export/ -m "<iso-week>"`.
  2. `kaggle kernels push -p kaggle/train` runs inside one session of at most 11 h:
     - feature build
     - a hyperparameter sweep (about 12 configs, early stopping)
     - final training of the main model
     - distillation, if flagged
     - offline evaluation against the LightGBM model trained the same week
     - output `model.onnx` and `metrics.json`
  3. Check `kaggle kernels status` every 5 min. This is an external job, so polling is the only option. Give up after 12 h.
  4. `kaggle kernels output` → `/data/models/<id>/`.
  5. **Promotion rule:** run the §7.5 backtest on the VM for the newest week. The new model becomes active only if it beats the current active model by ≥ 2 % on rolled-forward CRPS, with 80 % and 90 % coverage within ±5 points of target. Then `NOTIFY model_changed` and the engine hot-reloads.
  6. If any step fails, the current model stays, LightGBM is retrained on the VM, and the failure is logged and recorded as a metric.
- **GPU replay kernel** (`replay_kernel.py`, monthly, or when filter logic changes):
  - Re-runs the particle filter on 4 held-out weeks with the same per-particle code compiled for CUDA (§8.2).
  - Sweeps particle count *N* ∈ {1k, 5k, 10k, 20k, 50k, 100k, 1M} per train class, plus the observation scale and roughening.
  - Output: the accuracy-vs-*N* curve and recommended settings → `ml.filter_params`.
  - Rule: pick the smallest *N* within 1 % of the best CRPS.
- **Budget** (about 30 GPU-h/week): about 10–11 h weekly training and sweep. The replay kernel takes about 10 h, monthly. The rest is spare for experiments.

### 7.5 Evaluation harness (`msynd/eval`)
- **Per-leg scores:**
  - CRPS
  - pinball loss at each quantile
  - coverage of the 50/80/90 % ranges
  - PIT histogram (checks whether predicted ranges are calibrated)
  - median absolute error
  - broken down by train class, month and delay level
- **Rolled-forward ETA (multi-leg):**
  - From cut points every 30 min of a run, simulate 2,000 rollouts through the following legs.
  - Score the arrival-time distribution at downstream halts, 30 min / 2 h / 6 h / 12 h ahead.
- **Baselines:**
  - B0: the schedule
  - B1: persistence (keep today's delay)
  - B2: NTES expected times. Only available from the start of live collection; the snapshot used is the one closest to the cut point.
  - B3: LightGBM
  - M: the main model
- **Outputs:** a JSON report saved in `ml.model.metrics`, plus a Markdown summary in `reports/`.

---

## 8. Engine: the particle filter

### 8.1 State layout
- All particles for all active runs live in flat arrays in one CSR layout (`offset[r]`, `count[r]`):
  - `leg` int16: current leg index
  - `t_enter` float32: minutes since run start, at entry to the current leg
  - `dur` float32: sampled total leg time (dwell at the start + running), minutes
  - `logw` float32: log weight
- That's 14 bytes per particle. A cap of 40M particles means about 560 MB, plus an equal-sized **checkpoint copy** for rewinds (§8.3).
- **Particles only change at leg boundaries or at an observation.** Between those, position is a pure function `pos(leg, t_enter, dur, t)`:
  - dwell first: scheduled dwell, clipped to [1 min, scheduled] when late
  - then motion along the leg's **scheduled passing pattern**, scaled to fill the remaining time
- Because of this, moving particles forward costs in proportion to legs crossed, not to elapsed time.

### 8.2 The compiled per-particle function
- `advance_particle(leg, t_enter, dur, t_target, u[], cdf_tables, leg_meta) -> (leg, t_enter, dur)`:
  - Crosses leg boundaries until `t_enter + dur > t_target`.
  - At each new leg, it samples `dur = sched_T + Δd`, with Δd from the CDF at the particle's delay level.
  - It applies the minimum physical time `leg_km / max_kmh_leg` as a floor.
- **It takes random numbers as an input array.** RNG lives outside, so CPU and GPU give identical results for the same inputs, which is what the parity test checks.
- **It's compiled twice** from one Python source:
  - `numba.njit` for the CPU, wrapped by a `prange` loop
  - the Numba CUDA target (`numba-cuda`) device function for the GPU replay kernel
- **Rust fallback.** If the Phase 0 benchmark exceeds budget (§14), only this function and its loop are ported to Rust (PyO3/maturin). The array contract stays the same. The GPU replay keeps the Numba CUDA version.

### 8.3 Folding in observations, including late ones
- Observations reach us minutes late, and targeted polls bring a batch covering the last 30–40 min. So each run keeps a **checkpoint**: its particle arrays at `t_c`, the time of the last folded-in observation.
- On `obs_new` for run *r*:
  1. Load new events and positions for *r* with time after `t_c`, sorted by time.
  2. **Rewind:** restore the checkpoint.
  3. For each observation *o*, in time order:
     - Move the particles forward to `o.time`.
     - Compute each particle's predicted time at *o*'s station:
       - an arrival is the end of a leg
       - a departure is `t_enter` plus dwell
       - a pass is `t_enter` plus dwell plus the inverse profile at the station's position within the leg
       - for `obs.position`, compare the predicted distance at `o.ts` instead
     - **Weight:** `logw += log Laplace(actual − predicted; b)`.
       - `b = obs_scale_min` (default 1.0 min) for actual events
       - `b = 2.0` for positions
     - Normalize. If the effective sample size is below N/2, resample systematically, then **roughen**: add N(0, `roughen_min`²) to each particle's current `dur`.
  4. **Divergence reset:** if every weight underflows (an event the model can't explain, such as a breakdown, an accident or a held train):
     - re-initialize *r*'s particles at the observation, with the delay set to the observed delay
     - spread from the model
     - set `diverged_count++`
     - mark the train "uncertain" on the dashboard for 30 min
  5. Set the checkpoint to the last observation time, then move forward to now.
- **NTES expected times are never folded in.** They're predictions, not observations, and are kept only as a baseline.

### 8.4 Main loop (wall clock)

| Every | Work |
|---|---|
| on `obs_new` | §8.3 for the runs that were notified |
| 10 s | move all particles forward to *now* (leg crossings only); emit a live **delta frame** (§8.6) |
| 30 s | recompute poll priorities, then upsert the top 200 into `obs.poll_queue` |
| 2 min | **coupling refresh** (§8.5); rebuild CDF tables for every run's current leg and the next 2 legs |
| 5 min | **forecast** (§8.6) |
| on `model_changed` | load the new model and rebuild all CDF tables |
| 00:10 IST | start new runs and retire finished ones |

**Budget:**
- 10 s cycle: ≤ 1 s
- coupling refresh: ≤ 20 s, run in a worker thread. It's built from posterior averages, then swapped in all at once.
- **Model calls per refresh:** about 2,500 runs × 3 legs × 16 delay levels ≈ 120k queries.
  - LightGBM: about 1–3 s on 4 cores (estimate).
  - Neural ONNX: if it takes more than 20 s, use the distilled model.

**CDF table memory:** 2,500 × 3 × 16 × 302 bins stored as uint16 ≈ 73 MB.

### 8.5 Coupling (mean-field)
- Every 2 min, each run's posterior is summarized: mean distance along route, P10/P90 distance, mean delay, current segment, and expected entry times for its next segments.
- From these, build the **segment occupancy index**: for each segment, the list of (run, direction, expected entry/exit, priority, mean delay).
- **Recent-slowdown index:** for each segment, the mean over the last 2 h of (observed Δd minus predicted median), shrunk toward 0 by `n/(n+10)`. It's computed from runs that just finished legs touching that segment.
- The network-context and recent-slowdown features (§7.2) come from these indexes. The same feature code (`msynd/features/context.py`) is used offline (from history) and online (from posteriors).

### 8.6 Outputs
- **Wire format: "leg schedules".** For each train, *K* = 32 display particles, chosen at evenly spaced quantiles of predicted position. Each carries its list of `(leg, t_enter, dur)` over a time window. The client computes exact positions and trails with the same `pos()` function, so motion is smooth at 60 fps between frames.
- **Live delta frame** (every 10 s) → `sim.latest['frame']` + `NOTIFY sim_latest`:
  - only runs whose display particles changed: a leg crossing, an observation or a reset
  - plus a header per train: delay P10/P50/P90 at the next 3 halts, the uncertain flag and the age of the last observation
  - a full snapshot every 5 min, and on request at client connect
- **Forecast** (every 5 min) → `sim.latest['forecast']`:
  - each display particle's leg schedule from now to now + 6 h, produced by moving copies forward with no observations
  - about 4 MB raw (2,500 × 32 × ~8 legs × 6 B)
- **Rebuilt, not saved:** on startup the engine rebuilds every active run by replaying its observations with seeds `hash(run_id, batch_no)` (§8.8). There are no particle checkpoints on disk.

### 8.7 Particle counts
- **Starting values:**
  - default `N = 10,000` per run
  - adaptive per run after each resample: `N = clamp(2,000, 50,000, 400 × number of distinct 1-min bins occupied by the predicted next-halt time)`
- **Global cap:** 40M. If the cap binds, every run is scaled down in proportion.
- **Overrides:** per train class, from `ml.filter_params`, once the GPU replays (§7.4) have run.

### 8.8 Determinism
- Every random draw comes from `numpy.random.Generator(PCG64(seed))`, with `seed = hash(run_id, batch_no)`.
- So the same observations always give the same particles. That's what makes rebuild-on-restart and the tests possible.

---

## 9. API (`msynd/api/app.py`, FastAPI)

| Endpoint | Returns |
|---|---|
| `GET /api/static/network` | Stations, segment polylines, each train's route as a list of segment IDs, and per-leg scheduled passing patterns. Generated once per static import as one file (ETag). About 5–15 MB, cached by the browser |
| `WS /api/live` | On connect: a full snapshot. Then delta frames as `sim_latest` fires. Binary msgpack with typed-array fields; permessage-deflate on |
| `GET /api/forecast` | The latest forecast payload (gzip) |
| `GET /api/replay?from&to` | Ground-truth events for every train in the window (capped at 24 h), plus the matching runs. The client fills in between events with `pos()` |
| `GET /api/train/{number}` | Schedule; current arrival-time ranges for each upcoming halt (P10/P50/P90) next to the schedule and NTES's latest ETA; the observation timeline; run status |
| `GET /api/search?q` | Trains and stations by number, name or code |
| `GET /api/health` | Status of each service, the age of the last observation, the circuit breaker, the active model ID |
| `GET /api/metrics` | Recent `sim.metric` rows (request rate, errors, cycle times, divergences, effective sample size) |

**Exposure:**
- Caddy handles TLS through a free DuckDNS name or the user's domain, plus basic auth on everything.
- Postgres isn't exposed. The Oracle security list opens only ports 80 and 443 (and 22 restricted to the user's IP).

---

## 10. Dashboard (`web/`, Vite + TypeScript)

**Map and rendering:**
- **Base map:** MapLibre GL with a **self-hosted Protomaps PMTiles extract of India** in the dark style, served by Caddy with range requests. No third-party tile keys and no rate limits.
- **Layers** (deck.gl):
  - `PathLayer`: the rail network, drawn faint, with each segment tinted by its current recent-slowdown value. Stretches running slow glow amber or red.
  - `TripsLayer`: *K* streaks per train, `trailLength` = 10 min of simulated time.
    - Color: the train's current median delay (teal for on time → amber → red).
    - Opacity is about 1/*K* times a boost, so a tight cloud looks bright and solid while an uncertain one looks dim and spread out.
  - `ScatterplotLayer`: the median "head" of each train. Click it to open the side panel.

**Timeline bar:**
- **Speeds:** pause, 0.25×, 1×, 10×, 60×, 300×, 1000×.
- **Range:** from −24 h to +6 h, with a **LIVE** button.
- **Three color-coded zones** in the timeline:

| Zone | Source | Display |
|---|---|---|
| Past (more than 30 min ago) | `/api/replay` (ground truth) | Single solid streaks, no spread |
| Recent (last 30 min to now) | **in-browser buffer** of the last 30 min of live frames | What the filter believed at the time; rewind or slow-mo without load on the server |
| Future | `/api/forecast` | Streaks fan out as uncertainty grows |

- The in-browser clock `simTime` advances at speed × wall time.
- In LIVE mode, `simTime = now − 15 s`, so frames arrive before they're needed.

**Panels and controls:**
- **Train panel:**
  - name, number and class
  - a histogram of the current delay distribution
  - each upcoming halt with schedule, our P10–P90 range and median, and NTES's ETA
  - a strip of past observations (dots where reports arrived)
- **Search** (train or station); **filters** (class, zone, delay ≥ X).
- **Status display:**
  - trains live
  - network-wide median delay
  - NTES requests per second
  - active model and its age
  - last frame age

**Performance:**
- Typed arrays throughout, and attributes are updated only for changed trains.
- If the frame rate drops below 30 fps for 3 s, drop to *K* = 8.

**Parity:** `web/src/position.ts` ports `pos()`. A shared fixture `tests/fixtures/pos_cases.json` is checked by both the Python and the TypeScript tests.

---

## 11. Nightly online update (`msynd.ops.nightly`, 02:30 IST)
1. **Segment bias:** over the last 7 days of live legs, compute the leftover error (actual Δd minus active-model median) per (segment, 3-hour bucket), shrunk toward 0 by `n/(n+20)` → `ml.segment_bias`. The engine shifts the matching CDFs by it. This picks up new engineering works or speed restrictions between weekly retrains.
2. Delete `obs.raw` older than 14 days.
3. Run `pg_dump` of `ref`, `obs` (without `raw`) and `ml` to `/data/backups`, keeping 7 days.
4. Refresh the passing-station lists and segments added by the live collector, and re-export the static network file if it changed.

---

## 12. Operations
- **Docker Compose,** all images arm64:
  - `restart: unless-stopped`, with health checks
  - JSON logs, rotated (10 MB × 5)
  - volumes `/data/pg`, `/data/models`, `/data/backups`, `/data/tiles`
- **Secrets:** `.env` on the VM (chmod 600), holding the Kaggle API token, basic-auth hash, DuckDNS token and rate settings. Only `.env.example` is committed.
- **Packages:** Python 3.12; `httpx`, `selectolax` (HTML parsing), `zstandard`, `psycopg[binary,pool]`, `numpy`, `numba`, `lightgbm`, `onnxruntime`, `pyarrow`, `fastapi`, `uvicorn`, `msgpack`, `kaggle`. The import step also uses `osmium-tool` (apt) and `networkx`. All have aarch64 wheels or packages.
- **Health and alerting (v1):** `/api/health` plus the status display. Optionally a free external uptime ping on `/api/health`.
- **Off-site backup:** the weekly Kaggle dataset version.

---

## 13. Testing strategy
- **Parsers:** fixture tests for every NTES and etrain page type, including the ones captured in Phase 0.
- **Rate limiter:** a simulated 10-minute run must never exceed the limit or the burst.
- **Leakage:** a property test on a synthetic day checks that no feature for a leg starting at *t* reads a row with time ≥ *t*.
- **Per-particle function:**
  - a zero-variance CDF reproduces the schedule exactly
  - an observation collapses the cloud to within `b` of the observed time
  - an impossible observation triggers a reset
  - multi-leg crossing in one call
  - **CPU/GPU parity** with identical random-number arrays (run on Kaggle)
- **`pos()` parity:** Python vs TypeScript on a shared fixture.
- **Engine rebuild:** rebuilding a run from observations gives exactly the same particles as the live engine did.
- **Replay-day end-to-end:** a recorded day of observations is fed through the collector and engine at 60× speed. Pass conditions:
  - S1 coverage ≥ 95 %
  - 10 s cycles within budget
  - divergence resets < 2 % of runs
  - no unhandled exceptions
- **Model gate:** the promotion rule in §7.4 is the acceptance test for every new model.

---

## 14. Delivery phases

Durations are rough, for one developer working part-time.

### Phase 0 — Spikes and setup (2–4 days)
1. Set up the Oracle A1 VM (Mumbai or Hyderabad), Docker, firewall and DuckDNS. Bring up Postgres and run the migrations.
2. **NTES spike:** capture request and response for running status, live station, exceptions, schedule and map spot, and save them as fixtures. Confirm:
   - the session/CSRF flow
   - **whether boards include actual times for trains that have departed**
   - whether running status lists passing stations
   - whether map spot returns a timestamped lat/lon
3. **etrain spike:** confirm the `train-history?train=N` redirect, the `d=1y` response, and what the delay number means (§7.1), using 20 trains.
4. **Kaggle spike:** API token; push a GPU "hello" script kernel from the VM with the CLI. Confirm the metadata keys, including the accelerator setting, and the output download. Also confirm that `torch`, `onnx`, `lightgbm` and Numba CUDA import with internet off. If any don't, either phone-verify the account and set `enable_internet: true`, or ship the wheels inside the dataset.
5. **Kernel benchmark gate:** a Numba prototype of §8.2 on the VM with 40M particles. Measure a full forward move, with 10 % of particles crossing legs, and folding in a 30-observation batch for 200 runs.

**Decision rules:**

| Spike finding | Decision |
|---|---|
| Boards have no actual times | Boards are used only for status and discovery. Targeted polls get 90 % of the budget (§6.1) |
| Running status has no passing stations | Fine-grained routes come from OSM paths. Segments are halt-to-halt until mapped |
| Map spot has no lat/lon | Drop `obs.position` |
| Forward move > 1 s or folding-in > 2 s | Port §8.2 to Rust before Phase 4 |
| etrain number turns out to be departure delay | Swap the meaning in §7.1 |

**Exit criteria:** fixtures committed, and every decision above recorded in §17 of this spec.

### Phase 1 — Static network and backfill (≈1 week)
- `static-import` (GTFS, datameet, OSM segments, routes) and `backfill`.
- **Exit:**
  - ≥ 95 % of GTFS trains have routes with segments, polyline or straight-line
  - ≥ 3,500 trains have ≥ 300 days of history

### Phase 2 — Live collector (≈1 week)
- Session handling, request limits, board set cover and sweep, the targeted queue (round-robin filler), exceptions, raw retention, run lifecycle.
- **Exit:** 72 h of continuous running with:
  - the S4 load limit met
  - < 1 % of hours with no data
  - parse failures < 1 %

### Phase 3 — Baseline model and evaluation harness (≈1 week)
- The shared features package (offline mode), LightGBM quantile models, and §7.5 with baselines B0–B1. B2 is added once live data exists.
- **Exit:** LightGBM beats B1 on per-leg CRPS and on 2 h rolled-forward median absolute error.

### Phase 4 — Engine v1 (≈1–2 weeks)
- The per-particle function, state layout, folding-in with rewind, coupling refresh, poll ranking feeding the collector, frames, forecast, rebuild on restart.
- A **debug map** (a plain HTML page with dots) for development.
- **Exit:**
  - the replay-day test passes
  - live on the VM for 7 days, meeting S1, S3 and S5
  - S2 against NTES ETAs over those 7 days

### Phase 5 — Kaggle GPU pipeline (≈1–2 weeks)
- Export and bundling, the main neural model, ONNX, distillation if needed, weekly automation, promotion rule, and the GPU replay kernel feeding `ml.filter_params`.
- **Exit:**
  - two unattended weekly cycles complete
  - the promotion decision is logged with metrics
  - the particle-count curve is produced

### Phase 6 — Dashboard (≈1–2 weeks)
- API endpoints, the WebSocket, PMTiles, the deck.gl layers, the timeline with three zones and speed controls, panels, search and filters.
- **Exit:**
  - S7 met
  - dragging from −24 h to +6 h is smooth
  - the LIVE button snaps back correctly

### Phase 7 — Nightly corrections and hardening (ongoing)
- §11, tuning the circuit breaker, and fixing whatever the first month of live running shows up.

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| NTES blocks us or changes its page format | Global limit and backoff; circuit breaker; raw responses kept for re-parsing; fixtures catch drift; lower `NTES_RATE`. A second source (etrain live status pages) is noted but not built in v1 |
| Terms of service / legal | Personal, non-commercial use; polite request rates; honest UA; no redistribution of data; not public. A public launch would need licensed data, which is outside this spec |
| Kaggle quota shrinks or a run fails | LightGBM is retrained on the VM every week regardless; the current model is kept on failure |
| Oracle reclaims an idle Always Free instance | The engine keeps CPU busy. If a reclaim warning comes, upgrade the account to Pay-As-You-Go: still free within Always Free limits and not subject to idle reclaim |
| GTFS schedules go stale (yearly timetable revisions, new or special trains) | Weekly NTES schedule refresh for affected trains; NTES running status takes priority for that run |
| OSM route mapping errors | Only affects drawing (prediction is 1-D); `ref.route_override` |
| Unclear meaning of etrain's delay number | Phase 0 check against NTES actual times |
| Unmodeled disruptions (breakdowns, accidents, protests) | Divergence reset plus an "uncertain" flag; recent-slowdown feature; nightly segment bias |
| Late and out-of-order observations | Checkpoint plus rewind folding-in (§8.3) |
| Diverted trains | Show the train at its last observation, flagged "diverted"; no simulation past the point of diversion in v1 |
| Model drift (January timetable revision, fog season) | Weekly retrain, time-split evaluation, the `fog_north` and calendar features |

---

## 16. Mumbai suburban (separate spec, not part of v1)
- **Facts:**
  - About 3,000+ services a day on Central and Western Railway.
  - Not in NTES live running; etrain excludes suburban history.
  - Yatri has GPS on these services through its arrangement with Central and Western Railway, but that data isn't public.
- **Modeling note for that spec:** with trains every 3–4 min and fixed corridors, state should live per **corridor segment** (a line-delay field), not per train.
- **Data options to evaluate there:**
  1. Approach Yatri, Central or Western Railway for access.
  2. Timetable plus a line-level delay estimated from public disruption announcements.
  3. Crowdsourced GPS from your own users. Not viable for a personal project.

---

## 17. Decisions log
- **Purpose:** personal / research / portfolio, so polite reading of public pages is acceptable.
- **Hosting:** Oracle Cloud Always Free Ampere A1, in an India region.
- **Live scope in v1:** all long-distance trains in India from day 1. Mumbai locals excluded (§16).
- **Collection:** hybrid, combining board sweeps and targeted polls ranked by uncertainty, under one global limit.
- **Stack:**
  - Python and Postgres.
  - Numba per-particle function, ported to Rust only if the Phase 0 benchmark fails.
  - LightGBM baseline, plus a neighborhood-attention neural model on Kaggle GPUs.
  - FastAPI.
  - deck.gl and MapLibre with self-hosted PMTiles.
  - Docker Compose.
- **Kaggle usage:** weekly retrain and sweep, distillation when needed, and monthly GPU replay to tune particle counts and filter settings.
- **Particles:** start at 10k per train, adapt per run (2k–50k), global cap 40M, tuned from GPU replay data.
- **Not used:** PostGIS, Redis/Kafka and engine checkpoints on disk.
- **Replay:** the past is replayed from ground truth. The last 30 min of what the filter believed is buffered in the browser. Server-side storage of past beliefs is deferred, to add only if past-belief replay is wanted.
- **NTES expected times:** stored as a baseline, never used as observations.
- **Phase 0 outcomes:** to be appended here when Phase 0 completes.
