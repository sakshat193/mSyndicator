# m-syndicator — Live All-India Train Tracker: Design Spec (v2.2)

- **Date:** 2026-10-07
- **Status:** Draft v2.2, incorporating the third review; implementation gates remain unmeasured
- **What changed in v2.2:** signed O and two-sided R/O tails; effective event-time conditioning; per-key ETA confirmations; feasible, censored G2 cohorts; evidence-preserving fetch retention; per-key evidence merge; pass acceptance and replicate-based MC tolerance; versioned dirty acknowledgement after publication; API-read latency endpoint; restart spacing, parameter hashes, event-time matrices and partial-cancellation semantics. §19 maps R3 findings to amendments A1–A11.
- **What changed from v1:** this version answers [the adversarial review](../reviews/2026-10-07-live-train-tracker-adversarial-review.md). §19 maps each review finding to the section that resolves it.
- **What changed in v2.1** (second review, five blockers):
  - "exact conditioning" became block-wise weighted conditioning with a validation gate (§4.1, G7)
  - "not reported" evidence became a soft publication-lag likelihood (§3.3)
  - the as-of query now filters before picking the latest revision, and NTES ETAs record confirmations (§6)
  - O is now a residual against the rescheduled departure (§3.2)
  - frames are published right after each replay (§9.5)
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
- **Monte Carlo.** Each train is simulated as 10,000 "ghost trains".
  - At each station report, the ghosts are weighted by how well they explain the report, then resampled. Ghosts consistent with the report survive, along with their guesses about the unreported stretch before it.
  - "Not reported as of the last check" is soft evidence: it lowers the weight of ghosts that should already have been reported, using the measured publication-lag distribution.
  - A report fixes the train's state, so weights only live within the stretch between two reports. Degeneracy is limited to that stretch and is measured (§4.1). Correctness is checked against exact enumeration before any performance work (§15, G7).
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
| A6 | Engine latency (Oracle VM, all services running) | Time from evidence commit until the API process reads the delta frame containing it, server side before WebSocket send | p95 ≤ 5 s; full rebuild after a model change ≤ 3 min |
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
- The publication-lag distribution *S*: time from an event happening to it becoming visible, including its tail and the share of events never published.
- Whether any "not yet" display actually guarantees that the event hasn't happened.
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
- **Origin departure residual.** Model **O** predicts `actual dep − (scheduled dep + resched_min)`.
  - `resched_min` is the latest exception revision received **before** the actual departure was received; 0 if none.
  - Initialisation uses `scheduled + resched_min + O-draw`, so rescheduling is counted once.
  - "Rescheduled" is an O feature.
  - Regression case (second review): scheduled 100, rescheduled +20, actual 130 → label 10; initialisation centred on 130, not 150.
  - Signed-residual regression: scheduled 100, rescheduled +60, actual 150 → O = −10, and initialisation can draw 150. R and O use SIGNED bins; W alone is NONNEG (§3.4).
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
- **As-of rule.** Anything computed "as of *t*" (features, forecasts, baselines, evaluation) uses only revisions with `received_at ≤ t`. The query filters revisions **first**, then takes the latest eligible revision per key: `obs.event_asof(t)` in §6. Filtering the current-truth view would drop an event whose latest revision arrived after *t*.
- **Labels** use matured current revisions: the run is completed or expired and ≥ 6 h have passed since the event.
- **History rows** have no availability time. They supply only label statistics and features, are never replayed as a live stream, and any history-only backtest is labelled *oracle-latency* and never used for acceptance.
- **NTES baseline.** For each event key, use the latest `obs.ntes_eta_rev` row received ≤ *t*, eligible only if `t − received_at ≤ 30 min`. Every `ok` status response appends every ETA actually present, including unchanged values. Missing keys get no confirmation; token/session fetches never write ETA rows. `obs.fetch.run_id` measures poll age only. Issued forecasts preserve eligible B2 run-clock values in `sim.forecast_issue.b2` (NaN = ineligible), so evaluation does not depend on expired ETA rows (§6, §12).
- **Expected never becomes actual.** An actual requires the source's actual marker. A passed scheduled time does not count.
- **Not-reported evidence.** `obs.pending_rev` records "the source had **not reported** this event as of `received_at`". For example:
  - in running status: the next halt after the last report
  - on a board: a starred (expected) time

  This does **not** prove the event hasn't happened. It enters the engine only as a likelihood:

  `P(not reported by r | event at t) = 1` if t ≥ r, otherwise `S(r − t)`.

  - *S* is the publication-lag survival function measured in G2: the empirical lag distribution with an exponential tail, including the mass ε of events that are never published.
  - *S* is estimated per source (status page vs board).
  - A hard bound (S = 0 beyond some lag) requires an independently documented source guarantee. Zero counterexamples in a finite pilot never establish one. None is assumed.
  - **Merge per event key, as of t:** if any actual revision exists with `received_at ≤ t`, use the latest actual and ignore pending markers for that key. Otherwise use exactly one latest pending marker across sources; ties prefer `ntes_status`. Apply that source's S. Repeated markers are never multiplied; cross-source joint likelihood is not modelled (conservative). The repeated-pending witness gives 0.731059, not 0.880797.

### 3.4 Distribution format (the model contract)

**Bins**
- Each model returns a probability vector over fixed bins.
- **SIGNED (R and O):** edges at −60, −59, …, 240 min. That gives 300 unit bins, plus underflow (−∞, −60) and overflow [240, ∞): **302 bins**.
- **NONNEG (W only):** edges 0 … 180, giving 180 unit bins plus overflow [180, ∞): **181 bins**.

**Sampling** with a uniform *u*:
- Find the bin by inverse CDF, then place the value linearly inside the bin.
- SIGNED underflow draws `−60 − τ_lo·E`; overflow draws `240 + τ_hi·E`, with E ~ Exp(1). The physical floor applies afterwards. NONNEG overflow draws `180 + τ_hi·E`.
- Tail scales are fitted per class on training labels and stored in `ml.model.tail` as `{"<class>": {"lo": τ_lo, "hi": τ_hi}}`.

**Effective event-time distribution and truncation to [lo, hi)**
- Every event draw has form `T = b + a·max(f, c + s·Q_d(u))`: b is the particle's base time; a and s are positive pass scalings, otherwise 1; f is `60·km/V_MAX` for running, 0 for dwell, −∞ for origin; c is scheduled running minutes + bias for R, 0 for W, and `resched_min` for O (b is scheduled departure).
- `Q_d(u) = (1−w)·Q_j(u) + w·Q_{j+1}(u)` is the delay-interpolated quantile. `G(x) = P(T < x)` is the matching **left-limit CDF**, including floor atoms. Invert Q_d by bisection in the CDF coordinate below the median and the survival coordinate above it; use a closed form in joint exponential tails.
- Window mass `W = G(hi) − G(lo)`, evaluated as `S(lo) − S(hi)` in the upper half (never by subtracting `1 − F`). Here S is the effective event-time survival, distinct from publication-lag S in §3.3.
- Draw `T = Q_eff(G(lo) + u·W)` conditionally and clamp below hi. Use the survival coordinate in the upper half to preserve tiny positive masses.
- **Any W > 0 is valid**, however small; weight the particle by W. W = 0 gives only that particle zero weight. Only if every particle has zero weight does the flagged observation override apply (§4.1). There is no 1e-6 impossible-window threshold.
- Atom regression: D ~ U[0,10) with floor 5 gives `P(max(5,D) ∈ [5,6)) = 0.6`, not 0.1.

**LightGBM to CDF**
- Fit quantiles {0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98}, then sort them so they cannot cross.
- The CDF is piecewise linear between quantiles.
- Tails are exponential: `s_hi = max(0.5, (q.98 − q.90)/ln 5)` anchored at q.98, and `s_lo = max(0.5, (q.10 − q.02)/ln 5)` anchored at q.02 (minutes).
- Mass is rasterised into the bins. The excess beyond 240 becomes overflow with τ = s, which is consistent because the exponential is memoryless.

**Delay conditioning**
- Models are evaluated at 16 delay levels: `[-30,-10,0,5,10,15,20,30,45,60,90,120,180,240,360,600]`.
- A particle with delay *d* between levels L_j and L_{j+1} draws `(1−w)·Q_j(u) + w·Q_{j+1}(u)`, with the same *u* and `w = (d − L_j)/(L_{j+1} − L_j)`. Outside the range, the nearest level is used.
- Phase 3 measures the interpolation error against direct model calls.

---

## 4. Modelling approach

### 4.1 Block-wise weighted conditioning

R, W and O are first-order in the run's own delay. Their other features are known in advance: static data, calendar, and run context that has already been observed.

So an observed actual makes everything *before* it irrelevant to the future, up to the minute-display jitter. Weights are needed only within a **block**: the run of events from one piece of evidence up to and including the next event that carries evidence (an actual or a not-reported marker). The canonical event order is §3.2.

**For each block, in route order:**
1. **Propagate.** Draw the block's unreported events forward from each particle's state, using the prior (R / W / O).
2. **Weight at the evidence event *e*.**
   - **Actual displayed at minute *m*** (truncation per G2; rounding uses [m−½, m+½)):
     - draw *e* from the model truncated to [m, m+1)
     - weight = `G(m+1) − G(m)`, the effective prior mass including delay interpolation, shifts, scalings and floor atoms (§3.4); use stable survival subtraction in the upper half

     This is the locally optimal proposal, so after resampling, the unreported upstream times are posterior samples. Downstream truncation alone would not achieve that.
   - **Pass event at minute *m*.** Same rule. The pass also becomes the segment's anchor (§9.2).
   - **Not reported as of *r*.** Draw *e* from the prior and weight it by `P(not reported by r | e)` (§3.3).
3. **Resample** to equal weights with systematic resampling. Its single uniform comes from the counter RNG keyed by (run, block index). Ancestors keep their upstream draws.
4. **Monitor.**
   - Record ESS/N per block as a metric.
   - If ESS/N < 0.05, repeat the block with 4N proposals and resample back to N.
   - If every weight is 0 (the report is impossible under the model, e.g. faster than `V_MAX`), set *e* to the observation for all particles and raise `data_invariant`.

**Check (second review's example).** X ∈ {0, 10} and D ∈ {10, 20}, all equally likely. Evidence: not reported, with a hard survival so that only X + D > 15 survives.
- Weights: (0,10) gets 0; the other three get 1.
- So P(X = 10 | evidence) = 2/3, as the enumeration requires.

**Status.** As N → ∞ this converges to the model's exact posterior (sequential importance resampling). At finite N it is a Monte Carlo approximation, and its error is measured by G7 (§15) and the conditioning tests (§14). It is not claimed to be exact.

**Remaining model approximations**, each validated in G7 or Phase 3 rather than assumed:
- *ponytail:* a pass prior is the time-fraction point of the full segment R draw: `b = t_dep(A)`, `a = sched_to_pass/sched_run`, with s = 1, full-segment c and floor from §3.4. After pass k, draw `T_P + max(f_rem, sched_rem + r·(bias + Q_d′(u)))`, where `r = sched_rem/sched_run` and d′ is delay at the pass (slot 3+k). G7(c) compares this approximation against independent proportional sub-segment draws. If it fails, passes become display anchors only and are excluded from conditioning; record that decision in §18. Replace with a trained sub-segment model once live pass labels suffice.
- One publication-lag function *S* per source, not per station.

**Surprise.** Before weighting each actual, its PIT under the cloud is recorded. PIT < 0.001 or > 0.999 counts as a *surprise*. It flags the train "uncertain" for 30 min and feeds calibration monitoring.

### 4.2 Why many particles still matter

Every particle shares the observed past, but futures fan out over up to 40 segments. The P05/P95 forecasts and tail risks are Monte Carlo estimates, so more particles mean less Monte Carlo error.

- Default **N = 10,000 per run**.
- Phase 3 plots Monte Carlo error at P05/P50/P95 for N ∈ {1k, 5k, 10k, 50k, 100k}. The chosen N is the smallest whose MC error is ≤ 0.25 min at P05/P95 for the 6 h horizon.
- Particles exist only during a run's replay: scalar state plus an `N × E` float32 event-time matrix (4NE bytes), gathered on resampling. G8 measures both memory and replay CPU; no fixed 300 KB claim includes all upstream event times.

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
  run_id bigint,                           -- status poll age only; never confirms absent ETA keys
  started_at timestamptz NOT NULL, received_at timestamptz,
  http_status smallint,
  outcome text NOT NULL,                   -- inflight|ok|empty|invalid|transport_error|challenge
  body_zstd bytea);                        -- retention §12
CREATE INDEX ON obs.fetch (host, started_at);
CREATE INDEX ON obs.fetch (run_id, received_at) WHERE outcome = 'ok';

CREATE TABLE obs.event_rev (               -- actual times; insert when new or changed
  id bigserial PRIMARY KEY,
  run_id bigint NOT NULL REFERENCES obs.run,
  station text NOT NULL, occ smallint NOT NULL, kind obs.event_kind NOT NULL,
  actual_ts timestamptz NOT NULL,
  source obs.source NOT NULL,
  fetch_id bigint REFERENCES obs.fetch ON DELETE SET NULL,
  received_at timestamptz NOT NULL);
CREATE INDEX ON obs.event_rev (run_id, station, occ, kind, received_at);
CREATE INDEX ON obs.event_rev (received_at);

CREATE FUNCTION obs.event_asof(t timestamptz) RETURNS SETOF obs.event_rev
  LANGUAGE sql STABLE AS $$
  SELECT DISTINCT ON (run_id, station, occ, kind) *
  FROM obs.event_rev
  WHERE received_at <= t                   -- filter FIRST, then pick latest eligible revision
  ORDER BY run_id, station, occ, kind, received_at DESC, id DESC $$;

CREATE VIEW obs.event AS SELECT * FROM obs.event_asof('infinity');   -- current truth only; never filter this view by time

CREATE TABLE obs.pending_rev (             -- "not reported" as of received_at (a likelihood, §3.3)
  run_id bigint REFERENCES obs.run, station text, occ smallint, kind obs.event_kind,
  source obs.source NOT NULL,
  fetch_id bigint REFERENCES obs.fetch ON DELETE SET NULL, received_at timestamptz,
  PRIMARY KEY (run_id, station, occ, kind, received_at));

CREATE TABLE obs.ntes_eta_rev (            -- baseline only; append EVERY present ETA on each ok status page
                                           -- Freshness = this key's latest eligible received_at, including unchanged values
  run_id bigint REFERENCES obs.run, station text, occ smallint, kind obs.event_kind,
  expected_ts timestamptz NOT NULL,
  fetch_id bigint REFERENCES obs.fetch ON DELETE SET NULL, received_at timestamptz,
  PRIMARY KEY (run_id, station, occ, kind, received_at));

CREATE TABLE obs.exception_rev (
  train text, start_date date, detail jsonb NOT NULL,
  fetch_id bigint REFERENCES obs.fetch ON DELETE SET NULL, received_at timestamptz,
  PRIMARY KEY (train, start_date, received_at));

CREATE TABLE obs.history_daily (           -- quarantined until G4 records its meaning in §18
  train text, run_date date, station text, occ smallint,
  value_min smallint,                      -- NULL = cancelled/unknown
  fetch_id bigint REFERENCES obs.fetch ON DELETE SET NULL,
  PRIMARY KEY (train, run_date, station, occ));

CREATE SEQUENCE obs.dirty_seq;
CREATE TABLE obs.dirty_run (               -- acknowledged only after frame publication
  run_id bigint PRIMARY KEY,
  since timestamptz NOT NULL DEFAULT now(),
  ver bigint NOT NULL DEFAULT nextval('obs.dirty_seq')); -- every upsert sets a NEW ver

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
  tail jsonb NOT NULL,                     -- {"<class>": {"lo": τ_lo, "hi": τ_hi}} (§3.4)
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
  b2 real[] NOT NULL,                      -- one per halt, run clock; NaN = ineligible at issue
  PRIMARY KEY (run_id, issued_at));

CREATE TABLE sim.metric (
  ts timestamptz, name text, value double precision,
  PRIMARY KEY (ts, name));
```

**NOTIFY channels**
- `dirty`: wakes the engine.
- `sim_frame`: wakes the API.
- `model_changed`.

None of these is relied on for delivery. Durable state is in `obs.dirty_run` and `sim.frame`; listeners LISTEN, commit and reconcile at startup, then the engine polls every 1 s and the API every 5 s. Every FK to `obs.fetch` is nullable with `ON DELETE SET NULL`; retention never cascades into evidence.

---

## 7. Collector

### 7.1 The Dispatcher: the single owner of HTTP for each host
- **One owner per host.** At start it takes `pg_try_advisory_lock(hashtext('http:' || host))`. A second process for the same host refuses to start. Backfill, schedule refresh and exceptions are tasks inside the collector, not separate processes.
- **Spacing rule.** Request **starts** are ≥ `1/rate` apart, with no burst and one request in flight per host. At 1.0/s that guarantees ≤ 60 starts in any 60 s window (A5).
  - Every request goes through this rule and is logged in `obs.fetch`: pages, tokens, session pages, retries.
  - Insert `outcome = 'inflight'` at request start, update on completion. On startup wait until the last logged start + 1/rate, including interrupted requests.
  - Real httpx transports for NTES and etrain are built only with `MSYND_LIVE=1`; offline tests use `httpx.MockTransport`.
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

   `priority = P(next halt event is published by now) × width(P10..P90 of that event) + age/T_max`

   The probability is the cloud's mean of `1 − S(now − t)` (§3.3). This favours trains that probably have a new report waiting and whose estimate is wide.
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
- **Partial cancellation.** A cancelled tail ends simulation at the first cancelled halt; a cancelled head begins at the first operated halt with O applied to its scheduled departure. A cancelled middle is treated as a diversion.
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
| O | Origin departure residual against `scheduled + resched_min` (§3.2) | Live |
| C (cold start) | Composite `Δcomp` | History, only after G4 |

### 8.2 Features (`msynd/features`, versioned by `FEATURE_VER`)

One query is (run, segment *i*, delay level *d*). Everything except *d* is shared by all particles of the run.

| Group | Features |
|---|---|
| Train | class, train number (target-encoded, fold-local, smoothed), segment index, segments remaining, is-last |
| Segment | timetable km, scheduled running min, scheduled dwell at A, slack (scheduled running − fold-local P05 of observed running on this halt pair), zone at A |
| Calendar | hour of `sched_dep + d` (sin/cos), weekday, month, `fog_north` × northern zone, holiday ±1 day |
| Delay | *d*: departure delay at A for R; arrival delay for W |
| Observed run context | the run's last *observed* `Δrun` and how many segments ago it occurred. *Count of surprises so far is deferred to `FEATURE_VER` 2* (no as-of surprise store in v1; §18) |
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
  - **Head:** 302-way (R/O) or 181-way (W) softmax. Loss is cross-entropy on a Gaussian-smoothed target (σ = 1.5 bins) plus 0.5 × CRPS.
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
  - `params_hash`, covering per-source S, V_MAX and ESS threshold in `data/params.json`
- **Input:** a run's evidence, i.e. the current revisions of its events and pending markers.
- **Determinism.** *V* plus the evidence fully determines the run's output.

### 9.2 Per-particle state during replay

Structure-of-arrays plus an `N × E` float32 event-time matrix:

| Field | Type | Meaning |
|---|---|---|
| `seg` | int16 | current run segment |
| `t_arr_a` | f32 | arrival at A_i (departure for the origin) |
| `t_dep_a` | f32 | departure from A_i |
| `anchor_t` | f32 | last pass anchor time; `t_dep_a` if none |
| `anchor_km` | f32 | last pass anchor distance; `km(A_i)` if none |
| `t_arr_b` | f32 | arrival at A_{i+1} |
| `logw` | f32 | log-weight within the current block (§4.1); 0 after each resample |
| `times` | N × E f32 | full event-time matrix, gathered on every resample to preserve upstream paths |

*ponytail:* matrix gathers copy upstream times; use ancestor indices only if G8 identifies copies as a bottleneck.

**Draws.** Each event time comes from one draw per particle and block attempt:

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

`u = splitmix64(key(V, run_id) ⊕ pack(particle, segment, slot, k, attempt)) / 2⁶⁴`

- The generator is stateless and counter-based.
- Each resample's single uniform uses `pack(block, RESAMPLE)`.
- A replay is a pure function of (*V*, evidence). Results are therefore independent of batching, arrival order of evidence, and restarts.

### 9.4 Replaying a run

The function `replay(run) → summaries, display paths`:
1. **Load** *V*, the bound route, and revisions as of the replay cutoff. Merge each event key under §3.3: latest actual wins, otherwise one latest pending marker across sources (status wins ties). Order canonically: route index, then kind.
2. **Initialise** N particles. Origin departure = scheduled + `resched_min` + O-draw (O is the residual, §3.2). `resched_min` is as of the replay's evidence.
3. **Run the blocks of §4.1 in route order.** Each evidence event (actual or not-reported marker) closes a block with propagate, weight, resample. A not-reported marker downstream of the last actual is handled the same way, so it also reweights the unreported upstream draws.
4. **After the last evidence event:** draw forward without weights to the destination, or to `diverted_from`.
5. **Outputs:**
   - Per remaining halt: arrival and departure quantiles (P5, P10, P25, P50, P75, P90, P95) computed from **all N equal-weight particles**.
   - Per-run headline: next-halt quantiles, evidence state, last-event age, surprise flag, minimum block ESS/N.
   - **K = 32 display paths:** particles at indices `j·N/32`. Strided indices are used because systematic resampling clusters copies of an ancestor next to each other. Each path is the full event list to the destination.
6. **Discard** the particle arrays. The engine keeps only the outputs, about 10 KB per run.

**Cost estimate.** About 40 segments × N = 10k ≈ 0.4 M draws ≈ 10–20 ms per run on one core. A full rebuild of 2,500 runs on 4 cores takes about 10–15 s. Both are to be measured in G8. Weighting and resampling add O(N) per evidence event, which is small next to the draws.

### 9.5 Main loop

| Trigger | Work |
|---|---|
| Startup | LISTEN, commit, rebuild all active runs from the database, write a full frame |
| `NOTIFY dirty`, or every 1 s | Read `run_id, ver FROM obs.dirty_run ORDER BY since LIMIT 200`, replay, then in **one transaction** insert the delta frame, delete only rows whose `(run_id, ver)` still match, and `NOTIFY sim_frame`. A concurrent update with a new ver survives |
| Every 5 min | Append a **full frame** |
| Every 30 s | Recompute poll priorities and upsert `obs.poll_queue` |
| Hourly | Append `sim.forecast_issue` for running runs |
| `model_changed`, or a new bias generation | Mark all active runs dirty |

**Latency budget for A6 (p95 ≤ 5 s):**
- wake on NOTIFY, with the 1 s poll as fallback
- a consumed batch is capped at 200 runs (≈ 1 s of replay on 4 cores); the rest is left for the next pass
- frame write and API read (server side before WebSocket send)

Transient DB or IO failure acknowledges nothing, so dirty rows are retried. A deterministic replay exception publishes run state `error` and metric `replay_error`, then acknowledges the matching version in the publication transaction.

Idle notification test: evidence commit to API reading its delta < 2 s. With NOTIFY disabled: ≤ 1 s engine poll + 5 s API poll + replay time. This fallback bound is reported separately from the normal-notification A6 target.

A burst larger than that (a model change, a restart) is excluded from A6 and measured as "full rebuild".

**Time passing with no new evidence does not change a forecast.** The cloud moves along its precomputed paths, and the poll priority rises.

### 9.6 Model inference and cache
- **Per-run CDF cache**, keyed by (*V*, shared query features, segment, level). It covers every remaining segment × 16 levels × {R, W}, about 1,280 queries per run.
- **Batching.** Queries are batched per replay.
- **Rebuild trigger.** The cache is rebuilt when *V* or the run's observed context features change.
- **Bias.** Applied as a shift of the R CDF.
- **No network context in v1**, so future segments need only known features.

### 9.7 Performance gate (replaces "port to Rust")
- **Numba first.** `@njit` kernels for the draw and replay loops. Numba is already native code.
- **G8 profile.** On the Oracle VM, with all services running, profile each stage separately:
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
   | `obs.pending_rev` | 180 days |
   | `obs.ntes_eta_rev` | 35 days; issued B2 retained in `sim.forecast_issue.b2` |
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
- Python 3.12, `httpx`, `selectolax`, `zstandard`, `psycopg[binary,pool]`, `numpy`, `numba`, `lightgbm`, `onnxruntime`, `pyarrow`, `fastapi`, `uvicorn`, `websockets`, `msgpack`, `kaggle`
- `osmium-tool` and `networkx` for the import
- G5 smoke-tests the **pinned** image on the actual A1 VM.

**Storage budget.** These are estimates; G5 and Phase 2 replace them with measurements. The review measured uncompressed NTES pages at 0.4–1.4 MB.

| Item | Estimate |
|---|---|
| `obs.event_rev` with indexes | ~10 GB per year |
| Fetch bodies: 24 h of all `ok` responses at ~86k requests per day × zstd size | measure (≤ 15 GB assumed) |
| Fetch bodies: 14-day non-`ok` set plus sample | ≤ 5 GB |
| Forecast issues | ~3–4 GB (120 days) |
| Pending and ETA revisions | Measure Phase 2 row volume and indexed size; pending 180 days, per-key ETA confirmations 35 days |
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
| Conditioning (G7 suite) | Small enumerable models (§15 G7): engine posteriors match exact enumeration within MC tolerance, including the review's 2/3 case. Observed actuals: every particle inside the minute window. Invariants: speed ≤ `V_MAX`, dwell ≥ 0, monotone times |
| As-of query | Revision received 10:00 (time 09:50), corrected 10:30 (time 09:55): `event_asof(10:15)` returns 09:50; the current view returns 09:55 |
| NTES baseline | Unchanged-present ETA at 10:00 is eligible at 10:10; a missing key first seen 09:00 stays stale. Changed and post-cutoff confirmations, tokens and sessions tested separately |
| Rescheduling | Scheduled 100, rescheduled +20, actual 130 → O label 10; initialisation median ≈ 130. +60, actual 150 → signed O = −10 through labels, conversion, sampling and initialisation |
| Frame latency | Evidence committed → delta frame visible to the API in < 2 s with an idle engine |
| Distribution format | Overflow draws are finite and follow τ. LightGBM→CDF reproduces its quantiles to within half a bin. The interpolation test |
| Dispatcher | Simulated clock with tokens, retries and session recovery: ≤ 60 starts in any 60 s window. A second process cannot take the host lock |
| Validation | Empty 200 → `empty`. Identity mismatch → `invalid`. An expected time is never stored as an actual |
| WebSocket | A missed `gen` triggers a full resync. Slow-client coalescing. A tombstone removes a run |
| `pos()` parity | Python and TypeScript on the shared fixture |
| Restore | Dump, restore into a fresh container, rebuild; frames equal the original's for the same evidence |
| Retention | Deleting an old referenced fetch sets all evidence fetch_id references NULL without deleting evidence |
| Replay day (end to end) | A recorded day of fetch bodies goes through parsers and engine at 60×. Checks: A1, A6, `invalid` < 1 %, no unhandled exceptions |
| Model gate | §8.6 promotion is the acceptance test for every model |

---

## 15. Delivery phases

Each gate ends in pass, fail or unavailable, with fixtures or measurements and a decision recorded in §18. **Fixtures alone are not an exit.**

### Phase 0 — Contract and capacity gates (≈ 1–2 weeks, mostly waiting on pilot data)

| Gate | Work | If it fails |
|---|---|---|
| **G1 Data use** | Record terms and licences, with retrieval date, in `docs/sources.md`: NTES (incl. the Disclaimer page), etrain, the GTFS repo, OSM (ODbL), Protomaps. Cover automated collection, model training, export to Kaggle, and portfolio display | Change source or scope. *User decision*: e.g. no Kaggle export (train on the VM only), or no etrain backfill |
| **G2 NTES protocol and capacity** | 7-day pilot at <= 0.3 req/s. At least 40 distinct runs over the week in rotating cohorts; simultaneous n_sim <= floor(0.7 * 0.3 * 120 / h), i.e. 12 at h = 2, reserving 30 % for boards, exceptions, sessions and retries. Measure h, token lifetime, empty-200 behaviour, per-source publication-lag S using interval-censored lags; unfinished runs are right-censored, not never-published. Report epsilon with a 95 % Wilson CI. A hard bound requires an independently documented source guarantee, never zero pilot counterexamples. Rerun G7(d) at the epsilon CI upper bound and tail scale x2. Measure board actual retention on >= 30 train-station pairs (busy/quiet, midnight, late), board yield, pending semantics, passing prevalence (50 trains) and hourly active-run count | Sets h, S per source, board share, T_max and F. Unacceptable freshness is a user decision: accept, change rate/source or reduce scope |
| **G3 Fixtures** | Committed fixtures for every page type, including edge cases: cancelled, diverted, rescheduled, multi-day, empty | — |
| **G4 etrain meaning** | After G2 has ≥ 2 weeks of live NTES data and etrain covers those dates: for ≥ 20 trains, compare etrain values against NTES actual arrivals and departures where they differ by ≥ 5 min. Cover origin, intermediate long dwell, destination, overnight and multi-day, early running, non-daily, repeated stations. Settle `run_date` semantics | History stays quarantined. Cold start uses the live empirical fallback (§3.2). Not blocking: G4 may finish during Phase 3 |
| **G5 Platform** | Get the Oracle A1 allocation. Run the pinned arm64 image smoke test (imports, a LightGBM and ONNX round-trip). Measure the storage budget from G2 bodies. Run a restore drill into a fresh container | Fix the image, or use the other region. Development continues on the user's PC meanwhile |
| **G6 Kaggle** (only if G1 allows export) | Private dataset and verified privacy. CLI push, status and output. GPU confirmed. `torch`/`onnx`/`lightgbm` import offline (otherwise phone-verify and enable internet, or ship wheels in the dataset) | Train on the VM only; drop the neural candidate |
| **G7 Conditioning correctness** (before G8) | Real replay on 2-4 segment models, 2-5 outcomes per draw, N = 10k and 100k. Use 20 independent replicates salted into the RNG key; compare >= 5 CDF grid points, discrete marginals and continuous quantiles. Tolerance = 4*sqrt(se_rep**2 + se_ref**2), se_rep = sd/sqrt(20). Reference: exact enumeration for bin-aligned conditions; otherwise iid likelihood weighting over >= 2 M prior samples with computed SE, a different algorithm from the engine. Cases: (a) upstream 2/3; (b) actual bridge; (c) pass versus independent proportional sub-segment reference: next-halt P10/P50/P90 error <= max(2 min, 10 % of reference P10-P90 width), MC spread separate from mean approximation error; failure excludes passes from conditioning and is logged in section 18; (d) soft publication lag with long tail and epsilon > 0, plus epsilon CI upper bound and tail x2; (e) infeasible floor override, flagged, no NaNs; (f) three mixed blocks; (g) ESS fallback; also tiny positive windows, floor atoms, repeated/cross-source pending and actual precedence | Fix conditioning; Phase 4 waits. Failed pass approximation uses display anchors only |
| **G8 Engine profile** | Prototype §9.4 with LightGBM on synthetic runs, on the VM with all services running: per-run replay p50/p95/p99, full rebuild, frame build, evidence-commit-to-API-read latency | Apply §9.7 to the measured bottleneck |

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
  - Per-key ETA confirmation row volume and indexed storage size measured; replace the provisional storage budget.

### Phase 3 — Baseline models and evaluation (runs while live data accumulates)
- Features with the causal rules, LightGBM R/W/O, the empirical fallbacks, and C if G4 passes.
- The §8.6 harness with B0–B2. The Monte Carlo error curve picks *N*. Delay-level interpolation error measured.
- **Exit** (needs ≥ 4 weeks of live data): R/W beat B1 on CRPS at the 2 h horizon (CI > 0), with coverage within ±5 points.

### Phase 4 — Engine (≈ 1–2 weeks; can start on the cold-start models)
- Replay, block-wise conditioning (must pass G7), not-reported likelihood, dirty set, determinism, cache, summaries, frame log, poll priorities, issued forecasts, and a debug map.
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
- **Filtering:** block-wise weighted conditioning (sequential importance resampling that resets at each actual), N = 10k by default and set by the Monte Carlo error curve. Validated against enumeration (G7). "Not reported" is a soft likelihood from the measured publication lag.
- **Origin model:** a residual against the rescheduled departure.
- **Frames:** a delta is published right after each replay batch, plus a full frame every 5 min.
- **Recovery:** each dirty run is replayed in full. Durable dirty set and frame log. Counter-based random numbers.
- **No position observations:** the NTES map gives station markers, not GPS.
- **Coupling:** a gated experiment (C1, then C2).
- **Performance:** Numba plus a profile gate. Rust only for a measured kernel bottleneck.
- **Models:** LightGBM on the VM always. A neural candidate on Kaggle only if G1/G6 pass and it wins the promotion test.
- **Backup:** off-site to Oracle Object Storage. Kaggle is not a backup.
- **Phase 0 outcomes:** *to be appended here (G1–G8 results and decisions).*
  - **G7 (2026-10-07): pass.** All 13 cases plus both A4 sensitivity variants within A8 tolerance at N = 10k and 100k (`docs/gates/G7.md`). A7 accepted: P10/P50/P90 error ≤ 0.49 min against a 2 min limit, so passes stay in conditioning. A second pass chains from the previous pass (remainder anchored at Pᵢ; formula in G7.md).
- **Implementation decisions (2026-10-08, user):**
  - Features v1 drop "count of surprises so far"; it returns in `FEATURE_VER` 2 once surprises are stored with receipt times.
  - A training row's as-of cutoff is the `received_at` of the segment's departure event at A (the live moment a run forecast is issued).
  - W scheduled-dwell buckets are data-driven: quintile edges of scheduled dwell over the fold's training rows (duplicate edges merged), stored in the artifact. An empty or unseen bucket falls back to the class-pooled histogram, then the provisional prior.
  - `websockets` becomes a runtime dependency as uvicorn's WebSocket backend.
  - Model tables carry one tail pair per segment (R) and one scale per halt (W): the CDF cache takes the max of the per-level LightGBM scales. Per-level τ in the kernel is the upgrade path.

---

## 19. Review resolution map

| Review finding | Resolution |
|---|---|
| Phase 0: board retention | G2 measurement; boards limited to discovery and pending evidence if they fail (§7.2) |
| Phase 0: map GPS | Dropped; no position observations (§2, §18) |
| Phase 0: etrain meaning | G4 with discriminating cases; quarantine; never an R/W label (§3.2) |
| Phase 0: performance / Rust | Correctness gate G7, then profile gate G8; §9.7 |
| Collection arithmetic | Capacity stated (§7.2); *h* measured; fairness via *T_max*; rate change is a user decision |
| 1 Mixed clocks | §3.2 contract plus regression test |
| 2 Availability leakage | `received_at` revisions, as-of rule, NTES baseline age, fold-local fitting (§3.3, §8.3) |
| 3 Mutable sequences | Immutable schedule and route versions; (station, occ) keys; explicit route→segment membership (§3.1, §6) |
| 4 Checkpoint and late data / NOTIFY | Full replay per dirty run; transactional dirty set; LISTEN-then-reconcile (§9.4–9.5) |
| 5 State and likelihood contract | Explicit dwell; event-directed draws; floor in minutes with `V_MAX`; weights only within a block, no roughening, ESS fallback instead of an underflow reset; no position likelihood (§4.1, §9.2) |
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

**Second review (v2 → v2.1)**

| Finding | Resolution |
|---|---|
| R2-1 "Exact conditioning" was false | Block-wise weighting with locally optimal proposals and resampling; claimed only as consistent Monte Carlo; G7 enumeration gate (§4.1, §15) |
| R2-2 Pending evidence was too strong | Soft likelihood `S(r − t)` with tail and ε; hard bounds only for sources with a verified guarantee; G2 tests "not reported" vs "not happened" (§3.3, §15) |
| R2-3 As-of SQL order; ETA staleness | `obs.event_asof(t)` filters before picking the latest revision; ETA freshness from `ok` status fetches (§3.3, §6, §14) |
| R2-4 Rescheduling counted twice | O is a residual against `scheduled + resched_min`, with a regression test (§3.2, §8.1, §9.4, §14) |
| R2-5 Frame cadence vs A6 | Delta published right after each replay batch; 1 s fallback poll; batch cap; latency budget (§9.5) |

**Third review (v2.1 -> v2.2), R3 amendments**

| Finding / addition | Amendment and resolution |
|---|---|
| R3-1 Effective conditional law | A2: left-limit effective CDF including floors, scaling, bias and quantile interpolation; stable upper-tail mass; any positive window valid (section 3.4, section 4.1) |
| R3-2 Signed origin residual | A1: SIGNED 302 bins for R/O, NONNEG 181 for W; two exponential tails, floored LightGBM scales and matching neural heads (section 3.2, section 3.4, section 6, section 8.4) |
| R3-3 False ETA refresh | A3: append every present ETA per key; missing keys never refresh; 35-day retention and issued B2 array; measured row volume (section 3.3, section 6, sections 12-15) |
| R3-4 Pilot capacity and tail | A4: rotating >= 40 distinct runs, bounded simultaneous cohort, 30 % reserve, censoring, Wilson CI, independent guarantee for hard bound, sensitivity (section 3.3, section 15 G2/G7) |
| R3-5 Fetch retention | A5: every fetch FK nullable ON DELETE SET NULL, 180-day fetch rows, no cascade into evidence (section 6, section 12, section 14) |
| Repeated/cross-source pending | A6: latest actual wins, otherwise one latest marker across sources, status tie-break, source-specific S (section 3.3, section 6, section 9.4) |
| Pass approximation | A7: defined time-fraction prior and remainder, quantitative next-halt acceptance, display-only fallback (section 4.1, section 15 G7) |
| Monte Carlo tolerance | A8: 20 salted replicates, CDFs/marginals/quantiles, combined SE, independent reference (section 15 G7) |
| Dirty batches | A9: fresh sequence ver on upsert, bounded read, matching-version acknowledgement in frame/NOTIFY transaction, transient retry and deterministic error state (section 6, section 9.5) |
| Latency endpoint | A10: evidence commit to server-side API read; idle NOTIFY < 2 s, no-NOTIFY <= 1+5 s polls plus replay (section 1 A6, section 9.5, section 10, section 14) |
| Restart, versions, state, cancellation | A11: inflight start logging and restart spacing; params_hash; gathered N x E matrix; cancelled head/tail/middle rules (section 6, section 7.1, section 7.3, sections 9.1-9.2) |
