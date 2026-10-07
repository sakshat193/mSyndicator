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
  dist_src text NOT NULL CHECK (dist_src IN ('gtfs','gc','ntes')),
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
  outcome text NOT NULL CHECK (outcome IN ('inflight','ok','empty','invalid','transport_error','challenge')),                   -- inflight|ok|empty|invalid|transport_error|challenge
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
  cold_start boolean NOT NULL DEFAULT false,
  b2 real[] NOT NULL,                      -- one per halt, run clock; NaN = ineligible at issue
  PRIMARY KEY (run_id, issued_at));

CREATE TABLE sim.metric (
  ts timestamptz, name text, value double precision,
  PRIMARY KEY (ts, name));
