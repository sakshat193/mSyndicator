CREATE TABLE obs.pilot_bridge_cursor (
  singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  last_fetch_id bigint NOT NULL DEFAULT 0 CHECK (last_fetch_id >= 0)
);
