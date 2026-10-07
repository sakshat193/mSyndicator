-- Keep simultaneous markers from different sources for the A6 merge rule.
ALTER TABLE obs.pending_rev DROP CONSTRAINT pending_rev_pkey;
ALTER TABLE obs.pending_rev ADD PRIMARY KEY (run_id, station, occ, kind, received_at, source);
