ALTER TABLE obs.fetch
  ADD COLUMN mode text NOT NULL DEFAULT 'poll' CHECK (mode IN ('poll','harvest','probe','backfill')),
  ADD COLUMN flags text[] NOT NULL DEFAULT '{}';
