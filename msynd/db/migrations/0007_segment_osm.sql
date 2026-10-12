CREATE TABLE ref.segment_osm (
  stn_lo text NOT NULL,
  stn_hi text NOT NULL,
  polyline text NOT NULL,
  confidence real NOT NULL,
  PRIMARY KEY (stn_lo, stn_hi),
  CHECK (stn_lo <= stn_hi),
  CHECK (confidence > 0 AND confidence <= 1)
);
