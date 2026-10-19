CREATE TABLE ref.station_osm (
  code text PRIMARY KEY,
  lon double precision NOT NULL CHECK (lon BETWEEN 68 AND 98),
  lat double precision NOT NULL CHECK (lat BETWEEN 6 AND 37)
);
