import csv
import math
from pathlib import Path

from msynd.static.gtfs import ROOT

OVERRIDES = ROOT / "data/segment_overrides.csv"


def build_segments(conn, overrides_file=OVERRIDES):
    conn.execute("""
        INSERT INTO ref.segment(stn_lo,stn_hi,confidence)
        SELECT DISTINCT LEAST(a.station,b.station),GREATEST(a.station,b.station),0
        FROM ref.route_stop a JOIN ref.route_stop b
          ON b.route_version_id=a.route_version_id AND b.idx=a.idx+1
        ON CONFLICT(stn_lo,stn_hi) DO NOTHING
    """)
    conn.execute("""
        INSERT INTO ref.route_segment(route_version_id,idx,segment_id,forward)
        SELECT a.route_version_id,a.idx,s.id,a.station<=b.station
        FROM ref.route_stop a JOIN ref.route_stop b
          ON b.route_version_id=a.route_version_id AND b.idx=a.idx+1
        JOIN ref.segment s ON s.stn_lo=LEAST(a.station,b.station) AND s.stn_hi=GREATEST(a.station,b.station)
        ON CONFLICT(route_version_id,idx) DO NOTHING
    """)
    # Rebuild overrides from the versioned file, including removal of previous overrides.
    overrides = []
    with Path(overrides_file).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            confidence = float(row["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("override confidence must be in [0,1]")
            if row["stn_a"] > row["stn_b"]:
                raise ValueError("override polyline must run from canonical stn_a <= stn_b")
            overrides.append((row["polyline"] or None, confidence, row["stn_a"], row["stn_b"]))
    conn.execute("update ref.segment set polyline=NULL,confidence=0")
    for override in overrides:
        if conn.execute("update ref.segment set polyline=%s,confidence=%s where stn_lo=%s and stn_hi=%s", override).rowcount != 1:
            raise ValueError(f"override references unknown segment: {override[2:]}")
    return conn.execute("select count(*) from ref.segment").fetchone()[0]
