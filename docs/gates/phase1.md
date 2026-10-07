# Phase 1 static network measurement

Measured locally on 2026-10-07, PostgreSQL 16, using the pinned source revisions in `data/static_sources.json`.

| Measurement | Result |
|---|---:|
| GTFS train inventory | 10,594 |
| Schedule versions | 10,593 |
| Trains with drawable geometry | 10,593 (99.9906%) |
| Stations | 8,550 |
| Missing station coordinates | 0 |
| Missing calendar IDs decoded | 154 |
| Trips skipped for unparseable missing calendars | 0 |
| Trips skipped for inverted validity | 1 (`00816`, 2099-12-31 to 2027-08-30) |
| Undirected drawing segments | 15,949 |
| Geometry confidence 0 (straight lines) | 15,949 |
| Route distance source `gtfs` | 10,593 (100%) |

The first import preceded calendar decoding; the follow-up added 204 versions and deduplicated 10,389, for 10,593 total. All 154 missing IDs parsed. Operating days are decoded only when an explicit calendar row is absent. Inverted validity is retained as a reported source defect, not corrected by guessing.

The ≥95% drawable/schedule condition passes locally at confidence 0. OSM geometry enrichment and history completeness are not measured. The GTFS redistribution licence remains unclear (G1); this report grants no data-use permission.

Commands: `uv run python -m msynd.static.gtfs import`, then a pinned `--zip` re-import, then `uv run python -m msynd.static.export report`, with `DATABASE_URL` pointing to the local database.
