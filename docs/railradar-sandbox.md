# RailRadar sandbox integration (offline only)

The client and parser are prepared against the published response shape. No account was created and no RailRadar API request was sent. The implementation plan remains unchanged; the monitor owns revisions to Tasks 7–11 and 20–25.

- API documentation: https://railradar.in/docs
- Live response shape: https://railradar.in/docs/live-train-status
- Terms: https://railradar.in/terms
- Documented sandbox quota: 1,000 requests/month. This is a development pilot, not all-India collection capacity.

`RailRadarClient` uses the existing per-host dispatcher, a Bearer key, an explicit origin date, and one GET per status enquiry. It sends `authoritative=false` and retains passing stations. `from_env()` constructs a real transport only through the `MSYND_LIVE` guard; it requires `RAILRADAR_API_KEY`. Do not enable live mode for this offline task.

`parse_status(body, train, start_date)` accepts only the requested train and origin date, timezone-aware timestamps, monotone routes/times and explicit halt markers. It emits actual station events, pending markers, passing-station anchors and exception details. Event identity uses `(station, occurrence, kind)`, independent of route sequence numbers. Position telemetry and delay fields never become actual events. The documented live response does not define per-event ETA timestamps, so no ETA is inferred.

The synthetic fixture is labelled in `tests/fixtures/railradar/README.md`. Run the offline checks with:

```powershell
docker compose up -d db
$env:MSYND_TEST_DSN = 'postgresql://msynd:msynd@127.0.0.1:55432/postgres'
uv run pytest -q tests/collect/test_railradar.py
```

The sandbox scheduler, account-wide quota accounting, persistent event storage, production scope, retention/model-training permission, pilot and API-key provisioning belong to the monitor's revised tasks. No live entrypoint or scheduler is added here. Provider terms grant API use but do not explicitly settle historical retention, model training or third-party export.
