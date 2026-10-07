# Offline RailRadar fixture

`running.json` is a hand-written fixture based on the documented envelope and route fields at https://railradar.in/docs/live-train-status (read 2026-10-07). Dates were adjusted to exercise an IST midnight crossing. It is not a captured API response and contains no account, token, cookie or real telemetry.

The parser tests modify this fixture for wrong-run responses, repeated stations, passing events, invalid timestamps and extreme delays. No test sends network traffic.
