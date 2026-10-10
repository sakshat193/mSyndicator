import asyncio
from datetime import date, datetime
import json
from pathlib import Path

import httpx
import pytest

from msynd.collect.railradar import RailRadarClient, parse_status, InvalidResponse
from msynd.collect.dispatcher import Dispatcher
from msynd.db import connect
from tests.collect.test_dispatcher import FakeClock

FIXTURE = Path(__file__).parents[1] / "fixtures/railradar/running.json"


def payload():
    return json.loads(FIXTURE.read_text())


def test_documented_status_actuals_and_pending():
    status = parse_status(FIXTURE.read_bytes(), "12919", date(2026, 10, 7))
    assert [(e.station, e.occ, e.kind) for e in status.events] == [("INDB", 1, "dep"), ("UJN", 1, "arr"), ("UJN", 1, "dep")]
    assert [(e.station, e.kind) for e in status.pending] == [("MKSM", "arr"), ("MKSM", "dep")]
    assert status.events[0].actual_ts == datetime.fromisoformat("2026-10-08T00:07:00+05:30")
    assert status.start_date == date(2026, 10, 7)


@pytest.mark.parametrize("change", ["train", "date", "naive", "shape", "success", "order", "distance", "is_halt"])
def test_invalid_response_rejected(change):
    p = payload()
    if change == "train": p["data"]["trainNumber"] = "12920"
    if change == "date": p["data"]["startDate"] = "2026-10-06"
    if change == "naive": p["data"]["route"][0]["actualDeparture"] = "2026-10-08T00:07:00"
    if change == "shape": p["data"]["route"] = {}
    if change == "success": p["success"] = False
    if change == "order": p["data"]["route"][1]["actualArrival"] = "2026-10-07T22:00:00+05:30"
    if change == "distance": p["data"]["route"][1]["distance"] = -1
    if change == "is_halt": p["data"]["route"][1]["isHalt"] = "false"
    with pytest.raises(InvalidResponse):
        parse_status(json.dumps(p).encode(), "12919", date(2026, 10, 7))


def test_repeated_station_and_passing_keys():
    p = payload()
    route = p["data"]["route"]
    route[1]["stationCode"] = "INDB"
    route[1]["isHalt"] = False
    route[1]["actualArrival"] = route[1]["actualDeparture"]
    route[1]["scheduledArrival"] = route[1]["scheduledDeparture"]
    parsed = parse_status(json.dumps(p).encode(), "12919", date(2026, 10, 7))
    assert [(e.station, e.occ, e.kind) for e in parsed.events] == [("INDB", 1, "dep"), ("INDB", 2, "pass")]
    assert parsed.passing_stations[0].occ == 2
    passing = parsed.passing_stations[0]
    assert passing.after == ("INDB", 1) and passing.before == ("MKSM", 1)
    assert passing.fraction == pytest.approx(55 / 96)


def test_unbounded_passing_stations_dropped():
    p = payload()
    for row in p["data"]["route"][::2]:
        row["isHalt"] = False
    parsed = parse_status(json.dumps(p).encode(), "12919", date(2026, 10, 7))
    assert parsed.passing_stations == ()


def test_passing_station_equal_source_halt_km():
    p = payload()
    route = p["data"]["route"]
    route[1]["isHalt"] = False
    route[1]["actualArrival"] = route[1]["actualDeparture"]
    for row in route:
        row["distance"] = 0
    passing = parse_status(json.dumps(p).encode(), "12919", date(2026, 10, 7)).passing_stations[0]
    assert passing.after == ("INDB", 1) and passing.before == ("MKSM", 1)
    assert passing.fraction == 0


def test_delay_fields_and_position_never_become_actual():
    p = payload()
    p["data"]["route"][2]["delayArrival"] = 10
    p["data"]["route"][2]["delayDeparture"] = 12
    parsed = parse_status(json.dumps(p).encode(), "12919", date(2026, 10, 7))
    assert all(e.station != "MKSM" for e in parsed.events)
    assert len(parsed.events) == 3  # GPS/currentLocation is not station-time evidence.


def test_extreme_delay_kept_flagged():
    p = payload()
    p["data"]["route"] = [p["data"]["route"][0]]
    p["data"]["route"][0]["actualDeparture"] = "2026-10-09T01:00:00+05:30"
    parsed = parse_status(json.dumps(p).encode(), "12919", date(2026, 10, 7))
    assert len(parsed.events) == 1 and "delay_over_24h" in parsed.flags


@pytest.mark.db
def test_client_single_authenticated_request_and_wrong_run_invalid(db_url):
    async def run():
        requests = []

        def handle(request):
            requests.append(request)
            assert request.headers["Authorization"] == "Bearer offline-test-key"
            assert request.url.path == "/v1/trains/12919/live"
            assert dict(request.url.params) == {"date": "2026-10-07", "authoritative": "false", "haltsOnly": "false"}
            response = payload()
            if len(requests) == 2:
                response["data"]["startDate"] = "2026-10-06"
            return httpx.Response(200, json=response)

        d = Dispatcher("api.railradar.in", .3, lambda: connect(db_url), httpx.MockTransport(handle), clock=FakeClock())
        client = RailRadarClient(d, "offline-test-key")
        await d.start()
        try:
            result = await client.train_running("12919", date(2026, 10, 7))
            assert result.outcome == "ok" and len(requests) == 1
            invalid = await client.train_running("12919", date(2026, 10, 7))
            assert invalid.outcome == "invalid" and len(requests) == 2
            with connect(db_url) as c:
                assert c.execute("select outcome from obs.fetch where id=%s", (invalid.fetch_id,)).fetchone() == ("invalid",)
        finally:
            await d.close()
    asyncio.run(run())


def test_factory_live_guard_and_no_key_leak(monkeypatch):
    monkeypatch.delenv("MSYND_LIVE", raising=False)
    monkeypatch.setenv("RAILRADAR_API_KEY", "secret-test-key")
    with pytest.raises(RuntimeError, match="MSYND_LIVE") as err:
        RailRadarClient.from_env()
    assert "secret-test-key" not in str(err.value)


def test_request_boundary_validation():
    with pytest.raises(ValueError):
        RailRadarClient(None, "")


def test_non_2xx_with_valid_body_is_invalid():
    from msynd.collect.railradar import classify

    request = httpx.Request("GET", "https://api.railradar.in/v1/trains/12919/live?date=2026-10-07")
    assert classify("status", httpx.Response(200, json=payload(), request=request)) == "ok"
    assert classify("status", httpx.Response(404, json=payload(), request=request)) == "invalid"
