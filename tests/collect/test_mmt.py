import asyncio
from datetime import date, datetime, timezone
import json

import httpx
import pytest

from msynd.collect.dispatcher import Dispatcher
from msynd.config import Settings
from msynd.db import connect
from tests.collect.test_dispatcher import FakeClock


@pytest.mark.db
def test_single_post_with_origin_date_and_no_next_running_date(db_url):
    from msynd.collect.mmt import HOST, MmtClient

    async def run():
        requests = []

        def handle(request):
            requests.append(request)
            assert request.method == "POST"
            assert str(request.url) == "https://rails-ris.makemytrip.com/api/ris/train/livestatus/v2"
            assert json.loads(request.content) == {
                "trainNumber": "12951", "dateOfJourney": "07-10-2026", "findNextRunningDate": False,
            }
            assert request.headers["Content-Type"] == "application/json"
            assert request.headers["User-Agent"].startswith("m-syndicator/")
            return httpx.Response(200, json={"data": {}})

        d = Dispatcher(HOST, .2, lambda: connect(db_url), httpx.MockTransport(handle), clock=FakeClock())
        client = MmtClient(d)
        await d.start()
        try:
            result = await client.train_running("12951", date(2026, 10, 7))
            assert result.outcome == "ok" and len(requests) == 1
            with connect(db_url) as c:
                assert c.execute("select purpose,key,outcome from obs.fetch where id=%s", (result.fetch_id,)).fetchone() == (
                    "status", "12951/2026-10-07", "ok",
                )
        finally:
            await d.close()

    asyncio.run(run())


def test_factory_requires_live_flag(monkeypatch):
    from msynd.collect.mmt import MmtClient

    monkeypatch.delenv("MSYND_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="MSYND_LIVE"):
        MmtClient.from_env()


@pytest.mark.parametrize("status,body,headers,want", [
    (200, b'{"data": {}}', {}, "ok"),
    (403, b'{"error": "forbidden"}', {}, "challenge"),
    (200, b' <!DOCTYPE html><html>login</html>', {}, "challenge"),
    (200, b'<html>login</html>', {}, "challenge"),
    (200, b'login', {"Content-Type": "text/html; charset=utf-8"}, "challenge"),
    (200, b'{"captcha": true}', {}, "challenge"),
    (200, b'CAPTCHA required', {}, "challenge"),
    (200, b'{broken', {}, "invalid"),
    (404, b'{"error": "not found"}', {}, "invalid"),
])
def test_classification(status, body, headers, want):
    from msynd.collect.mmt import classify

    assert classify("status", httpx.Response(status, content=body, headers=headers)) == want


def test_only_status_purpose_is_accepted():
    from msynd.collect.mmt import classify

    assert classify("token", httpx.Response(200, json={"data": {}})) == "invalid"


def test_shared_types_preserve_railradar_and_default_etas():
    from msynd.collect import railradar, status

    for name in ("InvalidResponse", "Event", "Pending", "PassingStation", "ParsedStatus"):
        assert getattr(railradar, name) is getattr(status, name)
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    parsed = status.ParsedStatus("12951", date(2026, 10, 7), "running", now, (), (), (), (), ())
    assert parsed.etas == ()
    eta = status.Eta("NDLS", 1, "arr", now)
    assert (eta.station, eta.occ, eta.kind, eta.eta_ts) == ("NDLS", 1, "arr", now)


def test_mmt_rate_from_env(monkeypatch):
    monkeypatch.delenv("MMT_RATE", raising=False)
    assert Settings.from_env().mmt_rate == .2
    monkeypatch.setenv("MMT_RATE", ".1")
    assert Settings.from_env().mmt_rate == .1


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf"])
def test_mmt_rate_rejects_invalid_values(monkeypatch, rate):
    monkeypatch.setenv("MMT_RATE", rate)
    with pytest.raises(ValueError):
        Settings.from_env()


def test_client_validates_host_and_request():
    from msynd.collect.mmt import HOST, MmtClient

    async def run():
        d = Dispatcher("wrong-host", .2, None, httpx.MockTransport(lambda r: httpx.Response(200)))
        try:
            with pytest.raises(ValueError):
                MmtClient(d)
            d.host = HOST
            client = MmtClient(d)
            for train, start in [("1295", date(2026, 10, 7)), (12951, date(2026, 10, 7)), ("12951", "2026-10-07")]:
                with pytest.raises(ValueError):
                    await client.train_running(train, start)
        finally:
            await d.close()

    asyncio.run(run())
