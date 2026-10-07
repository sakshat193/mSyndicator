import asyncio
from datetime import datetime, timedelta, timezone
from itertools import cycle

import httpx
import pytest

from msynd.db import connect
from msynd.collect.dispatcher import Dispatcher, HostLocked, HostStopped, real_transport

pytestmark = pytest.mark.db


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.base = datetime.now(timezone.utc)

    def monotonic(self):
        return self.t

    def now(self):
        return self.base + timedelta(seconds=self.t)

    async def sleep(self, seconds):
        self.t += seconds


def dispatcher(url, clock, handler, **kwargs):
    return Dispatcher("h", 1, lambda: connect(url), httpx.MockTransport(handler), clock=clock, **kwargs)


def test_starts_spaced_with_tokens_retries_sessions(db_url):
    async def run():
        clock = FakeClock()
        script = cycle([httpx.Response(200, content=b"tok"), httpx.Response(503, headers={"Retry-After": "3"}), httpx.Response(200, content=b""), httpx.Response(200, content=b"<ok/>")])
        starts = []

        def handle(request):
            starts.append(clock.t)
            assert request.headers["User-Agent"].startswith("m-syndicator/")
            return next(script)

        d = dispatcher(db_url, clock, handle)
        await d.start()
        try:
            for i in range(150):
                await d.request("status" if i % 2 else "token", "GET", "https://h/x")
            assert all(b - a >= 1 - 1e-9 for a, b in zip(starts, starts[1:]))
            assert max(sum(0 <= b - a < 60 for b in starts) for a in starts) <= 60
        finally:
            await d.close()

    asyncio.run(run())


def test_second_process_cannot_lock(db_url):
    async def run():
        d = dispatcher(db_url, FakeClock(), lambda r: httpx.Response(200, content=b"ok"))
        other = dispatcher(db_url, FakeClock(), lambda r: httpx.Response(200, content=b"ok"))
        await d.start()
        try:
            with pytest.raises(HostLocked):
                await other.start()
        finally:
            await d.close()
            await other.close()
    asyncio.run(run())


def test_challenge_stops_host(db_url):
    async def run():
        d = dispatcher(db_url, FakeClock(), lambda r: httpx.Response(200, content=b"captcha"), classify=lambda p, r: "challenge")
        await d.start()
        try:
            with pytest.raises(HostStopped):
                await d.request("status", "GET", "https://h/x")
            with pytest.raises(HostStopped):
                await d.request("token", "GET", "https://h/x")
            assert d.state == "stopped"
        finally:
            await d.close()
    asyncio.run(run())


def test_breaker_pauses_10_min(db_url):
    async def run():
        clock = FakeClock()
        d = dispatcher(db_url, clock, lambda r: httpx.Response(200, content=b"bad"), classify=lambda p, r: "invalid")
        await d.start()
        try:
            for _ in range(20):
                await d.request("status", "GET", "https://h/x")
            assert d.state == "breaker"
            before = clock.t
            await d.request("status", "GET", "https://h/x")
            assert clock.t - before >= 600
        finally:
            await d.close()
    asyncio.run(run())


def test_empty_refreshes_session_once_then_retries_once(db_url):
    async def run():
        clock = FakeClock()
        calls = []
        script = iter([httpx.Response(200), httpx.Response(200, content=b"session"), httpx.Response(200, content=b"ok")])
        d = dispatcher(db_url, clock, lambda r: next(script))

        async def refresh():
            calls.append(1)
            await d.request("session", "GET", "https://h/")

        d.on_empty = refresh
        await d.start()
        try:
            result = await asyncio.wait_for(d.request("status", "GET", "https://h/x"), timeout=5)
            assert result.outcome == "ok" and calls == [1]
        finally:
            await d.close()
    asyncio.run(run())


def test_empty_session_recovery_does_not_recurse(db_url):
    async def run():
        requests, refreshes = [], []

        def handle(request):
            requests.append(request)
            if len(requests) > 10:
                raise AssertionError("session recovery recursed")
            return httpx.Response(200)

        d = dispatcher(db_url, FakeClock(), handle)

        async def refresh():
            refreshes.append(1)
            await d.request("session", "GET", "https://h/")

        d.on_empty = refresh
        await d.start()
        try:
            result = await asyncio.wait_for(d.request("status", "GET", "https://h/x"), timeout=5)
            assert result.outcome == "empty" and len(requests) == 4 and refreshes == [1]
        finally:
            await d.close()
    asyncio.run(run())


def test_every_request_logged_with_outcome(db_url):
    async def run():
        script = iter([httpx.ReadTimeout("timeout"), httpx.Response(200, content=b"ok")])
        ids = []

        def handle(request):
            with connect(db_url) as c:
                ids.append(c.execute("select id from obs.fetch where host='h' and outcome='inflight'").fetchone()[0])
            response = next(script)
            if isinstance(response, Exception):
                raise response
            return response

        d = dispatcher(db_url, FakeClock(), handle)
        await d.start()
        try:
            result = await d.request("status", "GET", "https://h/x")
            with connect(db_url) as c:
                assert c.execute("select outcome from obs.fetch where id=any(%s) order by id", (ids,)).fetchall() == [("transport_error",), ("ok",)]
            assert result.fetch_id == ids[-1]
        finally:
            await d.close()
    asyncio.run(run())


def test_real_transport_requires_live_flag(monkeypatch):
    monkeypatch.delenv("MSYND_LIVE", raising=False)
    with pytest.raises(RuntimeError):
        real_transport()


def test_restart_respects_last_start(db_url):
    async def run():
        clock = FakeClock()
        with connect(db_url) as c:
            c.execute("delete from obs.fetch where host='h'")
            c.execute("insert into obs.fetch(host,purpose,started_at,outcome) values ('h','token',%s,'inflight')", (clock.now() - timedelta(seconds=.2),))
        starts = []
        d = dispatcher(db_url, clock, lambda r: (starts.append(clock.t), httpx.Response(200, content=b"ok"))[1])
        await d.start()
        try:
            await d.request("status", "GET", "https://h/x")
            assert starts[0] >= .8 - 1e-6
        finally:
            await d.close()
    asyncio.run(run())
