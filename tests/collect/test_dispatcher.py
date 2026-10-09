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


def test_mmt_403_stops_host_and_logs_challenge(db_url):
    from msynd.collect.mmt import classify

    async def run():
        d = dispatcher(db_url, FakeClock(), lambda r: httpx.Response(403, content=b"forbidden"), classify=classify)
        await d.start()
        try:
            with pytest.raises(HostStopped):
                await d.request("status", "POST", "https://h/x", key="mmt-403")
            with pytest.raises(HostStopped):
                await d.request("status", "POST", "https://h/x")
            assert d.state == "stopped"
            assert d.conn.execute("select http_status,outcome from obs.fetch where key='mmt-403' order by id desc limit 1").fetchone() == (403, "challenge")
        finally:
            await d.close()

    asyncio.run(run())


def test_404_without_classifier_is_invalid(db_url):
    async def run():
        d = dispatcher(db_url, FakeClock(), lambda r: httpx.Response(404, content=b"not found"))
        await d.start()
        try:
            result = await d.request("status", "GET", "https://h/x")
            assert result.outcome == "invalid" and result.status == 404
            assert d.state == "ok"
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


@pytest.fixture
def budget_url(db_url):
    with connect(db_url) as conn:
        conn.execute("delete from obs.fetch where host='h'")
    return db_url


def test_budget_counts_every_attempt_including_retries(budget_url):
    from msynd.collect.dispatcher import BudgetExhausted

    async def run():
        starts = []
        def handle(request):
            starts.append(1)
            raise httpx.ReadTimeout('offline timeout')
        d = dispatcher(budget_url, FakeClock(), handle, budgets={'poll': 3})
        await d.start()
        try:
            with pytest.raises(BudgetExhausted) as exc:
                await d.request('status', 'GET', 'https://h/x')
            assert exc.value.mode == 'poll'
            assert starts == [1, 1, 1]
            assert d.conn.execute("select mode,outcome from obs.fetch where host='h'").fetchall() == [
                ('poll', 'transport_error')] * 3
        finally:
            await d.close()
    asyncio.run(run())


def test_budget_survives_restart(budget_url):
    from msynd.collect.dispatcher import BudgetExhausted

    async def run():
        clock = FakeClock()
        for restart in range(2):
            d = dispatcher(budget_url, clock, lambda r: httpx.Response(200, content=b'ok'), budgets={'poll': 1})
            await d.start()
            try:
                if restart:
                    with pytest.raises(BudgetExhausted):
                        await d.request('status', 'GET', 'https://h/x')
                    assert d.conn.execute("select count(*) from obs.fetch where host='h'").fetchone() == (1,)
                else:
                    await d.request('status', 'GET', 'https://h/x')
            finally:
                await d.close()
    asyncio.run(run())


@pytest.mark.parametrize('after_wait', [False, True])
def test_budget_resets_at_ist_midnight(budget_url, after_wait):
    async def run():
        clock = FakeClock()
        clock.base = datetime(2026, 10, 9, 18, 29, 59, tzinfo=timezone.utc)
        d = dispatcher(budget_url, clock, lambda r: httpx.Response(200, content=b'ok'), budgets={'poll': 1})
        await d.start()
        try:
            await d.request('status', 'GET', 'https://h/x')
            if not after_wait:
                clock.t = 61  # 00:01 IST on October 10, still October 9 UTC.
            await d.request('status', 'GET', 'https://h/x')
            assert d.conn.execute("select count(*) from obs.fetch where host='h'").fetchone() == (2,)
        finally:
            await d.close()
    asyncio.run(run())


def test_budget_per_mode(budget_url):
    from msynd.collect.dispatcher import BudgetExhausted

    async def run():
        d = dispatcher(budget_url, FakeClock(), lambda r: httpx.Response(200, content=b'ok'),
                       budgets={'harvest': 1, 'poll': 1})
        await d.start()
        try:
            await d.request('status', 'GET', 'https://h/x', mode='harvest')
            with pytest.raises(BudgetExhausted) as exc:
                await d.request('status', 'GET', 'https://h/x', mode='harvest')
            assert exc.value.mode == 'harvest'
            await d.request('status', 'GET', 'https://h/x')
            assert d.conn.execute("select mode from obs.fetch where host='h' order by id").fetchall() == [('harvest',), ('poll',)]
        finally:
            await d.close()
    asyncio.run(run())


def test_missing_mode_budget_is_zero(budget_url):
    from msynd.collect.dispatcher import BudgetExhausted

    async def run():
        d = dispatcher(budget_url, FakeClock(), lambda r: pytest.fail('budget must prevent transport'), budgets={})
        await d.start()
        try:
            with pytest.raises(BudgetExhausted):
                await d.request('status', 'GET', 'https://h/x')
            assert d.conn.execute("select count(*) from obs.fetch where host='h'").fetchone() == (0,)
        finally:
            await d.close()
    asyncio.run(run())


def test_no_budgets_unlimited(budget_url):
    async def run():
        d = dispatcher(budget_url, FakeClock(), lambda r: httpx.Response(200, content=b'ok'))
        await d.start()
        try:
            for mode in ('poll', 'harvest', 'probe', 'backfill'):
                await d.request('status', 'GET', 'https://h/x', mode=mode)
            assert d.conn.execute("select mode from obs.fetch where host='h' order by id").fetchall() == [
                ('poll',), ('harvest',), ('probe',), ('backfill',)]
        finally:
            await d.close()
    asyncio.run(run())


@pytest.mark.parametrize('max_attempts', [1, 2, 5])
@pytest.mark.parametrize('status', [503, 200])
def test_max_attempts_caps_retries(budget_url, max_attempts, status):
    async def run():
        d = dispatcher(budget_url, FakeClock(), lambda r: httpx.Response(status))
        await d.start()
        try:
            result = await d.request('status', 'GET', 'https://h/x', max_attempts=max_attempts)
            assert result.outcome == ('empty' if status == 200 else 'transport_error')
            assert d.conn.execute("select count(*) from obs.fetch where host='h'").fetchone() == (
                min(max_attempts, 2) if status == 200 else max_attempts,)
        finally:
            await d.close()
    asyncio.run(run())


@pytest.mark.parametrize('max_attempts', [0, 6, 1.5, True])
def test_invalid_max_attempts_rejected(budget_url, max_attempts):
    async def run():
        d = dispatcher(budget_url, FakeClock(), lambda r: pytest.fail('invalid request must not be sent'))
        await d.start()
        try:
            with pytest.raises(ValueError):
                await d.request('status', 'GET', 'https://h/x', max_attempts=max_attempts)
        finally:
            await d.close()
    asyncio.run(run())
