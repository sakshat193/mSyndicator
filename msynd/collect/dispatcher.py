import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import os
import random
import time

import httpx
import zstandard

from msynd import VERSION


class HostLocked(Exception):
    pass


class HostStopped(Exception):
    pass


class RealClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(asyncio.sleep)

    @staticmethod
    def now():
        return datetime.now(timezone.utc)


@dataclass
class FetchResult:
    fetch_id: int
    outcome: str
    status: int | None
    body: bytes
    received_at: datetime


def real_transport():
    if os.environ.get("MSYND_LIVE") != "1":
        raise RuntimeError("real source traffic requires MSYND_LIVE=1 and checkpoint approval")
    return httpx.AsyncHTTPTransport(retries=0)


class Dispatcher:
    def __init__(self, host, rate, conn_factory, transport, clock=None, rand=None, classify=None, on_empty=None):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive")
        self.host, self.rate = host, rate
        self.conn_factory = conn_factory
        self.clock = clock or RealClock()
        self.rand = rand or random.Random()
        self.classify, self.on_empty = classify, on_empty
        self.client = httpx.AsyncClient(transport=transport, headers={"User-Agent": f"m-syndicator/{VERSION} (personal research)"}, timeout=30, follow_redirects=False)
        self.lock = asyncio.Lock()
        self.conn = None
        self.last_start = -math.inf
        self.state = "ok"
        self.outcomes = deque()
        self.pause_until = 0.0

    async def start(self):
        if self.conn is not None:
            return
        conn = self.conn_factory()
        conn.autocommit = True
        try:
            if not conn.execute("select pg_try_advisory_lock(hashtext(%s))", ("http:" + self.host,)).fetchone()[0]:
                raise HostLocked(self.host)
            latest = conn.execute("select max(started_at) from obs.fetch where host=%s", (self.host,)).fetchone()[0]
            if latest is not None:
                wait = (latest - self.clock.now()).total_seconds() + 1 / self.rate
                if wait > 0:
                    await self.clock.sleep(wait)
            self.conn = conn
        except BaseException:
            conn.close()
            raise

    async def close(self):
        await self.client.aclose()
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _retry_after(self, response):
        value = response.headers.get("Retry-After", "0")
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - self.clock.now()).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return 0
        return max(0, seconds) if math.isfinite(seconds) else 0

    async def _attempt(self, purpose, method, url, key, run_id, kwargs):
        async with self.lock:
            if self.conn is None:
                raise RuntimeError("start the dispatcher first")
            if self.state == "stopped":
                raise HostStopped(self.host)
            wait = max(self.last_start + 1 / self.rate, self.pause_until) - self.clock.monotonic()
            if wait > 0:
                await self.clock.sleep(wait)
            if self.state == "breaker":
                self.state = "ok"
                self.outcomes.clear()
            self.last_start = self.clock.monotonic()
            fetch_id = self.conn.execute("insert into obs.fetch(host,purpose,key,run_id,started_at,outcome) values (%s,%s,%s,%s,%s,'inflight') returning id", (self.host, purpose, key, run_id, self.clock.now())).fetchone()[0]
            status, body, retry_after = None, b"", 0
            cancelled = None
            try:
                response = await self.client.request(method, url, **kwargs)
                status, body = response.status_code, response.content
                if status == 429 or status >= 500:
                    outcome, retry_after = "transport_error", self._retry_after(response)
                elif status == 200 and not body.strip():
                    outcome = "empty"
                elif not 200 <= status < 300:
                    outcome = "invalid"
                else:
                    outcome = self.classify(purpose, response) if self.classify else "ok"
                    if outcome not in ("ok", "invalid", "challenge"):
                        raise ValueError("classifier returned an invalid outcome")
            except httpx.TransportError:
                outcome = "transport_error"
            except BaseException as exc:
                outcome, cancelled = "invalid", exc
            received = self.clock.now()
            self.conn.execute("update obs.fetch set received_at=%s,http_status=%s,outcome=%s,body_zstd=%s where id=%s", (received, status, outcome, zstandard.compress(body), fetch_id))
            now = self.clock.monotonic()
            self.outcomes.append((now, outcome))
            while self.outcomes and self.outcomes[0][0] < now - 300:
                self.outcomes.popleft()
            if outcome == "challenge":
                self.state = "stopped"
                raise HostStopped(self.host)
            if len(self.outcomes) >= 20 and sum(o != "ok" for _, o in self.outcomes) / len(self.outcomes) > .2:
                self.state, self.pause_until = "breaker", now + 600
            if cancelled is not None:
                raise cancelled
            return FetchResult(fetch_id, outcome, status, body, received), retry_after

    async def request(self, purpose, method, url, *, key=None, run_id=None, **httpx_kwargs):
        if httpx.URL(url).host != self.host:
            raise ValueError("dispatcher URL must match its locked host")
        if httpx_kwargs.get("follow_redirects"):
            raise ValueError("redirect requests must be dispatched and logged individually")
        refreshed = False
        for attempt in range(5):
            result, retry_after = await self._attempt(purpose, method, url, key, run_id, httpx_kwargs)
            if result.outcome == "empty" and not refreshed:
                refreshed = True
                if self.on_empty is not None:
                    await self.on_empty()
                continue
            if result.outcome != "transport_error" or attempt == 4:
                return result
            await self.clock.sleep(max(retry_after, min(300, 2 * 2**attempt) * self.rand.uniform(.5, 1.5)))
        return result
