from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
import re

import httpx

from msynd.collect.dispatcher import Dispatcher, real_transport
from msynd.db import connect

HOST = "api.railradar.in"
BASE_URL = f"https://{HOST}/v1"
IST = timezone(timedelta(hours=5, minutes=30))


class InvalidResponse(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    station: str
    occ: int
    kind: str
    actual_ts: datetime


@dataclass(frozen=True)
class Pending:
    station: str
    occ: int
    kind: str


@dataclass(frozen=True)
class PassingStation:
    station: str
    occ: int
    dist_km: float
    sched_off: float | None


@dataclass(frozen=True)
class ParsedStatus:
    train: str
    start_date: date
    status: str
    last_updated_at: datetime
    events: tuple[Event, ...]
    pending: tuple[Pending, ...]
    passing_stations: tuple[PassingStation, ...]
    exceptions: tuple[dict, ...]
    flags: tuple[str, ...]


def timestamp(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidResponse("timestamp must be an ISO string")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise InvalidResponse("timestamp must include its UTC offset")
    return result


def validate_request(train, start):
    if not isinstance(train, str) or re.fullmatch(r"[0-9]{5}", train) is None or type(start) is not date:
        raise ValueError("require a five-digit train number and origin date")


def parse_status(body, train, start):
    """Documented /trains/{number}/live shape; delay/location fields are not actuals."""
    validate_request(train, start)
    try:
        response = json.loads(body)
        if not isinstance(response, dict) or response.get("success") is not True:
            raise InvalidResponse("unsuccessful response")
        data = response["data"]
        if not isinstance(data, dict) or data["trainNumber"] != train or date.fromisoformat(data["startDate"]) != start:
            raise InvalidResponse("response is for another train or origin date")
        updated = timestamp(data["lastUpdatedAt"])
        if updated is None or not isinstance(data["status"], str):
            raise InvalidResponse("missing source update time or lifecycle")
        route = data["route"]
        if not isinstance(route, list) or (not route and data["status"] != "cancelled"):
            raise InvalidResponse("missing route")
        exceptions = data.get("exceptions", [])
        if not isinstance(exceptions, list) or any(not isinstance(item, dict) for item in exceptions):
            raise InvalidResponse("invalid exceptions")
        events, pending, passing, flags, occurrences = [], [], [], set(), {}
        base = datetime.combine(start, time(), IST)
        last_sequence, last_distance, last_actual, last_scheduled = -1, -math.inf, None, None
        for row in route:
            if not isinstance(row, dict):
                raise InvalidResponse("invalid route row")
            station, sequence, halt = row["stationCode"], row["sequence"], row["isHalt"]
            if not isinstance(station, str) or re.fullmatch(r"[A-Z0-9-]{1,16}", station) is None:
                raise InvalidResponse("invalid station code")
            if type(sequence) is not int or sequence <= last_sequence or type(halt) is not bool:
                raise InvalidResponse("invalid route sequence or halt marker")
            distance = row["distance"]
            if type(distance) not in (int, float) or not math.isfinite(distance) or distance < max(0, last_distance):
                raise InvalidResponse("non-monotone route distance")
            last_sequence, last_distance = sequence, distance
            occurrences[station] = occurrences.get(station, 0)+1
            occ = occurrences[station]
            scheduled = [timestamp(row.get("scheduledArrival")), timestamp(row.get("scheduledDeparture"))]
            actual = [timestamp(row.get("actualArrival")), timestamp(row.get("actualDeparture"))]
            for ts in scheduled:
                if ts is not None:
                    if last_scheduled is not None and ts < last_scheduled:
                        raise InvalidResponse("non-monotone scheduled times")
                    last_scheduled = ts
            for ts in actual:
                if ts is not None:
                    if last_actual is not None and ts < last_actual:
                        raise InvalidResponse("non-monotone actual times")
                    last_actual = ts
            for observed, sched in zip(actual, scheduled):
                if observed is not None and sched is not None and (observed-sched).total_seconds() > 86400:
                    flags.add("delay_over_24h")
            if halt:
                for kind, observed, sched in zip(("arr", "dep"), actual, scheduled):
                    if observed is not None:
                        events.append(Event(station, occ, kind, observed))
                    elif sched is not None and row.get("status") == "upcoming":
                        pending.append(Pending(station, occ, kind))
            else:
                if all(actual) and actual[0] != actual[1]:
                    raise InvalidResponse("passing station has conflicting actual times")
                sched = scheduled[1] or scheduled[0]
                passing.append(PassingStation(station, occ, float(distance), (sched-base).total_seconds()/60 if sched else None))
                observed = actual[1] or actual[0]
                if observed is not None:
                    events.append(Event(station, occ, "pass", observed))
                elif sched is not None and row.get("status") == "upcoming":
                    pending.append(Pending(station, occ, "pass"))
        return ParsedStatus(train, start, data["status"], updated, tuple(events), tuple(pending), tuple(passing), tuple(exceptions), tuple(sorted(flags)))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, InvalidResponse):
            raise
        raise InvalidResponse("invalid RailRadar status schema") from error


def classify(purpose, response):
    if purpose != "status":
        return "invalid"
    try:
        train = response.request.url.path.split("/")[-2]
        start = date.fromisoformat(response.request.url.params["date"])
        parse_status(response.content, train, start)
        return "ok"
    except (InvalidResponse, ValueError, KeyError, IndexError):
        return "invalid"


class RailRadarClient:
    def __init__(self, dispatcher, api_key):
        if not isinstance(api_key, str) or not api_key or any(c.isspace() for c in api_key):
            raise ValueError("a RailRadar API key is required")
        if dispatcher.host != HOST:
            raise ValueError("RailRadar client requires its own host dispatcher")
        self.d, self._api_key = dispatcher, api_key
        if self.d.classify is not None and self.d.classify is not classify:
            raise ValueError("RailRadar dispatcher requires its status classifier")
        self.d.classify = classify

    @classmethod
    def from_env(cls):
        # Construction of the real transport is guarded before any account/key or DB access.
        transport = real_transport()
        return cls(Dispatcher(HOST, .3, connect, transport, classify=classify), os.environ.get("RAILRADAR_API_KEY", ""))

    async def train_running(self, train, start, run_id=None):
        validate_request(train, start)
        return await self.d.request("status", "GET", f"{BASE_URL}/trains/{train}/live", key=f"{train}/{start.isoformat()}", run_id=run_id,
            params={"date": start.isoformat(), "authoritative": "false", "haltsOnly": "false"},
            headers={"Authorization": f"Bearer {self._api_key}"})
