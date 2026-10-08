"""Pure parser for the captured MMT livestatus/v2 response."""

from datetime import datetime, time, timedelta, timezone
import json
import math
import re

from msynd.collect.railradar import IST, validate_request
from msynd.collect.status import Eta, Event, InvalidResponse, ParsedStatus, PassingStation, Pending


def _date(value):
    return datetime.strptime(value, "%d-%m-%Y %H:%M:%S").date()


def _clock(value, day):
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"\d{2}:\d{2}", value) is None:
        raise InvalidResponse("invalid clock time")
    return datetime.combine(day, time.fromisoformat(value), IST).astimezone(timezone.utc)


def validate(times):
    """Flagged actuals must remain monotone in halt page order."""
    if times != sorted(times):
        raise InvalidResponse("non-monotone event times")


def parse_status(body: bytes, train, start) -> ParsedStatus:
    validate_request(train, start)
    try:
        envelope = json.loads(body)
        if not isinstance(envelope, dict) or envelope.get("success") is not True:
            raise InvalidResponse("unsuccessful response")
        data = envelope["response"]
        details = data["trainDetails"]
        if details["trainNumber"] != train or _date(details["startDate"]) != start:
            raise InvalidResponse("response is for another train or origin date")
        departed, terminated = details["departed"], details["terminated"]
        if type(departed) is not bool or type(terminated) is not bool or (terminated and not departed):
            raise InvalidResponse("invalid lifecycle markers")
        updated = datetime.strptime(data["lastUpdated"], "%d-%m-%Y %H:%M:%S").replace(tzinfo=IST).astimezone(timezone.utc)
        halts = data["stations"]
        if not isinstance(halts, list) or not halts:
            raise InvalidResponse("missing route")
        events, pending, etas, passing, flags, occurrences = [], [], [], [], set(), {}
        rows = []
        intermediates = []
        for halt in halts:
            distance = float(halt["distance"])
            if not math.isfinite(distance) or distance < 0:
                raise InvalidResponse("invalid halt distance")
            rows.append((distance, True, halt))
            children = halt.get("intermediateStations") or []
            if not isinstance(children, list):
                raise InvalidResponse("invalid passing stations")
            for row in children:
                distance = float(row["distance"])
                if not math.isfinite(distance) or distance < 0:
                    raise InvalidResponse("invalid passing distance")
                intermediates.append((distance, False, row))
        rows.extend(sorted(intermediates, key=lambda r: r[0]))
        base = datetime.combine(start, time(), IST)
        for distance, halt, row in rows:
            station = row["station"]["code"]
            if not isinstance(station, str) or re.fullmatch(r"[A-Z0-9-]{1,16}", station) is None:
                raise InvalidResponse("invalid station code")
            occurrences[station] = occurrences.get(station, 0) + 1
            occ = occurrences[station]
            day = row["dayDetails"]["dayCnt"] if halt else row["day"]
            if type(day) is not int or day < 1:
                raise InvalidResponse("invalid route day")
            scheduled_day = start + timedelta(days=day-1)
            scheduled = []
            for kind, prefix, key, flag in (("arr", "Arr", "arrivalDetails", "arrived"),
                                            ("dep", "Dep", "departureDetails", "departed")):
                detail = row[key]
                marker = detail[flag]
                if marker is not None and type(marker) is not bool:
                    raise InvalidResponse("invalid actual marker")
                sched = _clock(detail[f"sch{prefix}Time"], scheduled_day)
                scheduled.append(sched)
                # sch*Date follows the actual's date once a delay crosses midnight (live 11004, 11020); dayCnt is the schedule.
                value = detail[f"act{prefix}Time"]
                actual_day = detail[f"act{prefix}Date"]
                if value is not None and actual_day is None:
                    raise InvalidResponse("time without date")
                observed = _clock(value, _date(actual_day)) if value is not None else None
                if marker is True and observed is None:
                    raise InvalidResponse("actual marker without time")
                delay = detail.get(f"{kind}Delay")
                if delay is not None:
                    if type(delay) not in (int, float) or not math.isfinite(delay):
                        raise InvalidResponse("invalid delay")
                    if delay > 1440:
                        flags.add("delay_gt_24h")
                if observed is not None and sched is not None and (observed-sched).total_seconds() > 86400:
                    flags.add("delay_gt_24h")
                if halt and not (row is halts[0] and kind == "arr"):
                    if marker is True:
                        events.append(Event(station, occ, kind, observed))
                    elif marker is False:
                        if sched is not None:
                            pending.append(Pending(station, occ, kind))
                        if observed is not None:
                            etas.append(Eta(station, occ, kind, observed))
                    elif observed is not None or sched is not None:
                        raise InvalidResponse("halt time without explicit marker")
            if not halt and any(t is not None for t in scheduled):
                sched = scheduled[1] or scheduled[0]
                passing.append(PassingStation(station, occ, distance, (sched-base).total_seconds()/60))
        # ponytail: unmapped; upgrade: map to the Task 21 detail form once a real cancelled body is captured.
        if details.get("cancelledFromStation") is not None or details.get("cancelledToStation") is not None:
            flags.add("exception_unparsed")
        parsed = ParsedStatus(train, start, "completed" if terminated else "running" if departed else "not_started",
                              updated, tuple(events), tuple(pending), tuple(passing), (), tuple(sorted(flags)), tuple(etas))
        validate([event.actual_ts for event in events])
        return parsed
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, InvalidResponse):
            raise
        raise InvalidResponse("invalid MMT status schema") from error
