from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from msynd.collect.status import InvalidResponse
from msynd.collect.parse_mmt import parse_status

FIXTURES = Path(__file__).parents[1] / "fixtures" / "mmt"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


def loaded(label="running_12723"):
    return json.loads((FIXTURES / f"{label}.json").read_text())


def parse_copy(body, train="12723", start=date(2026, 10, 8)):
    return parse_status(json.dumps(body).encode(), train, start)


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda e: e["label"])
def test_fixture_golden(entry):
    expected = json.loads((FIXTURES / f"{entry['label']}.expected.json").read_text())
    body = (FIXTURES / f"{entry['label']}.json").read_bytes()
    if expected == {"invalid": True}:
        with pytest.raises(InvalidResponse):
            parse_status(body, entry["train"], date.fromisoformat(entry["date"]))
    else:
        result = parse_status(body, entry["train"], date.fromisoformat(entry["date"]))
        assert json.loads(json.dumps(asdict(result), default=lambda v: v.isoformat())) == expected


def test_status_for_other_start_date_is_invalid():
    with pytest.raises(InvalidResponse):
        parse_copy(loaded(), start=date(2026, 10, 7))
    with pytest.raises(InvalidResponse):
        parse_copy(loaded(), train="12951")


def test_multiday_offsets_monotone():
    result = parse_copy(loaded("five_day_22503"), "22503", date(2026, 10, 5))
    times = [e.actual_ts for e in result.events]
    assert times == sorted(times)
    assert max(e.eta_ts for e in result.etas).astimezone(timezone(timedelta(hours=5, minutes=30))).date() == date(2026, 10, 9)


def test_repeated_station_occ():
    result = parse_copy(loaded("repeated_station_11014"), "11014")
    arrivals = [e for e in (*result.events, *result.etas) if e.station == "BNC" and e.kind == "arr"]
    assert sorted(e.occ for e in arrivals) == [1, 2]


def test_delay_over_24h_kept_flagged():
    body = loaded()
    for row in body["response"]["stations"]:
        for prefix, key in (("Arr", "arrivalDetails"), ("Dep", "departureDetails")):
            detail = row[key]
            if detail[f"act{prefix}Date"]:
                day = datetime.strptime(detail[f"act{prefix}Date"], "%d-%m-%Y %H:%M:%S") + timedelta(days=2)
                detail[f"act{prefix}Date"] = day.strftime("%d-%m-%Y %H:%M:%S")
                detail[f"{prefix.lower()}Delay"] = 2880
    result = parse_copy(body)
    assert result.events and "delay_gt_24h" in result.flags
    assert result.events[0].actual_ts == datetime(2026, 10, 10, 0, 31, tzinfo=timezone.utc)


def test_unflagged_time_is_eta_never_actual():
    result = parse_copy(loaded("not_started_12426"), "12426")
    assert result.events == ()
    assert result.etas and result.pending
    assert result.status == "not_started"


@pytest.mark.parametrize("label,train", [("running_12723", "12723"), ("not_started_12426", "12426")])
def test_origin_arrival_not_emitted(label, train):
    body = loaded(label)
    result = parse_copy(body, train)
    origin = body["response"]["stations"][0]["station"]["code"]
    for items in (result.events, result.pending, result.etas):
        assert not any(e.station == origin and e.occ == 1 and e.kind == "arr" for e in items)
    assert any(e.station == origin and e.occ == 1 and e.kind == "dep"
               for e in (*result.events, *result.pending, *result.etas))


def test_eta_before_last_actual_kept():
    body = loaded()
    row = next(r for r in body["response"]["stations"] if r["arrivalDetails"]["arrived"] is False)
    row["arrivalDetails"]["actArrTime"] = "00:00"
    row["departureDetails"]["actDepTime"] = "00:01"
    result = parse_copy(body)
    assert result.events == parse_copy(loaded()).events
    eta = next(e for e in result.etas if e.station == row["station"]["code"] and e.kind == "arr")
    assert eta.eta_ts == datetime(2026, 10, 7, 18, 30, tzinfo=timezone.utc)
    assert eta.eta_ts < result.events[-1].actual_ts


def test_non_monotone_flagged_actuals_invalid():
    body = loaded()
    row = body["response"]["stations"][1]
    assert row["arrivalDetails"]["arrived"] is True
    row["arrivalDetails"]["actArrTime"] = "00:01"
    with pytest.raises(InvalidResponse):
        parse_copy(body)


@pytest.mark.parametrize("field", ["cancelledFromStation", "cancelledToStation"])
def test_exception_unparsed_flag(field):
    body = loaded()
    body["response"]["trainDetails"][field] = "NGP"
    result = parse_copy(body)
    assert result.exceptions == ()
    assert result.flags == ("exception_unparsed",)
    assert result.events == parse_copy(loaded()).events


@pytest.mark.parametrize("body", [b"", b"{}", b"null", b"{broken"])
def test_empty_body_invalid(body):
    with pytest.raises(InvalidResponse):
        parse_status(body, "12723", date(2026, 10, 8))


@pytest.mark.parametrize("mutation", ["station", "distance", "flag", "time", "day"])
def test_invalid_route(mutation):
    body = loaded()
    row = body["response"]["stations"][1]
    if mutation == "station":
        row["station"]["code"] = ""
    elif mutation == "distance":
        row["distance"] = "nan"
    elif mutation == "flag":
        row["arrivalDetails"]["arrived"] = "true"
    elif mutation == "time":
        row["arrivalDetails"]["actArrTime"] = "00:01"
    else:
        row["dayDetails"]["dayCnt"] = 0
    with pytest.raises(InvalidResponse):
        parse_copy(body)


def test_null_intermediate_stations_parse_like_empty():
    # Live bodies send null, not [], for halts without passing stations.
    body = loaded("repeated_station_11014")
    expected = parse_status(json.dumps(body).encode(), "11014", date(2026, 10, 8))
    for halt in body["response"]["stations"]:
        if not halt["intermediateStations"]:
            halt["intermediateStations"] = None
    assert parse_status(json.dumps(body).encode(), "11014", date(2026, 10, 8)) == expected


def test_scheduled_date_follows_actual_past_midnight():
    # Live 11004: CHI sch dep 23:12 dayCnt 1, left 00:04 next day, and MMT moved schDepDate to the actual's date.
    body = loaded("running_12723")
    expected = parse_status(json.dumps(body).encode(), "12723", date(2026, 10, 8))
    for halt in body["response"]["stations"]:
        for detail in (halt["arrivalDetails"], halt["departureDetails"]):
            for key in ("schArrDate", "schDepDate"):
                if detail.get(key):
                    detail[key] = "09-10-2026 00:00:00"
    assert parse_status(json.dumps(body).encode(), "12723", date(2026, 10, 8)) == expected
