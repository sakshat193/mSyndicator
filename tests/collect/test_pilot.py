import asyncio
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random

import httpx
import pytest
import zstandard

from msynd.collect.dispatcher import Dispatcher
from msynd.collect.mmt import HOST, MmtClient
from msynd.db import connect
from tests.collect.test_dispatcher import FakeClock

FIXTURES = Path(__file__).parents[1] / "fixtures/mmt"


@pytest.fixture
def clean_pilot_log(db_url):
    # Fake UTC clocks must not inherit another test's later request start.
    with connect(db_url) as conn:
        conn.execute("DELETE FROM obs.fetch WHERE host=%s", (HOST,))


def test_cohort_bound():
    from msynd.collect.pilot import cohort_bound

    assert cohort_bound(.2, 1) == 16
    assert cohort_bound(.3, 2) == 12
    for rate in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            cohort_bound(rate, 1)


def test_s_fit_recovers_synthetic():
    from msynd.collect.pilot import fit_s

    rng = random.Random(11)
    lags = [rng.expovariate(math.log(2) / 10) for _ in range(9700)]
    intervals = [(2 * math.floor(x / 2), 2 * math.floor(x / 2) + 2) for x in lags]
    result = fit_s(intervals, unpublished=300)
    assert abs(result["p50"] - 10) <= 1
    assert abs(result["tau"] - 10 / math.log(2)) < 2
    assert result["eps_ci"][0] < .03 < result["eps_ci"][1]
    assert result["lag"][0] == 0 and result["survival"][0] == 1
    assert all(a >= b >= result["eps"] for a, b in zip(result["survival"], result["survival"][1:]))


def test_right_censored_not_counted_unpublished():
    from msynd.collect.pilot import analyse_rows

    result = analyse_rows(censor_rows(), polls=2, active_curve=[])
    assert result["right_censored"] == 1
    assert result["censored_lags"] == [4]  # last poll 01:16 minus later SC departure 01:12 UTC
    assert result["unpublished"] == 0
    assert result["excluded"] == 2  # KZJ arrival/departure have no later actual
    assert result["S"]["mmt_status"]["resolved"] == 2


def test_right_censoring_uses_earliest_of_two_later_actuals():
    from msynd.collect.pilot import analyse_rows

    rows = censor_rows()
    body = json.loads(zstandard.decompress(rows[-1]["body_zstd"]))
    body["response"]["stations"][2]["arrivalDetails"]["arrived"] = True  # KZJ, 02:52 UTC
    rows.append(row(json.dumps(body).encode(), "12723/2026-10-08",
                    datetime(2026, 10, 8, 3, tzinfo=timezone.utc)))
    result = analyse_rows(rows, polls=3, active_curve=[])
    assert result["right_censored"] == 1
    assert result["censored_lags"] == [108]  # 03:00 minus earliest later actual: SC departure 01:12


def test_terminated_missing_actual_with_later_actual_is_epsilon_counterexample():
    from msynd.collect.pilot import analyse_rows

    result = analyse_rows(censor_rows(terminated=True), polls=2, active_curve=[])
    assert result["unpublished"] == 1
    assert result["right_censored"] == 0 and result["censored_lags"] == []
    assert result["excluded"] == 2
    assert result["S"]["mmt_status"]["resolved"] == 3


@pytest.mark.parametrize("terminated", [False, True])
def test_missing_actual_without_later_actual_is_excluded(terminated):
    from msynd.collect.pilot import analyse_rows

    result = analyse_rows(censor_rows(terminated=terminated, later=False), polls=2, active_curve=[])
    assert result["excluded"] == 4
    assert result["unpublished"] == result["right_censored"] == 0


def test_right_censoring_affects_survival_and_tail_fit():
    from msynd.collect.pilot import fit_s

    baseline = fit_s([(0, 2), (2, 4)])
    censored = fit_s([(0, 2), (2, 4)], right_censored=[10])
    assert censored["survival"][-1] > baseline["survival"][-1]
    assert censored["tau"] > baseline["tau"]
    assert censored["eps_ci"] == baseline["eps_ci"]


def test_corrected_actual_recomputes_lag_with_first_receipt():
    from msynd.collect.pilot import analyse_rows

    rows = censor_rows()
    corrected = json.loads(zstandard.decompress(rows[-1]["body_zstd"]))
    corrected["response"]["stations"][0]["departureDetails"]["actDepTime"] = "06:07"  # 00:37 UTC
    rows.append(row(json.dumps(corrected).encode(), "12723/2026-10-08",
                    datetime(2026, 10, 8, 1, 18, tzinfo=timezone.utc)))
    result = analyse_rows(rows, polls=3, active_curve=[])
    assert result["lag_intervals"] == [[37, 39]]  # 01:14 and first receipt 01:16 minus corrected 00:37
    assert result["S"]["mmt_status"]["resolved"] == 2


def test_never_hard_bound():
    from msynd.collect.pilot import fit_s

    result = fit_s([(0, 2)] * 100)
    assert 0 < result["eps"] < .05
    assert result["tau"] > 0
    assert result["survival"][-1] > result["eps"]
    assert result["eps_ci"][0] <= 0 < result["eps_ci"][1]
    with pytest.raises(ValueError, match="lag"):
        fit_s([])


def test_epsilon_includes_published_first_sightings():
    from msynd.collect.pilot import fit_s

    result = fit_s([(0, 2)] * 7, unpublished=3, left_censored=90)
    assert result["resolved"] == 100
    assert result["eps_ci"][0] < .03 < result["eps_ci"][1]


def row(body, key, received, outcome="ok", purpose="status"):
    return dict(body_zstd=zstandard.compress(body), key=key, received_at=received,
                started_at=received, outcome=outcome, purpose=purpose)


def censor_rows(*, terminated=False, later=True):
    body = json.loads((FIXTURES / "running_12723.json").read_bytes())
    body["response"]["trainDetails"]["terminated"] = terminated
    halts = body["response"]["stations"] = body["response"]["stations"][:3]
    for halt in halts:
        halt["intermediateStations"] = []
        halt["arrivalDetails"]["arrived"] = False
        halt["departureDetails"]["departed"] = False
    halts[1]["departureDetails"]["departed"] = later
    # A deliberately future ETA must not replace the later flagged actual as t_L.
    halts[1]["arrivalDetails"]["actArrTime"] = "23:59"
    before = datetime(2026, 10, 8, 1, 14, tzinfo=timezone.utc)
    first = row(json.dumps(body).encode(), "12723/2026-10-08", before)
    halts[0]["departureDetails"]["departed"] = True
    return [first, row(json.dumps(body).encode(), "12723/2026-10-08", before + timedelta(minutes=2))]


def test_analysis_uses_flags_intervals_and_per_key_eta_revisions():
    from msynd.collect.pilot import analyse_rows
    from msynd.collect.parse_mmt import parse_status

    body = (FIXTURES / "running_12723.json").read_bytes()
    parsed = parse_status(body, "12723", date(2026, 10, 8))
    actual = parsed.events[-1]
    pending_body = json.loads(body)
    for halt in pending_body["response"]["stations"]:
        if halt["station"]["code"] == actual.station:
            halt["departureDetails"]["departed"] = False
    before = actual.actual_ts + timedelta(minutes=2)
    after = actual.actual_ts + timedelta(minutes=4)
    pending = json.dumps(pending_body).encode()
    again = after + timedelta(minutes=2)
    rows = [row(pending, "12723/2026-10-08", before), row(body, "12723/2026-10-08", after),
            row(pending, "12723/2026-10-08", again), row(b'{"data":{}}', "12723/2026-10-08", again)]
    result = analyse_rows(rows, polls=4, active_curve=[{"at": before.isoformat(), "count": 2500}])
    assert result["lag_intervals"] == [[2, 4]]
    assert result["left_censored"] == len(parsed.events) - 1
    assert result["pending_after_actual"] == 1
    assert result["right_censored"] == 0
    assert result["excluded"] == len(parsed.pending)
    assert result["rates"]["invalid"] == .25
    assert result["eta_confirmations"] > result["eta_changes"]
    assert result["passing_trains"] == 1
    assert result["active_curve"][0]["count"] == 2500


def test_pilot_command_requires_live_guard_before_transport(monkeypatch):
    from msynd.collect.pilot import main

    monkeypatch.delenv("MSYND_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="MSYND_LIVE"):
        main(["run", "--days", "7", "--rate", ".2"])


def test_provisional_params_are_usable():
    from msynd.engine.replay import _lag_table

    params = json.loads((Path(__file__).parents[2] / "data/params.json").read_text())
    assert params["provisional"] is True and params["minute"] == "trunc"
    table = params["S"]["mmt_status"]
    grid, values, tau, eps = _lag_table(table)
    assert eps == .05 and table["p50"] == 10
    assert tau == pytest.approx(10 / math.log(2))


@pytest.mark.db
def test_inventory_calendar_multiday_and_latest_version(db):
    from msynd.collect.pilot import inventory

    db.execute("insert into ref.train(number,name) values ('00011','pilot')")
    db.execute("insert into ref.station(code,name) values ('P11A','A'),('P11B','B')")
    version = db.execute("""insert into ref.schedule_version(train,valid_from,valid_to,days_of_run,source,content_hash)
        values ('00011','2026-01-01','2026-12-31',8,'gtfs','p11') returning id""").fetchone()[0]
    db.execute("insert into ref.schedule_stop values (%s,0,'P11A',1,null,60,0),(%s,1,'P11B',1,3000,null,100)", (version, version))
    newer = db.execute("""insert into ref.schedule_version(train,valid_from,valid_to,days_of_run,source,content_hash)
        values ('00011','2026-01-01','2026-12-31',8,'gtfs','p11new') returning id""").fetchone()[0]
    db.execute("insert into ref.schedule_stop values (%s,0,'P11A',1,null,120,0),(%s,1,'P11B',1,3030,null,100)", (newer, newer))
    now = datetime(2026, 10, 9, 20, tzinfo=timezone.utc)  # IST Saturday, Thursday-origin run
    candidates = inventory(db, now)
    run = next(r for r in candidates if r.train == "00011")
    assert run.start == date(2026, 10, 8)
    assert len(run.expected) == 2
    assert run.expected["P11A", 1, "dep"] == datetime(2026, 10, 7, 20, 30, tzinfo=timezone.utc)
    assert not any(r.train == "00011" for r in inventory(db, now + timedelta(days=1)))


@pytest.mark.db
def test_runner_keeps_bodies_classifies_invalid_and_samples_hourly(db_url, tmp_path, clean_pilot_log):
    from msynd.collect.pilot import PilotRun, run_pilot, analyse

    async def run():
        clock = FakeClock()
        clock.base = datetime(2026, 10, 8, 10, 17, tzinfo=timezone.utc)
        body = (FIXTURES / "running_12723.json").read_bytes()
        before = json.loads(body)
        for halt in before["response"]["stations"]:
            if halt["station"]["code"] == "NGP":
                halt["departureDetails"]["departed"] = False
        starts = []

        def handle(request):
            starts.append(clock.t)
            if len(starts) == 1:
                return httpx.Response(200, content=b"")
            if len(starts) == 2:
                return httpx.Response(200, json=before)
            return httpx.Response(200, content=body if len(starts) != 3 else b'{"data":{}}')

        d = Dispatcher(HOST, .2, lambda: connect(db_url), httpx.MockTransport(handle), clock=clock)
        client = MmtClient(d)
        candidate = PilotRun("12723", date(2026, 10, 8), {("NGP", 1, "dep"): clock.base})
        metadata = tmp_path / "pilot.json"
        await run_pilot(client, days=1/24, metadata=metadata, candidates=lambda now: [candidate])
        session = json.loads(metadata.read_text())
        assert len(session["active_curve"]) == 2
        assert len(starts) >= 4 and starts[1] - starts[0] >= 5
        assert starts[2] - starts[1] >= 120
        with connect(db_url) as conn:
            rows = conn.execute("select outcome,body_zstd from obs.fetch where host=%s and started_at >= %s", (HOST, clock.base)).fetchall()
            assert any(outcome == "invalid" for outcome, _ in rows)
            assert all(body_zstd is not None for _, body_zstd in rows)
            assert any(zstandard.decompress(b) == body for _, b in rows)
            out, params = tmp_path / "G2.md", tmp_path / "params.json"
            result = analyse(conn, metadata, out, params)
            assert result["h"] > 1 and result["rates"]["empty"] > 0
            assert "2500" not in out.read_text()
            assert "right_censored" in out.read_text()
            assert json.loads(params.read_text())["provisional"] is False

    asyncio.run(run())


@pytest.mark.db
def test_hourly_rotation_covers_valid_runs_and_new_trains(db_url, tmp_path, clean_pilot_log):
    from msynd.collect.pilot import PilotRun, run_pilot

    async def run():
        clock = FakeClock()
        clock.base = datetime(2026, 10, 8, 15, 35, tzinfo=timezone.utc)
        body = json.loads((FIXTURES / "not_started_12426.json").read_bytes())
        starts = []

        def handle(request):
            train = json.loads(request.content)["trainNumber"]
            body["response"]["trainDetails"]["trainNumber"] = train
            starts.append((clock.t, train))
            return httpx.Response(200, json=body)

        d = Dispatcher(HOST, .2, lambda: connect(db_url), httpx.MockTransport(handle), clock=clock)
        candidates = [PilotRun(f"{i:05}", date(2026, 10, 8), {
            (f"WINDOW{j}", 1, "dep"): clock.base + timedelta(hours=j) for j in range(3)
        }) for i in range(40)]
        metadata = tmp_path / "pilot.json"
        await run_pilot(MmtClient(d), days=7290/86400, metadata=metadata, candidates=lambda now: candidates)
        assert len({train for t, train in starts if t < 3600}) == 16
        assert len({train for t, train in starts}) == 40
        per_train = {}
        for t, train in starts:
            if train in per_train:
                assert t - per_train[train] >= 120
            per_train[train] = t
        assert len(json.loads(metadata.read_text())["active_curve"]) == 3

    asyncio.run(run())


@pytest.mark.db
def test_rotation_bounds_cohort_and_prioritizes_new_trains(db_url, tmp_path, clean_pilot_log):
    from msynd.collect.pilot import PilotRun, run_pilot

    async def run():
        clock = FakeClock()
        clock.base = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        trains = []

        def handle(request):
            trains.append(json.loads(request.content)["trainNumber"])
            return httpx.Response(400, content=(FIXTURES / "not_running_16318.json").read_bytes())

        d = Dispatcher(HOST, .2, lambda: connect(db_url), httpx.MockTransport(handle), clock=clock)
        candidates = [PilotRun(f"{i:05}", date(2026, 10, 8), {("A", 1, "dep"): clock.base}) for i in range(40)]
        metadata = tmp_path / "pilot.json"
        await run_pilot(MmtClient(d), days=1800/86400, metadata=metadata, candidates=lambda now: candidates)
        assert len(set(trains[:16])) == 16
        assert len(set(trains)) == 40  # invalid run dates leave the cohort
        assert json.loads(metadata.read_text())["max_cohort"] <= 16

    asyncio.run(run())


@pytest.mark.db
def test_semantic_invalids_count_toward_dispatcher_breaker(db_url, tmp_path, clean_pilot_log):
    from msynd.collect.pilot import PilotRun, run_pilot

    async def run():
        clock = FakeClock()
        clock.base = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        starts = []

        def handle(request):
            starts.append(clock.t)
            return httpx.Response(200, json={"data": {}})

        d = Dispatcher(HOST, .2, lambda: connect(db_url), httpx.MockTransport(handle), clock=clock)
        candidates = [PilotRun(f"{i:05}", date(2026, 10, 8), {("A", 1, "dep"): clock.base}) for i in range(16)]
        await run_pilot(MmtClient(d), days=800/86400, metadata=tmp_path / "pilot.json", candidates=lambda now: candidates)
        assert max(b-a for a, b in zip(starts, starts[1:])) >= 600

    asyncio.run(run())


@pytest.mark.db
@pytest.mark.parametrize("code", [1001, 1002])
def test_not_running_400_stops_polling_that_run(db_url, tmp_path, clean_pilot_log, code):
    from msynd.collect.pilot import PilotRun, run_pilot

    async def run():
        clock = FakeClock()
        clock.base = datetime(2026, 10, 8, 10, 17, tzinfo=timezone.utc)
        starts = []

        def handle(request):
            starts.append(clock.t)
            return httpx.Response(400, json={"response": None, "success": False, "error": {
                "errorCode": code, "errorDescription": "Train is not running on given date"}})

        d = Dispatcher(HOST, .2, lambda: connect(db_url), httpx.MockTransport(handle), clock=clock)
        candidate = PilotRun("00461", date(2026, 10, 8), {("X", 1, "dep"): clock.base})
        await run_pilot(MmtClient(d), days=1/24, metadata=tmp_path / "pilot.json", candidates=lambda now: [candidate])
        assert len(starts) == 1

    asyncio.run(run())
