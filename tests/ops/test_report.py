from datetime import datetime, timedelta, timezone

import pytest

from msynd.ops import report
from tests.helpers import insert_event, insert_fetch, make_run

SINCE = datetime(2026, 10, 7, tzinfo=timezone.utc)
UNTIL = SINCE + timedelta(days=1)


@pytest.mark.db
def test_a5_window_counts(db):
    for i in range(61):
        insert_fetch(db, host='busy', started_at=SINCE + timedelta(seconds=i * .9))
    for i in range(61):
        insert_fetch(db, host='paced', started_at=SINCE + timedelta(seconds=i))
        insert_fetch(db, host='before', started_at=SINCE - timedelta(seconds=59 - i * .9))
    # A window ending within the report must include starts just before since.
    insert_fetch(db, host='edge', started_at=SINCE - timedelta(seconds=1))
    insert_fetch(db, host='edge', started_at=SINCE)
    insert_fetch(db, host='edge', started_at=SINCE - timedelta(seconds=60))
    insert_fetch(db, host='future', started_at=UNTIL + timedelta(seconds=1))
    assert report.a5(db, SINCE, UNTIL) == [
        {'host': 'before', 'max_starts': 61, 'passes': False},
        {'host': 'busy', 'max_starts': 61, 'passes': False},
        {'host': 'edge', 'max_starts': 2, 'passes': True},
        {'host': 'paced', 'max_starts': 60, 'passes': True},
    ]


@pytest.mark.db
def test_a1_aggregates_metric_rows(db):
    for minute, values in enumerate(((100, 90, 80, 40, 38, 1), (200, 200, 100, 100, 100, 0))):
        for name, value in zip(('active', 'discovered', 'fresh', 'active_1h', 'discovered_1h', 'cold'), values):
            db.execute('INSERT INTO sim.metric VALUES (%s,%s,%s)',
                       (SINCE + timedelta(minutes=minute), 'a1_' + name, value))
    # Zero active is undefined, not 0% or 100%; incomplete minutes stay visible.
    db.execute("INSERT INTO sim.metric VALUES (%s,'a1_active',0)", (SINCE + timedelta(minutes=2),))
    db.execute("INSERT INTO sim.metric VALUES (%s,'a1_active',999)", (UNTIL + timedelta(minutes=1),))
    result = report.a1(db, SINCE, UNTIL)
    assert result['discovered'] == {'mean': .95, 'min': .9, 'minutes': 2}
    assert result['fresh']['mean'] == pytest.approx(.65)
    assert result['fresh']['min'] == .5
    assert result['discovered_1h'] == {'mean': .975, 'min': .95, 'minutes': 2}
    assert result['target_passes'] is True
    assert result['cold_share'] == .5
    assert result['minutes'] == 3
    assert result['complete_minutes'] == 2
    db.execute("UPDATE sim.metric SET value=37 WHERE name='a1_discovered_1h' AND ts=%s", (SINCE,))
    assert report.a1(db, SINCE, UNTIL)['target_passes'] is False


@pytest.mark.db
def test_empty_reports_do_not_pass(db):
    assert report.a1(db, SINCE, UNTIL)['target_passes'] is None
    assert report.a1(db, SINCE, UNTIL)['cold_share'] is None
    assert report.a5(db, SINCE, UNTIL) == []
    assert report.invalid(db, SINCE, UNTIL) == {'total': 0, 'invalid': 0, 'share': None, 'passes': None}


@pytest.mark.db
def test_a2_asof_event_zone_and_unseen_origin(db):
    observed = make_run(db)
    unseen = make_run(db, train='00001')
    db.execute("UPDATE ref.train SET class='SF' WHERE number IN ('12951','00001')")
    db.execute("UPDATE ref.station SET zone='WR' WHERE code='MMCT'")
    db.execute("UPDATE ref.station SET zone='NR' WHERE code='NDLS'")
    insert_event(db, observed, 'NDLS', 1, 'arr', SINCE, UNTIL - timedelta(minutes=10))
    insert_event(db, observed, 'MMCT', 1, 'dep', SINCE, UNTIL + timedelta(seconds=1))
    # Fetches and expected times do not refresh actual-event age.
    insert_fetch(db, run_id=unseen, started_at=UNTIL - timedelta(minutes=1), received_at=UNTIL)
    assert report.a2(db, SINCE, UNTIL) == [
        {'class': 'SF', 'zone': 'NR', 'runs': 1, 'unseen': 0, 'p50_min': 10., 'p90_min': 10.},
        {'class': 'SF', 'zone': 'WR', 'runs': 1, 'unseen': 1, 'p50_min': None, 'p90_min': None},
    ]


@pytest.mark.db
def test_a8_completed_runs_actual_keys_not_revisions(db):
    run = make_run(db, halts=[('A', None, 0, 0), ('B', 10, 12, 10),
                              ('A', 20, 22, 20), ('D', 30, None, 30)])
    db.execute("UPDATE ref.train SET class='SF' WHERE number='12951'")
    insert_event(db, run, 'A', 1, 'dep', SINCE, SINCE)
    insert_event(db, run, 'A', 1, 'dep', SINCE, SINCE + timedelta(minutes=1))
    insert_event(db, run, 'B', 1, 'arr', SINCE, SINCE)
    insert_event(db, run, 'D', 1, 'arr', SINCE, UNTIL)
    insert_event(db, run, 'A', 2, 'arr', SINCE, UNTIL + timedelta(seconds=1))
    make_run(db, train='00002')  # Scheduled-only is not completed.
    assert report.a8(db, SINCE, UNTIL) == [
        {'class': 'SF', 'runs': 1, 'expected': 6, 'missing': 3, 'mean_missing_rate': .5},
    ]


@pytest.mark.db
def test_invalid_share_all_completed_requests(db):
    for outcome in ('ok', 'invalid', 'empty', 'transport_error', 'challenge', 'inflight'):
        insert_fetch(db, outcome=outcome, started_at=SINCE, received_at=UNTIL if outcome != 'inflight' else None)
    insert_fetch(db, outcome='invalid', started_at=SINCE, received_at=UNTIL + timedelta(seconds=1))
    assert report.invalid(db, SINCE, UNTIL) == {'total': 5, 'invalid': 1, 'share': .2, 'passes': False}


@pytest.mark.db
def test_cli_writes_phase2_without_claiming_external_gates(db, tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def connection():
        yield db

    monkeypatch.setattr(report, 'connect', connection)
    out = tmp_path / 'gates' / 'phase2.md'
    report.main(['phase2', '--since', SINCE.isoformat(), '--until', UNTIL.isoformat(), '--out', str(out)])
    text = out.read_text(encoding='utf-8')
    assert 'A2' in text and 'A5' in text and 'A8' in text and 'invalid' in text
    assert 'Restore drill: unavailable' in text
    assert 'Continuous live collection: unavailable' in text
    assert 'Phase 2: unavailable' in text


@pytest.mark.parametrize('args', [
    ['a1', '--since', '2026-10-07', '--until', UNTIL.isoformat(), '--out', 'ignored.md'],
    ['a1', '--since', UNTIL.isoformat(), '--until', SINCE.isoformat(), '--out', 'ignored.md'],
])
def test_cli_rejects_ambiguous_or_reversed_time(args):
    with pytest.raises(SystemExit):
        report.main(args)


DAY = '2026-10-15'
IST_DAY = datetime(2026, 10, 14, 18, 30, tzinfo=timezone.utc)  # 2026-10-15 00:00 IST
HALTS = [('A', None, 0, 0), ('B', 100, 105, 200), ('C', 200, None, 400)]


def _runs(db, n, train='12951', cls='SF'):
    first = make_run(db, train=train, start='2026-10-14', halts=HALTS)
    db.execute('UPDATE ref.train SET class=%s WHERE number=%s', (cls, train))
    return [first] + [db.execute("""INSERT INTO obs.run(train,start_date,schedule_version_id,route_version_id)
        SELECT train,start_date-%s,schedule_version_id,route_version_id FROM obs.run WHERE id=%s
        RETURNING id""", (i, first)).fetchone()[0] for i in range(1, n)]


def _harvest(db, run, outcome='ok', status=200, at=IST_DAY + timedelta(hours=1)):
    from msynd.collect.mmt import HOST
    return insert_fetch(db, host=HOST, mode='harvest', run_id=run, started_at=at,
                        received_at=at, outcome=outcome, http_status=status)


def _events(db, run, kinds=(('A', 'dep'), ('B', 'arr'), ('B', 'dep'), ('C', 'arr'))):
    for station, kind in kinds:
        insert_event(db, run, station, 1, kind, IST_DAY, IST_DAY)


@pytest.mark.db
def test_ramp_not_running_keys_leave_denominator(db):
    ok, cancelled, other_400 = _runs(db, 3)
    _harvest(db, ok)
    _events(db, ok)
    _harvest(db, cancelled, outcome='invalid', status=400)
    db.execute("UPDATE obs.run SET lifecycle='cancelled' WHERE id=%s", (cancelled,))
    _harvest(db, other_400, outcome='invalid', status=400)  # a 400 that did not cancel the run stays due
    result = report.ramp(db, DAY)
    assert (result['keys'], result['not_running'], result['due'], result['usable']) == (3, 1, 2, 1)
    assert result['usable_pct'] == 50.
    assert result['by_class'] == {'SF': {'n': 2, 'usable_pct': 50.}}


@pytest.mark.db
def test_ramp_pairs_on_three_halt_route(db):
    run, = _runs(db, 1)
    _harvest(db, run)
    _events(db, run, (('A', 'dep'), ('B', 'arr'), ('C', 'arr')))  # B departure missing
    result = report.ramp(db, DAY)
    assert result['usable'] == 1
    assert result['r_pairs_pct'] == 50.  # A→B present, B→C lacks dep B
    assert result['w_pairs_pct'] == 0.   # B has arr but no dep


@pytest.mark.db
def test_ramp_ist_day_boundary(db):
    before, inside = _runs(db, 2)
    _harvest(db, before, at=IST_DAY - timedelta(seconds=1))
    _harvest(db, inside, at=IST_DAY)
    _harvest(db, inside, outcome='transport_error', status=None, at=IST_DAY + timedelta(hours=23, minutes=59))
    _harvest(db, before, at=IST_DAY + timedelta(days=1))
    result = report.ramp(db, DAY)
    assert (result['keys'], result['attempts']) == (1, 2)


@pytest.mark.db
def test_ramp_class_rule_needs_thirty_runs(db):
    for run in _runs(db, 29, train='00001', cls='PASS'):
        _harvest(db, run)  # ok answers, no destination arrival → unusable
    assert 'classes' not in report.ramp(db, DAY)['failing']
    extra, = _runs(db, 1, train='00002', cls='PASS')
    _harvest(db, extra)
    result = report.ramp(db, DAY)
    assert result['by_class']['PASS'] == {'n': 30, 'usable_pct': 0.}
    assert 'classes' in result['failing']


@pytest.mark.db
def test_ramp_passing_batch_and_one_challenge(db):
    for run in _runs(db, 300):
        _harvest(db, run)
        _events(db, run)
    result = report.ramp(db, DAY)
    assert (result['due'], result['usable_pct'], result['r_pairs_pct'], result['w_pairs_pct']) == (300, 100., 100., 100.)
    assert (result['passes'], result['failing']) == (True, [])
    _harvest(db, None, outcome='challenge', status=403)
    result = report.ramp(db, DAY)
    assert (result['challenges'], result['passes'], result['failing']) == (1, False, ['challenges'])


@pytest.mark.db
def test_ramp_empty_day_fails_every_measured_rule(db):
    result = report.ramp(db, DAY)
    assert result['keys'] == 0 and result['usable_pct'] is None and result['passes'] is False
    assert result['failing'] == ['due', 'usable', 'r_pairs', 'w_pairs']


@pytest.mark.db
def test_cli_ramp_prints_json(db, capsys, monkeypatch):
    import json
    from contextlib import contextmanager

    @contextmanager
    def connection():
        yield db

    monkeypatch.setattr(report, 'connect', connection)
    report.main(['ramp', '--day', DAY])  # main sets the read-only transaction first, so no seeded rows
    out = json.loads(capsys.readouterr().out)
    assert (out['day'], out['keys'], out['passes']) == (DAY, 0, False)
    with pytest.raises(SystemExit):
        report.main(['ramp'])
