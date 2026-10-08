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
