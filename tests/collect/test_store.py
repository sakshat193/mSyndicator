from dataclasses import replace
from datetime import date, datetime
import json
from pathlib import Path

import pytest
import psycopg

from msynd.collect.dispatcher import FetchResult
from msynd.collect.parse_mmt import parse_status
from msynd.collect.status import Eta, Event, ParsedStatus, PassingStation, Pending
from msynd.engine.evidence import load_run_evidence
from msynd.labels import IST
from tests.helpers import insert_fetch, make_run

pytestmark = pytest.mark.db
FIXTURES = Path(__file__).parents[1] / 'fixtures' / 'mmt'


def ist(clock):
    return datetime.fromisoformat(f'2026-10-07 {clock}').replace(tzinfo=IST)


def page(**kwargs):
    return replace(ParsedStatus('12951', date(2026, 10, 7), 'not_started',
                               ist('09:00'), (), (), (), (), ()), **kwargs)


def store(db, run, parsed, clock='09:00', purpose='status', outcome='ok'):
    from msynd.collect.store import store_status

    received = ist(clock)
    fetch_id = insert_fetch(db, purpose=purpose, outcome=outcome, run_id=run,
                            started_at=received, received_at=received)
    return store_status(db, run, FetchResult(fetch_id, outcome, 200, b'', received), parsed)


@pytest.mark.parametrize('bound', [False, True])
@pytest.mark.parametrize('flags', [('delay_gt_24h',), ('exception_unparsed', 'delay_gt_24h')])
def test_store_records_parser_flags(db, bound, flags):
    from msynd.collect.store import store_status

    run = make_run(db)
    received = ist('09:00')
    fetch = insert_fetch(db, run_id=run if bound else None, received_at=received, started_at=received)
    store_status(db, run, FetchResult(fetch, 'ok', 200, b'', received), page(flags=flags))
    assert db.execute('SELECT flags,run_id,received_at FROM obs.fetch WHERE id=%s', (fetch,)).fetchone() == (
        sorted(flags), run, received)


def test_parser_flags_roll_back_with_evidence(db):
    from msynd.collect.store import store_status

    run = make_run(db)
    received = ist('09:00')
    fetch = insert_fetch(db, received_at=received, started_at=received)
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        store_status(db, run, FetchResult(fetch, 'ok', 200, b'', received),
                     page(flags=('exception_unparsed',),
                          events=(Event('BRC', 1, 'invalid_kind', ist('08:30')),)))
    assert db.execute('SELECT flags,run_id FROM obs.fetch WHERE id=%s', (fetch,)).fetchone() == ([], None)


def test_eta_confirmation_per_key(db):
    from msynd.eval.baselines import b2_eta

    run = make_run(db)
    b = Eta('BRC', 1, 'arr', ist('10:30'))
    c = Eta('NDLS', 1, 'arr', ist('11:00'))
    assert store(db, run, page(etas=(b, c))).changed
    store(db, run, page(etas=(c,)), '10:00')
    store(db, run, page(etas=(b,)), '10:20')
    store(db, run, page(etas=(replace(b, eta_ts=ist('10:45')),)), '10:40')
    assert b2_eta(db, run, 'BRC', 1, 'arr', ist('10:10')) is None
    assert b2_eta(db, run, 'NDLS', 1, 'arr', ist('10:10')) == ist('11:00')
    assert b2_eta(db, run, 'BRC', 1, 'arr', ist('10:25')) == ist('10:30')
    assert b2_eta(db, run, 'BRC', 1, 'arr', ist('10:39')) == ist('10:30')
    assert b2_eta(db, run, 'BRC', 1, 'arr', ist('10:41')) == ist('10:45')
    assert b2_eta(db, run, 'NDLS', 1, 'arr', ist('10:30')) == ist('11:00')
    assert b2_eta(db, run, 'NDLS', 1, 'arr', ist('10:30:01')) is None
    assert b2_eta(db, run, 'BRC', 2, 'arr', ist('10:41')) is None
    assert db.execute('SELECT count(*) FROM obs.ntes_eta_rev').fetchone() == (5,)


@pytest.mark.parametrize('purpose', ['autosuggest', 'capture', 'token', 'session'])
def test_non_status_fetch_never_confirms(db, purpose):
    run = make_run(db)
    result = store(db, run, page(etas=(Eta('BRC', 1, 'arr', ist('10:30')),)), purpose=purpose)
    assert not result.changed
    assert db.execute('SELECT count(*) FROM obs.ntes_eta_rev').fetchone() == (0,)
    assert db.execute('SELECT count(*) FROM obs.dirty_run').fetchone() == (0,)


@pytest.mark.parametrize('outcome', ['empty', 'invalid', 'transport_error', 'inflight'])
def test_unsuccessful_fetch_writes_no_evidence(db, outcome):
    run = make_run(db)
    assert not store(db, run, page(events=(Event('BRC', 1, 'arr', ist('08:30')),)),
                     outcome=outcome).changed
    assert db.execute('SELECT count(*) FROM obs.event_rev').fetchone() == (0,)


def test_unchanged_actual_not_reinserted(db):
    run = make_run(db)
    actual = Event('BRC', 1, 'arr', ist('08:30'))
    assert store(db, run, page(events=(actual,))).changed
    version = db.execute('SELECT ver FROM obs.dirty_run WHERE run_id=%s', (run,)).fetchone()
    assert not store(db, run, page(events=(actual,)), '09:10').changed
    assert db.execute('SELECT ver FROM obs.dirty_run WHERE run_id=%s', (run,)).fetchone() == version
    assert store(db, run, page(events=(replace(actual, actual_ts=ist('08:35')),)), '09:20').changed
    assert db.execute('SELECT actual_ts,source FROM obs.event_rev ORDER BY received_at').fetchall() == [
        (ist('08:30'), 'mmt_status'), (ist('08:35'), 'mmt_status')]
    assert load_run_evidence(db, run, ist('09:15')).evidence[0].minute == 510
    assert load_run_evidence(db, run).evidence[0].minute == 515


def test_every_pending_marker_is_written_and_dirty(db):
    run = make_run(db)
    parsed = page(pending=(Pending('BRC', 1, 'arr'),))
    assert store(db, run, parsed).changed
    version = db.execute('SELECT ver FROM obs.dirty_run WHERE run_id=%s', (run,)).fetchone()[0]
    assert store(db, run, parsed, '09:10').changed
    assert db.execute('SELECT received_at,source FROM obs.pending_rev ORDER BY received_at').fetchall() == [
        (ist('09:00'), 'mmt_status'), (ist('09:10'), 'mmt_status')]
    assert db.execute('SELECT ver FROM obs.dirty_run WHERE run_id=%s', (run,)).fetchone()[0] > version


def fixture_run(db, label):
    entry = next(e for e in json.loads((FIXTURES / 'manifest.json').read_text()) if e['label'] == label)
    body = (FIXTURES / f'{label}.json').read_bytes()
    halts = []
    for idx, row in enumerate(json.loads(body)['response']['stations']):
        day = (row['dayDetails']['dayCnt'] - 1) * 1440
        def offset(value):
            if value is None:
                return None
            hour, minute = map(int, value.split(':'))
            return day + hour * 60 + minute
        halts.append((row['station']['code'],
                      None if idx == 0 else offset(row['arrivalDetails']['schArrTime']),
                      offset(row['departureDetails']['schDepTime']), float(row['distance'])))
    run = make_run(db, train=entry['train'], start=entry['date'], halts=halts)
    return run, parse_status(body, entry['train'], date.fromisoformat(entry['date']))


def test_expected_never_stored_as_actual(db):
    run, parsed = fixture_run(db, 'not_started_12426')
    assert store(db, run, parsed).changed
    assert db.execute('SELECT count(*) FROM obs.event_rev').fetchone() == (0,)
    assert db.execute('SELECT count(*) FROM obs.ntes_eta_rev').fetchone()[0] == len(parsed.etas) > 0
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('scheduled',)


def test_dirty_in_same_txn(db):
    run = make_run(db)
    before = db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()
    with pytest.raises(RuntimeError), db.transaction():
        store(db, run, page(events=(Event('BRC', 1, 'arr', ist('08:30')),),
                            pending=(Pending('NDLS', 1, 'arr'),),
                            etas=(Eta('NDLS', 1, 'arr', ist('11:00')),),
                            passing_stations=(PassingStation('PASS', 1, 196, None, ('MMCT', 1), ('BRC', 1), .5),)))
        assert db.execute('SELECT run_id FROM obs.dirty_run').fetchall() == [(run,)]
        raise RuntimeError('abort evidence transaction')
    for table in ('event_rev', 'pending_rev', 'ntes_eta_rev', 'dirty_run'):
        assert db.execute(f'SELECT count(*) FROM obs.{table}').fetchone() == (0,)
    assert db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone() == before
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('scheduled',)
    assert db.execute('SELECT count(*) FROM ref.route_version').fetchone() == (1,)


def test_repeated_station_keys(db):
    run, parsed = fixture_run(db, 'repeated_station_11014')
    store(db, run, parsed)
    rows = db.execute('''SELECT station,occ,kind FROM obs.event_rev WHERE run_id=%s
                        UNION ALL SELECT station,occ,kind FROM obs.ntes_eta_rev WHERE run_id=%s''',
                      (run, run)).fetchall()
    assert sorted(occ for station, occ, kind in rows if station == 'BNC' and kind == 'arr') == [1, 2]
    assert len(rows) == len(set(rows))


def test_enrichment_never_rekeys(db):
    run = make_run(db)
    schedule, old_route = db.execute('SELECT schedule_version_id,route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()
    other = db.execute('''INSERT INTO obs.run(train,start_date,schedule_version_id,route_version_id)
                         VALUES ('12951','2026-10-08',%s,%s) RETURNING id''', (schedule, old_route)).fetchone()[0]
    unrelated = make_run(db, train='99999')
    store(db, run, page(events=(Event('BRC', 1, 'arr', ist('08:30')),)))
    before = db.execute('SELECT id,station,occ,kind FROM obs.event_rev WHERE run_id=%s', (run,)).fetchall()
    parsed = page(passing_stations=(PassingStation('PASS', 1, 196, None, ('MMCT', 1), ('BRC', 1), .5),))
    assert store(db, run, parsed, '09:10').changed
    new_route = db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()[0]
    assert new_route != old_route
    assert db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (other,)).fetchone() == (new_route,)
    assert db.execute('SELECT run_id FROM obs.dirty_run ORDER BY run_id').fetchall() == [(run,), (other,)]
    assert db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (unrelated,)).fetchone()[0] != new_route
    assert db.execute('SELECT id,station,occ,kind FROM obs.event_rev WHERE run_id=%s', (run,)).fetchall() == before
    ev = load_run_evidence(db, run)
    assert [(e.station, e.occ, e.halt_idx) for e in ev.events if e.kind == 'arr'] == [('BRC', 1, 1), ('NDLS', 1, 2)]
    assert ev.events[1].kind == 'pass' and ev.events[1].sched_off == 150
    assert db.execute('SELECT count(*) FROM ref.route_segment WHERE route_version_id=%s', (new_route,)).fetchone() == (3,)
    assert not store(db, run, parsed, '09:20').changed
    assert db.execute('SELECT count(*) FROM ref.route_version').fetchone() == (3,)
    # A run still bound to the older version reuses already learnt knowledge.
    db.execute('UPDATE obs.run SET route_version_id=%s WHERE id=%s', (old_route, other))
    assert store(db, other, replace(parsed, start_date=date(2026, 10, 8)), '09:25').changed
    assert db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (other,)).fetchone() == (new_route,)
    assert db.execute('SELECT count(*) FROM ref.route_version').fetchone() == (3,)
    # New knowledge is a union, never a deletion of an earlier passing station.
    store(db, other, replace(parsed, start_date=date(2026, 10, 8),
                            passing_stations=(PassingStation('PASS2', 1, 250, 200, ('MMCT', 1), ('BRC', 1), 250 / 392),)), '09:30')
    route = db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()[0]
    assert db.execute('SELECT station FROM ref.route_stop WHERE route_version_id=%s AND NOT halt ORDER BY idx', (route,)).fetchall() == [('PASS',), ('PASS2',)]


def test_page_passes_use_gtfs_gap_and_heal_stored_placement(db):
    run = make_run(db, train='12028', halts=[('A', None, 300, 300), ('MAS', 360, None, 354)])
    route = db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()[0]
    db.execute("INSERT INTO ref.station(code,name) VALUES ('PER','PER'),('PCW','PCW')")
    db.execute("UPDATE ref.route_stop SET idx=3 WHERE route_version_id=%s AND station='MAS'", (route,))
    db.execute("INSERT INTO ref.route_stop VALUES (%s,1,'PCW',1,false,353,NULL),(%s,4,'PER',1,false,354,355)", (route, route))
    parsed = page(train='12028', passing_stations=(
        PassingStation('PER', 1, 354, 355, ('A', 1), ('MAS', 1), .9),
        PassingStation('VJM', 1, 355, 356, ('A', 1), ('MAS', 1), 11 / 12)))
    assert store(db, run, parsed).changed
    rows = db.execute('''SELECT station,dist_km,sched_off FROM ref.route_stop
        WHERE route_version_id=(SELECT route_version_id FROM obs.run WHERE id=%s) ORDER BY idx''', (run,)).fetchall()
    assert [r[0] for r in rows] == ['A', 'PER', 'VJM', 'PCW', 'MAS']
    assert rows[1][1] == pytest.approx(348.6)
    assert rows[2][1] == pytest.approx(349.5)
    assert rows[1][2] == 355 and rows[2][2] == 356
    events = load_run_evidence(db, run).events
    assert [e.km for e in events] == sorted(e.km for e in events)
    assert all(300 <= e.km <= 354 for e in events)
    assert not store(db, run, parsed).changed


@pytest.mark.parametrize('after,before', [(('MMCT', 1), ('NDLS', 1)), (('MISSING', 1), ('BRC', 1)),
                                         (('MMCT', 2), ('BRC', 1)), (('BRC', 1), ('MMCT', 1))])
def test_pass_with_incompatible_source_halts_skipped(db, after, before):
    run = make_run(db)
    assert not store(db, run, page(passing_stations=(PassingStation('PASS', 1, 196, None, after, before, .5),))).changed
    assert db.execute('SELECT count(*) FROM ref.route_stop WHERE NOT halt').fetchone() == (0,)


@pytest.mark.parametrize('halt_km', [392, 0])
def test_pass_endpoint_ties_keep_page_order_between_halts(db, halt_km):
    run = make_run(db, halts=[('MMCT', None, 0, 0), ('BRC', 300, None, halt_km)])
    route = db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()[0]
    db.execute('UPDATE ref.route_stop SET idx=5 WHERE route_version_id=%s AND halt AND idx=1', (route,))
    for idx, station in enumerate(('A1', 'A0', 'Z1', 'Z0'), 1):
        db.execute('INSERT INTO ref.station(code,name) VALUES (%s,%s)', (station, station))
        db.execute('INSERT INTO ref.route_stop VALUES (%s,%s,%s,1,false,0,NULL)', (route, idx, station))
    parsed = page(passing_stations=tuple(
        PassingStation(station, 1, km, None, ('MMCT', 1), ('BRC', 1), fraction)
        for station, km, fraction in [('Z0', 0, 0), ('A0', 0, 0), ('Z1', 392, 1), ('A1', 392, 1)]))
    assert store(db, run, parsed).changed
    rows = db.execute('''SELECT station FROM ref.route_stop
        WHERE route_version_id=(SELECT route_version_id FROM obs.run WHERE id=%s) ORDER BY idx''', (run,)).fetchall()
    assert rows == [('MMCT',), ('Z0',), ('A0',), ('Z1',), ('A1',), ('BRC',)]
    events = load_run_evidence(db, run).events
    assert [e.km for e in events] == sorted(e.km for e in events)
    assert not store(db, run, parsed).changed


def test_enrichment_orders_gap_by_km_and_clamps_to_halts(db):
    run = make_run(db)
    parsed = page(passing_stations=(
        PassingStation('FIRST', 1, 300, None, ('MMCT', 1), ('BRC', 1), 300 / 392),
        PassingStation('SECOND', 1, 196, None, ('MMCT', 1), ('BRC', 1), .5),
        PassingStation('BEYOND', 1, 500, None, ('MMCT', 1), ('BRC', 1), 1.5)))
    assert store(db, run, parsed).changed
    assert db.execute('''SELECT station,dist_km FROM ref.route_stop
        WHERE route_version_id=(SELECT route_version_id FROM obs.run WHERE id=%s) ORDER BY idx''', (run,)).fetchall() == [
            ('MMCT', 0), ('SECOND', 196), ('FIRST', 300), ('BEYOND', 392), ('BRC', 392), ('NDLS', 1384)]


def test_store_failure_rolls_back_partial_writes(db):
    run = make_run(db)
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        store(db, run, page(events=(Event('BRC', 1, 'arr', ist('08:30')),),
                            etas=(Eta('NDLS', 1, 'invalid_kind', ist('11:00')),),
                            passing_stations=(PassingStation('PASS', 1, 196, None, ('MMCT', 1), ('BRC', 1), .5),)))
    assert db.execute('SELECT count(*) FROM obs.event_rev').fetchone() == (0,)
    assert db.execute('SELECT count(*) FROM ref.route_version').fetchone() == (1,)
    assert db.execute('SELECT count(*) FROM obs.dirty_run').fetchone() == (0,)


def test_enrichment_rebinds_runs_on_different_compatible_versions(db):
    run_a = make_run(db)
    schedule, v1 = db.execute('SELECT schedule_version_id,route_version_id FROM obs.run WHERE id=%s', (run_a,)).fetchone()
    store(db, run_a, page(passing_stations=(PassingStation('PASS', 1, 196, None, ('MMCT', 1), ('BRC', 1), .5),)))
    v2 = db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run_a,)).fetchone()[0]
    assert v2 != v1
    run_b = db.execute('''INSERT INTO obs.run(train,start_date,schedule_version_id,route_version_id)
                         VALUES ('12951','2026-10-08',%s,%s) RETURNING id''', (schedule, v1)).fetchone()[0]
    unrelated = make_run(db, train='99999')
    before = db.execute('SELECT * FROM obs.run WHERE id=%s', (unrelated,)).fetchone()
    db.execute('DELETE FROM obs.dirty_run WHERE run_id=%s', (run_a,))

    assert store(db, run_a, page(passing_stations=(PassingStation('PASS2', 1, 250, 200, ('MMCT', 1), ('BRC', 1), 250 / 392),)), '09:10').changed
    v3 = db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run_a,)).fetchone()[0]
    assert v3 not in (v1, v2)
    assert db.execute('SELECT route_version_id FROM obs.run WHERE id=%s', (run_b,)).fetchone() == (v3,)
    assert db.execute('SELECT run_id FROM obs.dirty_run ORDER BY run_id').fetchall() == [(run_a,), (run_b,)]
    assert db.execute('SELECT * FROM obs.run WHERE id=%s', (unrelated,)).fetchone() == before
    assert db.execute('SELECT station FROM ref.route_stop WHERE route_version_id=%s AND NOT halt ORDER BY idx', (v3,)).fetchall() == [('PASS',), ('PASS2',)]


@pytest.mark.parametrize('label,want', [('running_12723', 'running'), ('completed_12951', 'completed')])
def test_fixture_lifecycle(db, label, want):
    run, parsed = fixture_run(db, label)
    store(db, run, parsed)
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == (want,)


def test_lifecycle_uses_actuals_and_never_regresses(db):
    run = make_run(db)
    store(db, run, page(status='completed'))
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('scheduled',)
    store(db, run, page(events=(Event('NDLS', 1, 'arr', ist('08:30')),)), '09:10')
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('completed',)
    assert not store(db, run, page(events=(Event('NDLS', 1, 'arr', ist('08:30')),)), '09:20').changed
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('completed',)


def test_actual_preserves_cancelled_and_revives_expired_unknown(db):
    for train, state, expected in [('12951', 'cancelled', 'cancelled'),
                                   ('12952', 'expired_unknown', 'running')]:
        run = make_run(db, train=train)
        db.execute('UPDATE obs.run SET lifecycle=%s WHERE id=%s', (state, run))
        assert store(db, run, page(train=train, events=(Event('BRC', 1, 'arr', ist('08:30')),))).changed
        assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == (expected,)
        assert db.execute('SELECT station,occ,kind,actual_ts FROM obs.event_rev WHERE run_id=%s', (run,)).fetchall() == [('BRC', 1, 'arr', ist('08:30'))]


def test_exception_on_status_page_is_stored_and_dirty(db):
    run = make_run(db)
    assert store(db, run, page(exceptions=({'kind': 'cancelled'},),
                               events=(Event('BRC', 1, 'arr', ist('08:30')),))).changed
    assert db.execute('SELECT detail FROM obs.exception_rev').fetchone() == (
        {'kind': 'cancelled', 'resched_min': 0, 'diverted_from': None, 'part_cancelled': None},)
    assert db.execute('SELECT lifecycle FROM obs.run WHERE id=%s', (run,)).fetchone() == ('cancelled',)
    assert db.execute('SELECT count(*) FROM obs.event_rev').fetchone() == (1,)
    assert db.execute('SELECT run_id FROM obs.dirty_run').fetchall() == [(run,)]


def test_page_exceptions_share_evidence_transaction(db):
    run = make_run(db)
    with pytest.raises(RuntimeError), db.transaction():
        store(db, run, page(exceptions=({'kind': 'rescheduled', 'resched_min': 60},),
                            events=(Event('MMCT', 1, 'dep', ist('08:30')),)))
        raise RuntimeError('rollback')
    for table in ('event_rev', 'exception_rev', 'dirty_run'):
        assert db.execute(f'SELECT count(*) FROM obs.{table}').fetchone() == (0,)
    assert db.execute('SELECT lifecycle,resched_min FROM obs.run WHERE id=%s', (run,)).fetchone() == ('scheduled', 0)


def test_multiple_page_exceptions_write_one_cumulative_revision(db):
    run = make_run(db)
    assert store(db, run, page(exceptions=({'kind': 'rescheduled', 'resched_min': 60},
                                          {'kind': 'diverted', 'diverted_from': 'BRC'}))).changed
    assert db.execute('SELECT detail FROM obs.exception_rev').fetchall() == [
        ({'kind': 'diverted', 'resched_min': 60, 'diverted_from': 'BRC', 'part_cancelled': None},)]
    assert load_run_evidence(db, run).resched_min == 60


def test_exception_can_bind_status_fetch_without_run_id(db):
    from msynd.collect.store import store_status

    run = make_run(db)
    received = ist('09:00')
    fetch = insert_fetch(db, received_at=received, started_at=received)
    assert store_status(db, run, FetchResult(fetch, 'ok', 200, b'', received),
                        page(exceptions=({'kind': 'rescheduled', 'resched_min': 60},))).changed
    assert db.execute('SELECT run_id FROM obs.fetch WHERE id=%s', (fetch,)).fetchone() == (run,)
    assert db.execute('SELECT resched_min FROM obs.run WHERE id=%s', (run,)).fetchone() == (60,)


def test_wrong_run_page_is_rejected(db):
    run = make_run(db)
    with pytest.raises(ValueError):
        store(db, run, page(train='00000', events=(Event('BRC', 1, 'arr', ist('08:30')),)))
    assert db.execute('SELECT count(*) FROM obs.event_rev').fetchone() == (0,)


def test_pending_tie_prefers_live_source(db):
    from msynd.config import settings

    run = make_run(db)
    store(db, run, page(pending=(Pending('BRC', 1, 'arr'),)))
    db.execute("INSERT INTO obs.pending_rev VALUES (%s,'BRC',1,'arr','ntes_status',NULL,%s)", (run, ist('09:00')))
    assert settings.live_source == 'mmt_status'
    assert load_run_evidence(db, run).evidence[0].source == 'mmt_status'
