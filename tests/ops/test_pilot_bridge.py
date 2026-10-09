from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
import pytest
import zstandard

from msynd.db import migrate
from tests.helpers import insert_fetch, make_run

RECEIVED = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)


@pytest.fixture
def bridge_dbs():
    admin_url = os.environ['MSYND_TEST_DSN']
    urls = []
    with psycopg.connect(admin_url, autocommit=True) as admin:
        try:
            for side in ('source', 'target'):
                name = f'msynd_test_bridge_{os.getpid()}_{side}'
                admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
                url = urlunsplit(urlsplit(admin_url)._replace(path=f'/{name}'))
                urls.append(url)
                migrate.apply(url)
            yield urls
        finally:
            for url in urls:
                admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(
                    sql.Identifier(urlsplit(url).path[1:])))


def body(train='12951'):
    def detail(prefix, marker, clock):
        return {marker: True if clock else None, f'sch{prefix}Time': clock,
                f'act{prefix}Time': clock,
                f'act{prefix}Date': '07-10-2026 00:00:00' if clock else None}
    stations = []
    for station, km, arr, dep in [('MMCT', 0, None, '06:00'), ('NDLS', 100, '06:20', None)]:
        stations.append(dict(station={'code': station}, distance=km, dayDetails={'dayCnt': 1},
                             arrivalDetails=detail('Arr', 'arrived', arr),
                             departureDetails=detail('Dep', 'departed', dep)))
    return json.dumps(dict(success=True, response=dict(
        trainDetails=dict(trainNumber=train, startDate='07-10-2026 00:00:00',
                          departed=True, terminated=True),
        lastUpdated='07-10-2026 06:30:00', stations=stations))).encode()


def seed(urls):
    with psycopg.connect(urls[1]) as target:
        run = make_run(target, halts=[('MMCT', None, 360, 0), ('NDLS', 380, None, 100)])
        target.execute('DELETE FROM obs.run WHERE id=%s', (run,))
    with psycopg.connect(urls[0]) as source:
        return insert_fetch(source, host='synthetic', key='12951/2026-10-07',
                            received_at=RECEIVED, started_at=RECEIVED,
                            http_status=200, body_zstd=zstandard.compress(body()))


@pytest.mark.db
def test_bridge_evidence_receipts_and_idempotence(bridge_dbs, capsys):
    from msynd.ops.pilot_bridge import bridge_once, open_source

    seed(bridge_dbs)
    assert bridge_once(*bridge_dbs) == 1
    with psycopg.connect(bridge_dbs[1]) as target:
        assert target.execute('SELECT train,start_date::text,lifecycle FROM obs.run').fetchall() == [
            ('12951', '2026-10-07', 'completed')]
        assert target.execute('SELECT purpose,key,started_at,received_at,outcome,body_zstd FROM obs.fetch').fetchall() == [
            ('status', '12951/2026-10-07', RECEIVED, RECEIVED, 'ok', None)]
        assert target.execute('SELECT station,kind,source,received_at FROM obs.event_rev ORDER BY id').fetchall() == [
            ('MMCT', 'dep', 'mmt_status', RECEIVED), ('NDLS', 'arr', 'mmt_status', RECEIVED)]
        assert target.execute('SELECT count(*) FROM obs.event_asof(%s)',
                              (datetime(2026, 10, 7, 0, tzinfo=timezone.utc),)).fetchone()[0] == 0
        assert target.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 1
    assert bridge_once(*bridge_dbs) == 0
    with psycopg.connect(bridge_dbs[1]) as target:
        assert target.execute('SELECT count(*) FROM obs.fetch').fetchone()[0] == 1
        assert target.execute('SELECT count(*) FROM obs.event_rev').fetchone()[0] == 2
    with open_source(bridge_dbs[0]) as source:
        assert source.read_only is True
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            source.execute("INSERT INTO ref.train VALUES ('99999','forbidden','UNKNOWN')")


@pytest.mark.db
def test_skip_unknown_and_parse_failures_advances_cursor(bridge_dbs, capsys):
    from msynd.ops.pilot_bridge import bridge_once

    seed(bridge_dbs)
    with psycopg.connect(bridge_dbs[0]) as source:
        for key, payload in [('99999/2026-10-07', body('99999')),
                             ('12951/2026-10-07', b'{}'),
                             ('bad-key', body()), ('12951/2026-10-07', b'not zstd')]:
            insert_fetch(source, host='synthetic', key=key, received_at=RECEIVED,
                         body_zstd=payload if payload == b'not zstd' else zstandard.compress(payload))
        insert_fetch(source, host='synthetic', key='ignored', purpose='token')
        insert_fetch(source, host='synthetic', key='ignored', outcome='invalid')
    assert bridge_once(*bridge_dbs) == 1
    summary = capsys.readouterr().out
    assert 'no_version=1' in summary and 'invalid=3' in summary
    assert bridge_once(*bridge_dbs) == 0
    assert 'no_version=0 invalid=0' in capsys.readouterr().out


@pytest.mark.db
def test_failed_store_rolls_back_row_and_cursor(bridge_dbs, monkeypatch):
    from msynd.ops import pilot_bridge as bridge

    seed(bridge_dbs)
    real_store = bridge.store_status

    def fail(*args):
        real_store(*args)
        raise RuntimeError('interrupted')

    monkeypatch.setattr(bridge, 'store_status', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        bridge.bridge_once(*bridge_dbs)
    with psycopg.connect(bridge_dbs[1]) as target:
        assert target.execute('SELECT count(*) FROM obs.fetch').fetchone()[0] == 0
        assert target.execute('SELECT count(*) FROM obs.event_rev').fetchone()[0] == 0
        assert target.execute('SELECT count(*) FROM obs.run').fetchone()[0] == 0
    monkeypatch.setattr(bridge, 'store_status', real_store)
    assert bridge.bridge_once(*bridge_dbs) == 1


@pytest.mark.db
@pytest.mark.parametrize('invalid_version', ['future', 'expired', 'weekday', 'no_route', None])
def test_bridge_uses_lifecycle_version_selection(bridge_dbs, invalid_version):
    from msynd.ops.pilot_bridge import bridge_once

    seed(bridge_dbs)
    with psycopg.connect(bridge_dbs[1]) as target:
        old = target.execute('SELECT id FROM ref.schedule_version').fetchone()[0]
        version = target.execute('''INSERT INTO ref.schedule_version
            (train,valid_from,days_of_run,source,content_hash)
            VALUES ('12951','2026-10-07',127,'gtfs','new') RETURNING id''').fetchone()[0]
        target.execute('''INSERT INTO ref.schedule_stop
            SELECT %s,idx,station,occ,arr_off,dep_off,dist_km FROM ref.schedule_stop WHERE version_id=%s''',
            (version, old))
        if invalid_version == 'future':
            target.execute("UPDATE ref.schedule_version SET valid_from='2026-10-08' WHERE id=%s", (version,))
        elif invalid_version == 'expired':
            target.execute("UPDATE ref.schedule_version SET valid_from='2026-10-06',valid_to='2026-10-06' WHERE id=%s", (version,))
        elif invalid_version == 'weekday':
            target.execute('UPDATE ref.schedule_version SET days_of_run=2 WHERE id=%s', (version,))
        if invalid_version != 'no_route':
            for digest in (b'first', b'last'):
                route = target.execute('''INSERT INTO ref.route_version(schedule_version_id,content_hash,dist_src)
                    VALUES (%s,%s,'gtfs') RETURNING id''', (version, digest)).fetchone()[0]
                target.execute('''INSERT INTO ref.route_stop SELECT %s,idx,station,occ,halt,dist_km,sched_off
                    FROM ref.route_stop WHERE route_version_id=(
                        SELECT id FROM ref.route_version WHERE schedule_version_id=%s LIMIT 1)''', (route, old))
    assert bridge_once(*bridge_dbs) == (0 if invalid_version == 'no_route' else 1)
    with psycopg.connect(bridge_dbs[1]) as target:
        run = target.execute('SELECT schedule_version_id,route_version_id FROM obs.run').fetchone()
        if invalid_version == 'no_route':
            assert run is None
        elif invalid_version is None:
            assert run == (version, route)
        else:
            assert run[0] == old


def test_cli_follow_sleeps_between_passes(monkeypatch):
    from msynd.ops import pilot_bridge as bridge

    calls = []

    def once(source, target):
        calls.append((source, target))
        if len(calls) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(bridge, 'bridge_once', once)
    monkeypatch.setattr(bridge, 'sleep', calls.append)
    with pytest.raises(KeyboardInterrupt):
        bridge.main(['--source', 'test-source', '--target', 'test-target', '--follow'])
    assert calls == [('test-source', 'test-target'), 60, ('test-source', 'test-target')]
    calls.clear()
    bridge.main(['--source', 'test-source', '--target', 'test-target'])
    assert calls == [('test-source', 'test-target')]


@pytest.mark.parametrize('workers', [None, '2'])
@pytest.mark.parametrize('check', ['s_tables', 'pending_replay'])
def test_engine_main_configuration(monkeypatch, workers, check):
    from msynd import db
    from msynd.config import Settings
    from msynd.engine import __main__ as engine

    monkeypatch.setenv('N_PARTICLES', '64')
    monkeypatch.setenv('K_PATHS', '8')
    if workers is None:
        monkeypatch.delenv('ENGINE_WORKERS', raising=False)
    else:
        monkeypatch.setenv('ENGINE_WORKERS', workers)
    monkeypatch.setattr(engine, 'settings', Settings.from_env())
    pool, conn = object(), object()
    calls = []

    @contextmanager
    def connect(url):
        yield conn

    @contextmanager
    def Pool(n):
        calls.append(n)
        yield pool

    def run_loop(connection, replay, models, params, *, pool):
        assert connection is conn
        assert (params.n, params.k) == (64, 8)
        assert pool is pool_object
        if check == 's_tables':
            assert params.s_tables == json.loads(
                (Path(__file__).resolve().parents[2] / 'data/params.json').read_text())['S']
        else:
            from msynd.engine.evidence import Evidence
            from msynd.engine.worker import _job
            from tests.engine.test_replay import fixture

            ev, tables, _ = fixture()
            ev = replace(ev, evidence=(Evidence(1, False, 202, 'mmt_status'),))
            record = dict(evidence=ev, origin_ts=0, lifecycle='running',
                          last_received=RECEIVED, asof=RECEIVED,
                          diverted_from=None, part_cancelled=False)
            _, frame, result = _job((record, tables, params, replay))
            assert frame['state'] == 'observed', frame
            assert result is not None
            assert frame['path_shape'] == (8, 4, 2)
        calls.append('loop')

    pool_object = pool
    monkeypatch.setattr(db, 'connect', connect)
    monkeypatch.setattr(engine.mp, 'get_context', lambda method: type('Context', (), {'Pool': staticmethod(Pool)})())
    monkeypatch.setattr(engine, 'run_loop', run_loop)
    engine.main()
    assert calls == [int(workers or 4), 'loop']
