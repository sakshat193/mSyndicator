import base64
import csv
from datetime import date, datetime, timedelta, timezone
import json
import math

import psycopg
from psycopg import sql
import pytest
import zstandard

from tests.helpers import insert_fetch, make_run
from tests.ops.test_pilot_bridge import body, bridge_dbs

pytestmark = pytest.mark.db
TABLES = ('station', 'train', 'schedule_version', 'schedule_stop', 'route_version',
          'route_stop', 'segment', 'route_segment', 'calendar')
FIRST = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)


def seed(source):
    with psycopg.connect(source) as conn:
        for train in ('12951', '12952'):
            run = make_run(conn, train=train,
                           halts=[('MMCT', None, 360, 0), ('NDLS', 620, None, 100)])
            conn.execute('DELETE FROM obs.run WHERE id=%s', (run,))
        segment = conn.execute("""INSERT INTO ref.segment(stn_lo,stn_hi,polyline,osm_km,confidence)
            VALUES ('MMCT','NDLS','test,line',101,.5) RETURNING id""").fetchone()[0]
        conn.execute('INSERT INTO ref.route_segment SELECT id,0,%s,true FROM ref.route_version', (segment,))
        conn.execute("INSERT INTO ref.calendar VALUES ('2026-10-07','test,holiday',true)")


def fetch(conn, received, train='12951', payload=None, **kwargs):
    return insert_fetch(conn, host='synthetic', key=f'{train}/2026-10-07',
                        started_at=received, received_at=received, http_status=200,
                        body_zstd=zstandard.compress(payload if payload is not None else body(train)), **kwargs)


def running_body(train='12951', departure='06:00'):
    payload = json.loads(body(train))
    payload['response']['trainDetails']['terminated'] = False
    payload['response']['stations'][0]['departureDetails']['actDepTime'] = departure
    arrival = payload['response']['stations'][1]['arrivalDetails']
    arrival.update(arrived=False, schArrTime='10:20', actArrTime='10:20')
    return json.dumps(payload).encode()


def test_export_ist_order_ref_and_read_only(bridge_dbs, tmp_path, monkeypatch):
    from msynd.ops import replay_day
    from msynd.ops.pilot_bridge import open_source
    from contextlib import contextmanager

    seed(bridge_dbs[0])
    with psycopg.connect(bridge_dbs[0]) as conn:
        late = fetch(conn, FIRST)
        early_time = datetime(2026, 10, 6, 18, 30, tzinfo=timezone.utc)
        early = fetch(conn, early_time)
        tied = fetch(conn, FIRST, train='12952')
        fetch(conn, early_time - timedelta(microseconds=1))
        fetch(conn, early_time + timedelta(days=1))
        fetch(conn, FIRST, purpose='token')
        fetch(conn, FIRST, outcome='invalid')

    @contextmanager
    def checked_source(dsn):
        with open_source(dsn) as conn:
            assert conn.read_only
            yield conn

    monkeypatch.setattr(replay_day, 'open_source', checked_source)
    replay_day.main(['export', '--source', bridge_dbs[0], '--day', '2026-10-07', '--out', str(tmp_path)])
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['day'] == '2026-10-07'
    assert manifest['fetch_count'] == 3
    assert manifest['fetch_id_range'] == [late, tied]
    rows = [json.loads(line) for line in (tmp_path / 'fetches.jsonl').read_text().splitlines()]
    assert [row['received_at'] for row in rows] == [early_time.isoformat(), FIRST.isoformat(), FIRST.isoformat()]
    assert [row['key'] for row in rows] == ['12951/2026-10-07', '12951/2026-10-07', '12952/2026-10-07']
    assert [base64.b64decode(row['body_zstd']) for row in rows] == [
        zstandard.compress(body()), zstandard.compress(body()), zstandard.compress(body('12952'))]
    assert set(manifest['ref_rows']) == set(TABLES)
    for table in TABLES:
        with (tmp_path / 'ref' / f'{table}.csv').open(newline='') as file:
            assert len(list(csv.reader(file))) - 1 == manifest['ref_rows'][table]
    with psycopg.connect(bridge_dbs[0]) as conn:
        assert conn.execute('SELECT count(*) FROM obs.fetch').fetchone()[0] == 7
        assert conn.execute('SELECT count(*) FROM obs.run').fetchone()[0] == 0


def test_replay_ref_round_trip_and_sequences(bridge_dbs, tmp_path):
    from msynd.ops.replay_day import export, replay
    from msynd.engine.replay import Params

    seed(bridge_dbs[0])
    # An empty day still restores its reference data and writes a report.
    export(bridge_dbs[0], date(2026, 10, 7), tmp_path)
    station_file = tmp_path / 'ref' / 'station.csv'
    with station_file.open(newline='') as file:
        station_rows = list(csv.reader(file))
    with station_file.open('w', newline='') as file:
        csv.writer(file).writerows(row[::-1] for row in station_rows)
    result = replay(tmp_path, bridge_dbs[1], speed=1e6, params=Params(n=200, k=4), workers=0)
    assert result['runs_discovered'] == result['runs_framed'] == 0
    assert result['runs_unframed'] == 0
    with psycopg.connect(bridge_dbs[0]) as source, psycopg.connect(bridge_dbs[1]) as target:
        for table in TABLES:
            query = sql.SQL('SELECT * FROM ref.{} ORDER BY 1,2').format(sql.Identifier(table))
            assert source.execute(query).fetchall() == target.execute(query).fetchall()
        version = target.execute("""INSERT INTO ref.schedule_version
            (train,valid_from,days_of_run,source,content_hash)
            VALUES ('12951','2026-10-08',127,'gtfs','new') RETURNING id""").fetchone()[0]
        assert version > 2
        route = target.execute("""INSERT INTO ref.route_version(schedule_version_id,dist_src,content_hash)
            VALUES (%s,'gtfs','new') RETURNING id""", (version,)).fetchone()[0]
        assert route > 2
        target.execute("INSERT INTO ref.station(code,name) VALUES ('ZZZ','new')")
        segment = target.execute("""INSERT INTO ref.segment(stn_lo,stn_hi,confidence)
            VALUES ('NDLS','ZZZ',0) RETURNING id""").fetchone()[0]
        assert segment > 1
    assert json.loads((tmp_path / 'report.json').read_text()) == result


def test_replay_evidence_frames_periodic_and_a6(bridge_dbs, tmp_path, monkeypatch):
    from msynd.ops import replay_day
    from msynd.engine.replay import Params

    seed(bridge_dbs[0])
    with psycopg.connect(bridge_dbs[0]) as conn:
        fetch(conn, FIRST, payload=running_body())
        fetch(conn, FIRST, train='12952', payload=running_body('12952'))
        fetch(conn, FIRST + timedelta(minutes=2), payload=b'{}')
        fetch(conn, FIRST + timedelta(minutes=3), train='99999')
        fetch(conn, FIRST + timedelta(minutes=62), payload=running_body(departure='06:01'))
        fetch(conn, FIRST + timedelta(minutes=63), train='12952')

    replay_day.export(bridge_dbs[0], date(2026, 10, 7), tmp_path)
    # Advance only on sleeps: compilation must not skip the synthetic day's quiet intervals.
    elapsed = [0.0]
    monkeypatch.setattr(replay_day, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(replay_day, 'sleep', lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    result = replay_day.replay(tmp_path, bridge_dbs[1], speed=1e6,
                              params=Params(n=200, k=4), workers=0)
    assert (result['applied'], result['invalid'], result['no_version']) == (4, 1, 1)
    assert result['invalid_share'] == 1 / 6
    assert result['runs_discovered'] == result['runs_framed'] == 2
    assert result['runs_unframed'] == 0
    assert result['replay_errors'] == 0
    assert result['a1']['minutes'] >= 63
    assert result['forecast_issues'] > 0
    assert result['a6']['count'] == 4
    assert all(math.isfinite(result['a6'][key]) and result['a6'][key] >= 0 for key in ('p50', 'p95', 'max'))
    assert json.loads((tmp_path / 'report.json').read_text()) == result
    with psycopg.connect(bridge_dbs[1]) as conn:
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM obs.event_rev').fetchone()[0] == 4
        assert conn.execute('SELECT min(received_at) FROM obs.event_rev').fetchone()[0] == FIRST
        times = [row[0] for row in conn.execute("SELECT DISTINCT ts FROM sim.metric WHERE name LIKE 'a1_%'")]
        assert FIRST <= min(times) < max(times) <= FIRST + timedelta(minutes=63)
        assert all(ts.second == ts.microsecond == 0 for ts in times)
        assert conn.execute('SELECT count(*) FROM obs.poll_queue').fetchone()[0] > 0
        assert conn.execute('SELECT min(issued_at) FROM sim.forecast_issue').fetchone()[0] == FIRST + timedelta(hours=1)


def test_replay_drains_completed_run_and_times_only_changed_stores(bridge_dbs, tmp_path):
    from msynd.ops.replay_day import export, replay
    from msynd.engine.replay import Params

    seed(bridge_dbs[0])
    with psycopg.connect(bridge_dbs[0]) as conn:
        fetch(conn, FIRST)
        fetch(conn, FIRST)
    export(bridge_dbs[0], date(2026, 10, 7), tmp_path)
    result = replay(tmp_path, bridge_dbs[1], speed=1e6, params=Params(n=200, k=4), workers=0)
    assert result['applied'] == 2
    assert result['runs_discovered'] == result['runs_framed'] == 1
    assert result['runs_unframed'] == 0
    assert result['a6']['count'] == 1
    assert 0 < result['a6']['p50'] <= result['a6']['p95'] <= result['a6']['max']
    with psycopg.connect(bridge_dbs[1]) as conn:
        assert conn.execute('SELECT count(*) FROM obs.event_rev').fetchone()[0] == 2
        assert conn.execute('SELECT count(*) FROM obs.dirty_run').fetchone()[0] == 0


def test_apply_status_is_shared_and_caller_owns_transaction(bridge_dbs):
    from msynd.ops.pilot_bridge import apply_status

    seed(bridge_dbs[1])
    with psycopg.connect(bridge_dbs[1]) as conn:
        row = ('synthetic', 'status', '12951/2026-10-07', FIRST, FIRST, 'ok', 200, zstandard.compress(body()))
        assert apply_status(conn, row) == 'applied'
        conn.rollback()
        assert conn.execute('SELECT count(*) FROM obs.run').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM obs.fetch').fetchone()[0] == 0
        assert apply_status(conn, row[:-1] + (b'bad zstd',)) == 'invalid'
        assert apply_status(conn, row[:2] + ('99999/2026-10-07',) + row[3:-1] + (zstandard.compress(body('99999')),)) == 'no_version'
