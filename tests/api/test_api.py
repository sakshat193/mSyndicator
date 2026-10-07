import gzip
import hashlib
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import msgpack
import psycopg
import pytest
from fastapi.testclient import TestClient

from msynd.engine.frames import write_frame
from tests.helpers import insert_event, make_run

pytestmark = pytest.mark.db
NOW = datetime.now(timezone.utc)


@pytest.fixture
def api(db_url, tmp_path):
    from msynd.api.app import create_app

    with psycopg.connect(db_url) as conn:
        conn.execute('TRUNCATE sim.frame, sim.metric')
        existing_stations = {r[0] for r in conn.execute('SELECT code FROM ref.station')}
        train = uuid4().hex[:10]
        run = make_run(conn, train=train)
        binding = conn.execute('SELECT schedule_version_id,route_version_id FROM obs.run WHERE id=%s', (run,)).fetchone()
        other = conn.execute("INSERT INTO obs.run(train,start_date,schedule_version_id,route_version_id) VALUES (%s,'2026-10-08',%s,%s) RETURNING id", (train, *binding)).fetchone()[0]
        conn.commit()
        body = b'{"stations":[],"segments":[],"routes":[]}'
        etag = hashlib.sha256(body).hexdigest()
        (tmp_path / f'network-{etag[:12]}.json.gz').write_bytes(gzip.compress(body, mtime=0))
        app = create_app(db_url, tmp_path)
        try:
            with TestClient(app) as client:
                yield conn, client, app, run, other, train, etag
        finally:
            conn.execute('DELETE FROM obs.event_rev WHERE run_id=ANY(%s)', ([run, other],))
            conn.execute('DELETE FROM obs.ntes_eta_rev WHERE run_id=ANY(%s)', ([run, other],))
            conn.execute('DELETE FROM obs.dirty_run WHERE run_id=ANY(%s)', ([run, other],))
            conn.execute('DELETE FROM obs.run WHERE id=ANY(%s)', ([run, other],))
            conn.execute('DELETE FROM ref.route_stop WHERE route_version_id=%s', (binding[1],))
            conn.execute('DELETE FROM ref.route_version WHERE id=%s', (binding[1],))
            conn.execute('DELETE FROM ref.schedule_stop WHERE version_id=%s', (binding[0],))
            conn.execute('DELETE FROM ref.schedule_version WHERE id=%s', (binding[0],))
            conn.execute('DELETE FROM ref.train WHERE number=%s', (train,))
            for station in {'MMCT', 'BRC', 'NDLS'} - existing_stations:
                conn.execute('DELETE FROM ref.station WHERE code=%s', (station,))
            conn.execute('TRUNCATE sim.frame, sim.metric')
            conn.commit()


def frame(conn, full, runs=None, tombstones=(), notify=True):
    gen = write_frame(conn, full, {}, runs or {}, tombstones)
    if notify:
        conn.execute('NOTIFY sim_frame')
    conn.commit()
    return gen


def wait_read(app, gen, timeout=2):
    deadline = time.monotonic() + timeout
    while app.state.frames.gen < gen and time.monotonic() < deadline:
        time.sleep(.01)
    assert app.state.frames.gen == gen


def receive(ws):
    return msgpack.unpackb(ws.receive_bytes(), raw=False, strict_map_key=False)


def test_gap_triggers_full_resync(api):
    conn, client, app, run, *_ = api
    full = frame(conn, True, {run: {'state': 'observed'}})
    delta = frame(conn, False, {run: {'state': 'stale'}})
    wait_read(app, delta)
    with client.websocket_connect('/api/live') as ws:
        ws.send_json({'gen': delta + 100})
        assert receive(ws)['gen'] == full
        assert receive(ws)['gen'] == delta
    with client.websocket_connect('/api/live') as ws:
        ws.send_json({'gen': full})
        result = receive(ws)
        assert result['gen'] == delta and not result['is_full']
        # A running client can explicitly request resynchronization.
        ws.send_json({'gen': 0})
        assert receive(ws)['is_full']


def test_slow_client_coalesces_to_full(api):
    conn, client, app, run, other, *_ = api
    full = frame(conn, True, {run: {'state': 'observed'}, other: {'state': 'observed'}})
    wait_read(app, full)
    queue = client.portal.call(app.state.frames.subscribe, full)
    try:
        frame(conn, False, {run: {'state': 'stale'}})
        frame(conn, False, tombstones=[other])
        last = frame(conn, False, {run: {'state': 'error'}})
        wait_read(app, last)
        result = msgpack.unpackb(client.portal.call(queue.get), raw=False, strict_map_key=False)
        assert result['is_full'] and result['gen'] == last
        assert result['runs'] == {run: {'state': 'error'}}
        with client.websocket_connect('/api/live') as ws:
            ws.send_json({'gen': full})
            assert receive(ws) == result
    finally:
        client.portal.call(app.state.frames.unsubscribe, queue)


def test_tombstone_forwarded(api):
    conn, client, app, run, *_ = api
    full = frame(conn, True, {run: {'state': 'observed'}})
    wait_read(app, full)
    with client.websocket_connect('/api/live') as ws:
        ws.send_json({'gen': full})
        delta = frame(conn, False, tombstones=[run])
        result = receive(ws)
        assert result['gen'] == delta and result['tombstones'] == [run]


def test_train_number_ambiguous_lists_runs(api):
    _, client, _, run, other, train, _ = api
    assert [r['id'] for r in client.get(f'/api/train/{train}').json()['runs']] == [run, other]
    assert [r['id'] for r in client.get(f'/api/train/{train}?date=2026-10-07').json()['runs']] == [run]
    assert client.get('/api/train/missing').status_code == 404


def test_run_endpoint_all_remaining_halts_and_eta_age(api):
    conn, client, app, run, *_ = api
    qs = [[None] * 7, [None] * 7], [[310.] * 7, [315.] * 7], [[950.] * 7, [None] * 7]
    insert_event(conn, run, 'MMCT', 1, 'dep', NOW, NOW - timedelta(minutes=2))
    insert_event(conn, run, 'MMCT', 1, 'dep', NOW + timedelta(minutes=1), NOW - timedelta(minutes=1))
    for age in (20, 10):
        conn.execute("INSERT INTO obs.ntes_eta_rev VALUES (%s,'BRC',1,'arr',%s,NULL,%s)", (run, NOW + timedelta(hours=1), NOW - timedelta(minutes=age)))
    gen = frame(conn, True, {run: {'state': 'observed', 'halt_q': qs}})
    wait_read(app, gen)
    result = client.get(f'/api/run/{run}').json()
    assert [s['idx'] for s in result['schedule']] == [0, 1, 2]
    assert [s['idx'] for s in result['halts']] == [1, 2]
    assert result['halts'][1]['arr'] == [950.] * 7
    assert len(result['eta']) == 1 and 600 <= result['eta'][0]['age'] < 620
    assert len(result['revisions']) == 2
    assert result['lifecycle'] == 'scheduled' and result['evidence_state'] == 'observed'
    assert client.get('/api/run/999999999').status_code == 404


def test_replay_window_max_24h(api):
    conn, client, _, run, *_ = api
    insert_event(conn, run, 'BRC', 1, 'arr', NOW, NOW - timedelta(minutes=2))
    insert_event(conn, run, 'BRC', 1, 'arr', NOW + timedelta(minutes=1), NOW - timedelta(minutes=1))
    conn.commit()
    def replay(start, end):
        return client.get('/api/replay', params={'from': start.isoformat(), 'to': end.isoformat()})
    result = replay(NOW - timedelta(hours=12), NOW + timedelta(hours=12))
    assert result.status_code == 200
    assert result.json()['label'] == 'station reports; motion reconstructed'
    assert len(result.json()['events']) == 1
    assert result.json()['runs'][0]['id'] == run
    assert replay(NOW, NOW + timedelta(hours=24, seconds=1)).status_code == 422
    assert replay(NOW, NOW - timedelta(seconds=1)).status_code == 422
    assert replay(NOW.replace(tzinfo=None), NOW).status_code == 422


def test_static_etag_304(api):
    _, client, _, *_, etag = api
    response = client.get('/api/static/network')
    assert response.status_code == 200 and response.json()['stations'] == []
    assert response.headers['etag'] == f'"{etag}"'
    assert client.get('/api/static/network', headers={'If-None-Match': f'"{etag}"'}).status_code == 304
    assert client.get('/api/static/network', params={'v': etag}).status_code == 200
    assert client.get('/api/static/network?v=../../anything').status_code == 404


def test_health_metrics_and_search(api):
    conn, client, _, run, _, train, _ = api
    for name, age, value in [('dispatcher_state:one', 2, 0), ('dispatcher_state:one', 1, 1), ('dispatcher_state:two', 1, 2)]:
        conn.execute('INSERT INTO sim.metric VALUES (%s,%s,%s)', (NOW - timedelta(minutes=age), name, value))
    conn.execute('INSERT INTO obs.dirty_run(run_id) VALUES (%s)', (run,))
    gen = frame(conn, True)
    result = client.get('/api/health').json()
    assert result['dispatchers']['one']['state'] == 'breaker'
    assert result['dispatchers']['two']['state'] == 'stopped'
    assert 'ts' in result['dispatchers']['one']
    assert result['dirty_backlog'] == 1 and result['last_frame']['gen'] == gen
    assert result['last_frame']['age'] >= 0
    assert isinstance(result['active_models'], list)
    assert len(client.get('/api/metrics').json()) == 3
    assert client.get('/api/search', params={'q': train}).json()['trains'][0]['number'] == train
    assert client.get('/api/search?q=MMCT').json()['stations'][0]['code'] == 'MMCT'


def test_notify_reads_before_send_and_poll_recovers(api):
    conn, _, app, *_ = api
    gen = frame(conn, True)
    wait_read(app, gen)  # No client: measures the server-side A10 endpoint.
    gen = frame(conn, False, notify=False)
    wait_read(app, gen, timeout=6)


def test_invalid_websocket_generation(api):
    _, client, *_ = api
    from starlette.websockets import WebSocketDisconnect
    for value in (-1, '1', True):
        with client.websocket_connect('/api/live') as ws:
            ws.send_json({'gen': value})
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_bytes()
            assert error.value.code == 1008


def test_restart_reads_retained_full_and_deltas(api, db_url, tmp_path):
    from msynd.api.app import create_app

    conn, _, _, run, other, *_ = api
    frame(conn, False, {999: {'state': 'expired'}})  # Before the retained baseline.
    frame(conn, True, {run: {'state': 'observed'}, other: {'state': 'observed'}})
    delta = frame(conn, False, {run: {'state': 'stale'}}, [other])
    with TestClient(create_app(db_url, tmp_path)) as client:
        with client.websocket_connect('/api/live') as ws:
            ws.send_json({'gen': delta + 10})
            first, second = receive(ws), receive(ws)
            assert first['is_full'] and 999 not in first['runs']
            assert second['gen'] == delta and second['tombstones'] == [other]


def test_coalesced_paths_preserve_absolute_times(api):
    import struct

    conn, client, app, run, other, *_ = api
    def paths(base, tick):
        return {'base_ts': base, 'paths': struct.pack('<HH', 2, tick), 'path_shape': [1, 1, 2]}
    full = frame(conn, True, {run: paths(1000, 10), other: paths(1000, 20)})
    wait_read(app, full)
    frame(conn, False, {run: paths(1060, 15)})
    frame(conn, False)
    last = frame(conn, False)
    wait_read(app, last)
    with client.websocket_connect('/api/live') as ws:
        ws.send_json({'gen': full})
        result = receive(ws)
        assert result['is_full'] and result['base_ts'] == 1000
        assert struct.unpack('<HH', result['runs'][run]['paths']) == (2, 25)
        assert struct.unpack('<HH', result['runs'][other]['paths']) == (2, 20)


def test_rolled_back_frame_generation_resyncs(api):
    conn, client, app, run, *_ = api
    full = frame(conn, True, {run: {'state': 'observed'}})
    wait_read(app, full)
    # PostgreSQL sequences aren't rolled back with a failed publication.
    conn.execute("SELECT nextval(pg_get_serial_sequence('sim.frame','gen'))")
    conn.rollback()
    delta = frame(conn, False, {run: {'state': 'stale'}})
    wait_read(app, delta)
    with client.websocket_connect('/api/live') as ws:
        ws.send_json({'gen': full})
        assert receive(ws)['gen'] == full
        assert receive(ws)['gen'] == delta
