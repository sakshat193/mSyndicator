import gzip
import hashlib
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import msgpack
import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from msynd.engine.frames import write_frame
from tests.helpers import insert_event, insert_fetch, make_run

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


@pytest.fixture
def halt_run():
    qs = np.arange(42, dtype='<f4').reshape(3, 2, 7)
    qs[0, 0] = np.nan
    qs[1, 1, 3] = 6139.6
    stored = {'state': 'observed', 'halt_q': qs.tobytes(), 'halt_shape': [3, 2, 7]}
    projected = {'state': 'observed',
                 'halt_p50': np.asarray([[np.nan, 10], [17, 6139.6], [31, 38]], '<f4').tobytes()}
    return stored, projected


def test_live_full_delta_and_resync_carry_only_halt_p50(api, halt_run):
    conn, client, app, run, other, *_ = api
    stored, projected = halt_run
    error = {'state': 'error', 'error': 'bad evidence'}
    full = frame(conn, True, {run: stored, other: error})
    wait_read(app, full)
    with client.websocket_connect('/api/live') as ws:
        ws.send_json({'gen': 0})
        baseline = receive(ws)
        assert baseline['is_full'] and baseline['runs'] == {run: projected, other: error}
        delta = frame(conn, False, {run: stored | {'state': 'stale'}})
        update = receive(ws)
        assert update['gen'] == delta and not update['is_full']
        assert update['runs'] == {run: projected | {'state': 'stale'}}
        ws.send_json({'gen': full})
        assert receive(ws) == update
        ws.send_json({'gen': 0})
        assert receive(ws) == baseline
        assert receive(ws) == update
    assert app.state.frames.runs == {run: projected | {'state': 'stale'}, other: error}
    assert 'halt_q' in stored and 'halt_shape' in stored


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


def test_slow_client_coalesces_to_full(api, halt_run):
    conn, client, app, run, other, *_ = api
    stored, projected = halt_run
    full = frame(conn, True, {run: stored, other: {'state': 'observed'}})
    wait_read(app, full)
    queue = client.portal.call(app.state.frames.subscribe, full)
    try:
        frame(conn, False, {run: {'state': 'stale'}})
        frame(conn, False, tombstones=[other])
        last = frame(conn, False, {run: stored | {'state': 'stale'}})
        wait_read(app, last)
        result = msgpack.unpackb(client.portal.call(queue.get), raw=False, strict_map_key=False)
        assert result['is_full'] and result['gen'] == last
        assert result['runs'] == {run: projected | {'state': 'stale'}}
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


def test_contiguous_full_is_delta_for_connected_client_and_full_for_resync(api):
    conn, client, app, run, other, *_ = api
    unchanged = {'state': 'schedule_only', 'base_ts': 1000,
                 'paths': b'\x00\x00\x00\x00', 'path_shape': [1, 1, 2],
                 'versions': {'route': 7}, 'halt_shape': [1, 2, 7],
                 'halt_q': np.asarray([[[np.nan] * 7, [1.5] * 7]], '<f4').tobytes(),
                 'next_q': np.full(7, np.nan, '<f4').tobytes(),
                 'headline': {'last_received': None}}
    projected = {key: value for key, value in unchanged.items() if key not in ('halt_q', 'halt_shape')}
    projected['halt_p50'] = np.asarray([[np.nan, 1.5]], '<f4').tobytes()
    full = frame(conn, True, {run: unchanged, other: unchanged | {'state': 'observed'},
                             999: {'state': 'expired'}})
    wait_read(app, full)
    queue = client.portal.call(app.state.frames.subscribe, full)
    try:
        # Reordering a decoded dictionary is not a payload change.
        same = dict(reversed(list(unchanged.items())))
        qs = np.frombuffer(same['halt_q'], '<f4').copy()
        qs[7] = -20  # A change outside P50 must not resend the run.
        same['halt_q'] = qs.tobytes()
        changed = unchanged | {'state': 'observed', 'paths': b'\x00\x00\x01\x00'}
        projected_changed = projected | {'state': 'observed', 'paths': b'\x00\x00\x01\x00'}
        updated = {run: same, other: changed, 1000: {'state': 'observed'}}
        projected_updated = {run: projected, other: projected_changed, 1000: {'state': 'observed'}}
        last = write_frame(conn, True, {'static': 'network-v2', 'model': 5}, updated, ())
        conn.execute('NOTIFY sim_frame')
        conn.commit()
        wait_read(app, last)
        result = msgpack.unpackb(client.portal.call(queue.get), raw=False, strict_map_key=False)
        baseline = app.state.frames.history[-1]
        assert result == baseline | {'is_full': False,
                                     'runs': {other: projected_changed, 1000: {'state': 'observed'}},
                                     'tombstones': [999]}
        assert baseline['is_full'] and baseline['runs'] == projected_updated
        assert app.state.frames.runs == projected_updated
        with client.websocket_connect('/api/live') as ws:
            ws.send_json({'gen': 0})
            assert receive(ws) == baseline
        with client.websocket_connect('/api/live') as ws:
            ws.send_json({'gen': last + 100})
            assert receive(ws) == baseline
        # A full frame with no changed runs must still advance connected clients.
        last = frame(conn, True, updated)
        wait_read(app, last)
        empty = msgpack.unpackb(client.portal.call(queue.get), raw=False, strict_map_key=False)
        assert empty['gen'] == last and not empty['is_full']
        assert empty['runs'] == {} and empty['tombstones'] == []
    finally:
        client.portal.call(app.state.frames.unsubscribe, queue)


def test_noncontiguous_full_stays_full_for_connected_client(api):
    conn, client, app, run, *_ = api
    full = frame(conn, True, {run: {'state': 'observed'}})
    wait_read(app, full)
    queue = client.portal.call(app.state.frames.subscribe, full)
    try:
        conn.execute("SELECT nextval(pg_get_serial_sequence('sim.frame','gen'))")
        conn.rollback()
        last = frame(conn, True, {run: {'state': 'stale'}})
        wait_read(app, last)
        result = msgpack.unpackb(client.portal.call(queue.get), raw=False, strict_map_key=False)
        assert result['is_full'] and result['gen'] == last
        assert result['runs'] == {run: {'state': 'stale'}}
    finally:
        client.portal.call(app.state.frames.unsubscribe, queue)


def test_train_number_ambiguous_lists_runs(api):
    _, client, _, run, other, train, _ = api
    assert [r['id'] for r in client.get(f'/api/train/{train}').json()['runs']] == [run, other]
    assert [r['id'] for r in client.get(f'/api/train/{train}?date=2026-10-07').json()['runs']] == [run]
    assert client.get('/api/train/missing').status_code == 404


def test_run_endpoint_all_remaining_halts_and_eta_age(api):
    conn, client, app, run, other, *_ = api
    now = conn.execute('SELECT clock_timestamp()').fetchone()[0]
    qs = [[None] * 7, [None] * 7], [[310.123456 + q for q in range(7)], [315.] * 7], [[950.] * 6 + [None], [None] * 7]
    insert_event(conn, run, 'MMCT', 1, 'dep', now, now - timedelta(minutes=2))
    insert_event(conn, run, 'MMCT', 1, 'dep', now + timedelta(minutes=1), now - timedelta(minutes=1))
    for age in (20, 10):
        conn.execute("INSERT INTO obs.ntes_eta_rev VALUES (%s,'BRC',1,'arr',%s,NULL,%s)", (run, now + timedelta(hours=1), now - timedelta(minutes=age)))
    gen = frame(conn, True, {run: {'state': 'observed', 'halt_shape': [3, 2, 7],
                                 'halt_q': np.asarray(qs, '<f4').tobytes()}})
    wait_read(app, gen)
    result = client.get(f'/api/run/{run}').json()
    assert [s['idx'] for s in result['schedule']] == [0, 1, 2]
    assert [s['idx'] for s in result['halts']] == [1, 2]
    assert result['halts'] == [
        {'idx': 1, 'station': 'BRC', 'occ': 1,
         'arr': [float(np.float32(310.123456 + q)) for q in range(7)], 'dep': [315.] * 7},
        {'idx': 2, 'station': 'NDLS', 'occ': 1, 'arr': [950.] * 6 + [None], 'dep': [None] * 7},
    ]
    assert client.get(f'/api/run/{other}').json()['halts'] == []
    assert len(result['eta']) == 1 and 600 <= result['eta'][0]['age'] < 620
    assert len(result['revisions']) == 2
    assert result['lifecycle'] == 'scheduled' and result['evidence_state'] == 'observed'
    assert client.get('/api/run/999999999').status_code == 404


def test_run_endpoint_reads_only_latest_full_and_following_deltas(api, monkeypatch):
    conn, client, app, run, other, *_ = api
    frame(conn, True, {run: {'state': 'observed'}})
    frame(conn, False, {other: {'state': 'observed'}})
    frame(conn, True, {run: {'state': 'observed'}})
    qs = np.full((3, 2, 7), 315., '<f4')
    gen = frame(conn, False, {run: {'state': 'stale', 'halt_shape': [3, 2, 7], 'halt_q': qs.tobytes()}})
    wait_read(app, gen)
    execute = psycopg.Connection.execute
    fetched = []

    def track(self, query, *args, **kwargs):
        cursor = execute(self, query, *args, **kwargs)
        if isinstance(query, str) and query.startswith('SELECT payload FROM sim.frame'):
            fetched.append(cursor.rowcount)
        return cursor

    monkeypatch.setattr(psycopg.Connection, 'execute', track)
    result = client.get(f'/api/run/{run}').json()
    assert result['evidence_state'] == 'stale'
    assert [h['idx'] for h in result['halts']] == [0, 1, 2]
    assert all(h['arr'] == h['dep'] == [315.] * 7 for h in result['halts'])
    result = client.get(f'/api/run/{other}').json()
    assert result['evidence_state'] == 'schedule_only' and result['halts'] == []
    assert fetched == [2, 2]


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


@pytest.mark.parametrize('seeded', [False, True])
def test_health_source_freshness_uses_only_recent_ok_mmt_status_by_mode(api, seeded):
    from msynd.collect.mmt import HOST

    conn, client, *_ = api
    poll = NOW - timedelta(minutes=20)
    harvest = NOW - timedelta(minutes=10)
    fetches = []
    try:
        if seeded:
            for values in [
                {'mode': 'poll', 'received_at': poll},
                {'mode': 'poll', 'received_at': poll - timedelta(minutes=1)},
                {'mode': 'poll', 'outcome': 'transport_error', 'received_at': NOW},
                {'mode': 'harvest', 'received_at': harvest},
                {'mode': 'harvest', 'outcome': 'invalid', 'received_at': NOW},
                {'mode': 'poll', 'host': 'other', 'received_at': NOW},
                {'mode': 'poll', 'purpose': 'token', 'received_at': NOW},
                {'mode': 'probe', 'received_at': NOW},
                {'mode': 'poll', 'started_at': NOW - timedelta(days=3), 'received_at': NOW},
            ]:
                fetches.append(insert_fetch(conn, **({'host': HOST, 'started_at': poll} | values)))
            conn.commit()
        source = client.get('/api/health').json()['source']
        assert set(source) == {'poll_ok_at', 'harvest_ok_at'}
        if seeded:
            assert datetime.fromisoformat(source['poll_ok_at']) == poll
            assert datetime.fromisoformat(source['harvest_ok_at']) == harvest
        else:
            assert source == {'poll_ok_at': None, 'harvest_ok_at': None}
    finally:
        conn.execute('DELETE FROM obs.fetch WHERE id=ANY(%s)', (fetches,))


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


def test_full_resync_preserves_per_run_time_bases(api):
    conn, _, app, run, other, *_ = api
    runs = {
        run: {'base_ts': NOW.timestamp(), 'paths': b'\x03\x00\x02\x00\x03\x00\x04\x00',
              'path_shape': [2, 1, 2]},
        other: {'base_ts': (NOW + timedelta(days=6)).timestamp(),
                'paths': b'\x03\x00\x02\x00\x03\x00\x04\x00', 'path_shape': [2, 1, 2]},
    }
    frame(conn, True, {run: runs[run]})
    last = frame(conn, False, {other: runs[other]})
    wait_read(app, last)

    result = msgpack.unpackb(app.state.frames.full(), raw=False, strict_map_key=False)
    assert result['is_full'] and result['gen'] == last
    assert result['base_ts'] == NOW.timestamp()
    assert result['runs'] == runs


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
        assert result['runs'][run]['base_ts'] == 1060
        assert struct.unpack('<HH', result['runs'][run]['paths']) == (2, 15)
        assert result['runs'][other]['base_ts'] == 1000
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
