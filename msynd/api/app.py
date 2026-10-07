"""Read-only API over the durable evidence and Task 31 frame log."""
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import date, datetime, timedelta
import gzip
import hashlib
import logging
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
import msgpack
import numpy as np
import psycopg
from psycopg.rows import dict_row
from starlette.responses import Response

from msynd.config import settings

log = logging.getLogger(__name__)


def unpack(payload):
    return msgpack.unpackb(bytes(payload), raw=False, strict_map_key=False)


class Frames:
    """One process snapshot and a bounded queue per connected client."""

    def __init__(self):
        self.gen = 0
        self.history = []
        self.runs = {}
        self.clients = set()

    def full(self):
        latest = self.history[-1]
        runs = {key: dict(value) for key, value in self.runs.items()}
        bases = [r['base_ts'] for r in runs.values() if 'paths' in r]
        base = min(bases) if bases else None
        for run in runs.values():
            if 'paths' in run:
                paths = np.frombuffer(run['paths'], dtype='<u2').reshape(run['path_shape']).copy()
                ticks = np.rint(paths[:, :, 1].astype(float) + (run['base_ts'] - base) / 6)
                if np.any(ticks < 0) or np.any(ticks > 65535):
                    raise ValueError('display paths outside uint16 frame range')
                paths[:, :, 1] = ticks
                run.update(paths=paths.tobytes(), base_ts=base)
        return msgpack.packb(latest | {'is_full': True, 'runs': runs,
                                     'base_ts': base, 'tombstones': []}, use_bin_type=True)

    def enqueue(self, queue, payload):
        if queue.full():
            while not queue.empty():
                queue.get_nowait()
            payload = self.full()
        queue.put_nowait(payload)

    def ingest(self, payloads):
        for payload in payloads:
            frame = unpack(payload)
            if frame['gen'] <= self.gen:
                continue
            if frame['is_full']:
                self.history.clear()
                self.runs.clear()
            self.history.append(frame)
            self.runs.update(frame['runs'])
            for key in frame['tombstones']:
                self.runs.pop(key, None)
            self.gen = frame['gen']  # A10 endpoint: read/decoded before any send.
            for queue in self.clients:
                self.enqueue(queue, bytes(payload))

    async def subscribe(self, gen):
        queue = asyncio.Queue(maxsize=2)
        self.clients.add(queue)
        self.resync(queue, gen)
        return queue

    async def unsubscribe(self, queue):
        self.clients.discard(queue)

    def resync(self, queue, gen):
        while not queue.empty():
            queue.get_nowait()
        if not self.history or gen == self.gen:
            return
        newer = [f for f in self.history if f['gen'] > gen]
        contiguous = (newer and len(newer) == self.gen - gen
                      and all(f['gen'] == gen + i + 1 for i, f in enumerate(newer)))
        selected = newer if contiguous else self.history
        for frame in selected:
            self.enqueue(queue, msgpack.packb(frame, use_bin_type=True))


def _read_frames(conn, gen):
    # Include the retained full baseline if startup or pruning overtook the reader.
    baseline = conn.execute('SELECT max(gen) FROM sim.frame WHERE is_full').fetchone()[0]
    first = conn.execute('SELECT min(gen) FROM sim.frame WHERE gen > %s', (gen,)).fetchone()[0]
    if first is None:
        return []
    if gen == 0 or first != gen + 1:
        if baseline is None:
            return []  # Startup waits for an engine full frame; never invent missing runs.
        gen = baseline - 1
    return [bytes(r[0]) for r in conn.execute('SELECT payload FROM sim.frame WHERE gen > %s ORDER BY gen', (gen,))]


def create_app(database_url=None, static_dir='data/static'):
    url = database_url or settings.database_url
    folder = Path(static_dir)
    frames = Frames()

    @asynccontextmanager
    async def lifespan(app):
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        stop = threading.Event()
        started = threading.Event()

        def publish(payloads):
            frames.ingest(payloads)
            if not ready.done():
                started.set()
                ready.set_result(None)

        def reader():
            gen = 0
            while not stop.is_set():
                try:
                    # Sync psycopg also works on Windows' default Proactor event loop.
                    with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
                        conn.execute('LISTEN sim_frame')
                        while not stop.is_set():
                            payloads = _read_frames(conn, gen)
                            if payloads:
                                gen = unpack(payloads[-1])['gen']
                            loop.call_soon_threadsafe(publish, payloads)
                            # Notifications wake immediately; timeout supplies the 5 s poll.
                            for _ in conn.notifies(timeout=5, stop_after=1):
                                break
                except psycopg.Error as error:
                    if not started.is_set():
                        loop.call_soon_threadsafe(ready.set_exception, error)
                        return
                    log.exception('frame reader database failure; retrying')
                    stop.wait(5)

        task = asyncio.create_task(asyncio.to_thread(reader))
        try:
            await ready
            yield
        finally:
            stop.set()
            # Wake the blocking LISTEN so shutdown doesn't wait for the poll interval.
            def wake():
                with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
                    conn.execute('NOTIFY sim_frame')
            with suppress(psycopg.Error):
                await asyncio.to_thread(wake)
            await task

    app = FastAPI(lifespan=lifespan)
    app.state.frames = frames

    def query(sql, args=()):
        with psycopg.connect(url, row_factory=dict_row, connect_timeout=5) as conn:
            return conn.execute(sql, args).fetchall()

    @app.websocket('/api/live')
    async def live(ws: WebSocket):
        await ws.accept()
        queue = None
        sender = None

        async def send():
            while True:
                await ws.send_bytes(await queue.get())

        try:
            while True:
                message = await ws.receive_json()
                gen = message.get('gen') if isinstance(message, dict) else None
                if type(gen) is not int or gen < 0:
                    await ws.close(code=1008)
                    return
                if queue is None:
                    queue = await frames.subscribe(gen)
                    sender = asyncio.create_task(send())
                else:
                    frames.resync(queue, gen)
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            if sender:
                sender.cancel()
                with suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await sender
            if queue is not None:
                await frames.unsubscribe(queue)

    @app.get('/api/static/network')
    def network(request: Request, v: str | None = None):
        paths = list(folder.glob('network-*.json.gz'))
        if v is not None:
            paths = [p for p in paths if p.name[8:-8] == v or p.name[8:-8] == v[:12]] if v.isalnum() else []
        if not paths:
            raise HTTPException(404, 'static version not found')
        path = max(paths, key=lambda p: (p.stat().st_mtime_ns, p.name))
        compressed = path.read_bytes()
        etag = '"' + hashlib.sha256(gzip.decompress(compressed)).hexdigest() + '"'
        if v is not None and v not in (etag.strip('"'), path.name[8:-8]):
            raise HTTPException(404, 'static version not found')
        tags = [tag.strip().removeprefix('W/') for tag in request.headers.get('if-none-match', '').split(',')]
        headers = {'ETag': etag}
        if etag in tags or '*' in tags:
            return Response(status_code=304, headers=headers)
        return Response(compressed, media_type='application/json', headers=headers | {'Content-Encoding': 'gzip'})

    @app.get('/api/train/{number}')
    def train(number: str, day: date | None = Query(None, alias='date')):
        runs = query('SELECT * FROM obs.run WHERE train=%s AND (%s::date IS NULL OR start_date=%s) ORDER BY start_date,id', (number, day, day))
        if not runs:
            raise HTTPException(404, 'train run not found')
        return {'runs': runs}

    @app.get('/api/run/{run_id}')
    def run(run_id: int):
        with psycopg.connect(url, row_factory=dict_row, connect_timeout=5) as conn:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            row = conn.execute('SELECT * FROM obs.run WHERE id=%s', (run_id,)).fetchone()
            if row is None:
                raise HTTPException(404, 'run not found')
            schedule = conn.execute('SELECT * FROM ref.schedule_stop WHERE version_id=%s ORDER BY idx', (row['schedule_version_id'],)).fetchall()
            revisions = conn.execute('SELECT * FROM obs.event_rev WHERE run_id=%s ORDER BY received_at,id', (run_id,)).fetchall()
            eta = conn.execute('''SELECT DISTINCT ON (station,occ,kind) *,
                extract(epoch FROM (clock_timestamp()-received_at)) AS age
                FROM obs.ntes_eta_rev WHERE run_id=%s AND received_at<=clock_timestamp()
                ORDER BY station,occ,kind,received_at DESC''', (run_id,)).fetchall()
            # Read the newest durable update, without requiring a registry/model cache.
            prediction = {}
            for record in conn.execute('SELECT payload FROM sim.frame ORDER BY gen DESC'):
                f = unpack(record['payload'])
                if run_id in f['tombstones']:
                    break
                if run_id in f['runs']:
                    prediction = f['runs'][run_id]
                    break
                if f['is_full']:
                    break
        halts = []
        qs = prediction.get('halt_q', [])
        for stop in schedule:
            idx = stop['idx']
            if idx < len(qs):
                arr, dep = qs[idx]
                if any(q is not None for q in arr + dep):
                    halts.append({'idx': idx, 'station': stop['station'], 'occ': stop['occ'], 'arr': arr, 'dep': dep})
        return row | {'schedule': schedule, 'halts': halts, 'eta': eta, 'revisions': revisions,
                      'evidence_state': prediction.get('state', 'schedule_only')}

    @app.get('/api/replay')
    def replay(start: datetime = Query(alias='from'), end: datetime = Query(alias='to')):
        if start.tzinfo is None or end.tzinfo is None or not timedelta(0) < end - start <= timedelta(hours=24):
            raise HTTPException(422, 'require a UTC-offset window of at most 24 hours')
        with psycopg.connect(url, row_factory=dict_row, connect_timeout=5) as conn:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            events = conn.execute('SELECT * FROM obs.event WHERE actual_ts >= %s AND actual_ts < %s ORDER BY run_id,actual_ts,station,occ,kind', (start, end)).fetchall()
            runs = conn.execute('SELECT * FROM obs.run WHERE id=ANY(%s) ORDER BY id', (sorted({e['run_id'] for e in events}),)).fetchall()
        return {'label': 'station reports; motion reconstructed', 'events': events, 'runs': runs}

    @app.get('/api/search')
    def search(q: str = Query(min_length=1, max_length=100)):
        pattern = '%' + q.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
        return {'trains': query('SELECT * FROM ref.train WHERE number ILIKE %s OR name ILIKE %s ORDER BY number LIMIT 50', (pattern, pattern)),
                'stations': query('SELECT * FROM ref.station WHERE code ILIKE %s OR name ILIKE %s ORDER BY code LIMIT 50', (pattern, pattern))}

    @app.get('/api/health')
    def health():
        rows = query("SELECT DISTINCT ON (name) * FROM sim.metric WHERE name LIKE 'dispatcher_state:%%' ORDER BY name,ts DESC")
        states = {0: 'ok', 1: 'breaker', 2: 'stopped'}
        dispatchers = {r['name'].split(':', 1)[1]: {'state': states[r['value']], 'ts': r['ts']} for r in rows if r['value'] in states}
        latest = query('SELECT gen,extract(epoch FROM (clock_timestamp()-ts)) AS age FROM sim.frame ORDER BY gen DESC LIMIT 1')
        return {'dispatchers': dispatchers,
                'active_models': query('SELECT id,role,kind,feature_ver,trained_at FROM ml.model WHERE active ORDER BY role'),
                'dirty_backlog': query('SELECT count(*) AS n FROM obs.dirty_run')[0]['n'],
                'last_frame': latest[0] if latest else None}

    @app.get('/api/metrics')
    def metrics():
        return query('SELECT * FROM sim.metric WHERE ts >= clock_timestamp()-interval \'15 minutes\' ORDER BY ts DESC,name')

    return app


app = create_app()
