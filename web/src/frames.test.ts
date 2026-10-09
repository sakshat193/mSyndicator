import { afterEach, expect, it, vi } from 'vitest';
import { encode } from '@msgpack/msgpack';
import { connectLive, FrameStore, type Frame } from './frames';
import { displayRun, type Network } from './layers';

const frame = (gen: number, is_full = false, runs = {}, tombstones: number[] = []): Frame =>
  ({ gen, is_full, runs, tombstones, ts: 1000, base_ts: null, versions: {} });

it('applies error runs without display paths in full and delta frames', () => {
  const store = new FrameStore();
  const run = { run_id: 42, state: 'error', error: 'bad evidence', versions: { route: 7 } };
  const network: Network = { stations: [], segments: [],
    routes: [{ id: 7, stops: [], segments: [] }] };
  for (const [gen, full] of [[10, true], [11, false]] as const) {
    expect(store.apply(frame(gen, full, { 42: run }))).toBe('ok');
    expect(store.gen).toBe(gen);
    expect(store.runs['42']).toEqual(run);
    expect(displayRun(store.runs['42'], network, 1000)).toBeNull();
  }
});

it('applies sequential deltas and tombstones, including absent runs', () => {
  const store = new FrameStore();
  store.apply(frame(10, true, { 1: { run_id: 1, state: 'observed' } }));
  expect(store.apply(frame(11, false, { 2: { run_id: 2, state: 'stale' } }, [1, 99]))).toBe('ok');
  expect(store.runs).toEqual({ 2: { run_id: 2, state: 'stale' } });
  expect(store.gen).toBe(11);
});

it('rejects gaps and out-of-order deltas without changing the snapshot', () => {
  const store = new FrameStore();
  store.apply(frame(10, true, { 1: { run_id: 1 } }));
  for (const gen of [12, 9, 10]) {
    expect(store.apply(frame(gen, false, { 2: { run_id: 2 } }, [1]))).toBe('gap');
    expect(store.gen).toBe(10);
    expect(store.runs).toEqual({ 1: { run_id: 1 } });
  }
});

it('full frames replace all runs even across a generation reset', () => {
  const store = new FrameStore();
  store.apply(frame(10, true, { 1: { run_id: 1 } }));
  expect(store.apply(frame(2, true, { 2: { run_id: 2 } }))).toBe('ok');
  expect(store.runs).toEqual({ 2: { run_id: 2 } });
  expect(store.gen).toBe(2);
});

it('rejects malformed frames before changing state', () => {
  const store = new FrameStore();
  store.apply(frame(10, true, { 1: { run_id: 1 } }));
  for (const bad of [{ ...frame(11), gen: -1 }, { ...frame(11), runs: null },
    { ...frame(11), tombstones: ['wrong'] }, { ...frame(11), is_full: 'yes' }]) {
    expect(() => store.apply(bad as unknown as Frame)).toThrow();
    expect(store.gen).toBe(10);
    expect(store.runs).toEqual({ 1: { run_id: 1 } });
  }
});

// Only the network boundary is fake; frames pass through the real msgpack codec/store.
class Socket {
  static all: Socket[] = [];
  binaryType = '';
  onopen?: () => void;
  onmessage?: (event: { data: ArrayBuffer }) => void;
  onclose?: () => void;
  onerror?: () => void;
  sent: string[] = [];
  constructor(public url: string) { Socket.all.push(this); }
  send(data: string) { this.sent.push(data); }
  close() { this.onclose?.(); }
  receive(value: unknown) { this.onmessage?.({ data: encode(value).slice().buffer }); }
}

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); Socket.all = []; });

it('decodes binary frames, requests gap resync and resumes after full replacement', () => {
  vi.stubGlobal('WebSocket', Socket);
  const store = new FrameStore();
  const disconnect = connectLive('ws://localhost/api/live', store);
  const socket = Socket.all[0];
  socket.onopen?.();
  expect(socket.sent).toEqual(['{"gen":0}']);
  const paths = new Uint8Array([2, 0, 10, 0]);
  socket.receive(frame(10, true, { 1: { run_id: 1, base_ts: 900, path_shape: [1, 1, 2], paths } }));
  expect(store.runs['1'].paths).toEqual(paths);
  socket.receive(frame(12));
  expect(socket.sent.at(-1)).toBe('{"gen":10}');
  expect(store.gen).toBe(10);
  socket.receive(frame(12, true));
  socket.receive(frame(13, false, { 2: { run_id: 2 } }));
  expect(store.gen).toBe(13);
  expect(store.runs).toEqual({ 2: { run_id: 2 } });
  disconnect();
});

it('reconnects exponentially, sends the last generation and cancels on disconnect', () => {
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', Socket);
  const store = new FrameStore();
  const disconnect = connectLive('ws://localhost/api/live', store);
  Socket.all[0].close();
  vi.advanceTimersByTime(999);
  expect(Socket.all).toHaveLength(1);
  vi.advanceTimersByTime(1);
  Socket.all[1].close();
  vi.advanceTimersByTime(1999);
  expect(Socket.all).toHaveLength(2);
  vi.advanceTimersByTime(1);
  Socket.all[2].onopen?.();
  Socket.all[2].receive(frame(7, true));
  Socket.all[2].close();
  vi.advanceTimersByTime(1000);
  Socket.all[3].onopen?.();
  expect(Socket.all[3].sent).toEqual(['{"gen":7}']);
  Socket.all[3].close();
  disconnect();
  vi.advanceTimersByTime(60_000);
  expect(Socket.all).toHaveLength(4);
});

it('reconnects after invalid binary input without mutating the store', () => {
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', Socket);
  const store = new FrameStore();
  const disconnect = connectLive('ws://localhost/api/live', store);
  Socket.all[0].onopen?.();
  Socket.all[0].receive({ gen: 1 });
  expect(store.gen).toBe(0);
  vi.advanceTimersByTime(1000);
  expect(Socket.all).toHaveLength(2);
  disconnect();
});

it('ignores messages from a closed socket while waiting to reconnect', () => {
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', Socket);
  const store = new FrameStore();
  const disconnect = connectLive('ws://localhost/api/live', store);
  const old = Socket.all[0];
  old.close();
  old.receive(frame(99, true));
  expect(store.gen).toBe(0);
  vi.advanceTimersByTime(1000);
  Socket.all[1].onopen?.();
  expect(Socket.all[1].sent).toEqual(['{"gen":0}']);
  disconnect();
});

it('resyncs through contiguous replay deltas and preserves each run base', () => {
  vi.stubGlobal('WebSocket', Socket);
  const store = new FrameStore();
  const disconnect = connectLive('ws://localhost/api/live', store);
  const socket = Socket.all[0];
  socket.receive(frame(10, true, { 1: { run_id: 1, base_ts: 900 } }));
  socket.receive(frame(12));
  socket.receive(frame(11, false, { 2: { run_id: 2, base_ts: 1000 } }));
  socket.receive(frame(12, false, {}, [2]));
  expect(store.gen).toBe(12);
  expect(store.runs).toEqual({ 1: { run_id: 1, base_ts: 900 } });
  expect(store.latest?.ts).toBe(1000);
  disconnect();
});

it('rejects inconsistent run ids and packed path lengths atomically', () => {
  const store = new FrameStore();
  const baseline = frame(10, true, { 1: { run_id: 1 } });
  store.apply(baseline);
  for (const run of [{ run_id: 3 },
    { run_id: 2, paths: new Uint8Array(3), path_shape: [1, 1, 2], base_ts: 900 }]) {
    expect(() => store.apply(frame(11, false, { 2: run }, [1]))).toThrow();
    expect(store.runs).toEqual({ 1: { run_id: 1 } });
    expect(store.gen).toBe(10);
    expect(store.latest).toBe(baseline);
  }
});

it('handles socket errors and caps repeated reconnect delays', () => {
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', Socket);
  const disconnect = connectLive('ws://localhost/api/live', new FrameStore());
  for (const delay of [1000, 2000, 4000, 8000, 16000, 30000, 30000]) {
    const count = Socket.all.length;
    Socket.all.at(-1)?.onerror?.();
    vi.advanceTimersByTime(delay - 1);
    expect(Socket.all).toHaveLength(count);
    vi.advanceTimersByTime(1);
    expect(Socket.all).toHaveLength(count + 1);
  }
  disconnect();
  vi.advanceTimersByTime(60_000);
  expect(Socket.all).toHaveLength(8);
});
