import { afterEach, expect, it, vi } from 'vitest';
import { encode } from '@msgpack/msgpack';
import type { Frame } from './frames';
import './layers';

const rendered = vi.hoisted(() => ({ layers: [] as any[], options: {} as any,
  draws: 0, zoom: 4 }));
vi.mock('maplibre-gl', () => ({
  Map: class { addControl() {} remove() {} getZoom() { return rendered.zoom; } }, NavigationControl: class {},
}));
vi.mock('@deck.gl/mapbox', () => ({ MapboxOverlay: class {
  constructor(options: unknown) { rendered.options = options; }
  setProps(props: { layers: any[] }) { rendered.layers = props.layers; rendered.draws++; }
} }));

class Socket {
  static current: Socket;
  binaryType = '';
  onmessage?: (event: { data: ArrayBuffer }) => void;
  constructor() { Socket.current = this; }
  send() {}
  close() {}
  receive(frame: Frame) { this.onmessage?.({ data: encode(frame).slice().buffer }); }
}

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); vi.resetModules(); });

it('decodes only on generations and reuses network and trips while animating heads', async () => {
  let draw: FrameRequestCallback;
  let simTime = 1600;
  rendered.draws = 0;
  rendered.zoom = 4;
  let statusWrites = 0;
  const status = { set textContent(_text: string) { statusWrites++; } };
  const panel = { hidden: true, querySelector: () => ({ addEventListener() {}, textContent: '' }) };
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { draw = callback; return 1; });
  vi.stubGlobal('window', { addEventListener() {} });
  vi.stubGlobal('document', { querySelector: (selector: string) => selector === '#connection' ? status : panel });
  vi.stubGlobal('location', { protocol: 'http:', host: '127.0.0.1' });
  vi.stubGlobal('WebSocket', Socket);
  vi.spyOn(Date, 'now').mockImplementation(() => (simTime + 15) * 1000);
  const fetch = vi.fn(async (url: string) => url.startsWith('/api/run/') ? { ok: false, status: 503 } :
    ({ ok: true, json: async () => ({
    stations: [{ code: 'A', lon: 70, lat: 20 }, { code: 'B', lon: 72, lat: 22 }],
    segments: [{ id: 1, stn_lo: 'A', stn_hi: 'B', polyline: null, derived: false, confidence: 0 }],
    routes: [{ id: 7, train: '00001', name: 'Express', class: 'EXP', stops: [
      { idx: 0, station: 'A', halt: true, dist_km: 0, sched_off: 0 },
      { idx: 1, station: 'B', halt: true, dist_km: 100, sched_off: 10 },
    ], segments: [] }],
  }) }));
  vi.stubGlobal('fetch', fetch);
  const run = { run_id: 9, versions: { route: 7 }, origin_ts: 1000, base_ts: 1000,
    path_shape: [1, 2, 2] as [number, number, number],
    paths: new Uint8Array([0, 0, 0, 0, 1, 0, 200, 0]) };
  await import('./main');
  // Let loadNetwork finish both awaits before driving requestAnimationFrame.
  await Promise.resolve();
  Socket.current.receive({ gen: 1, ts: 1600, is_full: true, versions: {}, base_ts: 1000,
    runs: { 9: run }, tombstones: [] });
  const reads = vi.spyOn(DataView.prototype, 'getUint16');
  draw!(100);
  const network = rendered.layers[0];
  const trails = rendered.layers.filter(layer => layer.id !== 'heads' && layer.id !== 'network');
  const arrays = trails.map(layer => layer.props.data);
  const timestamps = trails.map(layer => layer.props.getTimestamps(layer.props.data[0]));
  const heads = () => rendered.layers.find(layer => layer.id === 'heads');
  const decodedReads = reads.mock.calls.length;
  expect(decodedReads).toBeGreaterThan(0);
  expect(heads().props.data[0].head).toEqual([71, 21]);
  expect(rendered.options.pickingRadius).toBe(10);
  expect(statusWrites).toBe(1);
  simTime = 1900;
  draw!(116); // Settled heads redraw at 10 Hz only.
  expect(rendered.draws).toBe(1);
  draw!(200);
  expect(rendered.draws).toBe(2);
  expect(reads.mock.calls.length).toBe(decodedReads);
  expect(rendered.layers[0]).toBe(network);
  expect(heads().props.data[0].head[0]).toBeCloseTo(71 + 0.5 * (1 - Math.exp(-100 / 250)));
  expect(heads().props.data[0].head[1]).toBeCloseTo(21 + 0.5 * (1 - Math.exp(-100 / 250)));
  simTime = 3601;
  rendered.zoom = 5;
  draw!(212); // A visibly easing head redraws at frame rate.
  expect(rendered.draws).toBe(3);
  trails.forEach((layer, i) => {
    const animated = rendered.layers.find(current => current.id === layer.id);
    expect(animated.props.data).toBe(arrays[i]);
    expect(animated.props.getTimestamps(animated.props.data[0])).toBe(timestamps[i]);
    expect(animated.props.currentTime).toBe(3601);
    expect(animated.props.updateTriggers.getTimestamps).toBe(layer.props.updateTriggers.getTimestamps);
  });
  expect(rendered.layers.find(layer => layer.id === 'comet-1').props.trailLength).toBe(7200);
  expect(statusWrites).toBe(1);
  // An empty delta advances the generation while keeping the same RunFrame object.
  Socket.current.receive({ gen: 2, ts: 3601, is_full: false, versions: {}, base_ts: 1000,
    runs: {}, tombstones: [] });
  reads.mockClear(); // Exclude MessagePack's DataView reads when receiving the frame.
  const previousHead = heads().props.data[0].head;
  draw!(300);
  expect(reads.mock.calls.length).toBe(0);
  expect(rendered.layers[0]).toBe(network);
  trails.forEach((layer, i) => {
    expect(rendered.layers.find(current => current.id === layer.id).props.data).not.toBe(arrays[i]);
  });
  expect(heads().props.data[0].head[0]).toBeGreaterThan(previousHead[0]);
  expect(heads().props.data[0].head[0]).toBeLessThan(72);
  // Replacing this run must decode it, retaining the eased head by run_id.
  Socket.current.receive({ gen: 3, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: { 9: run }, tombstones: [] });
  reads.mockClear();
  draw!(400);
  expect(reads.mock.calls.length).toBe(decodedReads);
  expect(heads().props.data[0].head[0]).toBeLessThan(72);
  heads().props.onClick({ object: heads().props.data[0] });
  expect(panel.hidden).toBe(false);
  expect(fetch).toHaveBeenCalledWith('/api/run/9', expect.any(Object));
  Socket.current.receive({ gen: 4, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: {}, tombstones: [9] });
  draw!(500);
  expect(rendered.layers[0]).toBe(network);
  expect(rendered.layers.slice(1).every(layer => layer.props.data.length === 0)).toBe(true);
  expect(statusWrites).toBe(1);
  draw!(1150);
  expect(statusWrites).toBe(2);
  Socket.current.receive({ gen: 5, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: { 9: run }, tombstones: [] });
  simTime = 1600;
  draw!(1166);
  expect(heads().props.data[0].head).toEqual([71, 21]);
  const draws = rendered.draws;
  draw!(1180);
  expect(rendered.draws).toBe(draws);
});
