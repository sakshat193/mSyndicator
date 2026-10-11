import { afterEach, expect, it, vi } from 'vitest';
import { encode } from '@msgpack/msgpack';
import type { Frame } from './frames';
import './layers';

const rendered = vi.hoisted(() => ({ layers: [] as any[] }));
vi.mock('maplibre-gl', () => ({
  Map: class { addControl() {} remove() {} }, NavigationControl: class {},
}));
vi.mock('@deck.gl/mapbox', () => ({ MapboxOverlay: class {
  setProps(props: { layers: any[] }) { rendered.layers = props.layers; }
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
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { draw = callback; return 1; });
  vi.stubGlobal('window', { addEventListener() {} });
  vi.stubGlobal('document', { querySelector: () => ({ textContent: '',
    querySelector: () => ({ addEventListener() {}, textContent: '' }) }) });
  vi.stubGlobal('location', { protocol: 'http:', host: '127.0.0.1' });
  vi.stubGlobal('WebSocket', Socket);
  vi.spyOn(Date, 'now').mockImplementation(() => (simTime + 15) * 1000);
  vi.stubGlobal('fetch', async () => ({ ok: true, json: async () => ({
    stations: [{ code: 'A', lon: 70, lat: 20 }, { code: 'B', lon: 72, lat: 22 }],
    segments: [{ id: 1, stn_lo: 'A', stn_hi: 'B', polyline: null, confidence: 0 }],
    routes: [{ id: 7, stops: [
      { idx: 0, station: 'A', halt: true, dist_km: 0, sched_off: 0 },
      { idx: 1, station: 'B', halt: true, dist_km: 100, sched_off: 10 },
    ], segments: [] }],
  }) }));
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
  const trips = rendered.layers[1].props.data;
  const decodedReads = reads.mock.calls.length;
  expect(decodedReads).toBeGreaterThan(0);
  expect(rendered.layers[2].props.data[0].head).toEqual([71, 21]);
  simTime = 1900;
  draw!(200);
  expect(reads.mock.calls.length).toBe(decodedReads);
  expect(rendered.layers[0]).toBe(network);
  expect(rendered.layers[1].props.data).toBe(trips);
  expect(rendered.layers[2].props.data[0].head).toEqual([71.5, 21.5]);
  simTime = 3601;
  draw!(300);
  expect(rendered.layers[1].props.data).toBe(trips);
  expect(rendered.layers[1].props.currentTime).toBe(1);
  expect(rendered.layers[1].props.updateTriggers.getTimestamps).toBe(3600);
  expect(rendered.layers[1].props.getTimestamps(trips[0])).toEqual([-2600, -1400]);
  Socket.current.receive({ gen: 2, ts: 3601, is_full: false, versions: {}, base_ts: 1000,
    runs: { 9: run }, tombstones: [] });
  draw!(400);
  expect(reads.mock.calls.length).toBeGreaterThan(decodedReads);
  expect(rendered.layers[1].props.data).not.toBe(trips);
  Socket.current.receive({ gen: 3, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: {}, tombstones: [9] });
  draw!(500);
  expect(rendered.layers[1].props.data).toEqual([]);
  expect(rendered.layers[2].props.data).toEqual([]);
});
