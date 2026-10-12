import { afterEach, expect, it, vi } from 'vitest';
import { encode } from '@msgpack/msgpack';
import type { Frame, RunFrame } from './frames';
import './layers';

const rendered = vi.hoisted(() => ({ layers: [] as any[], options: {} as any,
  draws: 0, zoom: 4, mapOptions: {} as any, setup: [] as string[],
  ease: [] as any[], listeners: {} as Record<string, Function>,
  selectSearch: undefined as ((run: RunFrame) => void) | undefined }));
vi.mock('./search', async importOriginal => {
  const actual = await importOriginal<typeof import('./search')>();
  return { ...actual, initSearch: (...args: Parameters<typeof actual.initSearch>) => {
    rendered.selectSearch = args[3];
    return actual.initSearch(...args);
  } };
});
vi.mock('maplibre-gl', () => ({
  Map: class {
    constructor(options: unknown) { rendered.mapOptions = options; rendered.setup.push('map'); }
    addControl() {} remove() {} getZoom() { return rendered.zoom; }
    on(name: string, callback: Function) { rendered.listeners[`map:${name}`] = callback; }
    easeTo(options: unknown) { rendered.ease.push(options); }
    getCanvas() { return { style: {} }; }
  }, NavigationControl: class {},
  addProtocol() { rendered.setup.push('protocol'); },
  setWorkerUrl(url: string) { if (url.includes('maplibre-gl-worker')) rendered.setup.push('worker'); },
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
  rendered.setup = [];
  rendered.ease = [];
  rendered.listeners = {};
  let statusWrites = 0;
  const element = (): any => ({ hidden: true, textContent: '', value: '', style: { setProperty() {} }, dataset: {},
    setAttribute() {}, removeAttribute() {}, contains() { return false; }, scrollIntoView() {},
    addEventListener() {}, replaceChildren() {}, append() {}, focus() {},
    querySelector: () => element() });
  const status = { ...element(), set textContent(_text: string) { statusWrites++; } };
  const panel = element();
  const elements = new Map<string, any>([['#connection', status], ['#panel', panel]]);
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { draw = callback; return 1; });
  vi.stubGlobal('window', { innerWidth: 1000, innerHeight: 800,
    addEventListener: (name: string, callback: Function) => { rendered.listeners[name] = callback; } });
  vi.stubGlobal('document', { createElement: () => element(), querySelector: (selector: string) => {
    if (!elements.has(selector)) elements.set(selector, element());
    return elements.get(selector);
  } });
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
  expect(rendered.setup).toEqual(['worker', 'protocol', 'map']);
  expect(rendered.mapOptions.bounds).toEqual([[68, 6.5], [97.5, 36]]);
  // Let loadNetwork finish both awaits before driving requestAnimationFrame.
  await Promise.resolve();
  Socket.current.receive({ gen: 1, ts: 1600, is_full: true, versions: {}, base_ts: 1000,
    runs: { 9: run }, tombstones: [] });
  const reads = vi.spyOn(DataView.prototype, 'getUint16');
  draw!(100);
  const network = rendered.layers[0];
  const trails = rendered.layers.filter(layer => layer.id.startsWith('comet-') || layer.id === 'streaks');
  const arrays = trails.map(layer => layer.props.data);
  const timestamps = trails.map(layer => layer.props.getTimestamps(layer.props.data[0]));
  const heads = () => rendered.layers.find(layer => layer.id === 'heads');
  const head = () => heads().props.getPosition(heads().props.data[0]);
  const headData = heads().props.data;
  const positionTrigger = heads().props.updateTriggers.getPosition;
  const decodedReads = reads.mock.calls.length;
  expect(decodedReads).toBeGreaterThan(0);
  expect(head()).toEqual([71, 21]);
  expect(rendered.options.pickingRadius).toBe(10);
  expect(statusWrites).toBe(1);
  expect(elements.get('#train-count').textContent).toBe('1 live');
  simTime = 1900;
  draw!(116); // Settled heads redraw at 10 Hz only.
  expect(rendered.draws).toBe(1);
  draw!(200);
  expect(rendered.draws).toBe(2);
  expect(reads.mock.calls.length).toBe(decodedReads);
  expect(rendered.layers[0]).toBe(network);
  expect(head()[0]).toBeCloseTo(71 + 0.5 * (1 - Math.exp(-100 / 250)));
  expect(head()[1]).toBeCloseTo(21 + 0.5 * (1 - Math.exp(-100 / 250)));
  expect(heads().props.data).toBe(headData);
  expect(heads().props.updateTriggers.getPosition).not.toBe(positionTrigger);
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
  expect(rendered.layers.find(layer => layer.id === 'comet-2').props.trailLength).toBe(10800);
  expect(statusWrites).toBe(1);
  // An empty delta advances the generation while keeping the same RunFrame object.
  Socket.current.receive({ gen: 2, ts: 3601, is_full: false, versions: {}, base_ts: 1000,
    runs: {}, tombstones: [] });
  reads.mockClear(); // Exclude MessagePack's DataView reads when receiving the frame.
  const previousHead = head();
  draw!(300);
  expect(reads.mock.calls.length).toBe(0);
  expect(rendered.layers[0]).toBe(network);
  trails.forEach((layer, i) => {
    expect(rendered.layers.find(current => current.id === layer.id).props.data).not.toBe(arrays[i]);
  });
  expect(head()[0]).toBeGreaterThan(previousHead[0]);
  expect(head()[0]).toBeLessThan(72);
  // Replacing this run must decode it, retaining the eased head by run_id.
  Socket.current.receive({ gen: 3, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: { 9: run }, tombstones: [] });
  reads.mockClear();
  draw!(400);
  expect(reads.mock.calls.length).toBe(decodedReads);
  expect(head()[0]).toBeLessThan(72);
  heads().props.onClick({ object: heads().props.data[0] });
  expect(panel.hidden).toBe(false);
  expect(fetch).toHaveBeenCalledWith('/api/run/9', expect.any(Object));
  expect(rendered.ease.at(-1)).toMatchObject({ center: expect.any(Array), zoom: 6 });
  Socket.current.receive({ gen: 4, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: {}, tombstones: [9] });
  draw!(500);
  expect(rendered.layers[0]).toBe(network);
  expect(rendered.layers.slice(1).every(layer => layer.props.data.length === 0)).toBe(true);
  expect(statusWrites).toBe(1);
  draw!(1150);
  expect(statusWrites).toBe(2);
  expect(elements.get('#train-count').textContent).toBe('0 live');
  Socket.current.receive({ gen: 5, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: { 9: run }, tombstones: [] });
  simTime = 1600;
  draw!(1166);
  expect(head()).toEqual([71, 21]);
  const draws = rendered.draws;
  draw!(1180);
  expect(rendered.draws).toBe(draws);
  const trailArrays = rendered.layers.filter(l => l.id.startsWith('comet-') || l.id === 'streaks')
    .map(l => l.props.data);
  heads().props.onClick({ object: heads().props.data[0] });
  draw!(1192);
  expect(rendered.draws).toBe(draws + 1); // Selection wakes the idle overlay.
  const highlight = rendered.layers.find(l => l.id === 'selection');
  expect(highlight.props.data[0]).toEqual([[70, 20], [72, 22]]);
  heads().props.onHover({ object: heads().props.data[0], x: 100, y: 100 });
  draw!(1204);
  expect(rendered.draws).toBe(draws + 2); // Hover also wakes it.
  expect(rendered.layers.find(l => l.id === 'selection')).toBe(highlight);
  rendered.layers.filter(l => l.id.startsWith('comet-') || l.id === 'streaks')
    .forEach((l, i) => expect(l.props.data).toBe(trailArrays[i]));
  rendered.listeners.keydown({ key: 'Escape' });
  draw!(1216);
  expect(rendered.layers.some(l => l.id === 'selection')).toBe(false);
  expect(panel.hidden).toBe(true);
  const pending = { run_id: 42, state: 'schedule_only', origin_ts: 1000, versions: { route: 7 } };
  Socket.current.receive({ gen: 6, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: { 42: pending }, tombstones: [] });
  rendered.selectSearch!(pending);
  expect(panel.hidden).toBe(false); // Live store membership is sufficient, even before a path is drawable.
  expect(fetch).toHaveBeenCalledWith('/api/run/42', expect.any(Object));
  draw!(1228);
  expect(panel.hidden).toBe(false);
  expect(rendered.layers.some(l => l.id === 'selection')).toBe(false);
  Socket.current.receive({ gen: 7, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: {}, tombstones: [42] });
  draw!(1240);
  expect(panel.hidden).toBe(true);
  const labels = () => rendered.layers.find(l => l.id === 'station-labels');
  rendered.zoom = 7.1;
  draw!(1500);
  const atSeven = labels();
  expect(atSeven.props.getSize).toBe(12);
  rendered.zoom = 7.49;
  rendered.listeners['map:move']();
  draw!(1512);
  expect(labels()).toBe(atSeven);
  rendered.zoom = 7.74;
  draw!(1612);
  const atSevenHalf = labels();
  expect(atSevenHalf).not.toBe(atSeven);
  rendered.zoom = 7.99;
  draw!(1712);
  expect(labels()).toBe(atSevenHalf);
  rendered.listeners['map:move']();
  draw!(1724);
  expect(labels()).toBe(atSevenHalf); // Panning reuses the layer and its decluttered data.
  const counted = Array.from({ length: 1831 }, (_, i) => ({ ...run, run_id: i + 100,
    state: i < 5 ? (i === 4 ? 'stale' : 'observed') : 'schedule_only' }));
  Socket.current.receive({ gen: 8, ts: 3601, is_full: true, versions: {}, base_ts: 1000,
    runs: Object.fromEntries(counted.map(r => [r.run_id, r])), tombstones: [] });
  draw!(2150);
  expect(elements.get('#train-count').textContent).toBe('5 live · 1,826 by timetable');
  Socket.current.receive({ gen: 9, ts: 3601, is_full: true, versions: {}, base_ts: 1000,
    runs: { 9: { ...run, state: 'schedule_only' }, 42: pending }, tombstones: [] });
  draw!(3150);
  expect(elements.get('#train-count').textContent).toBe('0 live · 1 by timetable');
  Socket.current.receive({ gen: 10, ts: 3601, is_full: true, versions: {}, base_ts: 1000,
    runs: { 9: { ...run, state: 'observed' }, 10: { ...run, run_id: 10, state: 'schedule_only' } }, tombstones: [] });
  draw!(3250);
  const timetable = rendered.layers.find(l => l.id === 'timetable');
  expect(timetable.props.data).toHaveLength(1);
  const timetableHead = heads().props.data.find((d: any) => d.run.run_id === 10);
  const liveTrails = rendered.layers.filter(l => l.id.startsWith('comet-') || l.id === 'streaks');
  Socket.current.receive({ gen: 11, ts: 3601, is_full: false, versions: {}, base_ts: 1000,
    runs: { 9: { ...run, state: 'stale' } }, tombstones: [] });
  draw!(3350);
  const nextTimetable = rendered.layers.find(l => l.id === 'timetable');
  expect(nextTimetable.props.data[0]).toBe(timetable.props.data[0]);
  expect(nextTimetable.props.dataComparator(nextTimetable.props.data, timetable.props.data)).toBe(true);
  expect(heads().props.data.find((d: any) => d.run.run_id === 10)).toBe(timetableHead);
  for (const layer of liveTrails) {
    const changed = rendered.layers.find(l => l.id === layer.id);
    expect(changed.props.dataComparator(changed.props.data, layer.props.data)).toBe(false);
  }
  const sameHeads = heads().props.data;
  const sameHeadColors = heads().props.updateTriggers.getFillColor;
  const beforeRedraw = heads().props.updateTriggers.getPosition;
  draw!(3450);
  expect(heads().props.data).toBe(sameHeads);
  expect(heads().props.updateTriggers.getFillColor).toBe(sameHeadColors);
  expect(heads().props.updateTriggers.getPosition).not.toBe(beforeRedraw);
});
