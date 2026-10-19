import { afterEach, expect, it, vi } from 'vitest';
import { encode } from '@msgpack/msgpack';
import type { Frame, RunFrame } from './frames';
import './layers';

const rendered = vi.hoisted(() => ({ layers: [] as any[], options: {} as any,
  draws: 0, zoom: 4, moving: false, mapOptions: {} as any, setup: [] as string[],
  ease: [] as any[], fits: [] as any[], listeners: {} as Record<string, Function>,
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
    fitBounds(bounds: unknown, options: unknown) { rendered.fits.push({ bounds, options }); }
    isMoving() { return rendered.moving; }
    project([lon, lat]: [number, number]) { return { x: (lon - 69) * 50, y: (23 - lat) * 50 }; }
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

it.each([
  { browser: 'Chromium', navigator: { userAgentData: {} }, refraction: { refract: '' } },
  { browser: 'other engines', navigator: {}, refraction: {} },
])('gates refraction for $browser and reuses network and trips while animating heads', async ({ navigator: browserNavigator, refraction }) => {
  let draw: FrameRequestCallback;
  let simTime = 1600;
  rendered.draws = 0;
  rendered.zoom = 4;
  rendered.setup = [];
  rendered.ease = [];
  rendered.fits = [];
  rendered.listeners = {};
  let statusWrites = 0;
  const element = (): any => ({ hidden: true, textContent: '', value: '', offsetWidth: 180, offsetHeight: 80,
    style: { setProperty() {} }, dataset: {},
    attributes: {}, listeners: {}, children: {},
    setAttribute(name: string, value: string) { this.attributes[name] = value; },
    removeAttribute(name: string) { delete this.attributes[name]; if (name === 'title') delete this.title; },
    contains() { return false; }, scrollIntoView() {},
    getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; },
    addEventListener(name: string, callback: Function) { this.listeners[name] = callback; },
    removeEventListener(name: string) { delete this.listeners[name]; },
    replaceChildren() {}, append() {}, focus() {},
    querySelector(selector: string) { return this.children[selector] ??= element(); } });
  let statusText = '';
  const status = { ...element(), get textContent() { return statusText; },
    set textContent(text: string) { statusText = text; statusWrites++; } };
  const panel = element();
  const elements = new Map<string, any>([['#connection', status], ['#panel', panel],
    ['#live', element()], ['#legend', element()]]);
  let healthTick: () => Promise<void>;
  const clearInterval = vi.fn();
  vi.stubGlobal('setInterval', (callback: () => Promise<void>, ms: number) => {
    expect(ms).toBe(60000);
    healthTick = callback;
    return 7;
  });
  vi.stubGlobal('clearInterval', clearInterval);
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { draw = callback; return 1; });
  vi.stubGlobal('cancelAnimationFrame', () => {});
  let reducedMotion = false;
  const matchMedia = vi.fn(() => ({ matches: reducedMotion }));
  vi.stubGlobal('window', { innerWidth: 1000, innerHeight: 800, matchMedia,
    addEventListener: (name: string, callback: Function) => { rendered.listeners[name] = callback; } });
  const body = { dataset: {} as Record<string, string | undefined> };
  vi.stubGlobal('navigator', browserNavigator);
  vi.stubGlobal('localStorage', {
    getItem: () => 'shown', setItem() {},
  });
  vi.stubGlobal('document', { body, createElement: () => element(),
    addEventListener: (name: string, callback: Function) => { rendered.listeners[`document:${name}`] = callback; },
    removeEventListener: (name: string) => { delete rendered.listeners[`document:${name}`]; },
    querySelector: (selector: string) => {
    if (!elements.has(selector)) elements.set(selector, element());
    return elements.get(selector);
  } });
  vi.stubGlobal('location', { protocol: 'http:', host: '127.0.0.1' });
  vi.stubGlobal('WebSocket', Socket);
  vi.spyOn(Date, 'now').mockImplementation(() => (simTime + 15) * 1000);
  let healthRequests = 0;
  const fetch = vi.fn(async (url: string) => {
    if (url === '/api/health') {
      if (++healthRequests === 1) throw new Error('Health unavailable');
      if (healthRequests === 3) return { ok: false, status: 503 };
      return { ok: true, json: async () => ({
        source: { poll_ok_at: new Date(415000).toISOString(), harvest_ok_at: new Date(1015000).toISOString() },
        dispatchers: { 'rails-ris.makemytrip.com': { state: 'breaker', ts: new Date(1615000).toISOString() } },
      }) };
    }
    return url.startsWith('/api/run/') ? { ok: false, status: 503 } :
    ({ ok: true, json: async () => ({
    stations: [{ code: 'A', lon: 70, lat: 20 }, { code: 'B', lon: 72, lat: 22 }],
    segments: [{ id: 1, stn_lo: 'A', stn_hi: 'B', polyline: null, derived: false, confidence: 0 }],
    routes: [{ id: 7, train: '00001', name: 'Express', class: 'EXP', stops: [
      { idx: 0, station: 'A', halt: true, dist_km: 0, sched_off: 0 },
      { idx: 1, station: 'B', halt: true, dist_km: 100, sched_off: 10 },
    ], segments: [] }],
  }) });
  });
  vi.stubGlobal('fetch', fetch);
  const run = { run_id: 9, versions: { route: 7 }, origin_ts: 1000, base_ts: 1000,
    path_shape: [1, 2, 2] as [number, number, number],
    paths: new Uint8Array([0, 0, 0, 0, 1, 0, 200, 0]) };
  await import('./main');
  expect(body.dataset).toEqual({ hud: 'shown', ...refraction });
  expect(rendered.setup).toEqual(['worker', 'protocol', 'map']);
  expect(rendered.mapOptions.bounds).toEqual([[68, 6.5], [97.5, 36]]);
  expect(rendered.options).toEqual({ interleaved: false, useDevicePixels: 1,
    deviceProps: { webgl: { antialias: false, preserveDrawingBuffer: false } }, layers: [] });
  // Let loadNetwork finish both awaits before driving requestAnimationFrame.
  await Promise.resolve();
  const count = elements.get('#train-count');
  count.title = 'Previous count';
  draw!(-1000);
  expect(count.textContent).toBe('Loading…');
  expect(count.title).toBeUndefined();
  expect(fetch).toHaveBeenCalledWith('/api/health');
  expect(status.textContent).toBe('live data unknown');
  expect(status.dataset.freshness).toBe('unknown');
  expect(status.title).toBe('frame unknown · harvest unknown');
  await healthTick!();
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
  const point = (id = 9) => { // Where the mocked map projects a run's drawn head.
    const [lon, lat] = heads().props.getPosition(heads().props.data.find((d: any) => d.run.run_id === id));
    return { x: (lon - 69) * 50, y: (23 - lat) * 50 };
  };
  const pointer = (name: string, at: { x: number; y: number }) => rendered.listeners[`map:${name}`]({ point: at });
  const headData = heads().props.data;
  const positionTrigger = heads().props.updateTriggers.getPosition;
  const decodedReads = reads.mock.calls.length;
  expect(decodedReads).toBeGreaterThan(0);
  expect(head()).toEqual([71, 21]);
  expect(statusWrites).toBe(2);
  expect(elements.get('#train-count').textContent).toBe('1 train');
  expect(elements.get('#train-count').title).toBe('1 live · 0 by timetable');
  expect(status.textContent).toBe('live data 20 min ago · paused');
  expect(status.dataset.freshness).toBe('old');
  expect(status.title).toBe('frame 15 s ago · harvest 10 min ago');
  await healthTick!(); // A failed refresh leaves the last known poll timestamp aging.
  simTime = 1900;
  draw!(116); // Settled heads redraw at 10 Hz only.
  expect(rendered.draws).toBe(2);
  draw!(200);
  expect(rendered.draws).toBe(3);
  expect(reads.mock.calls.length).toBe(decodedReads);
  expect(rendered.layers[0]).toBe(network);
  expect(head()[0]).toBeCloseTo(71 + 0.5 * (1 - Math.exp(-100 / 250)));
  expect(head()[1]).toBeCloseTo(21 + 0.5 * (1 - Math.exp(-100 / 250)));
  expect(heads().props.data).toBe(headData);
  expect(heads().props.updateTriggers.getPosition).not.toBe(positionTrigger);
  simTime = 3601;
  rendered.zoom = 5;
  draw!(212); // A visibly easing head redraws at frame rate.
  expect(rendered.draws).toBe(4);
  trails.forEach((layer, i) => {
    const animated = rendered.layers.find(current => current.id === layer.id);
    expect(animated.props.data).toBe(arrays[i]);
    expect(animated.props.getTimestamps(animated.props.data[0])).toBe(timestamps[i]);
    expect(animated.props.currentTime).toBe(3601);
    expect(animated.props.updateTriggers.getTimestamps).toBe(layer.props.updateTriggers.getTimestamps);
  });
  expect(rendered.layers.find(layer => layer.id === 'comets').props.trailLength).toBe(10800);
  expect(statusWrites).toBe(2);
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
  pointer('click', point());
  expect(panel.hidden).toBe(false);
  expect(fetch).toHaveBeenCalledWith('/api/run/9', expect.any(Object));
  expect.soft(rendered.fits.at(-1)).toEqual({ bounds: [[70, 20], [72, 22]],
    options: { padding: { top: 16, right: 16, bottom: 16, left: 16 }, maxZoom: 10, linear: true, duration: 600 } });
  Socket.current.receive({ gen: 4, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: {}, tombstones: [9] });
  draw!(500);
  expect(rendered.layers[0]).toBe(network);
  expect(rendered.layers.slice(1).every(layer => layer.props.data.length === 0)).toBe(true);
  expect(statusWrites).toBe(2);
  draw!(1150);
  expect(statusWrites).toBe(3);
  expect(status.textContent).toBe('live data 53 min ago · paused');
  expect(status.dataset.freshness).toBe('old');
  expect(status.title).toBe('frame 15 s ago · harvest 43 min ago');
  expect(elements.get('#train-count').textContent).toBe('0 trains');
  expect(elements.get('#train-count').title).toBe('0 live · 0 by timetable');
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
  pointer('click', point());
  draw!(1192);
  expect(rendered.draws).toBe(draws + 1); // Selection wakes the idle overlay.
  expect(rendered.layers[0].id).toBe('network');
  expect(rendered.layers[0].props.opacity).toBe(0.1);
  expect(rendered.layers[0].props.data).toBe(network.props.data);
  const selectedHead = rendered.layers.find(l => l.id === 'selected-head');
  expect(selectedHead).toBeDefined();
  expect(selectedHead.props.data).toEqual([heads().props.data[0]]);
  expect(selectedHead.props.getPosition(selectedHead.props.data[0])).toBe(head());
  expect(selectedHead.props.updateTriggers.getPosition).toBe(heads().props.updateTriggers.getPosition);
  const layerIds = rendered.layers.map(l => l.id);
  expect(layerIds.indexOf('selection-pulse')).toBeLessThan(layerIds.indexOf('comets'));
  expect(layerIds.indexOf('selected-trail')).toBe(layerIds.indexOf('heads') + 1);
  expect(layerIds.indexOf('selected-head')).toBe(layerIds.indexOf('station-labels') - 1);
  const highlight = rendered.layers.find(l => l.id === 'selection');
  expect(highlight.props.data[0]).toEqual([[70, 20], [72, 22]]);
  const toggle = elements.get('#hud-toggle');
  const content = panel.querySelector('[data-content]');
  toggle.listeners.click();
  expect(body.dataset.hud).toBe('hidden');
  expect(panel.hidden).toBe(false); // CSS hides the panel without clearing its selection or content.
  expect(panel.querySelector('[data-content]')).toBe(content);
  expect(rendered.layers.find(l => l.id === 'selection')).toBe(highlight);
  rendered.listeners['document:keydown']({ key: 'h' });
  expect(body.dataset.hud).toBe('shown');
  expect(panel.hidden).toBe(false);
  expect(rendered.layers.find(l => l.id === 'selection')).toBe(highlight);
  const hover = elements.get('#hover');
  expect(point()).toEqual({ x: 100, y: 100 });
  pointer('mousemove', { x: 106, y: 108 });
  expect(hover.hidden).toBe(false);
  expect(hover.style.transform).toBe('translate(122px, 124px)');
  window.innerHeight = 150;
  pointer('mousemove', point());
  expect(hover.style.transform).toBe('translate(116px, 54px)');
  window.innerWidth = 200;
  window.innerHeight = 800;
  pointer('mousemove', point());
  expect(hover.style.transform).toBe('translate(12px, 116px)');
  window.innerWidth = 1000;
  pointer('mousemove', { x: 111, y: 100 });
  expect(hover.hidden).toBe(true);
  pointer('mousemove', point());
  rendered.listeners['map:mouseout']();
  expect(hover.hidden).toBe(true);
  rendered.moving = true;
  pointer('mousemove', point());
  expect(hover.hidden).toBe(true); // A dragged map keeps its head under the cursor; no tooltip mid-pan.
  rendered.moving = false;
  draw!(1204);
  expect(rendered.draws).toBe(draws + 1); // Hover leaves the idle overlay asleep.
  expect(rendered.layers.find(l => l.id === 'selection').props.data).toBe(highlight.props.data);
  rendered.layers.filter(l => l.id.startsWith('comet-') || l.id === 'streaks')
    .forEach((l, i) => expect(l.props.data).toBe(trailArrays[i]));
  rendered.listeners.keydown({ key: 'Escape' });
  draw!(1216);
  expect(rendered.layers[0]).toBe(network);
  expect(rendered.layers[0].props.opacity).toBe(1);
  expect(rendered.layers.some(l => l.id === 'selection')).toBe(false);
  expect(rendered.layers.some(l => l.id === 'selected-trail' || l.id === 'selected-head')).toBe(false);
  expect(panel.hidden).toBe(true);
  const pending = { run_id: 42, state: 'schedule_only', origin_ts: 1000, versions: { route: 7 } };
  Socket.current.receive({ gen: 6, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: { 42: pending }, tombstones: [] });
  const fitsBeforePending = rendered.fits.length, easeBeforePending = rendered.ease.length;
  rendered.selectSearch!(pending);
  expect(panel.hidden).toBe(false); // Live store membership is sufficient, even before a path is drawable.
  expect(fetch).toHaveBeenCalledWith('/api/run/42', expect.any(Object));
  draw!(1228);
  expect(panel.hidden).toBe(false);
  expect(rendered.layers[0].props.opacity).toBe(0.1);
  expect(rendered.layers.some(l => l.id === 'selection')).toBe(false);
  expect(rendered.fits).toHaveLength(fitsBeforePending);
  expect(rendered.ease).toHaveLength(easeBeforePending);
  const panelToggle = panel.querySelector('[data-toggle]');
  panelToggle.listeners.click?.();
  expect.soft(rendered.fits).toHaveLength(fitsBeforePending);
  panelToggle.listeners.click?.();
  expect.soft(rendered.fits).toHaveLength(fitsBeforePending);
  Socket.current.receive({ gen: 7, ts: 3601, is_full: false, versions: {}, base_ts: null,
    runs: {}, tombstones: [42] });
  draw!(1240);
  expect(panel.hidden).toBe(true);
  const labels = () => rendered.layers.find(l => l.id === 'station-labels');
  rendered.zoom = 7.1;
  draw!(1500);
  const atSeven = labels();
  expect(atSeven.props.getSize).toBe(13);
  rendered.zoom = 7.49;
  rendered.listeners['map:move']();
  draw!(1512);
  expect(labels()).toBe(atSeven);
  rendered.zoom = 7.74;
  draw!(1524);
  const atSevenHalf = labels();
  expect.soft(atSevenHalf).not.toBe(atSeven);
  rendered.zoom = 7.99;
  draw!(1536);
  expect(labels()).toBe(atSevenHalf);
  const beforePan = rendered.draws;
  rendered.listeners['map:move']();
  draw!(1548);
  expect.soft(rendered.draws).toBe(beforePan);
  expect.soft(labels()).toBe(atSevenHalf); // Panning reuses the layer and its decluttered data.
  const counted = Array.from({ length: 1831 }, (_, i) => ({ ...run, run_id: i + 100,
    state: i < 5 ? (i === 4 ? 'stale' : 'observed') : 'schedule_only' }));
  Socket.current.receive({ gen: 8, ts: 3601, is_full: true, versions: {}, base_ts: 1000,
    runs: Object.fromEntries(counted.map(r => [r.run_id, r])), tombstones: [] });
  draw!(2150);
  expect(elements.get('#train-count').textContent).toBe('1,831 trains');
  expect(elements.get('#train-count').title).toBe('5 live · 1,826 by timetable');
  Socket.current.receive({ gen: 9, ts: 3601, is_full: true, versions: {}, base_ts: 1000,
    runs: { 9: { ...run, state: 'schedule_only' }, 42: pending }, tombstones: [] });
  draw!(3150);
  expect(elements.get('#train-count').textContent).toBe('1 train');
  expect(elements.get('#train-count').title).toBe('0 live · 1 by timetable');
  Socket.current.receive({ gen: 10, ts: 3601, is_full: true, versions: {}, base_ts: 1000,
    runs: { 9: { ...run, state: 'observed' }, 10: { ...run, run_id: 10, state: 'schedule_only' } }, tombstones: [] });
  draw!(3250);
  const timetable = rendered.layers.find(l => l.id === 'timetable-10');
  expect(timetable.props.data).toHaveLength(1);
  const timetableHead = heads().props.data.find((d: any) => d.run.run_id === 10);
  const liveTrails = rendered.layers.filter(l => l.id.startsWith('comet-') || l.id === 'streaks');
  Socket.current.receive({ gen: 11, ts: 3601, is_full: false, versions: {}, base_ts: 1000,
    runs: { 9: { ...run, state: 'stale' } }, tombstones: [] });
  draw!(3350);
  const nextTimetable = rendered.layers.find(l => l.id === 'timetable-10');
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
  const beforeSelection = heads().props.updateTriggers.getPosition;
  pointer('click', point(9));
  draw!(3462);
  expect.soft(heads().props.updateTriggers.getPosition).toBe(beforeSelection);
  expect(rendered.layers.find(l => l.id === 'selected-head').props.updateTriggers.getPosition).toBe(beforeSelection);
  const pulse = () => rendered.layers.find(l => l.id === 'selection-pulse');
  const firstPulse = pulse();
  const selectedDraws = rendered.draws;
  draw!(3480);
  expect(rendered.draws).toBe(selectedDraws);
  draw!(3500); // A selected route redraws at ~30 Hz so its pulse keeps moving.
  expect(rendered.draws).toBe(selectedDraws + 1);
  expect(pulse().props.currentTime).toBeGreaterThan(firstPulse.props.currentTime);
  expect(pulse().props.data).toBe(firstPulse.props.data);
  expect.soft(heads().props.updateTriggers.getPosition).toBe(beforeSelection);

  rendered.listeners.keydown({ key: 'Escape' });
  draw!(4000);
  const headAt = vi.spyOn(await import('./layers'), 'headAt');
  const easeHead = vi.spyOn(await import('./layers'), 'easeHead');
  const version = () => heads().props.updateTriggers.getPosition;
  const idleVersion = version(), idleDraws = rendered.draws;
  for (const tick of [4001, 4016, 4099]) draw!(tick);
  expect(rendered.draws).toBe(idleDraws);
  expect(headAt).not.toHaveBeenCalled();
  draw!(4100);
  expect(rendered.draws).toBe(idleDraws + 1);
  expect.soft(version()).toBe(idleVersion + 1);
  expect(headAt).toHaveBeenCalledTimes(2);
  expect(easeHead).toHaveBeenCalledTimes(2);

  const movingVersion = version(), movingDraws = rendered.draws;
  headAt.mockClear(); easeHead.mockClear();
  pointer('mousemove', point());
  expect(hover.hidden).toBe(false);
  rendered.listeners['map:move']();
  expect(hover.hidden).toBe(true);
  draw!(4116);
  expect.soft(rendered.draws).toBe(movingDraws);
  expect.soft(version()).toBe(movingVersion);
  expect.soft(headAt).not.toHaveBeenCalled();
  expect.soft(easeHead).not.toHaveBeenCalled();

  headAt.mockClear(); easeHead.mockClear();
  rendered.zoom = 8.1;
  rendered.listeners['map:move']();
  rendered.listeners['map:move']();
  const zoomDraws = rendered.draws;
  draw!(4132);
  expect(rendered.draws).toBe(zoomDraws + 1);
  expect.soft(version()).toBe(movingVersion);
  expect.soft(headAt).not.toHaveBeenCalled();
  expect.soft(easeHead).not.toHaveBeenCalled();
  draw!(4148);
  expect(rendered.draws).toBe(zoomDraws + 1);
  draw!(4200);
  expect.soft(version()).toBe(movingVersion + 1);

  const frameVersion = version(), frameDraws = rendered.draws;
  Socket.current.receive({ gen: 12, ts: 1600, is_full: false, versions: {}, base_ts: null,
    runs: {}, tombstones: [] });
  headAt.mockClear(); easeHead.mockClear();
  draw!(4208);
  expect.soft(rendered.draws).toBe(frameDraws + 1);
  expect.soft(version()).toBe(frameVersion);
  expect.soft(headAt).not.toHaveBeenCalled();
  expect.soft(easeHead).not.toHaveBeenCalled();

  pointer('click', point());
  headAt.mockClear(); easeHead.mockClear();
  draw!(4224);
  expect.soft(version()).toBe(frameVersion);
  const pulseDraws = rendered.draws, pulseTime = pulse().props.currentTime;
  draw!(4256);
  expect.soft(rendered.draws).toBe(pulseDraws);
  draw!(4257);
  expect(rendered.draws).toBe(pulseDraws + 1);
  expect(pulse().props.currentTime).toBeGreaterThan(pulseTime);
  expect.soft(version()).toBe(frameVersion);
  draw!(4290);
  expect(rendered.draws).toBe(pulseDraws + 2);
  expect.soft(version()).toBe(frameVersion);
  expect.soft(headAt).not.toHaveBeenCalled();
  expect.soft(easeHead).not.toHaveBeenCalled();

  rendered.listeners.keydown({ key: 'Escape' });
  draw!(4500);
  simTime = 1900;
  draw!(4600);
  const easingHead = head(), easingVersion = version();
  expect(easingHead[0]).toBeCloseTo(71 + 0.5 * (1 - Math.exp(-100 / 250)));
  draw!(4604);
  expect.soft(version()).toBe(easingVersion + 1);
  expect.soft(head()[0]).toBeCloseTo(easingHead[0] + (71.5 - easingHead[0]) * (1 - Math.exp(-4 / 250)), 8);
  let tick = 4604;
  while (head()[0] !== 71.5 && tick < 8604) {
    const previous = head()[0], previousVersion = version();
    draw!(tick += 16);
    expect(head()[0]).toBeGreaterThan(previous);
    expect(head()[0]).toBeLessThanOrEqual(71.5);
    expect(version()).not.toBe(previousVersion);
  }
  expect(head()).toEqual([71.5, 21.5]);
  const settledVersion = version(), settledDraws = rendered.draws;
  headAt.mockClear(); easeHead.mockClear();
  draw!(tick + 1);
  draw!(tick + 99);
  expect(rendered.draws).toBe(settledDraws);
  expect(headAt).not.toHaveBeenCalled();
  expect(easeHead).not.toHaveBeenCalled();
  draw!(tick + 100);
  expect(rendered.draws).toBe(settledDraws + 1);
  expect.soft(version()).toBe(settledVersion + 1);
  expect(headAt).toHaveBeenCalledTimes(2);
  expect(easeHead).toHaveBeenCalledTimes(2);

  const rect = (left: number, top: number, width: number, height: number) =>
    ({ left, top, width, height, right: left + width, bottom: top + height });
  const empty = rect(0, 0, 0, 0), live = elements.get('#live'), legend = elements.get('#legend');
  const wide = { width: 1400, height: 900, live: rect(16, 16, 300, 164),
    legend: rect(16, 830, 300, 54), panel: rect(1004, 16, 380, 868), bar: rect(1004, 16, 380, 100),
    hud: 'shown', reduce: false, collapsed: false };
  const phone = { width: 390, height: 844, live: rect(12, 12, 300, 168),
    legend: rect(12, 778, 300, 54), panel: rect(12, 440, 366, 392), bar: rect(12, 732, 366, 100),
    hud: 'shown', reduce: false, collapsed: false };
  for (const test of [
    { ...wide, name: 'wide', padding: { top: 196, right: 412, bottom: 86, left: 16 } },
    { ...phone, name: 'phone', padding: { top: 196, right: 16, bottom: 534.4, left: 16 } },
    { ...phone, name: 'short phone panel', panel: rect(12, 752, 366, 80),
      padding: { top: 196, right: 16, bottom: 534.4, left: 16 } },
    { ...wide, name: 'hidden wide HUD', hud: 'hidden', live: rect(16, 16, 300, 60),
      padding: { top: 92, right: 16, bottom: 16, left: 16 } },
    { ...phone, name: 'hidden phone HUD', hud: 'hidden', live: rect(12, 12, 300, 36),
      padding: { top: 64, right: 16, bottom: 16, left: 16 } },
    { ...wide, name: 'no horizontal room', panel: rect(32, 16, 1352, 868),
      padding: { top: 16, right: 16, bottom: 16, left: 16 } },
    { ...phone, name: 'no vertical room', live: rect(12, 12, 300, 396),
      padding: { top: 16, right: 16, bottom: 16, left: 16 } },
    { ...wide, name: 'reduced motion', reduce: true,
      padding: { top: 196, right: 412, bottom: 86, left: 16 } },
    { ...wide, name: 'collapsed wide', collapsed: true,
      padding: { top: 196, right: 16, bottom: 86, left: 16 } },
    { ...wide, name: 'bar below wide HUD', collapsed: true, live: rect(16, 16, 300, 60),
      padding: { top: 132, right: 16, bottom: 86, left: 16 } },
    { ...phone, name: 'collapsed phone', collapsed: true,
      padding: { top: 196, right: 16, bottom: 128, left: 16 } },
    { ...phone, name: 'collapsed phone without room', collapsed: true, live: rect(12, 12, 300, 720),
      padding: { top: 16, right: 16, bottom: 16, left: 16 } },
    { ...wide, name: 'hidden collapsed HUD', collapsed: true, hud: 'hidden', live: rect(16, 16, 300, 60),
      padding: { top: 92, right: 16, bottom: 16, left: 16 } },
  ]) {
    window.innerWidth = test.width;
    window.innerHeight = test.height;
    body.dataset.hud = test.hud;
    reducedMotion = test.reduce;
    live.getBoundingClientRect = () => test.live;
    legend.getBoundingClientRect = () => test.hud === 'hidden' || (test.width < 640 && !panel.hidden) ? empty : test.legend;
    panel.getBoundingClientRect = () => panel.hidden || test.hud === 'hidden' ? empty :
      panel.dataset.collapsed !== undefined ? test.bar : test.panel;
    rendered.listeners.keydown({ key: 'Escape' });
    if ((panel.dataset.collapsed !== undefined) !== test.collapsed) panelToggle.listeners.click?.();
    rendered.zoom = 14;
    const fits = rendered.fits.length;
    rendered.selectSearch!(run);
    expect.soft(rendered.fits.length, test.name).toBe(fits + 1);
    expect.soft(rendered.fits.at(-1), test.name).toEqual({ bounds: [[70, 20], [72, 22]],
      options: { padding: test.padding, maxZoom: 10, linear: true, duration: test.reduce ? 0 : 600 } });
  }
  expect.soft(matchMedia).toHaveBeenCalledWith('(prefers-reduced-motion: reduce)');
  expect.soft(rendered.ease).toHaveLength(0);

  body.dataset.hud = 'shown';
  reducedMotion = false;
  for (const test of [wide, phone]) {
    window.innerWidth = test.width;
    window.innerHeight = test.height;
    live.getBoundingClientRect = () => test.live;
    legend.getBoundingClientRect = () => test.width < 640 ? empty : test.legend;
    panel.getBoundingClientRect = () => panel.dataset.collapsed !== undefined ? test.bar : test.panel;
    const fits = rendered.fits.length;
    panelToggle.listeners.click?.();
    expect.soft(rendered.fits.length).toBe(fits + 1);
    expect.soft(rendered.fits.at(-1)).toEqual({ bounds: [[70, 20], [72, 22]], options: {
      padding: { top: 196, right: test.width < 640 ? 16 : 412,
        bottom: test.width < 640 ? 534.4 : 86, left: 16 }, maxZoom: 10, linear: true, duration: 600 } });
    panelToggle.listeners.click?.();
    expect.soft(rendered.fits.length).toBe(fits + 2);
    expect.soft(rendered.fits.at(-1)).toEqual({ bounds: [[70, 20], [72, 22]], options: {
      padding: { top: 196, right: 16, bottom: test.width < 640 ? 128 : 86, left: 16 },
      maxZoom: 10, linear: true, duration: 600 } });
  }

  const fits = rendered.fits.length;
  draw!(10000);
  const requests = () => fetch.mock.calls.filter(([url]) => url === '/api/run/9').length;
  const beforeGeneration = requests();
  Socket.current.receive({ gen: 13, ts: 1900, is_full: false, versions: {}, base_ts: 1000,
    runs: { 9: { ...run, state: 'observed' } }, tombstones: [] });
  draw!(10016);
  expect(requests()).toBe(beforeGeneration + 1); // Re-showing the panel must not move the camera.
  expect(rendered.fits).toHaveLength(fits);
  Socket.current.receive({ gen: 14, ts: 1900, is_full: false, versions: {}, base_ts: 1000,
    runs: { 9: { ...run, versions: { route: 99 } } }, tombstones: [] });
  draw!(10032);
  expect(requests()).toBe(beforeGeneration + 2);
  expect(rendered.fits).toHaveLength(fits);
  panelToggle.listeners.click?.();
  expect.soft(rendered.fits).toHaveLength(fits);
  panelToggle.listeners.click?.();
  expect.soft(rendered.fits).toHaveLength(fits);
  Socket.current.receive({ gen: 15, ts: 1900, is_full: false, versions: {}, base_ts: 1000,
    runs: { 9: run }, tombstones: [] });
  draw!(10048);
  expect(rendered.fits).toHaveLength(fits);
  rendered.listeners.keydown({ key: 'Escape' });
  expect(rendered.fits).toHaveLength(fits);
  panelToggle.listeners.click?.();
  expect.soft(rendered.fits).toHaveLength(fits);
  rendered.listeners.pagehide();
  expect(clearInterval).toHaveBeenCalledWith(7);
  expect(toggle.listeners.click).toBeUndefined();
  expect(rendered.listeners['document:keydown']).toBeUndefined();
});
