import { expect, it } from 'vitest';
import { TripsLayer } from '@deck.gl/geo-layers';
import { ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import { indexNetwork, networkLayer, displayRun, trailData, trainLayers,
  delayColor, speedColor, trailLength, easeHead, headAt, labelBucket, labelStations, stationLabelLayer, selectionLayers, PULSE_MS,
  nearestRun, dataComparator, glyphShift, glyphMetrics, padGlyphAlpha, type Network } from './layers';

const GLOW_SCALE = Math.sqrt(6.125 / 3.5);
const timetableIds = Array.from({ length: 16 }, (_, c) => `timetable-${c}`);

function packQuantiles(quantiles: (number | null)[][]): Uint8Array {
  const values = quantiles.flat(), packed = new Uint8Array(values.length * 4);
  const bytes = new DataView(packed.buffer);
  values.forEach((q, i) => bytes.setFloat32(i * 4, q ?? NaN, true));
  return packed;
}

const network: Network = {
  stations: [{ code: 'A', lon: 70, lat: 20 }, { code: 'B', lon: 72, lat: 22 },
    { code: 'missing', lon: null, lat: null }, { code: 'invalid', lon: 181, lat: 20 }],
  segments: [{ id: 1, stn_lo: 'A', stn_hi: 'B', polyline: [[70, 20], [72, 22]], derived: false, confidence: 0 },
    { id: 2, stn_lo: 'A', stn_hi: 'missing', polyline: null, derived: false, confidence: 0 }],
  routes: [{ id: 7, train: '00001', name: 'Express', class: 'EXP', stops: [
    { idx: 0, station: 'A', halt: true, dist_km: 0, sched_off: 0 },
    { idx: 1, station: 'B', halt: true, dist_km: 100, sched_off: 10 },
  ], segments: [] }],
};

it('indexes stations, routes and stops with the same results as linear lookup', () => {
  const index = indexNetwork(network);
  expect(indexNetwork(network)).toBe(index);
  for (const code of ['A', 'B', 'missing', 'invalid', 'unknown']) {
    const station = network.stations.find(s => s.code === code);
    const point = station && typeof station.lon === 'number' && Math.abs(station.lon) <= 180
      && typeof station.lat === 'number' && Math.abs(station.lat) <= 90
      ? [station.lon, station.lat] : null;
    expect(index.stations.get(code) ?? null).toEqual(point);
  }
  for (const id of [7, 99]) {
    const route = network.routes.find(r => r.id === id);
    expect(index.routes.get(id)).toBe(route);
    for (const idx of [0, 1, 99])
      expect(index.stops.get(id)?.get(idx)).toBe(route?.stops.find(s => s.idx === idx));
  }
});

it('builds one network layer and geometry array per network', () => {
  const layer = networkLayer(network);
  expect(layer.props.data).toEqual([[[70, 20], [72, 22]]]);
  expect(networkLayer(network)).toBe(layer);
  expect(networkLayer(network).props.data).toBe(layer.props.data);
  expect(networkLayer({ ...network })).not.toBe(layer);
  expect(layer.props.getColor).toEqual([0, 200, 255, 36]);
  expect(layer.props.getWidth).toBe(1.5);
});

it('uses indexed geometry when decoding the existing P50 fixture', () => {
  const run = { run_id: 9, versions: { route: 7 }, origin_ts: 1000, base_ts: 1000,
    path_shape: [2, 2, 2] as [number, number, number],
    paths: new Uint8Array([0, 0, 0, 0, 1, 0, 100, 0, 0, 0, 0, 0, 1, 0, 200, 0]),
    halt_p50: packQuantiles([[null, 0], [20, null]]) };
  const result = displayRun(run, network, 1600);
  expect(result?.head).toEqual([71, 21]);
  expect(result?.trips).toEqual([
    { path: [[70, 20], [72, 22]], timestamps: [1000, 1600] },
    { path: [[70, 20], [72, 22]], timestamps: [1000, 2200] },
  ]);
});

function runFixture(k = 2) {
  const paths = new Uint8Array(k * 8);
  const bytes = new DataView(paths.buffer);
  for (let p = 0; p < k; p++) {
    bytes.setUint16(p * 8 + 4, 1, true);
    bytes.setUint16(p * 8 + 6, 100 + p, true);
  }
  return { run_id: 9, versions: { route: 7 }, origin_ts: 1000, base_ts: 1000,
    path_shape: [k, 2, 2] as [number, number, number], paths };
}

const chain: [number, number][] = [[70, 20], [70, 21], [70, 22], [72, 22]];
const chained: Network = { ...network,
  stations: [...network.stations, { code: 'X', lon: 70, lat: 21 }, { code: 'Y', lon: 70, lat: 22 }],
  segments: [{ ...network.segments[0], polyline: chain, derived: true }],
  routes: [{ ...network.routes[0], segments: [{ idx: 0, segment_id: 1, forward: true }] }],
};

it('expands comets and all particle streaks along the chain with length-based timestamps', () => {
  const display = displayRun(runFixture(), chained, 1300)!;
  const trails = trailData([display], 1300);
  expect(trails.comets[0].path).toEqual(chain);
  expect(trails.comets[0].timestamps).toEqual([1000, 1150.75, 1301.5, 1603]);
  expect(display.trips.map(t => t.path)).toEqual([chain, chain]);
  expect(trails.streaks.map(t => t.path)).toEqual([chain, chain]);
  expect(trails.streaks.map(t => t.timestamps)).toEqual([
    [1000, 1150, 1300, 1600], [1000, 1151.5, 1303, 1606],
  ]);
  for (const trip of [...trails.comets, ...trails.streaks])
    expect(trip.timestamps.every((t, i) => i === 0 || t >= trip.timestamps[i - 1])).toBe(true);
});

it('walks the chain by the pos fraction, including its midpoint instead of the chord', () => {
  const display = displayRun(runFixture(1), chained, 1300)!;
  expect(display.head).toEqual([70, 22]);
  expect(headAt(display, 1450)).toEqual([71, 22]);
  expect(headAt(display, 900)).toEqual(chain[0]);
  expect(headAt(display, 1700)).toEqual(chain.at(-1));
});

it('orients a reversed segment from the route start without mutating stored geometry', () => {
  const reversed: Network = { ...chained, routes: [{ ...chained.routes[0],
    stops: chained.routes[0].stops.map(s => ({ ...s, station: s.idx === 0 ? 'B' : 'A' })),
    segments: [{ idx: 0, segment_id: 1, forward: false }],
  }] };
  const display = displayRun(runFixture(1), reversed, 1300)!;
  const path = [...chain].reverse();
  expect(display.trips[0].path).toEqual(path);
  expect(trailData([display], 1300).comets[0].path).toEqual(path);
  expect(display.trips[0].timestamps).toEqual([1000, 1300, 1450, 1600]);
  expect(display.head).toEqual([70, 22]);
  expect(headAt(display, 1150)).toEqual([71, 22]);
  expect(chained.segments[0].polyline).toEqual(chain);
});

function eventRun(events: [number, number][]) {
  const paths = new Uint8Array(events.length * 4);
  const bytes = new DataView(paths.buffer);
  events.forEach(([idx, time], e) => {
    bytes.setUint16(e * 4, idx, true);
    bytes.setUint16(e * 4 + 2, time, true);
  });
  return { ...runFixture(1), paths, path_shape: [1, events.length, 2] as [number, number, number] };
}

it('joins every route segment across skipped events and keeps repeated-stop dwell', () => {
  const skipped: Network = { ...chained, segments: [
    { ...chained.segments[0], polyline: null, stn_hi: 'X' },
    { ...chained.segments[0], id: 2, stn_lo: 'B', stn_hi: 'X', polyline: [chain[3], chain[2], chain[1]] },
  ], routes: [{ ...chained.routes[0], stops: [chained.routes[0].stops[0],
    { idx: 1, station: 'X', halt: false, dist_km: 25, sched_off: null },
    { ...chained.routes[0].stops[1], idx: 2 }], segments: [
    { idx: 0, segment_id: 1, forward: true }, { idx: 1, segment_id: 2, forward: false },
  ] }] };
  const display = displayRun(eventRun([[0, 0], [0, 20], [2, 100]]), skipped, 1100)!;
  const path = [chain[0], ...chain];
  expect(display.trips[0].path).toEqual(path);
  expect(display.trips[0].timestamps).toEqual([1000, 1120, 1240, 1360, 1600]);
  expect(trailData([display], 1100).comets[0].path).toEqual(path);
  expect(display.head).toEqual(chain[0]);
  expect(headAt(display, 1360)).toEqual(chain[2]);
  expect(displayRun(eventRun([[0, 50], [2, 20]]), skipped, 1100)).toBeNull();
  expect(displayRun(eventRun([[0, 0], [2, 100]]), { ...skipped,
    stations: skipped.stations.filter(s => s.code !== 'X') }, 1100)).toBeNull();
});

it('keeps zero-length segment timestamps finite and event endpoints intact', () => {
  const stationary: Network = { ...network, stations: [
    { code: 'A', lon: 70, lat: 20 }, { code: 'B', lon: 70, lat: 20 }],
    segments: [], routes: network.routes };
  const display = displayRun(runFixture(1), stationary, 1300)!;
  expect(display.trips[0]).toEqual({ path: [[70, 20], [70, 20]], timestamps: [1000, 1600] });
  expect(display.head).toEqual([70, 20]);
});

it('hides derived segments and straight chords over 150 km while retaining stored polylines', () => {
  const drawing: Network = { ...chained, stations: [...chained.stations,
    { code: 'C', lon: 70.1, lat: 20 }], segments: [
    chained.segments[0], // Derived.
    { ...network.segments[0], id: 2, polyline: null }, // Long chord.
    { ...network.segments[0], id: 3, polyline: chain }, // Stored long polyline.
    { ...network.segments[0], id: 4, stn_hi: 'C', polyline: null }, // Short chord.
  ] };
  expect(networkLayer(drawing).props.data).toEqual([chain, [[70, 20], [70.1, 20]]]);
});

it('draws one fading glow comet with additive blending and unpicked heads', () => {
  const display = displayRun(runFixture(), network, 1300)!;
  const trails = trailData([display], 1300);
  const layers = trainLayers([display], trails, 1300, 4);
  const trips = layers.filter(layer => layer instanceof TripsLayer);
  const comets = trips.filter(layer => layer.id === 'comets');
  expect(trips.map(layer => layer.id)).toEqual(['comets', 'streaks', ...timetableIds]);
  expect(comets).toHaveLength(1);
  expect(comets[0].props.getWidth).toBe(7 * GLOW_SCALE);
  expect(comets[0].props.trailLength).toBe(21600);
  expect((comets[0].props.getColor as Function)(trails.comets[0])).toEqual([...display.color, 255]);
  for (const layer of trips) {
    expect(layer.props.fadeTrail).toBe(true);
    expect(layer.props.widthUnits).toBe('pixels');
    expect(layer.props.parameters).toMatchObject({ blend: true,
      blendColorOperation: 'add', blendColorSrcFactor: 'src-alpha', blendColorDstFactor: 'one',
      blendAlphaOperation: 'add', blendAlphaSrcFactor: 'one', blendAlphaDstFactor: 'one',
      depthWriteEnabled: false });
  }
  const streaks = trips.find(layer => layer.id === 'streaks')!;
  expect(streaks.props.getWidth).toBe(1.5);
  expect(streaks.props.trailLength).toBe(32400);
  expect((streaks.props.getColor as Function)(trails.streaks[0])[3]).toBe(24);
  const heads = layers.find(layer => layer instanceof ScatterplotLayer)!;
  expect((heads.props.getRadius as Function)(display)).toBe(5.5 * GLOW_SCALE);
  expect(heads.props.radiusUnits).toBe('pixels');
  expect(heads.props.pickable).toBe(false); // GPU picking stalls on readPixels; main.ts picks on the CPU.
  expect((heads.props.getFillColor as Function)(display)).toEqual([...display.color, 230]);
  expect(heads.props.parameters).toEqual(comets[0].props.parameters);
});

it('picks the nearest drawn head within ten pixels, keeping the first on ties', () => {
  const a = displayRun(runFixture(), network, 1300)!;
  const b = { ...a, run: { ...a.run, run_id: a.run.run_id + 1 } };
  const c = { ...a, run: { ...a.run, run_id: a.run.run_id + 2 } };
  const [lon, lat] = a.head; // a has no drawn head yet, so its target counts.
  const heads = new Map<number, [number, number]>([[b.run.run_id, [lon + 0.1, lat]], [c.run.run_id, [lon, lat]]]);
  const project = ([x, y]: [number, number]) => ({ x: (x - lon) * 100, y: (y - lat) * 100 });
  expect(nearestRun([a, b, c], heads, project, 4, 0)).toBe(a.run);
  expect(nearestRun([a, b, c], heads, project, 8, 0)).toBe(b.run);
  expect(nearestRun([a, b, c], heads, project, 21, 0)).toBeNull();
});

it('appends Gaussian glow after the TripsLayer fade and guards both glow snippets during picking', () => {
  const display = displayRun(runFixture(), network, 1300)!;
  const layers = trainLayers([display], trailData([display], 1300), 1300, 4);
  const filter = 'fs:DECKGL_FILTER_COLOR';
  const glow = `if (picking.isActive < 0.5) {
  float age = clamp((trips.currentTime - vTime) / trips.trailLength, 0.0, 1.0);
  float x = geometry.uv.x / mix(1.0, 0.35, age);
  color.rgb = mix(color.rgb, vec3(1.0), 0.15 * exp(-24.5 * x * x));
  color.a *= exp(-6.125 * x * x) * (1.0 - age);
}`;
  const blob = `if (picking.isActive < 0.5) {
  float r2 = dot(geometry.uv, geometry.uv);
  color.rgb = mix(color.rgb, vec3(1.0), 0.2 * exp(-21.0 * r2));
  color.a *= exp(-6.125 * r2);
}`;
  expect(Math.exp(-6.125) * 255).toBeLessThan(1);
  // The wider geometry preserves the old 10% halo contour and white-hot core.
  const contour = Math.sqrt(-Math.log(0.1) / 3.5);
  expect(Math.exp(-6.125 * (contour / GLOW_SCALE) ** 2)).toBeCloseTo(0.1);
  expect(24.5 / GLOW_SCALE ** 2).toBeCloseTo(14);
  expect(21 / GLOW_SCALE ** 2).toBeCloseTo(12);
  for (const layer of layers.filter(l => l.id !== 'streaks')) {
    Object.assign(layer, { context: { device: { type: 'webgl' }, defaultShaderModules: [] } });
    const shaders = layer.getShaders();
    if (layer instanceof TripsLayer) {
      const original = TripsLayer.prototype.getShaders.call(layer);
      expect(shaders.inject[filter]).toBe(original.inject[filter] + glow);
      for (const key of ['vs:#decl', 'fs:#decl'])
        expect(shaders.inject[key]).toBe(original.inject[key]);
      expect(shaders.inject['vs:#main-end']).toContain(original.inject['vs:#main-end']);
    } else {
      expect(shaders.inject[filter]).toBe(blob);
    }
    expect(shaders.inject[filter]).toContain('if (picking.isActive < 0.5) {');
  }
  const pulse = selectionLayers(display, 0, 900)[1];
  Object.assign(pulse, { context: { device: { type: 'webgl' }, defaultShaderModules: [] } });
  expect(pulse.getShaders().inject[filter]).toContain(glow);
});

it('culls whole segments outside each trips window and gives only the pulse a soft Gaussian lead', () => {
  const display = displayRun(runFixture(), network, 1300)!;
  const layers = [...trainLayers([display], trailData([display], 1300), 1300, 4),
    selectionLayers(display, 0, 900)[1]].filter(l => l instanceof TripsLayer);
  expect(layers.map(l => l.id)).toEqual(['comets', 'streaks', ...timetableIds, 'selection-pulse']);
  const filter = 'fs:DECKGL_FILTER_COLOR';
  for (const layer of layers) {
    Object.assign(layer, { context: { device: { type: 'webgl' }, defaultShaderModules: [] } });
    const shaders = layer.getShaders(), original = TripsLayer.prototype.getShaders.call(layer);
    expect(shaders.inject['vs:#main-end']).toBe(original.inject['vs:#main-end'] + `
float lead = ${layer.id === 'selection-pulse' ? '0.1' : '0.0'} * trips.trailLength;
if (max(instanceTimestamps, instanceNextTimestamps) < trips.currentTime - trips.trailLength ||
    min(instanceTimestamps, instanceNextTimestamps) > trips.currentTime + lead) {
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}
`);
    expect(shaders.inject['fs:#main-start']).toBeUndefined(); // Discard stays after analytic coverage.
    if (layer.id === 'selection-pulse') {
      expect(shaders.inject[filter]).toContain(`float lead = 0.1 * trips.trailLength;
if (vTime > trips.currentTime + lead || (trips.fadeTrail && vTime < trips.currentTime - trips.trailLength)) {
  discard;
}
if (vTime > trips.currentTime) {
  float front = (vTime - trips.currentTime) / lead;
  color.a *= exp(-6.125 * front * front);
} else if (trips.fadeTrail) {
  color.a *= 1.0 - (trips.currentTime - vTime) / trips.trailLength;
}
`);
      expect(shaders.inject[filter]).not.toContain('if(vTime > trips.currentTime ||');
    } else {
      expect(shaders.inject[filter]).toContain(original.inject[filter]);
      expect(shaders.inject[filter]).not.toContain('float front');
      if (layer.id === 'streaks') expect(shaders.inject[filter]).toBe(original.inject[filter]);
    }
  }
});

it.each([[3, 0.5, 0.35, 0.6, 0.45], [4, 0.5, 0.35, 0.6, 0.45],
  [8, 1, 1, 1, 1], [10, 1, 1, 1, 1]])(
  'scales timetable glow and heads with clamped zoom %s', (zoom, widthScale, trailOpacity, radiusScale, headOpacity) => {
    const live = displayRun({ ...runFixture(), state: 'observed' }, network, 1300)!;
    const timetable = displayRun({ ...runFixture(), run_id: 10, state: 'schedule_only' }, network, 1300)!;
    const layers = trainLayers([live, timetable], trailData([live, timetable], 1300), 1300, zoom);
    const core = layers.filter(l => l instanceof TripsLayer).find(l => l.id === 'timetable-10')!;
    expect(core.props.getWidth).toBe(6 * GLOW_SCALE);
    expect(core.props.widthScale).toBe(widthScale);
    expect(core.props.opacity).toBe(trailOpacity);
    const heads = layers.find(l => l instanceof ScatterplotLayer)!;
    expect((heads.props.getRadius as Function)(timetable)).toBe(4.5 * GLOW_SCALE);
    expect((heads.props.getRadius as Function)(live)).toBe(5.5 * GLOW_SCALE);
    expect(heads.props.radiusScale).toBe(radiusScale);
    expect(heads.props.opacity).toBe(headOpacity);
  });

it('omits only schedule-only expanded trips while preserving median fallback and trail data', () => {
  const run = runFixture();
  const live = displayRun({ ...run, state: 'observed' }, chained, 1300)!;
  const timetable = displayRun({ ...run, state: 'schedule_only' }, chained, 1300)!;
  expect(timetable.trips).toEqual([]);
  expect(live.trips).toHaveLength(run.path_shape[0]);
  expect(live.eventTimes).toEqual([1000, 1603]);
  expect(timetable).toEqual({ ...live, run: timetable.run, trips: [] });
  const core = { path: chain, runId: 9, color: live.color, timestamps: [1000, 1150.75, 1301.5, 1603] };
  expect(trailData([live], 1300)).toEqual({ base: 0, comets: [core], timetable: [], streaks: [
    { path: chain, runId: 9, color: live.color, timestamps: [1000, 1150, 1300, 1600] },
    { path: chain, runId: 9, color: live.color, timestamps: [1000, 1151.5, 1303, 1606] },
  ] });
  expect(trailData([timetable], 1300)).toEqual({ base: 0, comets: [], timetable: [core], streaks: [] });
});

it('makes median comet timestamps monotone without accepting non-monotone particle paths', () => {
  const run = { ...runFixture(), halt_p50: packQuantiles([[null, 20], [10, null]]) };
  const display = displayRun(run, network, 1300)!;
  const trails = trailData([display], 1300);
  expect(trails.comets[0].path).toBe(display.points);
  expect(trails.comets[0].timestamps).toEqual([2200, 2200]);
  const invalid = runFixture();
  new DataView(invalid.paths.buffer).setUint16(2, 200, true);
  expect(displayRun(invalid, network, 1300)).toBeNull();
  expect(displayRun(runFixture(), { ...network, stations: [] }, 1300)).toBeNull();
});

it.each([
  [-20, [0, 240, 255]], [0, [0, 240, 255]], [7.5, [126, 239, 132.5]],
  [15, [252, 238, 10]], [37.5, [253.5, 140, 59.5]],
  [60, [255, 42, 109]], [120, [255, 42, 109]],
])('interpolates and clamps the neon colour for %s minutes of delay', (delay, colour) => {
  expect(delayColor(delay as number)).toEqual(colour);
});

it('uses next-stop delay at decode time, falling back to cyan when unknown', () => {
  const run = { ...runFixture(), halt_p50: packQuantiles([[null, 0], [25, null]]) };
  expect(displayRun(run, network, 1300)?.color).toEqual([252, 238, 10]);
  expect(displayRun(run, network, 2600)?.color).toEqual([0, 240, 255]);
  expect(displayRun({ ...run, origin_ts: undefined }, network, 1300)?.color).toEqual([0, 240, 255]);
  const unknownSchedule = { ...network, routes: [{ ...network.routes[0],
    stops: network.routes[0].stops.map(s => ({ ...s, sched_off: null })) }] };
  expect(displayRun(run, unknownSchedule, 1300)?.color).toEqual([0, 240, 255]);
});

it('reads little-endian P50s from unaligned halt_p50 subarrays', () => {
  const packed = packQuantiles([[null, .25], [25.125, null]]);
  const buffer = new Uint8Array(packed.byteLength + 2).fill(255);
  buffer.set(packed, 1);
  const run = { ...runFixture(), halt_p50: buffer.subarray(1, -1) };
  const display = displayRun(run, network, 1300)!;
  expect(display.eventTimes).toEqual([1015, 2507.5]);
  expect(display.delay).toBe(15.125);
});

it('falls back to display medians for NaN, missing or short halt P50s', () => {
  const run = { ...runFixture(), halt_p50: packQuantiles([[null, null], [null, null]]) };
  const display = displayRun(run, network, 1300)!;
  expect(display.eventTimes).toEqual([1000, 1603]);
  expect(display.delay).toBeCloseTo(.05);
  const short = { ...run, halt_p50: packQuantiles([[null, .25]]) };
  expect(displayRun(short, network, 1300)!.eventTimes).toEqual([1015, 1603]);
  const empty = { ...run, halt_p50: new Uint8Array(0) };
  expect(displayRun(empty, network, 1300)!.eventTimes).toEqual([1000, 1603]);
  const { halt_p50, ...absent } = run;
  expect(displayRun(absent, network, 1300)!.eventTimes).toEqual([1000, 1603]);
  const old = { ...absent, halt_shape: [2, 2, 7], halt_q: new Uint8Array(112) };
  expect(displayRun(old, network, 1300)!.eventTimes).toEqual([1000, 1603]);
});

it.each([[-10, 32400], [3, 32400], [4, 32400], [6, 8100], [7, 4050], [10, 2700], [20, 2700]])(
  'clamps the trail to the zoom-dependent sim seconds at zoom %s', (zoom, length) => {
    expect(trailLength(zoom)).toBe(length);
  });

it('snaps a new head and eases existing heads partway to the target before converging', () => {
  const target: [number, number] = [72, 22];
  expect(easeHead(undefined, target, 16)).toEqual(target);
  const head = easeHead([70, 20], target, 250);
  expect(head[0]).toBeCloseTo(72 - 2 / Math.E);
  expect(head[1]).toBeCloseTo(22 - 2 / Math.E);
  expect(easeHead(head, target, 0)).toEqual(head);
  const converged = easeHead(head, target, 5000);
  expect(converged[0]).toBeCloseTo(72, 6);
  expect(converged[1]).toBeCloseTo(22, 6);
  expect(easeHead(head, target, 250, 0.01)).not.toBe(target);
  expect(easeHead(head, target, 2000, 0.01)).toBe(target);
});

it.each([
  [1, [1600]], [3, [1600, 1606, 1612]],
  [10, [1600, 1606, 1612, 1618, 1630, 1636, 1642, 1648]],
  [32, [1600, 1624, 1648, 1672, 1696, 1720, 1744, 1768]],
])('keeps at most eight strided streaks from each run with %s particles', (k, times) => {
  const display = displayRun(runFixture(k as number), network, 1300)!;
  const trails = trailData([display, { ...display, run: { ...display.run, run_id: 10 } }], 1300);
  expect(trails.comets).toHaveLength(2);
  expect(trails.streaks.map(t => t.timestamps[1])).toEqual([...(times as number[]), ...(times as number[])]);
});

const labelNetwork: Network = { stations: [
  { code: 'A', name: 'ALPHA CENTRAL', lon: 70, lat: 20 },
  { code: 'B', name: 'BETA', lon: 71, lat: 21 },
  { code: 'C', name: 'GAMMA', lon: 72, lat: 22 },
  { code: 'D', name: 'DELTA', lon: 73, lat: 23 },
  { code: 'E', name: 'EPSILON', lon: 74, lat: 24 },
  { code: 'F', name: 'ZETA', lon: 75, lat: 25 },
  { code: 'G', name: 'ETA', lon: 76, lat: 26 },
  { code: 'M', name: 'MISSING', lon: null, lat: null },
  { code: 'I', name: 'INVALID', lon: 181, lat: 20 },
  { code: 'Z', name: 'UNUSED', lon: 77, lat: 27 }], segments: [],
  routes: Array.from({ length: 400 }, (_, id) => ({ ...network.routes[0], id,
    stops: ['T', 'A', 'A', ...(id < 250 ? ['B'] : []), ...(id < 150 ? ['C'] : []),
      ...(id < 80 ? ['D'] : []), ...(id < 40 ? ['E'] : []), ...(id < 20 ? ['F'] : []),
      ...(id < 1 ? ['G', 'M', 'I'] : []), 'T']
      .map((station, idx, stops) => ({ idx, station, halt: true,
        dist_km: idx * 300 / (stops.length - 1), sched_off: idx * 5 })) })) };

it.each([[4.9, ['A']], [5, ['A', 'B']], [5.9, ['A', 'B']], [6, ['A', 'B', 'C']],
  [6.9, ['A', 'B', 'C']], [7, ['A', 'B', 'C', 'D']], [7.9, ['A', 'B', 'C', 'D']],
  [8, ['A', 'B', 'C', 'D', 'E']], [8.9, ['A', 'B', 'C', 'D', 'E']],
  [9, ['A', 'B', 'C', 'D', 'E', 'F']], [9.9, ['A', 'B', 'C', 'D', 'E', 'F']],
  [10, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z']]])(
  'labels only valid stations meeting the prominence threshold at zoom %s', (zoom, codes) => {
    expect(labelStations(labelNetwork, zoom as number).map(s => s.code)).toEqual(codes);
  });

it('counts only long-distance routes once per station and adds two for their terminals', () => {
  const calls = (codes: string[], distance: number) => codes.map((station, idx) => ({
    idx, station, halt: true, dist_km: idx * distance / (codes.length - 1), sched_off: idx * 5,
  }));
  const indexed = indexNetwork({ ...labelNetwork, routes: [
    { ...network.routes[0], id: 1, stops: calls(['A', 'B'], 299.99) },
    { ...network.routes[0], id: 2, stops: calls(['A', 'C', 'C', 'B'], 300) },
    { ...network.routes[0], id: 3, stops: calls(['C', 'A', 'D'], 600) },
    { ...network.routes[0], id: 4, stops: calls(['E', 'F', 'E'], 300) },
  ] });
  expect(Object.fromEntries(indexed.prominence)).toEqual({ A: 4, B: 3, C: 4, D: 3, E: 3, F: 1 });
  expect(indexed.routeCounts.get('A')).toBe(3);
});

it('ranks a long-distance terminal ahead of a suburban-only station with many more routes', () => {
  const suburban: Network = { stations: [
    { code: 'U', lon: 78, lat: 17 }, { code: 'L', lon: 80, lat: 20 },
  ], segments: [], routes: Array.from({ length: 501 }, (_, id) => ({ ...network.routes[0], id,
    stops: [
      { idx: 0, station: id === 500 ? 'L' : 'U', halt: true, dist_km: 0, sched_off: 0 },
      { idx: 1, station: 'T', halt: true, dist_km: id === 500 ? 300 : 50, sched_off: 10 },
    ] })) };
  expect(indexNetwork(suburban).routeCounts.get('U')).toBe(500);
  expect(labelStations(suburban, 10).map(s => s.code)).toEqual(['L', 'U']);
});

it('breaks prominence ties by route count, then station code', () => {
  const tied: Network = { ...network, segments: [], stations: [
    { code: 'B', lon: 80, lat: 20 }, { code: 'C', lon: 81, lat: 21 }, { code: 'A', lon: 82, lat: 22 },
  ], routes: [
    { ...network.routes[0], id: 1, stops: ['T', 'A', 'B', 'C', 'T'].map((station, idx) => ({
      idx, station, halt: true, dist_km: idx * 100, sched_off: idx * 5,
    })) },
    { ...network.routes[0], id: 2, stops: [
      { idx: 0, station: 'C', halt: true, dist_km: 0, sched_off: 0 },
      { idx: 1, station: 'T', halt: true, dist_km: 50, sched_off: 5 },
    ] },
  ] };
  expect(labelStations(tied, 10).map(s => s.code)).toEqual(['C', 'A', 'B']);
});

it('declutters nearby stations by prominence until zoom separates their padded labels', () => {
  const nearby: Network = { ...labelNetwork, stations: [
    { code: 'B', name: 'BETA', lon: 78.52, lat: 17.52 },
    { code: 'A', name: 'ALPHA CENTRAL', lon: 78.5, lat: 17.5 },
  ] };
  expect(labelStations(nearby, 8).map(s => s.code)).toEqual(['A']);
  expect(labelStations(nearby, 12).map(s => s.code)).toEqual(['A', 'B']);
});

it('declutters a dense network exactly like checking every accepted label, once per bucket', () => {
  let seed = 7;
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const dense: Network = { ...labelNetwork, stations: Array.from({ length: 1500 }, (_, i) => ({
    code: `S${i}`, name: 'X'.repeat(1 + Math.floor(random() * 30)), lon: 72 + random() * 4, lat: 18 + random() * 4 })) };
  dense.routes = [{ ...network.routes[0], id: 1, stops: dense.stations.map((s, idx) => ({
    idx, station: s.code, halt: true, dist_km: idx, sched_off: idx })) }];
  // From zoom 10 every station qualifies; terminals rank first, then station code.
  const terminal = (s: { code: string }) => Number(s.code === 'S0' || s.code === 'S1499');
  const order = [...dense.stations].sort((a, b) => terminal(b) - terminal(a) || a.code.localeCompare(b.code));
  for (const zoom of [10, 11.5, 13]) {
    const world = 512 * 2 ** zoom, boxes: number[][] = [];
    const expected = order.filter(s => {
      const x = (s.lon! + 180) / 360 * world, lat = s.lat! * Math.PI / 180;
      const y = (1 - Math.log(Math.tan(Math.PI / 4 + lat / 2)) / Math.PI) / 2 * world;
      const half = s.name!.length * 7 / 2 + 4, box = [x - half, y + 6, x + half, y + 28];
      if (boxes.some(b => box[0] <= b[2] && box[2] >= b[0] && box[1] <= b[3] && box[3] >= b[1])) return false;
      boxes.push(box);
      return true;
    }).map(s => s.code);
    const labels = labelStations(dense, zoom);
    expect(labels.map(s => s.code)).toEqual(expected);
    expect(labelStations(dense, zoom + 0.25)).toBe(labels); // Same bucket: no re-declutter.
  }
});

it('uses half-zoom buckets for label selection', () => {
  expect(labelBucket(7.74)).toBe(7.5);
  expect(labelBucket(8)).toBe(8);
  const bucketed: Network = { ...labelNetwork, stations: [
    { code: 'B', name: 'BETA', lon: 78.73, lat: 17.5 },
    { code: 'A', name: 'ALPHA CENTRAL', lon: 78.5, lat: 17.5 },
  ] };
  expect(labelStations(bucketed, 7.5).map(s => s.code)).toEqual(['A']);
  expect(labelStations(bucketed, 7.74).map(s => s.code)).toEqual(['A']);
  expect(labelStations(bucketed, 8).map(s => s.code)).toEqual(['A', 'B']);
});

it.each([[0.25, ['A']], [0.27, ['A', 'B']]])(
  'declutters wider station labels separated by %s degrees', (distance, codes) => {
    const nearby: Network = { ...labelNetwork, stations: [
      { code: 'B', name: 'BETA', lon: 78.5 + (distance as number), lat: 17.5 },
      { code: 'A', name: 'ALPHA CENTRAL', lon: 78.5, lat: 17.5 },
    ] };
    expect(labelStations(nearby, 7.5).map(s => s.code)).toEqual(codes);
  });

it.each([[-6, 4, 24, 44], [0, 10, 30, 50], [3, 13, 33, 53], [-12, 0, 20, 40]])(
  'fits the padded glyph image into its atlas cell with left bearing %s', (glyphLeft, shift, width, imageWidth) => {
    const glyph = { data: new Uint8ClampedArray(40 * 50), width: 40, height: 50,
      glyphLeft, glyphWidth: 20, glyphHeight: 30, glyphTop: 22, glyphAdvance: 18.25 };
    expect(glyphShift(glyphLeft, 10)).toBe(shift);
    const metrics = glyphMetrics(glyph, 10);
    expect(metrics).toEqual({ advance: 18.25, width, ascent: 22, descent: 8 });
    expect(metrics.width + 2 * 10).toBe(imageWidth);
    expect(padGlyphAlpha(glyph.data, glyph.width, glyph.height, shift)).toHaveLength(imageWidth * 50 * 4);
  });

it.each([0, 2])('pads each glyph row by %s columns with alpha only', shift => {
  const alpha = new Uint8ClampedArray([17, 64, 128, 255]);
  const empty = [0, 0, 0, 0, 0, 0, 0, 0];
  const expected = shift === 0 ? [0, 0, 0, 17, 0, 0, 0, 64, 0, 0, 0, 128, 0, 0, 0, 255]
    : [...empty, 0, 0, 0, 17, 0, 0, 0, 64, ...empty, 0, 0, 0, 128, 0, 0, 0, 255];
  expect(padGlyphAlpha(alpha, 2, 2, shift)).toEqual(new Uint8ClampedArray(expected));
});

it.each([-10, -6, 0, 3])('keeps ink at the same compensated pen position for left bearing %s', glyphLeft => {
  expect(glyphShift(glyphLeft, 10) + 10).toBe(2 * 10 + glyphLeft);
});

it('reuses the station font renderer across label layers', () => {
  const first = stationLabelLayer([], 4), second = stationLabelLayer([], 7);
  expect(first.props._getFontRenderer).toBeTypeOf('function');
  expect(second.props._getFontRenderer).toBe(first.props._getFontRenderer);
});

it.each([[4, 12], [6.99, 12], [7, 13], [8, 13]])(
  'compensates the station glyph padding in screen pixels at zoom %s', (zoom, size) => {
    const layer = stationLabelLayer([], zoom);
    expect(layer.props.getPixelOffset).toEqual([-2 * layer.props.fontSettings.buffer! * size / 64, 10]);
    expect(layer.props.getTextAnchor).toBe('middle');
    expect(layer.props.getAlignmentBaseline).toBe('top');
  });

it.each([[4, 12], [6.99, 12], [7, 13], [8, 13]])(
  'labels title case station names with a legible halo at zoom %s', (zoom, size) => {
    const layer = stationLabelLayer(labelStations(labelNetwork, zoom), zoom);
    expect(layer).toBeInstanceOf(TextLayer);
    expect((layer.props.getText as Function)(labelNetwork.stations[0])).toBe('Alpha Central');
    const halo = 0.75 * layer.props.outlineWidth;
    expect(halo * (layer.props.getSize as number) / 64).toBeGreaterThanOrEqual(1.5);
    expect(halo).toBeLessThanOrEqual(layer.props.fontSettings.buffer!);
    expect(layer.props.outlineWidth).toBeLessThan(layer.props.fontSettings.radius!);
    expect(layer.props.getSize).toBe(size);
    expect(layer.props.getColor).toEqual([214, 224, 238, 255]);
    expect(layer.props.characterSet).toBe('auto');
    expect(layer.props.fontSettings).toEqual({ sdf: true, buffer: 10, radius: 16 });
    expect(layer.props.fontWeight).toBe(600);
    expect(layer.props.outlineColor).toEqual([5, 7, 13, 255]);
  });

it.each([
  [29.9, [160, 16, 32, 210]], [30, [255, 32, 48, 210]], [59.9, [255, 32, 48, 210]],
  [60, [255, 128, 32, 210]], [89.9, [255, 128, 32, 210]],
  [90, [255, 208, 64, 210]], [109.9, [255, 208, 64, 210]], [110, [255, 244, 216, 210]],
])('uses the lower-inclusive speed band at %s km/h without cyan', (kmh, color) => {
  expect(speedColor).toBeTypeOf('function');
  const actual = speedColor(kmh as number);
  expect(actual).toEqual(color);
  expect(actual[1]).toBeLessThanOrEqual(actual[0]);
  expect(actual[2]).toBeLessThanOrEqual(actual[0]);
});

it('colours each gap by stop distance and event duration and caches the data per display', () => {
  const distances = [0, 5, 20, 65, 165, 225];
  const mixed: Network = { ...network,
    stations: distances.map((_, i) => ({ code: String(i), lon: 70 + i, lat: 20 })),
    segments: [{ ...network.segments[0], polyline: [[70, 20], [70.5, 21], [71, 20]] }],
    routes: [{ ...network.routes[0], stops: distances.map((dist_km, idx) =>
      ({ idx, station: String(idx), halt: true, dist_km, sched_off: null })),
    segments: [{ idx: 0, segment_id: 1, forward: true }] }],
  };
  const run = { ...eventRun(distances.map((_, i) => [i, i * 100])),
    halt_p50: packQuantiles([[null, 0], [15, null], [35, null], [71, null], [131, null], [161, null]]) };
  const selected = displayRun(run, mixed, 1300)!;
  const [route] = selectionLayers(selected, 0, 900);
  const data = route.props.data as any[];
  expect(data).toHaveLength(5);
  expect(route.props.getColor).toBeTypeOf('function');
  expect(data.map((gap, index) => (route.props.getColor as Function)(gap, { index }))).toEqual([
    [160, 16, 32, 210], [255, 32, 48, 210], [255, 128, 32, 210],
    [255, 208, 64, 210], [255, 244, 216, 210],
  ]);
  data.forEach((gap, i) => expect((route.props.getPath as Function)(gap)).toBe(selected.gaps[i].path));
  const same = selectionLayers(selected, PULSE_MS / 2, 1500)[0];
  const fresh = selectionLayers(displayRun(run, mixed, 1300)!, 0, 900)[0];
  expect(same.props.data).toBe(data);
  expect(fresh.props.data).not.toBe(data);
  data.forEach((gap, index) => {
    const color = (route.props.getColor as Function)(gap, { index });
    expect((same.props.getColor as Function)(gap, { index })).toBe(color);
    expect((fresh.props.getColor as Function)((fresh.props.data as any[])[index], { index })).not.toBe(color);
  });
});

it.each([10, 5])('keeps the selected gap red without positive duration (arrival %s min)', arrival => {
  const selected = displayRun({ ...runFixture(),
    halt_p50: packQuantiles([[null, 10], [arrival, null]]) }, chained, 1300)!;
  const [route] = selectionLayers(selected, 0, 900);
  expect(route.props.getColor).toBeTypeOf('function');
  expect((route.props.getColor as Function)((route.props.data as any[])[0], { index: 0 })).toEqual([255, 32, 48, 210]);
});

it('highlights the selected gaps without copying their paths and keeps the pulse last', () => {
  const selected = displayRun(runFixture(), chained, 1300)!;
  const layers = selectionLayers(selected, 0, 900); // Before departure: all ahead.
  const [route, pulse] = layers;
  expect(layers.map(layer => layer.id)).toEqual(['selection', 'selection-pulse']);
  expect(route.id).toBe('selection');
  expect(route.props.data as any[]).toHaveLength(1);
  expect((route.props.getPath as Function)((route.props.data as any[])[0])).toBe(selected.gaps[0].path);
  expect(route.props.getWidth).toBe(3);
  expect(route.props.widthUnits).toBe('pixels');
  expect((route.props.getColor as Function)((route.props.data as any[])[0], { index: 0 })).toEqual([255, 244, 216, 210]);
  expect(pulse).toBeInstanceOf(TripsLayer);
  expect(pulse.id).toBe('selection-pulse');
  expect(pulse.props.getWidth).toBe(16 * GLOW_SCALE);
  const [trip] = pulse.props.data as any[];
  expect(trip.path).toEqual(selected.points);
  // Halts repeat a point; zero-length segments would splay the wide glow into rays.
  const halted = { ...selected, points: [selected.points[0], ...selected.points] };
  expect((selectionLayers(halted, 0, 900)[1].props.data as any[])[0].path).toEqual(selected.points);
  expect(trip.timestamps[0]).toBe(0);
  expect(trip.timestamps.every((t: number, i: number) => !i || t > trip.timestamps[i - 1])).toBe(true);
  expect((pulse.props as any).jointRounded).toBe(true);
  const total = trip.timestamps.at(-1), tail = (pulse.props as any).trailLength;
  expect(total).toBeGreaterThan(0);
  expect(tail).toBeGreaterThan(0);
  expect(tail).toBeLessThan(total);
  expect((pulse.props as any).currentTime).toBe(0); // Origin.
  const half = selectionLayers(selected, PULSE_MS / 2, 900);
  // The head reaches the destination and the tail drains into it before the next pulse.
  expect((half[1].props as any).currentTime).toBeCloseTo((total + tail) / 2);
  expect((selectionLayers(selected, PULSE_MS * 3, 900)[1].props as any).currentTime).toBe(0);
  expect(half[0].props.data).toBe(route.props.data);
  expect(half[1].props.data).toBe(pulse.props.data);
  // The pulse dims behind the train and brightens once it passes it.
  const color = pulse.props.getColor as Function;
  const [behind, ahead] = selectionLayers(selected, 0, 1300)[1].props.data as any[];
  const head = headAt(selected, 1300);
  expect(behind.path.at(-1)[0]).toBeCloseTo(head[0]);
  expect(behind.path.at(-1)[1]).toBeCloseTo(head[1]);
  expect(ahead.path[0]).toEqual(behind.path.at(-1));
  expect(ahead.timestamps[0]).toBe(behind.timestamps.at(-1));
  expect([...behind.path, ...ahead.path.slice(1)].length).toBe(trip.path.length + 1);
  for (const t of [behind, ahead]) expect(t.timestamps.every((x: number, i: number) => !i || x > t.timestamps[i - 1])).toBe(true);
  expect(ahead.timestamps.at(-1)).toBe(total);
  expect(color(behind)[3]).toBeLessThan(color(ahead)[3]);
  expect(color(trip)).toEqual(color(ahead));
  const [arrived, none] = selectionLayers(selected, 0, 1700)[1].props.data as any[];
  expect(arrived.path).toEqual(selected.points);
  expect(color(arrived)).toEqual(color(behind));
  expect(none).toBeUndefined();
});

it.each([4, 6, 8])('selection changes only opacity on the base train layers at zoom %s', zoom => {
  const displays = ['observed', 'stale', 'schedule_only'].map((state, i) =>
    displayRun({ ...runFixture(8), run_id: i + 9, state }, network, 1300)!);
  const trails = trailData(displays, 1300), heads = new Map<number, [number, number]>();
  const first = trainLayers(displays, trails, 1300, zoom, null, heads, 7);
  for (const selected of [9, 10, 11, null]) {
    const layers = trainLayers(displays, trails, 1300, zoom, selected, heads, 7);
    for (const before of first) {
      const layer = layers.find(l => l.id === before.id)!;
      const { opacity, ...props } = layer.props;
      const { opacity: original, ...originalProps } = before.props;
      expect(opacity).toBeCloseTo(original! * (selected === null ? 1 : 0.3));
      for (const key of Object.keys(originalProps)) {
        const value = (props as any)[key], previous = (originalProps as any)[key];
        if (typeof previous !== 'function') expect(value).toEqual(previous);
      }
      expect(layer.props.data).toBe(before.props.data);
      expect(layer.props.updateTriggers!.getColor).toBeUndefined();
      expect(layer.props.updateTriggers!.getFillColor).toBeUndefined();
      for (const item of layer.props.data as any[]) {
        const color = layer instanceof ScatterplotLayer ? layer.props.getFillColor : layer.props.getColor;
        const alpha = layer.id === 'heads' ? 230 : layer.id === 'comets' ? 255 : layer.id === 'streaks' ? 24 : 190;
        expect((color as Function)(item)).toEqual([...item.color, alpha]);
      }
    }
  }
});

it.each(['observed', 'stale', 'schedule_only'])('overlays only the selected %s cached trail and drawn head', state => {
  const display = displayRun({ ...runFixture(8), state }, chained, 4900)!;
  const other = displayRun({ ...runFixture(), run_id: 10, state }, chained, 4900)!;
  const displays = [other, display], trails = trailData(displays, 4900);
  const heads = new Map<number, [number, number]>([[9, [70.5, 21.5]], [10, [71, 22]]]);
  for (const zoom of [4, 6, 8]) {
    const base = trainLayers(displays, trails, 4900, zoom, null, heads, 7);
    const layers = trainLayers(displays, trails, 4900, zoom, 9, heads, 7);
    expect(layers.slice(0, base.length).map(l => l.id)).toEqual(base.map(l => l.id));
    expect(layers).toHaveLength(base.length + 2);
    const [trail, head] = layers.slice(base.length) as [TripsLayer, ScatterplotLayer];
    const source = base.find(l => l.id === (state === 'schedule_only' ? 'timetable-9' : 'comets'))!;
    const core = (state === 'schedule_only' ? trails.timetable : trails.comets).find(t => t.runId === 9)!;
    expect(trail).toBeInstanceOf(TripsLayer);
    expect(trail.constructor).toBe(source.constructor);
    expect(trail.props.data).toHaveLength(1);
    expect((trail.props.data as any[])[0]).toBe(core);
    for (const key of ['currentTime', 'trailLength', 'fadeTrail', 'getWidth', 'widthUnits', 'widthScale', 'opacity', 'parameters'])
      expect((trail.props as any)[key]).toEqual((source.props as any)[key]);
    expect((trail.props.getColor as Function)(core)).toEqual([...display.color, state === 'schedule_only' ? 190 : 255]);
    expect((trail.props.getPath as Function)(core)).toBe(core.path);
    expect((trail.props.getTimestamps as Function)(core)).toBe(core.timestamps);
    expect((layers.find(l => l.id === source.id)!.props.data as any[]).find(t => t.runId === 9)).toBe(core);
    const originalHead = base.find(l => l.id === 'heads')!;
    expect(head).toBeInstanceOf(ScatterplotLayer);
    expect(head.constructor).toBe(originalHead.constructor);
    expect(head.props.data).toEqual([display]);
    expect((head.props.data as any[])[0]).toBe(display);
    expect((head.props.getPosition as Function)(display)).toBe(heads.get(9));
    expect(head.props.updateTriggers!.getPosition).toBe(7);
    expect(head.props.opacity).toBe(originalHead.props.opacity);
    expect(head.props.parameters).toEqual(originalHead.props.parameters);
    const next = trainLayers(displays, trails, 4901, zoom, 9, heads, 8).slice(base.length);
    expect((next[0].props.data as any[])[0]).toBe(core);
    expect(next[0].props.dataComparator!(next[0].props.data, trail.props.data)).toBe(true);
    expect(next[1].props.dataComparator!(next[1].props.data, head.props.data)).toBe(true);
    expect(next[1].props.updateTriggers!.getPosition).toBe(8);
    expect(layers.find(l => l.id === 'heads')!.props.data).toBe(displays);
    for (const selected of [null, 999])
      expect(trainLayers(displays, trails, 4900, zoom, selected, heads, 7).map(l => l.id)).toEqual(base.map(l => l.id));
  }
});

it.each(['observed', 'stale'])('keeps %s glow and eight streaks separate from the delay-coloured timetable core', state => {
  const live = displayRun({ ...runFixture(32), state }, network, 1300)!;
  const timetable = displayRun({ ...runFixture(8), run_id: 10, state: 'schedule_only' }, network, 1300)!;
  const onlyTimetable = trailData([timetable], 1300);
  expect(onlyTimetable.comets).toEqual([]);
  expect(onlyTimetable.streaks).toEqual([]);
  expect(onlyTimetable.timetable).toHaveLength(1);
  expect(onlyTimetable.timetable[0].color).toEqual(delayColor(timetable.delay));
  const trails = trailData([live, timetable], 1300);
  expect(trails.comets.map(t => t.runId)).toEqual([9]);
  expect(trails.streaks).toHaveLength(8);
  expect(trails.streaks.every(t => t.runId === 9)).toBe(true);
  expect(trails.comets[0].color).toEqual(live.color);
  const layers = trainLayers([live, timetable], trails, 1300, 4);
  const core = layers.filter(l => l instanceof TripsLayer).find(l => l.id === 'timetable-10')!;
  expect(core.props.data).toEqual(trails.timetable);
  expect((core.props.data as any[])[0]).toBe(trails.timetable[0]);
  expect(core.props.getWidth).toBe(6 * GLOW_SCALE);
  expect(core.props.widthScale).toBe(0.5);
  expect(core.props.opacity).toBe(0.35);
  expect(core.props.trailLength).toBe(21600);
  expect((core.props.getColor as Function)(trails.timetable[0])).toEqual([...timetable.color, 190]);
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getRadius as Function)(timetable)).toBe(4.5 * GLOW_SCALE);
  expect(heads.props.radiusScale).toBe(0.6);
  expect(heads.props.opacity).toBe(0.45);
  expect((heads.props.getFillColor as Function)(timetable)).toEqual([...timetable.color, 230]);
});

it.each(['schedule_only', 'observed', 'stale'])('keeps %s heads in their delay colour before shader glow', state => {
  const run = { ...runFixture(), state, halt_p50: packQuantiles([[null, 0], [25, null]]) };
  const display = displayRun(run, network, 1300)!;
  const layers = trainLayers([display], trailData([display], 1300), 1300, 4);
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getFillColor as Function)(display)).toEqual([252, 238, 10, 230]);
});

it('an unchanged display yields the identical trail object across a delta', () => {
  const live = displayRun(runFixture(), network, 1300)!;
  const timetable = displayRun({ ...runFixture(), run_id: 10, state: 'schedule_only' }, network, 1300)!;
  const first = trailData([live, timetable], 1300);
  const next = trailData([{ ...live }, timetable], 1400);
  expect(next.timetable[0]).toBe(first.timetable[0]);
});

it('draws sixteen ordered timetable chunks with the same glow props and cached data on repeated pushes', () => {
  const displays = [33, 0, 17, 32, 1, 16, 15, 31, 10, 26].map(run_id =>
    displayRun({ ...runFixture(), run_id, state: 'schedule_only' }, network, 4900)!);
  const trails = trailData(displays, 4900);
  const layers = trainLayers(displays, trails, 4900, 6, 17);
  expect(layers.map(l => l.id)).toEqual(['comets', 'streaks', ...timetableIds, 'heads', 'selected-trail', 'selected-head']);
  const chunks = layers.filter(l => l instanceof TripsLayer).filter(l => l.id.startsWith('timetable-'));
  expect(chunks).toHaveLength(16);
  const grouped = new Map([[0, [0, 32, 16]], [1, [33, 17, 1]], [10, [10, 26]], [15, [15, 31]]]);
  const next = trainLayers(displays, trails, 4900.1, 7, 33);
  chunks.forEach((layer, c) => {
    expect(layer.id).toBe(`timetable-${c}`);
    expect(layer.constructor).toBe(layers[0].constructor);
    const data = layer.props.data as any[];
    expect(data.map(t => t.runId)).toEqual(grouped.get(c) ?? []);
    for (const trail of data) {
      expect(trail).toBe(trails.timetable.find(t => t.runId === trail.runId));
      expect((layer.props.getPath as Function)(trail)).toBe(trail.path);
      expect((layer.props.getTimestamps as Function)(trail)).toBe(trail.timestamps);
      expect((layer.props.getColor as Function)(trail)).toEqual([...trail.color, 190]);
    }
    expect(layer.props.dataComparator).toBe(dataComparator);
    expect(layer.props.currentTime).toBe(1300);
    expect(layer.props.trailLength).toBe(5400);
    expect(layer.props.fadeTrail).toBe(true);
    expect(layer.props.getWidth).toBe(6 * GLOW_SCALE);
    expect(layer.props.widthScale).toBe(0.75);
    expect(layer.props.opacity).toBeCloseTo(0.675 * 0.3);
    expect(layer.props.widthUnits).toBe('pixels');
    expect(layer.props.updateTriggers!.getColor).toBeUndefined();
    expect(layer.props.parameters).toEqual(layers[0].props.parameters);
    expect(next.find(l => l.id === layer.id)!.props.data).toBe(data);
  });
});

it.each(['changing', 'removing'])('%s one run changes only its timetable chunk', action => {
  const displays = Array.from({ length: 32 }, (_, run_id) =>
    displayRun({ ...runFixture(), run_id, state: 'schedule_only' }, network, 1300)!);
  const first = trainLayers(displays, trailData(displays, 1300), 1300, 4)
    .filter(l => l instanceof TripsLayer && l.id.startsWith('timetable-'));
  const updated = displays.flatMap(d => d.run.run_id !== 10 ? [d] : action === 'removing' ? []
    : [displayRun({ ...d.run, halt_p50: packQuantiles([[null, 0], [20, null]]) }, network, 1400)!]);
  const next = trainLayers(updated, trailData(updated, 1400), 1400, 4)
    .filter(l => l instanceof TripsLayer && l.id.startsWith('timetable-'));
  expect(first).toHaveLength(16);
  expect(next).toHaveLength(16);
  next.forEach((layer, c) => {
    const previous = first[c].props.data as any[], data = layer.props.data as any[];
    expect(layer.id).toBe(first[c].id);
    expect(layer.props.dataComparator!(data, previous)).toBe(c !== 10);
    if (c === 10) {
      expect(data).not.toBe(previous);
      expect(data.map(t => t.runId)).toEqual(action === 'removing' ? [26] : [10, 26]);
      if (action === 'changing') expect(data[0].timestamps).toEqual([1000, 2200]);
    } else expect(data).toBe(previous);
  });
});

it('reuses each unchanged display\'s trail objects within an hour and rebases them in a new hour', () => {
  const live = displayRun(runFixture(32), network, 1300)!;
  const timetable = displayRun({ ...runFixture(8), state: 'schedule_only' }, network, 1300)!;
  const first = trailData([live, timetable], 1300);
  const same = trailData([live, timetable], 3599);
  const changed = trailData([{ ...live }, timetable], 1300);
  expect(changed.comets[0]).not.toBe(first.comets[0]);
  expect(changed.streaks[0]).not.toBe(first.streaks[0]);
  expect(changed.timetable[0]).toBe(first.timetable[0]);
  const nextHour = trailData([live, timetable], 3600);
  for (const key of ['comets', 'streaks', 'timetable'] as const) {
    first[key].forEach((trail, i) => {
      expect(same[key][i]).toBe(trail);
      expect(nextHour[key][i]).not.toBe(trail);
      expect(nextHour[key][i].timestamps).toEqual(trail.timestamps.map(t => t - 3600));
    });
  }
});

const longTimes = [0, 1800, 3600, 3600, 7200, 10800, 32400, 39600, 39600, 43200, 72000];

function windowDisplay(timestamps: number[], state = 'observed') {
  const path: [number, number][] = timestamps.map((_, i) => [70 + i / 100, 20]);
  return { ...displayRun({ ...runFixture(1), state }, network, 1300)!, points: path, times: timestamps,
    trips: state === 'schedule_only' ? [] : [{ path, timestamps }] };
}

it.each(['schedule_only', 'observed', 'stale'])('windows long %s trails on aligned bracketing vertices', state => {
  const display = windowDisplay(longTimes, state);
  // Core bracketing must use the running max, including repeated times at both bounds.
  display.times = [0, 1800, 3600, 3000, 7200, 10800, 32400, 39600, 39000, 43200, 72000];
  const trails = trailData([display], 36123);
  expect(trails.base).toBe(36000);
  const kept = [...trails.comets, ...trails.streaks, ...trails.timetable];
  expect(kept).toHaveLength(state === 'schedule_only' ? 1 : 2);
  for (const trail of kept) {
    expect(trail.path).toEqual(display.points.slice(3, 8));
    expect(trail.timestamps).toEqual([-32400, -28800, -25200, -3600, 3600]);
    expect(trail.timestamps[0] + trails.base).toBeLessThanOrEqual(3600);
    expect(trail.timestamps.at(-1)! + trails.base).toBeGreaterThanOrEqual(39600);
    expect(trail.path).toHaveLength(trail.timestamps.length);
  }
  expect(display.points).toHaveLength(11);
  expect(display.trips[0]?.timestamps ?? longTimes).toEqual(longTimes);
});

it('starts each window at the last vertex at or before the longest trail', () => {
  const trails = trailData([windowDisplay([0, 3000, 3600, 4000, 36000, 40000])], 36123);
  for (const trail of [...trails.comets, ...trails.streaks])
    expect(trail.timestamps).toEqual([-32400, -32000, 0, 4000]);
});

it.each([
  ['before', [0, 600, 1200, 1800], [2, 3], [-34800, -34200]],
  ['after', [43200, 46800, 50400, 54000], [0, 1], [7200, 10800]],
] as const)('keeps the nearest two vertices for a trail wholly %s the span', (_, times, indices, expected) => {
  const display = windowDisplay([...times]);
  const trails = trailData([display], 36123);
  expect(trails.comets).toHaveLength(1);
  expect(trails.streaks).toHaveLength(1);
  for (const trail of [...trails.comets, ...trails.streaks]) {
    expect(trail.path).toEqual(indices.map(i => display.points[i]));
    expect(trail.timestamps).toEqual(expected);
    expect(trail.path).toHaveLength(2);
  }
});

it.each(['schedule_only', 'observed', 'stale'])('reuses %s windows within an hour and moves them at the next base', state => {
  const display = windowDisplay(longTimes, state);
  const first = trailData([display], 36123);
  const same = trailData([display], 39599);
  const next = trailData([display], 39600);
  expect(next.base).toBe(39600);
  for (const key of ['comets', 'streaks', 'timetable'] as const) {
    expect(next[key]).toHaveLength(first[key].length);
    first[key].forEach((trail, i) => {
      expect(same[key][i]).toBe(trail);
      expect(next[key][i]).not.toBe(trail);
      expect(trail.path).toEqual(display.points.slice(3, 8));
      expect(next[key][i].path).toEqual(display.points.slice(4, 10));
      expect(next[key][i].timestamps).toEqual([-32400, -28800, -7200, 0, 0, 3600]);
    });
  }
});

it('compares layer data by length and element identity, including heads', () => {
  const a = {}, b = {};
  expect(dataComparator([], [])).toBe(true);
  expect(dataComparator([a, b], [a, b])).toBe(true);
  expect(dataComparator([a], [a, b])).toBe(false);
  expect(dataComparator([a, b], [b, a])).toBe(false);
  expect(dataComparator([a], [{}])).toBe(false);
  expect(dataComparator([a], undefined)).toBe(false);
  const display = displayRun(runFixture(), network, 1300)!;
  for (const layer of trainLayers([display], trailData([display], 1300), 1300, 4))
    expect(layer.props.dataComparator).toBe(dataComparator);
});
