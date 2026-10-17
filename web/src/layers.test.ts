import { expect, it } from 'vitest';
import { TripsLayer } from '@deck.gl/geo-layers';
import { ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import { indexNetwork, networkLayer, displayRun, trailData, trainLayers,
  delayColor, trailLength, easeHead, headAt, labelBucket, labelStations, stationLabelLayer, selectionLayers, PULSE_MS,
  trainAlpha, dataComparator, glyphShift, glyphMetrics, padGlyphAlpha, type Network } from './layers';

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

it('draws one fading glow comet with additive blending and pickable heads', () => {
  const display = displayRun(runFixture(), network, 1300)!;
  const trails = trailData([display], 1300);
  let selected: unknown;
  const layers = trainLayers([display], trails, 1300, 4, run => { selected = run; });
  const trips = layers.filter(layer => layer instanceof TripsLayer);
  const comets = trips.filter(layer => layer.id === 'comets');
  expect(trips.map(layer => layer.id)).toEqual(['comets', 'streaks', 'timetable']);
  expect(comets).toHaveLength(1);
  expect(comets[0].props.getWidth).toBe(7);
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
  expect((heads.props.getRadius as Function)(display)).toBe(5.5);
  expect(heads.props.radiusUnits).toBe('pixels');
  expect(heads.props.pickable).toBe(true);
  expect((heads.props.getFillColor as Function)(display)).toEqual([...display.color, 230]);
  expect(heads.props.parameters).toEqual(comets[0].props.parameters);
  (heads.props.onClick as Function)({ object: display });
  expect(selected).toBe(display.run);
});

it('appends glow after the TripsLayer fade and guards both glow snippets during picking', () => {
  const display = displayRun(runFixture(), network, 1300)!;
  const layers = trainLayers([display], trailData([display], 1300), 1300, 4, () => {});
  const filter = 'fs:DECKGL_FILTER_COLOR';
  const glow = `if (picking.isActive < 0.5) {
  float age = clamp((trips.currentTime - vTime) / trips.trailLength, 0.0, 1.0);
  float x = geometry.uv.x / mix(1.0, 0.35, age);
  color.rgb = mix(color.rgb, vec3(1.0), 0.15 * exp(-14.0 * x * x));
  color.a *= exp(-3.5 * x * x) * (1.0 - age);
}`;
  const blob = `if (picking.isActive < 0.5) {
  float r2 = dot(geometry.uv, geometry.uv);
  color.rgb = mix(color.rgb, vec3(1.0), 0.2 * exp(-12.0 * r2));
  color.a *= exp(-3.5 * r2);
}`;
  for (const layer of layers.filter(l => l.id !== 'streaks')) {
    Object.assign(layer, { context: { device: { type: 'webgl' }, defaultShaderModules: [] } });
    const shaders = layer.getShaders();
    if (layer instanceof TripsLayer) {
      const original = TripsLayer.prototype.getShaders.call(layer);
      expect(shaders.inject[filter]).toBe(original.inject[filter] + glow);
      for (const key of Object.keys(original.inject).filter(key => key !== filter))
        expect(shaders.inject[key]).toBe(original.inject[key]);
    } else {
      expect(shaders.inject[filter]).toBe(blob);
    }
    expect(shaders.inject[filter]).toContain('if (picking.isActive < 0.5) {');
  }
});

it.each([[3, 0.5, 0.35, 0.6, 0.45], [4, 0.5, 0.35, 0.6, 0.45],
  [8, 1, 1, 1, 1], [10, 1, 1, 1, 1]])(
  'scales timetable glow and heads with clamped zoom %s', (zoom, widthScale, trailOpacity, radiusScale, headOpacity) => {
    const live = displayRun({ ...runFixture(), state: 'observed' }, network, 1300)!;
    const timetable = displayRun({ ...runFixture(), run_id: 10, state: 'schedule_only' }, network, 1300)!;
    const layers = trainLayers([live, timetable], trailData([live, timetable], 1300), 1300, zoom, () => {});
    const core = layers.filter(l => l instanceof TripsLayer).find(l => l.id === 'timetable')!;
    expect(core.props.getWidth).toBe(6);
    expect(core.props.widthScale).toBe(widthScale);
    expect(core.props.opacity).toBe(trailOpacity);
    const heads = layers.find(l => l instanceof ScatterplotLayer)!;
    expect((heads.props.getRadius as Function)(timetable)).toBe(4.5);
    expect((heads.props.getRadius as Function)(live)).toBe(5.5);
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

it('dims other trains to 30 percent and highlights the selected expanded path without copying it', () => {
  const selected = displayRun(runFixture(), chained, 1300)!;
  const other = { ...selected, run: { ...selected.run, run_id: 10 } };
  const trails = trailData([selected, other], 1300);
  expect(trainAlpha(10, null)).toBe(1);
  expect(trainAlpha(9, 9)).toBe(1);
  expect(trainAlpha(10, 9)).toBe(0.3);
  const [route, pulse] = selectionLayers(selected, 0);
  expect(route.id).toBe('selection');
  expect(route.props.data).toEqual([selected.points]);
  expect((route.props.data as any[])[0]).toBe(selected.points);
  expect(route.props.getWidth).toBe(3);
  expect(route.props.getColor).toEqual([255, 32, 48, 210]); // Red stands apart from the cyan network.
  expect(pulse).toBeInstanceOf(TripsLayer);
  expect(pulse.id).toBe('selection-pulse');
  const [trip] = pulse.props.data as any[];
  expect(trip.path).toEqual(selected.points);
  // Halts repeat a point; zero-length segments would splay the wide glow into rays.
  const halted = { ...selected, points: [selected.points[0], ...selected.points] };
  expect((selectionLayers(halted, 0)[1].props.data as any[])[0].path).toEqual(selected.points);
  expect(trip.timestamps[0]).toBe(0);
  expect(trip.timestamps.every((t: number, i: number) => !i || t > trip.timestamps[i - 1])).toBe(true);
  expect((pulse.props as any).jointRounded).toBe(true);
  const total = trip.timestamps.at(-1), tail = (pulse.props as any).trailLength;
  expect(total).toBeGreaterThan(0);
  expect(tail).toBeGreaterThan(0);
  expect(tail).toBeLessThan(total);
  expect((pulse.props as any).currentTime).toBe(0); // Origin.
  const half = selectionLayers(selected, PULSE_MS / 2);
  // The head reaches the destination and the tail drains into it before the next pulse.
  expect((half[1].props as any).currentTime).toBeCloseTo((total + tail) / 2);
  expect((selectionLayers(selected, PULSE_MS * 3)[1].props as any).currentTime).toBe(0);
  expect(half[0].props.data).toBe(route.props.data);
  expect(half[1].props.data).toBe(pulse.props.data);
  const layers = trainLayers([selected, other], trails, 1300, 4, () => {}, 9);
  for (const layer of layers.filter(l => l instanceof TripsLayer)) {
    const data = layer.props.data as any[];
    if (!data.length) continue;
    const color = layer.props.getColor as Function;
    const otherIndex = layer.id === 'streaks' ? 2 : 1;
    expect(color(data[otherIndex])[3]).toBeCloseTo(color(data[0])[3] * 0.3);
  }
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getFillColor as Function)(other)[3]).toBeCloseTo(69);
  expect((heads.props.getFillColor as Function)(selected)[3]).toBe(230);
});

it.each([
  ['schedule_only', null, 1], ['schedule_only', 9, 1], ['schedule_only', 10, 0.3],
  ['observed', null, 1], ['observed', 9, 1], ['observed', 10, 0.3],
  ['stale', null, 1], ['stale', 9, 1], ['stale', 10, 0.3],
])('draws %s heads, comets and streaks with selection %s at alpha %s', (state, selected, alpha) => {
  expect(trainAlpha(9, selected as number | null)).toBeCloseTo(alpha as number);
  const display = displayRun({ ...runFixture(8), state }, network, 1300)!;
  const layers = trainLayers([display], trailData([display], 1300), 1300, 4, () => {}, selected as number | null);
  const trips = layers.filter(l => l instanceof TripsLayer);
  trips.forEach(layer => {
    const data = layer.props.data as any[];
    const color = layer.props.getColor as Function;
    const alphas: Record<string, number> = { comets: 255, streaks: 24, timetable: 190 };
    const baseAlpha = alphas[layer.id];
    for (const trip of data) expect(color(trip)[3]).toBeCloseTo(baseAlpha! * (alpha as number));
    expect(layer.props.updateTriggers!.getColor).toBe(selected);
  });
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getFillColor as Function)(display)).toEqual([...display.color, 230 * (alpha as number)]);
  expect(heads.props.updateTriggers!.getFillColor).toBe(selected);
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
  const layers = trainLayers([live, timetable], trails, 1300, 4, () => {});
  const core = layers.filter(l => l instanceof TripsLayer).find(l => l.id === 'timetable')!;
  expect(core.props.data).toBe(trails.timetable);
  expect(core.props.getWidth).toBe(6);
  expect(core.props.widthScale).toBe(0.5);
  expect(core.props.opacity).toBe(0.35);
  expect(core.props.trailLength).toBe(21600);
  expect((core.props.getColor as Function)(trails.timetable[0])).toEqual([...timetable.color, 190]);
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getRadius as Function)(timetable)).toBe(4.5);
  expect(heads.props.radiusScale).toBe(0.6);
  expect(heads.props.opacity).toBe(0.45);
  expect((heads.props.getFillColor as Function)(timetable)).toEqual([...timetable.color, 230]);
});

it.each(['schedule_only', 'observed', 'stale'])('keeps %s heads in their delay colour before shader glow', state => {
  const run = { ...runFixture(), state, halt_p50: packQuantiles([[null, 0], [25, null]]) };
  const display = displayRun(run, network, 1300)!;
  const layers = trainLayers([display], trailData([display], 1300), 1300, 4, () => {});
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getFillColor as Function)(display)).toEqual([252, 238, 10, 230]);
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
  for (const layer of trainLayers([display], trailData([display], 1300), 1300, 4, () => {}))
    expect(layer.props.dataComparator).toBe(dataComparator);
});
