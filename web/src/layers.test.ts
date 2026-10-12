import { expect, it } from 'vitest';
import { TripsLayer } from '@deck.gl/geo-layers';
import { ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import { indexNetwork, networkLayer, displayRun, trailData, trainLayers,
  delayColor, trailLength, easeHead, headAt, labelBucket, labelStations, stationLabelLayer, selectionLayer,
  trainAlpha, dataComparator, type Network } from './layers';

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
    halt_q: [[Array(7).fill(null), Array(7).fill(0)],
      [Array(7).fill(20), Array(7).fill(null)]] };
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

it('draws brighter fading comet tiers with additive blending and pickable heads', () => {
  const display = displayRun(runFixture(), network, 1300)!;
  const trails = trailData([display], 1300);
  let selected: unknown;
  const layers = trainLayers([display], trails, 1300, 4, run => { selected = run; });
  const trips = layers.filter(layer => layer instanceof TripsLayer);
  const comets = trips.filter(layer => layer.id.startsWith('comet-'));
  expect(comets).toHaveLength(3);
  expect(comets.map(layer => layer.props.getWidth)).toEqual([8, 4, 2]);
  expect(comets.map(layer => layer.props.trailLength)).toEqual([5400, 10800, 21600]);
  const alphas = comets.map(layer => (layer.props.getColor as Function)(trails.comets[0])[3]);
  expect(alphas).toEqual([64, 150, 255]);
  expect(alphas[0]).toBeGreaterThan(0);
  expect(alphas[1]).toBeGreaterThan(alphas[0]);
  expect(alphas[2]).toBeGreaterThan(alphas[1]);
  for (const layer of trips) {
    expect(layer.props.fadeTrail).toBe(true);
    expect(layer.props.widthUnits).toBe('pixels');
    expect(layer.props.parameters).toMatchObject({ blend: true,
      blendColorOperation: 'add', blendColorSrcFactor: 'src-alpha', blendColorDstFactor: 'one',
      blendAlphaOperation: 'add', blendAlphaSrcFactor: 'one', blendAlphaDstFactor: 'one' });
  }
  const streaks = trips.find(layer => layer.id === 'streaks')!;
  expect(streaks.props.getWidth).toBe(1.5);
  expect(streaks.props.trailLength).toBe(32400);
  expect((streaks.props.getColor as Function)(trails.streaks[0])[3]).toBe(24);
  const heads = layers.find(layer => layer instanceof ScatterplotLayer)!;
  expect((heads.props.getRadius as Function)(display)).toBe(3);
  expect(heads.props.radiusUnits).toBe('pixels');
  expect(heads.props.pickable).toBe(true);
  const whiteMix = (heads.props.getFillColor as Function)(display);
  display.color.forEach((channel, i) => expect(whiteMix[i]).toBeCloseTo(channel * 0.4 + 153));
  (heads.props.onClick as Function)({ object: display });
  expect(selected).toBe(display.run);
});

it('makes median comet timestamps monotone without accepting non-monotone particle paths', () => {
  const run = { ...runFixture(), halt_q: [[Array(7).fill(null), Array(7).fill(20)],
    [Array(7).fill(10), Array(7).fill(null)]] };
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
  const run = { ...runFixture(), halt_q: [[Array(7).fill(null), Array(7).fill(0)],
    [Array(7).fill(25), Array(7).fill(null)]] };
  expect(displayRun(run, network, 1300)?.color).toEqual([252, 238, 10]);
  expect(displayRun(run, network, 2600)?.color).toEqual([0, 240, 255]);
  expect(displayRun({ ...run, origin_ts: undefined }, network, 1300)?.color).toEqual([0, 240, 255]);
  const unknownSchedule = { ...network, routes: [{ ...network.routes[0],
    stops: network.routes[0].stops.map(s => ({ ...s, sched_off: null })) }] };
  expect(displayRun(run, unknownSchedule, 1300)?.color).toEqual([0, 240, 255]);
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

it('labels title case station names in a small deck font atlas', () => {
  const layer = stationLabelLayer(labelStations(labelNetwork, 4), 4);
  expect(layer).toBeInstanceOf(TextLayer);
  expect((layer.props.getText as Function)(labelNetwork.stations[0])).toBe('Alpha Central');
  expect(layer.props.getSize).toBe(11);
  expect(layer.props.getColor).toEqual([150, 168, 190, 210]);
  expect(layer.props.characterSet).toBe('auto');
  expect(layer.props.fontSettings.sdf).toBe(true);
  expect(layer.props.outlineWidth).toBeGreaterThan(0);
  expect(layer.props.outlineColor).toEqual([5, 7, 13, 230]);
  expect(stationLabelLayer(labelStations(labelNetwork, 7), 7).props.getSize).toBe(12);
});

it('dims other trains to 30 percent and highlights the selected expanded path without copying it', () => {
  const selected = displayRun(runFixture(), chained, 1300)!;
  const other = { ...selected, run: { ...selected.run, run_id: 10 } };
  const trails = trailData([selected, other], 1300);
  expect(trainAlpha(10, null)).toBe(1);
  expect(trainAlpha(9, 9)).toBe(1);
  expect(trainAlpha(10, 9)).toBe(0.3);
  const highlight = selectionLayer(selected);
  expect(highlight.props.data).toEqual([selected.points]);
  expect((highlight.props.data as any[])[0]).toBe(selected.points);
  expect(highlight.props.getWidth).toBe(3);
  expect(highlight.props.getColor).toEqual([0, 240, 255, 190]);
  const layers = trainLayers([selected, other], trails, 1300, 4, () => {}, 9);
  for (const layer of layers.filter(l => l instanceof TripsLayer)) {
    const data = layer.props.data as any[];
    if (!data.length) continue;
    const color = layer.props.getColor as Function;
    const otherIndex = layer.id === 'streaks' ? 2 : 1;
    expect(color(data[otherIndex])[3]).toBeCloseTo(color(data[0])[3] * 0.3);
  }
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getFillColor as Function)(other)[3]).toBeCloseTo(76.5);
  expect((heads.props.getFillColor as Function)(selected)[3]).toBe(255);
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
    const alphas: Record<string, number> = { 'comet-8': 64, 'comet-4': 150, 'comet-2': 255, streaks: 24, timetable: 230 };
    const baseAlpha = alphas[layer.id];
    for (const trip of data) expect(color(trip)[3]).toBeCloseTo(baseAlpha! * (alpha as number));
    expect(layer.props.updateTriggers!.getColor).toBe(selected);
  });
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getFillColor as Function)(display)[3]).toBeCloseTo(255 * (alpha as number));
  expect(heads.props.updateTriggers!.getFillColor).toBe(selected);
});

it.each(['observed', 'stale'])('keeps %s glow and eight streaks separate from the neutral timetable core', state => {
  const live = displayRun({ ...runFixture(32), state }, network, 1300)!;
  const timetable = displayRun({ ...runFixture(8), run_id: 10, state: 'schedule_only' }, network, 1300)!;
  const onlyTimetable = trailData([timetable], 1300);
  expect(onlyTimetable.comets).toEqual([]);
  expect(onlyTimetable.streaks).toEqual([]);
  expect(onlyTimetable.timetable).toHaveLength(1);
  expect(onlyTimetable.timetable[0].color).toEqual([200, 214, 235]);
  const trails = trailData([live, timetable], 1300);
  expect(trails.comets.map(t => t.runId)).toEqual([9]);
  expect(trails.streaks).toHaveLength(8);
  expect(trails.streaks.every(t => t.runId === 9)).toBe(true);
  expect(trails.comets[0].color).toEqual(live.color);
  const layers = trainLayers([live, timetable], trails, 1300, 4, () => {});
  const core = layers.filter(l => l instanceof TripsLayer).find(l => l.id === 'timetable')!;
  expect(core.props.data).toBe(trails.timetable);
  expect(core.props.getWidth).toBe(2);
  expect(core.props.trailLength).toBe(21600);
  expect((core.props.getColor as Function)(trails.timetable[0])).toEqual([200, 214, 235, 230]);
  const heads = layers.find(l => l instanceof ScatterplotLayer)!;
  expect((heads.props.getRadius as Function)(timetable)).toBe(2.5);
  expect((heads.props.getFillColor as Function)(timetable)).toEqual([233, 238.6, 247, 255]);
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
