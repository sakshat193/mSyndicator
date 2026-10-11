import { expect, it } from 'vitest';
import { TripsLayer } from '@deck.gl/geo-layers';
import { ScatterplotLayer } from '@deck.gl/layers';
import { indexNetwork, networkLayer, displayRun, trailData, trainLayers,
  delayColor, trailLength, easeHead, type Network } from './layers';

const network: Network = {
  stations: [{ code: 'A', lon: 70, lat: 20 }, { code: 'B', lon: 72, lat: 22 },
    { code: 'missing', lon: null, lat: null }, { code: 'invalid', lon: 181, lat: 20 }],
  segments: [{ id: 1, stn_lo: 'A', stn_hi: 'B', polyline: null, confidence: 0 },
    { id: 2, stn_lo: 'A', stn_hi: 'missing', polyline: null, confidence: 0 }],
  routes: [{ id: 7, stops: [
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
  expect(layer.props.getColor).toEqual([0, 200, 255, 28]);
  expect(layer.props.getWidth).toBe(1);
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

it('draws three fading comet tiers with additive blending and tiny pickable heads', () => {
  const display = displayRun(runFixture(), network, 1300)!;
  const trails = trailData([display], 1300);
  let selected: unknown;
  const layers = trainLayers([display], trails, 1300, 4, run => { selected = run; });
  const trips = layers.filter(layer => layer instanceof TripsLayer);
  const comets = trips.filter(layer => layer.id.startsWith('comet-'));
  expect(comets).toHaveLength(3);
  expect(comets.map(layer => layer.props.getWidth)).toEqual([4, 2, 1]);
  expect(comets.map(layer => layer.props.trailLength)).toEqual([3600, 7200, 14400]);
  const alphas = comets.map(layer => (layer.props.getColor as Function)(trails.comets[0])[3]);
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
  expect(streaks.props.getWidth).toBe(1);
  expect((streaks.props.getColor as Function)(trails.streaks[0])[3]).toBeLessThan(alphas[0]);
  const heads = layers.find(layer => layer instanceof ScatterplotLayer)!;
  expect(heads.props.getRadius).toBe(1.5);
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

it.each([[-10, 21600], [3, 21600], [4, 14400], [5, 7200], [8, 900], [9, 600], [20, 600]])(
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
