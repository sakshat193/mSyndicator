import { expect, it } from 'vitest';
import { indexNetwork, networkLayer, displayRun, type Network } from './layers';

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
