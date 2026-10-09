import { expect, it } from 'vitest';
import { liveTime, zone } from './timeline';
import { displayRun, type Network } from './layers';
import { haltRows } from './panel';

it('LIVE follows wall time fifteen seconds behind, in UTC seconds', () => {
  expect(liveTime(1_000_000)).toBe(985);
  expect(liveTime(1_020_000)).toBe(1005);
});

it('selects past, recent and future at the thirty-minute buffer boundaries', () => {
  expect(zone(8199, 10_000)).toBe('past');
  expect(zone(8200, 10_000)).toBe('recent');
  expect(zone(10_000, 10_000)).toBe('recent');
  expect(zone(10_001, 10_000)).toBe('future');
});

const network: Network = {
  stations: [{ code: 'A', lon: 70, lat: 20 }, { code: 'B', lon: 72, lat: 22 }],
  segments: [{ id: 1, stn_lo: 'A', stn_hi: 'B', polyline: null, confidence: 0 }],
  routes: [{ id: 7, stops: [
    { idx: 0, station: 'A', halt: true, dist_km: 0, sched_off: 0 },
    { idx: 1, station: 'B', halt: true, dist_km: 100, sched_off: 10 },
  ], segments: [{ idx: 0, segment_id: 1, forward: true }] }],
};

it('decodes little-endian paths and uses full-particle P50 rather than a streak median', () => {
  const run = { run_id: 9, versions: { route: 7 }, origin_ts: 1000, base_ts: 1000,
    path_shape: [2, 2, 2] as [number, number, number],
    paths: new Uint8Array([0, 0, 0, 0, 1, 0, 100, 0, 0, 0, 0, 0, 1, 0, 200, 0]),
    halt_q: [[Array(7).fill(null), Array(7).fill(0)],
      [Array(7).fill(20), Array(7).fill(null)]] };
  const result = displayRun(run, network, 1600);
  expect(result?.head).toEqual([71, 21]);
  expect(result?.trips.map(t => t.timestamps)).toEqual([[1000, 1600], [1000, 2200]]);
  expect(displayRun({ ...run, versions: { route: 99 } }, network, 1600)).toBeNull();
});

it('keeps repeated-station dwell at the station and ignores missing coordinates', () => {
  const run = { run_id: 9, versions: { route: 7 }, origin_ts: 1000, base_ts: 1000,
    path_shape: [1, 3, 2] as [number, number, number],
    paths: new Uint8Array([0, 0, 0, 0, 0, 0, 100, 0, 1, 0, 200, 0]) };
  expect(displayRun(run, network, 1300)?.head).toEqual([70, 20]);
  expect(displayRun(run, { ...network, stations: [] }, 1300)).toBeNull();
});

it('uses P10/P50/P90 from run-detail halts and preserves unavailable arrivals', () => {
  const rows = haltRows([{ station: 'B', occ: 2,
    arr: [1, 2, 3, 4, 5, 6, 7], dep: [null, null, null, null, null, null, null] }], 1000, 1100);
  expect(rows).toEqual([{ station: 'B', occ: 2, arr: [1120, 1240, 1360], dep: [null, null, null] }]);
  expect(haltRows([{ station: 'A', occ: 1, arr: Array(7).fill(0), dep: Array(7).fill(1) }], 1000, 1100)).toEqual([]);
});
