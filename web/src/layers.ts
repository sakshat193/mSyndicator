import { PathLayer, ScatterplotLayer } from '@deck.gl/layers';
import { TripsLayer } from '@deck.gl/geo-layers';
import type { RunFrame } from './frames';
import { pos } from './position';

type Point = [number, number];
type Stop = { idx: number; station: string; halt: boolean; dist_km: number; sched_off: number | null };
export interface Network {
  stations: { code: string; lon: number | null; lat: number | null }[];
  segments: { id: number; stn_lo: string; stn_hi: string; polyline: string | null; confidence: number }[];
  routes: { id: number; stops: Stop[];
    segments: { idx: number; segment_id: number; forward: boolean }[] }[];
}
type Trip = { path: Point[]; timestamps: number[] };
type Display = { run: RunFrame; head: Point; trips: Trip[] };

function coordinate(network: Network, code: string): Point | null {
  const s = network.stations.find(s => s.code === code);
  return s && typeof s.lon === 'number' && Number.isFinite(s.lon) && Math.abs(s.lon) <= 180
    && typeof s.lat === 'number' && Number.isFinite(s.lat) && Math.abs(s.lat) <= 90
    ? [s.lon, s.lat] : null;
}

/** Decode each run against its own base, including unchanged runs in delta frames. */
export function displayRun(run: RunFrame, network: Network, simTime: number): Display | null {
  const routeId = (run.versions as { route?: number } | undefined)?.route;
  const route = network.routes.find(r => r.id === routeId);
  if (!route || !run.paths || !run.path_shape || run.base_ts === undefined) return null;
  const [k, count] = run.path_shape;
  if (!k || !count) return null;
  const bytes = new DataView(run.paths.buffer, run.paths.byteOffset, run.paths.byteLength);
  const stops: Stop[] = [];
  const points: Point[] = [];
  for (let e = 0; e < count; e++) {
    const stop = route.stops.find(s => s.idx === bytes.getUint16(e * 4, true));
    const point = stop && coordinate(network, stop.station);
    if (!stop || !point) return null; // Missing geometry must not bridge an unknown station.
    stops.push(stop);
    points.push(point);
  }
  const trips: Trip[] = [];
  for (let p = 0; p < k; p++) {
    const timestamps = Array.from({ length: count }, (_, e) =>
      run.base_ts! + bytes.getUint16((p * count + e) * 4 + 2, true) * 6);
    if (timestamps.some((t, e) => e > 0 && t < timestamps[e - 1])) return null;
    trips.push({ path: points, timestamps });
  }
  const halts = route.stops.filter(s => s.halt);
  const qs = run.halt_q as (number | null)[][][] | undefined;
  const times = stops.map((s, e) => {
    const h = halts.findIndex(h => h.idx === s.idx);
    // Arrival then departure at intermediate halts; origin has departure only.
    const kind = h === 0 || (e > 0 && stops[e - 1].idx === s.idx) ? 1 : 0;
    const q = h >= 0 ? qs?.[h]?.[kind]?.[3] : null;
    if (typeof q === 'number' && Number.isFinite(q) && typeof run.origin_ts === 'number')
      return run.origin_ts + q * 60;
    // ponytail: past/pass anchors use display medians; upgrade when frames expose their full P50.
    const sorted = trips.map(t => t.timestamps[e]).sort((a, b) => a - b);
    return (sorted[Math.floor((k - 1) / 2)] + sorted[Math.floor(k / 2)]) / 2;
  });
  let head = points[0];
  for (let e = 1; e < count; e++) {
    if (simTime >= times[e]) { head = points[e]; continue; }
    const a = stops[e - 1], b = stops[e];
    const km = pos(simTime, times[e - 1], times[e - 1], a.dist_km,
      times[e], a.dist_km, b.dist_km, null);
    const f = b.dist_km === a.dist_km ? 0 : (km - a.dist_km) / (b.dist_km - a.dist_km);
    head = [points[e - 1][0] + f * (points[e][0] - points[e - 1][0]),
      points[e - 1][1] + f * (points[e][1] - points[e - 1][1])];
    break;
  }
  return { run, head, trips };
}

export function networkLayer(network: Network) {
  // ponytail: straight station geometry for part 1; upgrade with Task 37's OSM geometry contract.
  const paths = network.segments.flatMap(s => {
    const a = coordinate(network, s.stn_lo), b = coordinate(network, s.stn_hi);
    return a && b ? [[a, b]] : [];
  });
  return new PathLayer<Point[]>({ id: 'network', data: paths, getPath: p => p,
    getColor: [63, 78, 92], getWidth: 1, widthUnits: 'pixels' });
}

export function trainLayers(displays: Display[], simTime: number, select: (run: RunFrame) => void) {
  // Relative timestamps avoid float32 precision loss in TripsLayer's GPU attributes.
  const base = Math.floor(simTime / 3600) * 3600;
  const trips = displays.flatMap(d => d.trips.map(t => ({ ...t, k: d.trips.length })));
  return [new TripsLayer<Trip & { k: number }>({ id: 'streaks', data: trips,
    getPath: t => t.path, getTimestamps: t => t.timestamps.map(ts => ts - base),
    currentTime: simTime - base, trailLength: 600, getWidth: 2, widthUnits: 'pixels',
    getColor: t => [54, 206, 186, Math.min(255, 768 / t.k)] }),
  new ScatterplotLayer<Display>({ id: 'heads', data: displays, getPosition: d => d.head,
    getRadius: 5, radiusUnits: 'pixels', getFillColor: [80, 235, 214],
    pickable: true, onClick: info => { if (info.object) select(info.object.run); } })];
}
