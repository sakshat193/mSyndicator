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
type Display = { run: RunFrame; head: Point; trips: Trip[];
  stops: Stop[]; points: Point[]; times: number[] };
type NetworkIndex = {
  stations: Map<string, Point | null>;
  routes: Map<number, Network['routes'][number]>;
  stops: Map<number, Map<number, Stop>>;
  halts: Map<number, Map<number, number>>;
};
const indices = new WeakMap<Network, NetworkIndex>();
const networkLayers = new WeakMap<Network, PathLayer<Point[]>>();

export function indexNetwork(network: Network): NetworkIndex {
  const existing = indices.get(network);
  if (existing) return existing;
  const index: NetworkIndex = { stations: new Map(), routes: new Map(),
    stops: new Map(), halts: new Map() };
  for (const s of network.stations) if (!index.stations.has(s.code)) {
    index.stations.set(s.code, typeof s.lon === 'number' && Number.isFinite(s.lon) && Math.abs(s.lon) <= 180
      && typeof s.lat === 'number' && Number.isFinite(s.lat) && Math.abs(s.lat) <= 90
      ? [s.lon, s.lat] : null);
  }
  for (const route of network.routes) if (!index.routes.has(route.id)) {
    index.routes.set(route.id, route);
    const stops = new Map<number, Stop>(), halts = new Map<number, number>();
    let h = 0;
    for (const stop of route.stops) {
      if (!stops.has(stop.idx)) stops.set(stop.idx, stop);
      if (stop.halt) {
        if (!halts.has(stop.idx)) halts.set(stop.idx, h);
        h++;
      }
    }
    index.stops.set(route.id, stops);
    index.halts.set(route.id, halts);
  }
  indices.set(network, index);
  return index;
}

/** Decode each run against its own base, including unchanged runs in delta frames. */
export function displayRun(run: RunFrame, network: Network, simTime: number): Display | null {
  const routeId = (run.versions as { route?: number } | undefined)?.route;
  const index = indexNetwork(network);
  const route = routeId === undefined ? undefined : index.routes.get(routeId);
  if (!route || !run.paths || !run.path_shape || run.base_ts === undefined) return null;
  const [k, count] = run.path_shape;
  if (!k || !count) return null;
  const bytes = new DataView(run.paths.buffer, run.paths.byteOffset, run.paths.byteLength);
  const stops: Stop[] = [];
  const points: Point[] = [];
  for (let e = 0; e < count; e++) {
    const stop = index.stops.get(route.id)!.get(bytes.getUint16(e * 4, true));
    const point = stop && index.stations.get(stop.station);
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
  const halts = index.halts.get(route.id)!;
  const qs = run.halt_q as (number | null)[][][] | undefined;
  const times = stops.map((s, e) => {
    const h = halts.get(s.idx) ?? -1;
    // Arrival then departure at intermediate halts; origin has departure only.
    const kind = h === 0 || (e > 0 && stops[e - 1].idx === s.idx) ? 1 : 0;
    const q = h >= 0 ? qs?.[h]?.[kind]?.[3] : null;
    if (typeof q === 'number' && Number.isFinite(q) && typeof run.origin_ts === 'number')
      return run.origin_ts + q * 60;
    // ponytail: past/pass anchors use display medians; upgrade when frames expose their full P50.
    const sorted = trips.map(t => t.timestamps[e]).sort((a, b) => a - b);
    return (sorted[Math.floor((k - 1) / 2)] + sorted[Math.floor(k / 2)]) / 2;
  });
  return { run, trips, stops, points, times, head: headAt({ stops, points, times }, simTime) };
}

export function headAt({ stops, points, times }: Pick<Display, 'stops' | 'points' | 'times'>,
                       simTime: number): Point {
  let head = points[0];
  for (let e = 1; e < points.length; e++) {
    if (simTime >= times[e]) { head = points[e]; continue; }
    const a = stops[e - 1], b = stops[e];
    const km = pos(simTime, times[e - 1], times[e - 1], a.dist_km,
      times[e], a.dist_km, b.dist_km, null);
    const f = b.dist_km === a.dist_km ? 0 : (km - a.dist_km) / (b.dist_km - a.dist_km);
    head = [points[e - 1][0] + f * (points[e][0] - points[e - 1][0]),
      points[e - 1][1] + f * (points[e][1] - points[e - 1][1])];
    break;
  }
  return head;
}

export function networkLayer(network: Network) {
  const existing = networkLayers.get(network);
  if (existing) return existing;
  const index = indexNetwork(network);
  // ponytail: straight station geometry for part 1; upgrade with Task 37's OSM geometry contract.
  const paths = network.segments.flatMap(s => {
    const a = index.stations.get(s.stn_lo), b = index.stations.get(s.stn_hi);
    return a && b ? [[a, b]] : [];
  });
  const layer = new PathLayer<Point[]>({ id: 'network', data: paths, getPath: p => p,
    getColor: [63, 78, 92], getWidth: 1, widthUnits: 'pixels' });
  networkLayers.set(network, layer);
  return layer;
}

export function trainLayers(displays: Display[], trips: (Trip & { k: number })[],
                            simTime: number, select: (run: RunFrame) => void) {
  // Relative timestamps avoid float32 precision loss in TripsLayer's GPU attributes.
  const base = Math.floor(simTime / 3600) * 3600;
  return [new TripsLayer<Trip & { k: number }>({ id: 'streaks', data: trips,
    getPath: t => t.path, getTimestamps: t => t.timestamps.map(ts => ts - base),
    updateTriggers: { getTimestamps: base },
    currentTime: simTime - base, trailLength: 600, getWidth: 2, widthUnits: 'pixels',
    getColor: t => [54, 206, 186, Math.min(255, 768 / t.k)] }),
  new ScatterplotLayer<Display>({ id: 'heads', data: displays, getPosition: d => d.head,
    getRadius: 5, radiusUnits: 'pixels', getFillColor: [80, 235, 214],
    pickable: true, onClick: info => { if (info.object) select(info.object.run); } })];
}
