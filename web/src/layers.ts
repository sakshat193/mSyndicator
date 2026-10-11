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
type NeonColor = [number, number, number];
type Trail = Trip & { color: NeonColor };
type Trails = { base: number; comets: Trail[]; streaks: Trail[] };
type Display = { run: RunFrame; head: Point; trips: Trip[];
  stops: Stop[]; points: Point[]; times: number[]; color: NeonColor };
type NetworkIndex = {
  stations: Map<string, Point | null>;
  routes: Map<number, Network['routes'][number]>;
  stops: Map<number, Map<number, Stop>>;
  halts: Map<number, Map<number, number>>;
};
const indices = new WeakMap<Network, NetworkIndex>();
const ADDITIVE = { blend: true, blendColorOperation: 'add', blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one', blendAlphaOperation: 'add', blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one', depthWriteEnabled: false } as const;
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
  const e = times.findIndex(t => t > simTime);
  const scheduled = stops[e]?.sched_off;
  const delay = e >= 0 && typeof run.origin_ts === 'number' && Number.isFinite(run.origin_ts)
    && typeof scheduled === 'number' && Number.isFinite(scheduled)
    ? (times[e] - run.origin_ts) / 60 - scheduled : 0;
  return { run, trips, stops, points, times, color: delayColor(delay),
    head: headAt({ stops, points, times }, simTime) };
}

export function delayColor(delay: number): NeonColor {
  const cyan: NeonColor = [0, 240, 255], yellow: NeonColor = [252, 238, 10],
    magenta: NeonColor = [255, 42, 109];
  const a = delay <= 15 ? cyan : yellow, b = delay <= 15 ? yellow : magenta;
  const f = delay <= 15 ? Math.max(0, delay) / 15 : Math.min(1, (delay - 15) / 45);
  return a.map((c, i) => c + (b[i] - c) * f) as NeonColor;
}

export function trailLength(zoom: number): number {
  return Math.max(600, Math.min(21600, 14400 * 2 ** (4 - zoom)));
}

/** Returns `target` itself once within `eps` degrees, so callers can tell when a head has settled. */
export function easeHead(head: Point | undefined, target: Point, dt: number, eps = 0): Point {
  if (!head) return target;
  const f = 1 - Math.exp(-dt / 250);
  const next: Point = [head[0] + (target[0] - head[0]) * f, head[1] + (target[1] - head[1]) * f];
  return Math.abs(next[0] - target[0]) <= eps && Math.abs(next[1] - target[1]) <= eps ? target : next;
}

export function trailData(displays: Display[], simTime: number): Trails {
  // Rebase only on generations to retain float32 precision without rebuilding animation attributes.
  const base = Math.floor(simTime / 3600) * 3600;
  const comets = displays.map(d => {
    let latest = -Infinity;
    return { path: d.points, color: d.color,
      timestamps: d.times.map(ts => { latest = Math.max(latest, ts); return latest - base; }) };
  });
  // ponytail: all K = 32 streaks cost about 4× in attribute rebuilds; raise the subset when rebuilds become incremental.
  const streaks = displays.flatMap(d => {
    const count = Math.min(8, d.trips.length);
    return Array.from({ length: count }, (_, j) => {
      const trip = d.trips[Math.floor(j * d.trips.length / count)];
      return { path: trip.path, color: d.color, timestamps: trip.timestamps.map(ts => ts - base) };
    });
  });
  return { base, comets, streaks };
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
    getColor: [0, 200, 255, 28], getWidth: 1, widthUnits: 'pixels' });
  networkLayers.set(network, layer);
  return layer;
}

export function trainLayers(displays: Display[], trails: Trails, simTime: number,
                            zoom: number, select: (run: RunFrame) => void) {
  const length = trailLength(zoom);
  return [...[4, 2, 1].map((width, i) => new TripsLayer<Trail>({ id: `comet-${width}`,
    data: trails.comets, getPath: t => t.path, getTimestamps: t => t.timestamps,
    currentTime: simTime - trails.base, trailLength: length / width, fadeTrail: true,
    getWidth: width, widthUnits: 'pixels', getColor: t => [...t.color, [48, 112, 224][i]],
    parameters: ADDITIVE })),
  new TripsLayer<Trail>({ id: 'streaks', data: trails.streaks,
    getPath: t => t.path, getTimestamps: t => t.timestamps,
    currentTime: simTime - trails.base, trailLength: length, fadeTrail: true,
    getWidth: 1, widthUnits: 'pixels', getColor: t => [...t.color, 16], parameters: ADDITIVE }),
  new ScatterplotLayer<Display>({ id: 'heads', data: displays, getPosition: d => d.head,
    getRadius: 1.5, radiusUnits: 'pixels',
    getFillColor: d => d.color.map(c => c + (255 - c) * 0.6) as NeonColor,
    pickable: true, onClick: info => { if (info.object) select(info.object.run); } })];
}
