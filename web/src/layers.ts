import { PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import { TripsLayer } from '@deck.gl/geo-layers';
import type { RunFrame } from './frames';
import { pos } from './position';

type Point = [number, number];
type Stop = { idx: number; station: string; halt: boolean; dist_km: number; sched_off: number | null };
export interface Network {
  stations: { code: string; name?: string; lon: number | null; lat: number | null }[];
  segments: { id: number; stn_lo: string; stn_hi: string; polyline: Point[] | null;
    derived: boolean; confidence: number }[];
  routes: { id: number; train: string; name: string; class: string; stops: Stop[];
    segments: { idx: number; segment_id: number; forward: boolean }[] }[];
}
type Trip = { path: Point[]; timestamps: number[] };
type Gap = { path: Point[]; lengths: number[] };
type NeonColor = [number, number, number];
type Trail = Trip & { runId: number; color: NeonColor };
type Trails = { base: number; comets: Trail[]; streaks: Trail[]; timetable: Trail[] };
type Display = { run: RunFrame; head: Point; trips: Trip[];
  stops: Stop[]; points: Point[]; times: number[]; eventTimes: number[]; gaps: Gap[];
  delay: number; color: NeonColor };
type NetworkIndex = {
  stations: Map<string, Point | null>;
  names: Map<string, string>;
  routeCounts: Map<string, number>;
  prominence: Map<string, number>;
  routes: Map<number, Network['routes'][number]>;
  stops: Map<number, Map<number, Stop>>;
  halts: Map<number, Map<number, number>>;
  geometry: Map<number, Map<number, Point[]>>;
};
const indices = new WeakMap<Network, NetworkIndex>();
const ADDITIVE = { blend: true, blendColorOperation: 'add', blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one', blendAlphaOperation: 'add', blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one', depthWriteEnabled: false } as const;
const networkLayers = new WeakMap<Network, PathLayer<Point[]>>();
const TIMETABLE: NeonColor = [200, 214, 235];
const trailCache = new WeakMap<Display, { base: number; core: Trail; streaks: Trail[] }>();

export function indexNetwork(network: Network): NetworkIndex {
  const existing = indices.get(network);
  if (existing) return existing;
  const index: NetworkIndex = { stations: new Map(), names: new Map(), routeCounts: new Map(),
    prominence: new Map(), routes: new Map(),
    stops: new Map(), halts: new Map(), geometry: new Map() };
  for (const s of network.stations) if (!index.stations.has(s.code)) {
    index.names.set(s.code, titleCase(s.name || s.code));
    index.stations.set(s.code, typeof s.lon === 'number' && Number.isFinite(s.lon) && Math.abs(s.lon) <= 180
      && typeof s.lat === 'number' && Number.isFinite(s.lat) && Math.abs(s.lat) <= 90
      ? [s.lon, s.lat] : null);
  }
  const segments = new Map(network.segments.map(s => [s.id, s]));
  for (const route of network.routes) if (!index.routes.has(route.id)) {
    index.routes.set(route.id, route);
    const longDistance = (route.stops.at(-1)?.dist_km ?? 0) >= 300;
    const terminals = new Set([route.stops[0]?.station, route.stops.at(-1)?.station]);
    for (const station of new Set(route.stops.map(s => s.station))) {
      index.routeCounts.set(station, (index.routeCounts.get(station) ?? 0) + 1);
      if (longDistance) index.prominence.set(station,
        (index.prominence.get(station) ?? 0) + 1 + (terminals.has(station) ? 2 : 0));
    }
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
    const routeSegments = new Map(route.segments.map(s => [s.idx, s]));
    const geometry = new Map<number, Point[]>();
    for (const stop of stops.values()) {
      const next = stops.get(stop.idx + 1);
      const a = index.stations.get(stop.station), b = next && index.stations.get(next.station);
      if (!a || !b) continue;
      const ref = routeSegments.get(stop.idx);
      const polyline = ref && segments.get(ref.segment_id)?.polyline;
      geometry.set(stop.idx, polyline ? (ref!.forward ? polyline : [...polyline].reverse()) : [a, b]);
    }
    index.geometry.set(route.id, geometry);
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
  const gaps: Gap[] = [];
  for (let e = 1; e < count; e++) {
    const a = stops[e - 1].idx, b = stops[e].idx;
    let path = [points[e - 1], points[e]];
    if (a < b) {
      path = [points[e - 1]];
      for (let i = a; i < b; i++) {
        const segment = index.geometry.get(route.id)!.get(i);
        if (!segment) return null;
        path.push(...segment.slice(1));
      }
    }
    const lengths = [0];
    for (let v = 1; v < path.length; v++)
      lengths.push(lengths[v - 1] + Math.hypot(path[v][0] - path[v - 1][0], path[v][1] - path[v - 1][1]));
    gaps.push({ path, lengths });
  }
  const halts = index.halts.get(route.id)!;
  const qs = run.halt_q as (number | null)[][][] | undefined;
  const eventTimes = stops.map((s, e) => {
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
  const e = eventTimes.findIndex(t => t > simTime);
  const scheduled = stops[e]?.sched_off;
  const delay = e >= 0 && typeof run.origin_ts === 'number' && Number.isFinite(run.origin_ts)
    && typeof scheduled === 'number' && Number.isFinite(scheduled)
    ? (eventTimes[e] - run.origin_ts) / 60 - scheduled : 0;
  const expanded = [points[0], ...gaps.flatMap(g => g.path.slice(1))];
  const expandedTrips = trips.map(t => ({ path: expanded, timestamps: expandTimes(t.timestamps, gaps) }));
  const times = expandTimes(eventTimes, gaps);
  return { run, trips: expandedTrips, stops, points: expanded, times, eventTimes, gaps, delay, color: delayColor(delay),
    head: headAt({ stops, points: expanded, eventTimes, gaps }, simTime) };
}

function expandTimes(times: number[], gaps: Gap[]): number[] {
  return [times[0], ...gaps.flatMap((g, e) => {
    const total = g.lengths[g.lengths.length - 1];
    return g.lengths.slice(1).map((length, i) => i === g.path.length - 2 ? times[e + 1]
      : times[e] + (times[e + 1] - times[e]) * (total ? length / total : 0));
  })];
}

export function delayColor(delay: number): NeonColor {
  const cyan: NeonColor = [0, 240, 255], yellow: NeonColor = [252, 238, 10],
    magenta: NeonColor = [255, 42, 109];
  const a = delay <= 15 ? cyan : yellow, b = delay <= 15 ? yellow : magenta;
  const f = delay <= 15 ? Math.max(0, delay) / 15 : Math.min(1, (delay - 15) / 45);
  return a.map((c, i) => c + (b[i] - c) * f) as NeonColor;
}

export function trailLength(zoom: number): number {
  return Math.max(2700, Math.min(32400, 32400 * 2 ** (4 - zoom)));
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
  const comets: Trail[] = [], streaks: Trail[] = [], timetable: Trail[] = [];
  // ponytail: no viewport culling or LOD; add culling above ~1 M timetable vertices.
  for (const d of displays) {
    const scheduled = d.run.state === 'schedule_only';
    let cached = trailCache.get(d);
    if (!cached || cached.base !== base) {
      let latest = -Infinity;
      const core = { path: d.points, runId: d.run.run_id, color: scheduled ? TIMETABLE : d.color,
        timestamps: d.times.map(ts => { latest = Math.max(latest, ts); return latest - base; }) };
      // ponytail: draw eight of K streaks; raise the subset when rebuilds become incremental.
      const count = scheduled ? 0 : Math.min(8, d.trips.length);
      cached = { base, core, streaks: Array.from({ length: count }, (_, j) => {
        const trip = d.trips[Math.floor(j * d.trips.length / count)];
        return { path: trip.path, runId: d.run.run_id, color: d.color,
          timestamps: trip.timestamps.map(ts => ts - base) };
      }) };
      trailCache.set(d, cached);
    }
    (scheduled ? timetable : comets).push(cached.core);
    streaks.push(...cached.streaks);
  }
  return { base, comets, streaks, timetable };
}

export function dataComparator(newData: unknown, oldData?: unknown): boolean {
  return Array.isArray(newData) && Array.isArray(oldData)
    && newData.length === oldData.length && newData.every((item, i) => item === oldData[i]);
}

export function headAt({ stops, points, eventTimes: times, gaps }: Pick<Display, 'stops' | 'points' | 'eventTimes' | 'gaps'>,
                       simTime: number): Point {
  let head = points[0];
  for (let e = 1; e < stops.length; e++) {
    const { path, lengths } = gaps[e - 1];
    if (simTime >= times[e]) { head = path[path.length - 1]; continue; }
    const a = stops[e - 1], b = stops[e];
    const km = pos(simTime, times[e - 1], times[e - 1], a.dist_km,
      times[e], a.dist_km, b.dist_km, null);
    const f = b.dist_km === a.dist_km ? 0 : (km - a.dist_km) / (b.dist_km - a.dist_km);
    const distance = f * lengths[lengths.length - 1];
    head = path[0];
    for (let v = 1; v < path.length; v++) {
      if (distance >= lengths[v]) { head = path[v]; continue; }
      const fraction = (distance - lengths[v - 1]) / (lengths[v] - lengths[v - 1]);
      head = [path[v - 1][0] + fraction * (path[v][0] - path[v - 1][0]),
        path[v - 1][1] + fraction * (path[v][1] - path[v - 1][1])];
      break;
    }
    break;
  }
  return head;
}

export function networkLayer(network: Network) {
  const existing = networkLayers.get(network);
  if (existing) return existing;
  const index = indexNetwork(network);
  // ponytail: hidden long chords remain only as train trails; Task 37 replaces them.
  const paths = network.segments.flatMap(s => {
    const a = index.stations.get(s.stn_lo), b = index.stations.get(s.stn_hi);
    if (s.derived || !a || !b) return [];
    return s.polyline ? [s.polyline] : greatCircle(a, b) <= 150 ? [[a, b]] : [];
  });
  const layer = new PathLayer<Point[]>({ id: 'network', data: paths, getPath: p => p,
    getColor: [0, 200, 255, 36], getWidth: 1.5, widthUnits: 'pixels' });
  networkLayers.set(network, layer);
  return layer;
}

function greatCircle(a: Point, b: Point): number {
  const radians = Math.PI / 180;
  const h = Math.sin((b[1] - a[1]) * radians / 2) ** 2
    + Math.cos(a[1] * radians) * Math.cos(b[1] * radians) * Math.sin((b[0] - a[0]) * radians / 2) ** 2;
  return 6371.0088 * 2 * Math.asin(Math.min(1, Math.sqrt(h)));
}

export const LABEL_THRESHOLDS = [[5, 400], [6, 250], [7, 150], [8, 80],
  [9, 40], [10, 20], [Infinity, 0]] as const;

export function labelBucket(zoom: number): number {
  return Math.floor(zoom * 2) / 2;
}

export function labelThreshold(zoom: number): number {
  return LABEL_THRESHOLDS.find(([limit]) => zoom < limit)![1];
}

export function labelStations(network: Network, zoom: number) {
  const bucket = labelBucket(zoom), index = indexNetwork(network), threshold = labelThreshold(bucket);
  const candidates = network.stations.filter(s => index.stations.get(s.code)
    && (index.prominence.get(s.code) ?? 0) >= threshold)
    .sort((a, b) => (index.prominence.get(b.code) ?? 0) - (index.prominence.get(a.code) ?? 0)
      || (index.routeCounts.get(b.code) ?? 0) - (index.routeCounts.get(a.code) ?? 0)
      || a.code.localeCompare(b.code));
  const world = 512 * 2 ** bucket;
  const boxes: [number, number, number, number][] = [];
  return candidates.filter(s => {
    const x = (s.lon! + 180) / 360 * world;
    const lat = Math.max(-85.051129, Math.min(85.051129, s.lat!)) * Math.PI / 180;
    const y = (1 - Math.log(Math.tan(Math.PI / 4 + lat / 2)) / Math.PI) / 2 * world;
    // ponytail: fixed label estimates; use measured glyph bounds if fonts or label layout change.
    const halfWidth = titleCase(s.name || s.code).length * 6.5 / 2 + 4;
    const box: [number, number, number, number] = [x - halfWidth, y + 6, x + halfWidth, y + 28];
    if (boxes.some(b => box[0] <= b[2] && box[2] >= b[0] && box[1] <= b[3] && box[3] >= b[1]))
      return false;
    boxes.push(box);
    return true;
  });
}

export function titleCase(name: string): string {
  return name.toLowerCase().replace(/\b\p{L}/gu, letter => letter.toUpperCase());
}

export function stationName(network: Network, code: string): string {
  return indexNetwork(network).names.get(code) ?? code;
}

export function stationLabelLayer(stations: Network['stations'], zoom: number) {
  return new TextLayer<Network['stations'][number]>({ id: 'station-labels', data: stations,
    getPosition: s => [s.lon!, s.lat!], getText: s => titleCase(s.name || s.code),
    getSize: zoom < 7 ? 11 : 12, sizeUnits: 'pixels', getColor: [150, 168, 190, 210], characterSet: 'auto',
    // SDF preserves glyph bearings so letters such as j are not clipped by the atlas.
    fontSettings: { sdf: true }, outlineWidth: 0.18, outlineColor: [5, 7, 13, 230],
    fontFamily: "Inter, 'Segoe UI Variable', system-ui, sans-serif", getPixelOffset: [0, 10],
    getTextAnchor: 'middle', getAlignmentBaseline: 'top', pickable: false });
}

export function trainAlpha(runId: number, selected: number | null): number {
  return selected === null || selected === runId ? 1 : 0.3;
}

export function selectionLayer(display: Display) {
  return new PathLayer<Point[]>({ id: 'selection', data: [display.points], getPath: p => p,
    getWidth: 3, widthUnits: 'pixels', getColor: [0, 240, 255, 190] });
}

export function trainLayers(displays: Display[], trails: Trails, simTime: number,
                            zoom: number, select: (run: RunFrame) => void, selected: number | null = null,
                            hover?: (run: RunFrame | null, x: number, y: number) => void,
                            heads?: ReadonlyMap<number, Point>, redraw?: number) {
  const length = trailLength(zoom);
  return [...[8, 4, 2].map((width, i) => new TripsLayer<Trail>({ id: `comet-${width}`,
    data: trails.comets, dataComparator, getPath: t => t.path, getTimestamps: t => t.timestamps,
    currentTime: simTime - trails.base, trailLength: length / [6, 3, 1.5][i], fadeTrail: true,
    getWidth: width, widthUnits: 'pixels', getColor: t => [...t.color, [64, 150, 255][i] * trainAlpha(t.runId, selected)],
    updateTriggers: { getColor: selected },
    parameters: ADDITIVE })),
  new TripsLayer<Trail>({ id: 'streaks', data: trails.streaks, dataComparator,
    getPath: t => t.path, getTimestamps: t => t.timestamps,
    currentTime: simTime - trails.base, trailLength: length, fadeTrail: true,
    getWidth: 1.5, widthUnits: 'pixels', getColor: t => [...t.color, 24 * trainAlpha(t.runId, selected)],
    updateTriggers: { getColor: selected }, parameters: ADDITIVE }),
  new TripsLayer<Trail>({ id: 'timetable', data: trails.timetable, dataComparator,
    getPath: t => t.path, getTimestamps: t => t.timestamps,
    currentTime: simTime - trails.base, trailLength: length / 1.5, fadeTrail: true,
    getWidth: 2, widthUnits: 'pixels', getColor: t => [...t.color, 230 * trainAlpha(t.runId, selected)],
    updateTriggers: { getColor: selected }, parameters: ADDITIVE }),
  new ScatterplotLayer<Display>({ id: 'heads', data: displays, dataComparator,
    getPosition: d => heads?.get(d.run.run_id) ?? d.head,
    getRadius: d => d.run.state === 'schedule_only' ? 2.5 : 3, radiusUnits: 'pixels',
    getFillColor: d => [...(d.run.state === 'schedule_only' ? TIMETABLE : d.color).map(c => c + (255 - c) * 0.6),
      255 * trainAlpha(d.run.run_id, selected)] as [number, number, number, number],
    updateTriggers: { getPosition: redraw, getFillColor: selected },
    pickable: true, onClick: info => { if (info.object) select(info.object.run); },
    onHover: info => hover?.(info.object?.run ?? null, info.x, info.y) })];
}
