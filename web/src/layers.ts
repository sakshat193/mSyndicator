import { PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import type { TextLayerProps } from '@deck.gl/layers';
import { TripsLayer } from '@deck.gl/geo-layers';
import TinySDF from '@mapbox/tiny-sdf';
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
const GLOW_SCALE = Math.sqrt(6.125 / 3.5);
// deck.gl applies opacity^(1/2.2), so 0.01 draws at ~12 % alpha.
const DIM = 0.01, SELECTED_GLOW = 1.6;
// Gaussian cross-section: thin white-hot core, soft hue halo; the halo narrows down the tail.
const GLOW = `if (picking.isActive < 0.5) {
  float age = clamp((trips.currentTime - vTime) / trips.trailLength, 0.0, 1.0);
  float x = geometry.uv.x / mix(1.0, 0.35, age);
  color.rgb = mix(color.rgb, vec3(1.0), 0.15 * exp(-24.5 * x * x));
  color.a *= exp(-6.125 * x * x) * (1.0 - age);
}`;
const BLOB = `if (picking.isActive < 0.5) {
  float r2 = dot(geometry.uv, geometry.uv);
  color.rgb = mix(color.rgb, vec3(1.0), 0.2 * exp(-21.0 * r2));
  color.a *= exp(-6.125 * r2);
}`;
class CulledTrips<D> extends TripsLayer<D, { lead?: number }> {
  static layerName = 'CulledTrips';
  getShaders() {
    const shaders = super.getShaders(), lead = (this.props.lead ?? 0).toFixed(1);
    const inject = { ...shaders.inject, 'vs:#main-end': shaders.inject['vs:#main-end'] + `
float lead = ${lead} * trips.trailLength;
if (max(instanceTimestamps, instanceNextTimestamps) < trips.currentTime - trips.trailLength ||
    min(instanceTimestamps, instanceNextTimestamps) > trips.currentTime + lead) {
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}
` };
    // Trips discards must follow PathLayer's analytic-coverage derivatives.
    if (this.props.lead) inject['fs:DECKGL_FILTER_COLOR'] = `float lead = ${lead} * trips.trailLength;
if (vTime > trips.currentTime + lead || (trips.fadeTrail && vTime < trips.currentTime - trips.trailLength)) {
  discard;
}
if (vTime > trips.currentTime) {
  float front = (vTime - trips.currentTime) / lead;
  color.a *= exp(-6.125 * front * front);
} else if (trips.fadeTrail) {
  color.a *= 1.0 - (trips.currentTime - vTime) / trips.trailLength;
}
`;
    return { ...shaders, inject };
  }
}
class GlowTrips<D> extends CulledTrips<D> {
  static layerName = 'GlowTrips';
  getShaders() {
    const shaders = super.getShaders();
    const filter = 'fs:DECKGL_FILTER_COLOR';
    return { ...shaders, inject: { ...shaders.inject, [filter]: shaders.inject[filter] + GLOW } };
  }
}
class GlowDots<D> extends ScatterplotLayer<D> {
  static layerName = 'GlowDots';
  getShaders() {
    const shaders = super.getShaders();
    return { ...shaders, inject: { ...shaders.inject, 'fs:DECKGL_FILTER_COLOR': BLOB } };
  }
}
const networkLayers = new WeakMap<Network, { full: PathLayer<Point[]>; dimmed: PathLayer<Point[]> }>();
const trailCache = new WeakMap<Display, { base: number; core: Trail; streaks: Trail[] }>();
const timetableCache = new WeakMap<Trails, Trail[][]>();
let previousTimetable: Trail[][] = [];

const OVERVIEW_M = 75;
export const DETAIL_ZOOM = 10;

export function simplify(points: Point[], metres: number): Point[] {
  if (points.length < 3) return points;
  const origin = points[0], scale = Math.cos((origin[1] + points.at(-1)![1]) / 2 * Math.PI / 180);
  const projected = points.map(([lon, lat]) => [(lon - origin[0]) * 111195.08 * scale, (lat - origin[1]) * 111195.08]);
  const keep = new Set([0, points.length - 1]), pending: [number, number][] = [[0, points.length - 1]];
  while (pending.length) {
    const [first, last] = pending.pop()!;
    const [x, y] = projected[first], dx = projected[last][0] - x, dy = projected[last][1] - y;
    const length2 = dx * dx + dy * dy;
    let farthest = metres * metres, selected = -1;
    for (let i = first + 1; i < last; i++) {
      const px = projected[i][0] - x, py = projected[i][1] - y;
      const fraction = length2 ? Math.min(1, Math.max(0, (px * dx + py * dy) / length2)) : 0;
      const deviation = (px - fraction * dx) ** 2 + (py - fraction * dy) ** 2;
      if (deviation > farthest) { farthest = deviation; selected = i; }
    }
    if (selected !== -1) { keep.add(selected); pending.push([first, selected], [selected, last]); }
  }
  return [...keep].sort((a, b) => a - b).map(i => points[i]);
}

export function overview(network: Network): Network {
  return { ...network, segments: network.segments.map(s => s.polyline
    ? { ...s, polyline: simplify(s.polyline, OVERVIEW_M) } : s) };
}

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
  const p50 = run.halt_p50 && new DataView(run.halt_p50.buffer, run.halt_p50.byteOffset, run.halt_p50.byteLength);
  const eventTimes = stops.map((s, e) => {
    const h = halts.get(s.idx) ?? -1;
    // Arrival then departure at intermediate halts; origin has departure only.
    const kind = h === 0 || (e > 0 && stops[e - 1].idx === s.idx) ? 1 : 0;
    const q = p50 && h >= 0 && h < p50.byteLength / 8
      ? p50.getFloat32((h * 2 + kind) * 4, true) : NaN;
    if (typeof q === 'number' && Number.isFinite(q) && typeof run.origin_ts === 'number')
      return run.origin_ts + q * 60;
    // ponytail: past/pass or missing halt P50s use path medians; upgrade when all anchors carry P50.
    const sorted = trips.map(t => t.timestamps[e]).sort((a, b) => a - b);
    return (sorted[Math.floor((k - 1) / 2)] + sorted[Math.floor(k / 2)]) / 2;
  });
  const e = eventTimes.findIndex(t => t > simTime);
  const scheduled = stops[e]?.sched_off;
  const delay = e >= 0 && typeof run.origin_ts === 'number' && Number.isFinite(run.origin_ts)
    && typeof scheduled === 'number' && Number.isFinite(scheduled)
    ? (eventTimes[e] - run.origin_ts) / 60 - scheduled : 0;
  const expanded = [points[0], ...gaps.flatMap(g => g.path.slice(1))];
  const expandedTrips = run.state === 'schedule_only' ? []
    : trips.map(t => ({ path: expanded, timestamps: expandTimes(t.timestamps, gaps) }));
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

export function speedColor(kmh: number): [number, number, number, number] {
  if (kmh < 30) return [160, 16, 32, 210];
  if (kmh < 60) return [255, 32, 48, 210];
  if (kmh < 90) return [255, 128, 32, 210];
  if (kmh < 110) return [255, 208, 64, 210];
  return [255, 244, 216, 210];
}

const MAX_TRAIL_LENGTH = 32400;

export function trailLength(zoom: number): number {
  return Math.max(2700, Math.min(MAX_TRAIL_LENGTH, MAX_TRAIL_LENGTH * 2 ** (4 - zoom)));
}

/** Returns `target` itself once within `eps` degrees, so callers can tell when a head has settled. */
export function easeHead(head: Point | undefined, target: Point, dt: number, eps = 0): Point {
  if (!head) return target;
  const f = 1 - Math.exp(-dt / 250);
  const next: Point = [head[0] + (target[0] - head[0]) * f, head[1] + (target[1] - head[1]) * f];
  return Math.abs(next[0] - target[0]) <= eps && Math.abs(next[1] - target[1]) <= eps ? target : next;
}

// ponytail: linear picking fits ~2,000 heads; use a screen-space grid if counts or pointer cost grow.
export function nearestRun(displays: Display[], heads: ReadonlyMap<number, Point>,
                           project: (point: Point) => { x: number; y: number }, x: number, y: number, radius = 10): RunFrame | null {
  let nearest: RunFrame | null = null, distance = radius * radius;
  for (const d of displays) {
    const p = project(heads.get(d.run.run_id) ?? d.head);
    const squared = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (squared <= distance && (nearest === null || squared < distance)) {
      nearest = d.run;
      distance = squared;
    }
  }
  return nearest;
}

export function trailData(displays: Display[], simTime: number): Trails {
  // Rebase only on generations to retain float32 precision without rebuilding animation attributes.
  const base = Math.floor(simTime / 3600) * 3600;
  const comets: Trail[] = [], streaks: Trail[] = [], timetable: Trail[] = [];
  function windowTrail(path: Point[], timestamps: number[], d: Display): Trail {
    let start = 0;
    while (start + 1 < timestamps.length && timestamps[start + 1] <= base - MAX_TRAIL_LENGTH) start++;
    let end = start + 1;
    while (end < timestamps.length && timestamps[end - 1] < base + 3600) end++;
    if (end - start < 2) {
      start = Math.max(0, Math.min(start, timestamps.length - 2));
      end = Math.min(timestamps.length, start + 2);
    }
    return { path: start === 0 && end === path.length ? path : path.slice(start, end),
      timestamps: timestamps.slice(start, end).map(ts => ts - base), runId: d.run.run_id, color: d.color };
  }
  // ponytail: drawable time is windowed; add viewport culling or coarse low-zoom LOD above ~1 M windowed vertices.
  for (const d of displays) {
    const scheduled = d.run.state === 'schedule_only';
    let cached = trailCache.get(d);
    if (!cached || cached.base !== base) {
      let latest = -Infinity;
      const core = windowTrail(d.points, d.times.map(ts => { latest = Math.max(latest, ts); return latest; }), d);
      // ponytail: draw eight of K streaks; raise the subset when rebuilds become incremental.
      const count = scheduled ? 0 : Math.min(8, d.trips.length);
      cached = { base, core, streaks: Array.from({ length: count }, (_, j) => {
        const trip = d.trips[Math.floor(j * d.trips.length / count)];
        return windowTrail(trip.path, trip.timestamps, d);
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

export function networkLayer(network: Network, selected: number | null = null) {
  const key = selected === null ? 'full' : 'dimmed';
  const existing = networkLayers.get(network);
  if (existing) return existing[key];
  const index = indexNetwork(network);
  // ponytail: hidden long chords remain only as train trails; Task 37 replaces them.
  const paths = network.segments.flatMap(s => {
    const a = index.stations.get(s.stn_lo), b = index.stations.get(s.stn_hi);
    if (s.derived || !a || !b) return [];
    return s.polyline ? [s.polyline] : greatCircle(a, b) <= 150 ? [[a, b]] : [];
  });
  const layer = new PathLayer<Point[]>({ id: 'network', data: paths, getPath: p => p,
    getColor: [0, 200, 255, 36], getWidth: 1.5, widthUnits: 'pixels' });
  const layers = { full: layer, dimmed: layer.clone({ opacity: DIM }) };
  networkLayers.set(network, layers);
  return layers[key];
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

const labelCache = new WeakMap<Network, Map<number, Network['stations']>>();
const LABEL_CELL = 128; // px; a candidate only checks accepted boxes sharing one of its cells

export function labelStations(network: Network, zoom: number) {
  const bucket = labelBucket(zoom), index = indexNetwork(network);
  let cache = labelCache.get(network);
  if (!cache) labelCache.set(network, cache = new Map());
  const cached = cache.get(bucket);
  if (cached) return cached;
  const threshold = labelThreshold(bucket);
  const candidates = network.stations.filter(s => index.stations.get(s.code)
    && (index.prominence.get(s.code) ?? 0) >= threshold)
    .sort((a, b) => (index.prominence.get(b.code) ?? 0) - (index.prominence.get(a.code) ?? 0)
      || (index.routeCounts.get(b.code) ?? 0) - (index.routeCounts.get(a.code) ?? 0)
      || a.code.localeCompare(b.code));
  const world = 512 * 2 ** bucket;
  const grid = new Map<number, [number, number, number, number][]>();
  const cells = (box: [number, number, number, number]) => {
    const keys: number[] = [];
    for (let cx = Math.floor(box[0] / LABEL_CELL); cx <= Math.floor(box[2] / LABEL_CELL); cx++)
      for (let cy = Math.floor(box[1] / LABEL_CELL); cy <= Math.floor(box[3] / LABEL_CELL); cy++)
        keys.push(cx * 65536 + cy);
    return keys;
  };
  const labels = candidates.filter(s => {
    const x = (s.lon! + 180) / 360 * world;
    const lat = Math.max(-85.051129, Math.min(85.051129, s.lat!)) * Math.PI / 180;
    const y = (1 - Math.log(Math.tan(Math.PI / 4 + lat / 2)) / Math.PI) / 2 * world;
    // ponytail: fixed label estimates; use measured glyph bounds if fonts or label layout change.
    const halfWidth = index.names.get(s.code)!.length * 7 / 2 + 4;
    const box: [number, number, number, number] = [x - halfWidth, y + 6, x + halfWidth, y + 28];
    const keys = cells(box);
    if (keys.some(k => grid.get(k)?.some(b => box[0] <= b[2] && box[2] >= b[0] && box[1] <= b[3] && box[3] >= b[1])))
      return false;
    for (const k of keys) {
      const boxes = grid.get(k);
      if (boxes) boxes.push(box); else grid.set(k, [box]);
    }
    return true;
  });
  cache.set(bucket, labels);
  return labels;
}

export function titleCase(name: string): string {
  return name.toLowerCase().replace(/\b\p{L}/gu, letter => letter.toUpperCase());
}

export function stationName(network: Network, code: string): string {
  return indexNetwork(network).names.get(code) ?? code;
}

type Glyph = ReturnType<TinySDF['draw']>;

export function glyphShift(glyphLeft: number, buffer: number): number {
  return Math.max(0, buffer + glyphLeft);
}

export function glyphMetrics(t: Glyph, buffer: number) {
  return { advance: t.glyphAdvance, width: t.glyphWidth + glyphShift(t.glyphLeft, buffer),
    ascent: t.glyphTop, descent: t.glyphHeight - t.glyphTop };
}

export function padGlyphAlpha(alpha: Uint8ClampedArray, width: number, height: number, shift: number) {
  const paddedWidth = width + shift, rgba = new Uint8ClampedArray(paddedWidth * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++)
    rgba[(y * paddedWidth + x + shift) * 4 + 3] = alpha[y * width + x];
  return rgba;
}

// ponytail: deck shares this atlas with identical font settings, ignoring the renderer;
// drop this renderer when deck measures ink width and honours glyphLeft.
function stationFontRenderer(settings: Parameters<NonNullable<TextLayerProps['_getFontRenderer']>>[0]) {
  const { fontSize, buffer, radius, cutoff, fontFamily, fontWeight } = settings;
  const sdf = new TinySDF({ fontSize, buffer, radius, cutoff, fontFamily, fontWeight: String(fontWeight) });
  // TinySDF exposes ctx at runtime but omits it from its types.
  const ctx = (sdf as TinySDF & { ctx: CanvasRenderingContext2D }).ctx;
  const glyphs = new Map<string, Glyph>();
  function glyph(ch: string): Glyph {
    let t = glyphs.get(ch);
    if (!t) {
      t = sdf.draw(ch);
      glyphs.set(ch, t);
    }
    return t;
  }
  return {
    measure(ch?: string) {
      if (ch !== undefined) return glyphMetrics(glyph(ch), buffer);
      const metrics = ctx.measureText('A');
      return { advance: 0, width: 0, ascent: Math.ceil(metrics.fontBoundingBoxAscent),
        descent: Math.ceil(metrics.fontBoundingBoxDescent) };
    },
    draw(ch: string) {
      const t = glyph(ch), shift = glyphShift(t.glyphLeft, buffer);
      const data = new ImageData(padGlyphAlpha(t.data, t.width, t.height, shift), t.width + shift, t.height);
      return { data, left: buffer, top: buffer };
    }
  };
}

export function stationLabelLayer(stations: Network['stations'], zoom: number) {
  const size = zoom < 7 ? 12 : 13, buffer = 10;
  return new TextLayer<Network['stations'][number]>({ id: 'station-labels', data: stations,
    getPosition: s => [s.lon!, s.lat!], getText: s => titleCase(s.name || s.code),
    getSize: size, sizeUnits: 'pixels', getColor: [214, 224, 238, 255], characterSet: 'auto',
    _getFontRenderer: stationFontRenderer,
    // Halo is 0.75 * outlineWidth atlas px; it must be <= buffer and < radius.
    fontSettings: { sdf: true, buffer, radius: 16 }, outlineWidth: 12, outlineColor: [5, 7, 13, 255],
    fontWeight: 600,
    fontFamily: "Inter, 'Segoe UI Variable', system-ui, sans-serif", getPixelOffset: [-2 * buffer * size / 64, 10],
    getTextAnchor: 'middle', getAlignmentBaseline: 'top', pickable: false });
}

export const PULSE_MS = 3000;
const PULSE_TAIL = 0.25; // of the route length
type Pulse = Trip & { color: [number, number, number, number] };
const BEHIND: Pulse['color'] = [255, 64, 40, 80], AHEAD: Pulse['color'] = [255, 72, 48, 255];
const selectionCache = new WeakMap<Display, { route: Point[][]; colors: ReturnType<typeof speedColor>[];
  path: Point[]; timestamps: number[];
  along: number[]; at?: number; pulse?: Pulse[] }>();

/** Speed-coloured route plus a glow that runs origin to destination and drains into it every PULSE_MS,
 * dim behind the train and bright once it passes it. */
export function selectionLayers(display: Display, tick: number, simTime: number) {
  let cached = selectionCache.get(display);
  if (!cached) {
    // Halts repeat a point; zero-length segments would splay the wide glow into rays.
    const path = display.points.filter((p, i, all) => !i || p[0] !== all[i - 1][0] || p[1] !== all[i - 1][1]);
    let length = 0;
    const timestamps = path.map((p, i) => i ? length += Math.hypot(p[0] - path[i - 1][0], p[1] - path[i - 1][1]) : 0);
    length = 0;
    const along = display.points.map((p, i, all) => i ? length += Math.hypot(p[0] - all[i - 1][0], p[1] - all[i - 1][1]) : 0);
    const colors = display.gaps.map((_, e) => {
      const seconds = display.eventTimes[e + 1] - display.eventTimes[e];
      return seconds > 0
        ? speedColor((display.stops[e + 1].dist_km - display.stops[e].dist_km) / (seconds / 3600))
        : speedColor(30);
    });
    cached = { route: display.gaps.map(g => g.path), colors, path, timestamps, along };
    selectionCache.set(display, cached);
  }
  const { path, timestamps, along, colors } = cached, times = display.times;
  const total = timestamps.at(-1) || 1, tail = total * PULSE_TAIL;
  // headAt is linear in time between expanded vertices, so this is the head's distance along the route.
  const e = times.findIndex(t => t > simTime);
  const at = e < 0 ? total : e === 0 ? 0
    : along[e - 1] + (along[e] - along[e - 1]) * (simTime - times[e - 1]) / (times[e] - times[e - 1]);
  if (cached.at !== at || !cached.pulse) {
    cached.at = at;
    if (at <= 0) cached.pulse = [{ path, timestamps, color: AHEAD }];
    else if (at >= total) cached.pulse = [{ path, timestamps, color: BEHIND }];
    else {
      const i = timestamps.findIndex(t => t > at), f = (at - timestamps[i - 1]) / (timestamps[i] - timestamps[i - 1]);
      const head: Point = [path[i - 1][0] + f * (path[i][0] - path[i - 1][0]), path[i - 1][1] + f * (path[i][1] - path[i - 1][1])];
      const back = path.slice(0, i), backTimes = timestamps.slice(0, i);
      if (f > 0) { back.push(head); backTimes.push(at); }
      cached.pulse = [{ path: back, timestamps: backTimes, color: BEHIND },
        { path: [back.at(-1)!, ...path.slice(i)], timestamps: [backTimes.at(-1)!, ...timestamps.slice(i)], color: AHEAD }];
    }
  }
  return [new PathLayer<Point[]>({ id: 'selection', data: cached.route, getPath: p => p,
    getWidth: 3, widthUnits: 'pixels', getColor: (_, { index }) => colors[index] }),
  new GlowTrips<Pulse>({ id: 'selection-pulse', data: cached.pulse, lead: 0.1,
    getPath: t => t.path, getTimestamps: t => t.timestamps, getColor: t => t.color,
    currentTime: (tick % PULSE_MS) / PULSE_MS * (total + tail), trailLength: tail, fadeTrail: true,
    getWidth: 16 * GLOW_SCALE, widthUnits: 'pixels', jointRounded: true, capRounded: true, parameters: ADDITIVE })];
}

export function trainLayers(displays: Display[], trails: Trails, simTime: number,
                            zoom: number, selected: number | null = null,
                            heads?: ReadonlyMap<number, Point>, headsVersion?: number) {
  let chunks = timetableCache.get(trails);
  if (!chunks) {
    chunks = Array.from({ length: 16 }, () => [] as Trail[]);
    for (const trail of trails.timetable) chunks[trail.runId % 16].push(trail);
    chunks = chunks.map((chunk, c) => dataComparator(chunk, previousTimetable[c]) ? previousTimetable[c] : chunk);
    timetableCache.set(trails, chunks);
    previousTimetable = chunks;
  }
  const length = trailLength(zoom), opacity = selected === null ? 1 : DIM;
  // Dense India-wide views stay thin and dim; zoomed-in trails widen and brighten.
  const k = Math.min(1, Math.max(0, (zoom - 4) / 4));
  const comets = new GlowTrips<Trail>({ id: 'comets', data: trails.comets, dataComparator, opacity,
    getPath: t => t.path, getTimestamps: t => t.timestamps,
    currentTime: simTime - trails.base, trailLength: length / 1.5, fadeTrail: true,
    getWidth: 7 * GLOW_SCALE, widthUnits: 'pixels', getColor: t => [...t.color, 255], parameters: ADDITIVE });
  const streaks = new CulledTrips<Trail>({ id: 'streaks', data: trails.streaks, dataComparator, opacity,
    getPath: t => t.path, getTimestamps: t => t.timestamps,
    currentTime: simTime - trails.base, trailLength: length, fadeTrail: true,
    getWidth: 1.5, widthUnits: 'pixels', getColor: t => [...t.color, 24], parameters: ADDITIVE });
  const timetable = chunks.map((data, c) => new GlowTrips<Trail>({ id: `timetable-${c}`, data, dataComparator,
    getPath: t => t.path, getTimestamps: t => t.timestamps,
    currentTime: simTime - trails.base, trailLength: length / 1.5, fadeTrail: true,
    getWidth: 6 * GLOW_SCALE, widthScale: 0.5 + 0.5 * k, opacity: (0.35 + 0.65 * k) * opacity, widthUnits: 'pixels',
    getColor: t => [...t.color, 190], parameters: ADDITIVE }));
  const dots = new GlowDots<Display>({ id: 'heads', data: displays, dataComparator,
    getPosition: d => heads?.get(d.run.run_id) ?? d.head,
    getRadius: d => (d.run.state === 'schedule_only' ? 4.5 : 5.5) * GLOW_SCALE, radiusScale: 0.6 + 0.4 * k,
    opacity: (0.45 + 0.55 * k) * opacity, radiusUnits: 'pixels',
    getFillColor: d => [...d.color, 230] as [number, number, number, number],
    updateTriggers: { getPosition: headsVersion }, parameters: ADDITIVE, pickable: false });
  const layers = [comets, streaks, ...timetable, dots];
  const display = displays.find(d => d.run.run_id === selected);
  if (display) {
    const scheduled = display.run.state === 'schedule_only';
    const trail = (scheduled ? trails.timetable : trails.comets).find(t => t.runId === selected);
    if (trail) layers.push((scheduled ? timetable[display.run.run_id % 16] : comets).clone({
      id: 'selected-trail', data: [trail], opacity: 1, widthScale: SELECTED_GLOW }));
    layers.push(dots.clone({ id: 'selected-head', data: [display], opacity: 1, radiusScale: SELECTED_GLOW }));
  }
  return layers;
}
