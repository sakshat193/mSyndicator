/// <reference types="vite/client" />
import { Map as MapLibreMap, NavigationControl } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { connectLive, FrameStore, type RunFrame } from './frames';
import { displayRun, headAt, easeHead, indexNetwork, networkLayer, trailData, trainLayers,
  type Network } from './layers';
import { liveTime } from './timeline';
import { runPanel } from './panel';

const map = new MapLibreMap({ container: 'map', center: [79, 22], zoom: 4,
  style: { version: 8, sources: {}, layers: [{ id: 'background', type: 'background',
    paint: { 'background-color': '#05070d' } }] } });
map.addControl(new NavigationControl());
const overlay = new MapboxOverlay({ interleaved: false, layers: [], pickingRadius: 10 });
map.addControl(overlay);
const store = new FrameStore();
const disconnect = connectLive(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/live`, store);
const showRun = runPanel(document.querySelector<HTMLElement>('#panel')!);
const status = document.querySelector<HTMLElement>('#connection')!;
let network: Network | undefined;
let background: ReturnType<typeof networkLayer> | undefined;
let displays: NonNullable<ReturnType<typeof displayRun>>[] = [];
const decoded = new WeakMap<RunFrame, ReturnType<typeof displayRun>>();
const drawnHeads = new Map<number, ReturnType<typeof headAt>>();
let trails: ReturnType<typeof trailData> = { base: 0, comets: [], streaks: [] };
let generation = -1;
let lastDraw = -Infinity;
let settling = false;
let lastStatus = -Infinity;
let animation: number;

async function loadNetwork() {
  try {
    const response = await fetch('/api/static/network');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.stations) || !Array.isArray(data.segments) || !Array.isArray(data.routes))
      throw new Error('Invalid network');
    indexNetwork(data);
    background = networkLayer(data);
    network = data;
  } catch (error) {
    status.textContent = `Network unavailable: ${error instanceof Error ? error.message : 'request failed'}`;
  }
}
void loadNetwork();

function draw(tick: number) {
  const simTime = liveTime();
  if (network && background) {
    if (generation !== store.gen) {
      generation = store.gen;
      displays = Object.values(store.runs).flatMap(run => {
        if (!decoded.has(run)) decoded.set(run, displayRun(run, network!, simTime));
        const display = decoded.get(run);
        return display ? [display] : [];
      });
      trails = trailData(displays, simTime);
      const ids = new Set(displays.map(d => d.run.run_id));
      for (const id of drawnHeads.keys()) if (!ids.has(id)) drawnHeads.delete(id);
      settling = true;
    }
    // Live motion is far below a pixel per frame, so redraw at ~70 Hz only while a head is
    // still visibly easing and at 10 Hz otherwise; each deck redraw costs ~14-20 ms on an iGPU.
    if (tick - lastDraw >= (settling ? 12 : 100)) {
      const dt = tick - lastDraw;
      lastDraw = tick;
      const zoom = map.getZoom();
      const eps = 0.05 * 360 / (512 * 2 ** zoom); // 0.05 px in degrees
      settling = false;
      const heads = displays.map(d => {
        const target = headAt(d, simTime);
        const head = easeHead(drawnHeads.get(d.run.run_id), target, dt, eps);
        if (head !== target) settling = true;
        drawnHeads.set(d.run.run_id, head);
        return { ...d, head };
      });
      overlay.setProps({ layers: [background, ...trainLayers(heads, trails, simTime, zoom,
        run => { void showRun(run, liveTime()); })] });
    }
    if (tick - lastStatus >= 1000) {
      lastStatus = tick;
      const age = store.latest ? Math.max(0, Date.now() / 1000 - store.latest.ts) : null;
      status.textContent = age === null ? 'Waiting for live frames' :
        `${displays.length} trains · frame age ${Math.floor(age)} s`;
    }
  }
  animation = requestAnimationFrame(draw);
}
animation = requestAnimationFrame(draw);
window.addEventListener('pagehide', () => { disconnect(); cancelAnimationFrame(animation); map.remove(); }, { once: true });
