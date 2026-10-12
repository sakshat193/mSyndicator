/// <reference types="vite/client" />
import { Map as MapLibreMap, NavigationControl, addProtocol, setWorkerUrl } from 'maplibre-gl';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url';
import { Protocol } from 'pmtiles';
import 'maplibre-gl/dist/maplibre-gl.css';
import '@fontsource-variable/inter';
import './style.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { connectLive, FrameStore, type RunFrame } from './frames';
import { displayRun, headAt, easeHead, indexNetwork, networkLayer, trailData, trainLayers,
  labelBucket, labelStations, stationLabelLayer, selectionLayer, delayColor, type Network } from './layers';
import { liveTime } from './timeline';
import { hoverText, runPanel } from './panel';
import { mapOptions } from './basemap';
import { initSearch } from './search';
import { initHud, liveStatus, trainCount } from './hud';

// ponytail: userAgentData gates SVG backdrop filters to Chromium; drop once @supports can detect them.
if ('userAgentData' in navigator) document.body.dataset.refract = '';

const disposeHud = initHud(document, {
  getItem: key => localStorage.getItem(key),
  setItem: (key, value) => localStorage.setItem(key, value),
});

setWorkerUrl(workerUrl);
const protocol = new Protocol();
addProtocol('pmtiles', protocol.tile);
const map = new MapLibreMap({ container: 'map', ...mapOptions });
map.on('error', event => {
  if (!('sourceId' in event) || event.sourceId !== 'india') console.error(event.error);
});
map.addControl(new NavigationControl({ showCompass: false }), 'bottom-right');
const overlay = new MapboxOverlay({ interleaved: false, layers: [], pickingRadius: 10 });
map.addControl(overlay);
const store = new FrameStore();
const disconnect = connectLive(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/live`, store);
const panelElement = document.querySelector<HTMLElement>('#panel')!;
const panel = runPanel(panelElement, () => network, clearSelection);
const status = document.querySelector<HTMLElement>('#connection')!;
const clockElement = document.querySelector<HTMLElement>('#clock')!;
const count = document.querySelector<HTMLElement>('#train-count')!;
const tooltip = document.querySelector<HTMLElement>('#hover')!;
const hoverTitle = tooltip.querySelector<HTMLElement>('[data-hover-title]')!;
const hoverRoute = tooltip.querySelector<HTMLElement>('[data-hover-route]')!;
const hoverDelay = tooltip.querySelector<HTMLElement>('[data-hover-delay]')!;
const hoverEvidence = tooltip.querySelector<HTMLElement>('[data-hover-evidence]')!;
const istClock = new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata',
  hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
let network: Network | undefined;
let background: ReturnType<typeof networkLayer> | undefined;
let labels: ReturnType<typeof stationLabelLayer> | undefined;
let labelZoom = -1;
let selected: number | null = null;
let panelRun: RunFrame | undefined;
let highlight: ReturnType<typeof selectionLayer> | undefined;
let hovered: { id: number; x: number; y: number } | undefined;
let displays: NonNullable<ReturnType<typeof displayRun>>[] = [];
const decoded = new WeakMap<RunFrame, ReturnType<typeof displayRun>>();
const drawnHeads = new Map<number, ReturnType<typeof headAt>>();
let trails: ReturnType<typeof trailData> = { base: 0, comets: [], streaks: [], timetable: [] };
let generation = -1;
let lastDraw = -Infinity;
let settling = false;
let lastStatus = -Infinity;
let animation: number;

function clearSelection() {
  selected = null;
  panelRun = undefined;
  highlight = undefined;
  panel.close();
  settling = true;
}

function selectRun(run: RunFrame) {
  run = store.runs[run.run_id];
  if (!run) return;
  const display = displays.find(d => d.run.run_id === run.run_id);
  selected = run.run_id;
  panelRun = run;
  highlight = display ? selectionLayer(display) : undefined;
  settling = true;
  tooltip.hidden = true;
  hovered = undefined;
  void panel.show(run, liveTime());
  if (display) map.easeTo({ center: headAt(display, liveTime()), zoom: Math.max(6, map.getZoom()) });
}

function renderHover() {
  const display = hovered && displays.find(d => d.run.run_id === hovered!.id);
  if (!display || !network || !hovered) { tooltip.hidden = true; return; }
  const text = hoverText(display.run, network, display.delay, Date.now() / 1000);
  hoverTitle.textContent = text.title;
  hoverRoute.textContent = text.route;
  hoverDelay.textContent = text.delay;
  hoverDelay.style.setProperty('--delay-color', `rgb(${delayColor(display.delay).join(',')})`);
  hoverEvidence.textContent = text.evidence;
  tooltip.hidden = false;
  const width = tooltip.offsetWidth, height = tooltip.offsetHeight;
  const gutter = window.innerWidth < 640 ? 12 : 16;
  const x = Math.max(gutter, Math.min(hovered.x + 16, window.innerWidth - width - gutter));
  const y = Math.max(gutter, Math.min(hovered.y + 16, window.innerHeight - height - gutter));
  tooltip.style.transform = `translate(${x}px, ${y}px)`;
}

function hoverRun(run: RunFrame | null, x: number, y: number) {
  hovered = run ? { id: run.run_id, x, y } : undefined;
  map.getCanvas().style.cursor = run ? 'pointer' : '';
  renderHover();
  settling = true;
}

const closeSearch = initSearch(document.querySelector<HTMLInputElement>('#search')!,
  document.querySelector<HTMLElement>('#search-results')!, () => store.runs, selectRun);
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') { closeSearch(); clearSelection(); hoverRun(null, 0, 0); }
});
map.on('move', () => { settling = true; hoverRun(null, 0, 0); });

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
      const chosen = displays.find(d => d.run.run_id === selected);
      if (selected !== null && !store.runs[selected]) clearSelection();
      else if (chosen) {
        highlight = selectionLayer(chosen);
        if (panelRun !== chosen.run) { panelRun = chosen.run; void panel.show(chosen.run, simTime); }
      } else if (selected !== null) {
        highlight = undefined;
        const run = store.runs[selected];
        if (panelRun !== run) { panelRun = run; void panel.show(run, simTime); }
      }
      renderHover();
      settling = true;
    }
    // Live motion is far below a pixel per frame, so redraw at ~70 Hz only while a head is
    // still visibly easing and at 10 Hz otherwise; each deck redraw costs ~14-20 ms on an iGPU.
    if (tick - lastDraw >= (settling ? 12 : 100)) {
      const dt = tick - lastDraw;
      lastDraw = tick;
      const zoom = map.getZoom();
      const bucket = labelBucket(zoom);
      if (bucket !== labelZoom) {
        labelZoom = bucket;
        labels = stationLabelLayer(labelStations(network, bucket), bucket);
      }
      const eps = 0.05 * 360 / (512 * 2 ** zoom); // 0.05 px in degrees
      settling = false;
      for (const d of displays) {
        const target = headAt(d, simTime);
        const head = easeHead(drawnHeads.get(d.run.run_id), target, dt, eps);
        if (head !== target) settling = true;
        drawnHeads.set(d.run.run_id, head);
      }
      overlay.setProps({ layers: [background, ...(highlight ? [highlight] : []),
        ...trainLayers(displays, trails, simTime, zoom, selectRun, selected, hoverRun, drawnHeads, tick),
        ...(labels ? [labels] : [])] });
    }
  }
  if (tick - lastStatus >= 1000) {
    lastStatus = tick;
    const now = Date.now();
    clockElement.textContent = istClock.format(now);
    if (store.latest) {
      const total = trainCount(displays);
      count.textContent = total.text;
      count.title = total.title;
    } else {
      count.textContent = 'Loading…';
      count.removeAttribute('title');
    }
    if (network) {
      const age = store.latest ? Math.max(0, Math.floor(now / 1000 - store.latest.ts)) : null;
      const live = liveStatus(age);
      status.textContent = live.text;
      status.dataset.freshness = live.freshness;
    }
    if (panelRun) panel.update(panelRun, simTime);
    renderHover();
  }
  animation = requestAnimationFrame(draw);
}
animation = requestAnimationFrame(draw);
window.addEventListener('pagehide', () => {
  disposeHud(); closeSearch(); panel.close(); disconnect(); cancelAnimationFrame(animation); map.remove();
}, { once: true });
