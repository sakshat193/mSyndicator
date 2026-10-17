/// <reference types="vite/client" />
import { Map as MapLibreMap, NavigationControl, addProtocol, setWorkerUrl } from 'maplibre-gl';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url';
import { Protocol } from 'pmtiles';
import 'maplibre-gl/dist/maplibre-gl.css';
import '@fontsource-variable/inter';
import './style.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { connectLive, FrameStore, type RunFrame } from './frames';
import { displayRun, headAt, easeHead, nearestRun, indexNetwork, networkLayer, trailData, trainLayers,
  labelBucket, labelStations, stationLabelLayer, selectionLayers, delayColor, type Network } from './layers';
import { liveTime } from './timeline';
import { hoverText, runPanel } from './panel';
import { mapOptions } from './basemap';
import { initSearch } from './search';
import { initHud, sourceStatus, trainCount } from './hud';

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
const overlay = new MapboxOverlay({ interleaved: false, useDevicePixels: 1,
  deviceProps: { webgl: { antialias: false, preserveDrawingBuffer: false } }, layers: [] });
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
let highlight: NonNullable<ReturnType<typeof displayRun>> | undefined;
let hovered: { id: number; x: number; y: number } | undefined;
let hoverWidth = 0, hoverHeight = 0;
let displays: NonNullable<ReturnType<typeof displayRun>>[] = [];
const decoded = new WeakMap<RunFrame, ReturnType<typeof displayRun>>();
const drawnHeads = new Map<number, ReturnType<typeof headAt>>();
let trails: ReturnType<typeof trailData> = { base: 0, comets: [], streaks: [], timetable: [] };
let generation = -1;
let lastDraw = -Infinity;
let lastHeads = -Infinity;
let lastZoom = -1;
let headsVersion = 0;
let redraw = false;
let settling = false;
let lastStatus = -Infinity;
let animation: number;
let health: {
  source?: { poll_ok_at: string | null; harvest_ok_at: string | null };
  dispatchers?: Record<string, { state: string }>;
} | undefined;

async function loadHealth() {
  try {
    const response = await fetch('/api/health');
    if (response.ok) health = await response.json();
  } catch {}
}
void loadHealth();
const healthInterval = setInterval(loadHealth, 60000);

function clearSelection() {
  selected = null;
  panelRun = undefined;
  highlight = undefined;
  panel.close();
  redraw = true;
}

function selectRun(run: RunFrame) {
  run = store.runs[run.run_id];
  if (!run) return;
  const display = displays.find(d => d.run.run_id === run.run_id);
  selected = run.run_id;
  panelRun = run;
  highlight = display;
  redraw = true;
  tooltip.hidden = true;
  hovered = undefined;
  void panel.show(run, liveTime());
  if (display) {
    const live = document.querySelector<HTMLElement>('#live')!.getBoundingClientRect();
    const legend = document.querySelector<HTMLElement>('#legend')!.getBoundingClientRect();
    const panel = panelElement.getBoundingClientRect();
    const { innerWidth, innerHeight } = window;
    let padding = { top: live.bottom + 16, right: 16,
      bottom: legend.height ? innerHeight - legend.top + 16 : 16, left: 16 };
    if (panel.width && panel.height) {
      if (innerWidth < 640) padding.bottom = Math.max(padding.bottom,
        innerHeight - Math.min(panel.top, panel.bottom - 0.6 * innerHeight) + 16);
      else padding.right = innerWidth - panel.left + 16;
    }
    if (padding.left + padding.right >= innerWidth || padding.top + padding.bottom >= innerHeight)
      padding = { top: 16, right: 16, bottom: 16, left: 16 };
    let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity;
    for (const [lon, lat] of display.points) {
      west = Math.min(west, lon); south = Math.min(south, lat);
      east = Math.max(east, lon); north = Math.max(north, lat);
    }
    map.fitBounds([[west, south], [east, north]], { padding, maxZoom: 10, linear: true,
      duration: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 600 });
  }
}

function renderHover(refresh = false) {
  if (!network || !hovered) { tooltip.hidden = true; return; }
  if (refresh) {
    const display = displays.find(d => d.run.run_id === hovered!.id);
    if (!display) { tooltip.hidden = true; return; }
    const text = hoverText(display.run, network, display.delay, Date.now() / 1000);
    hoverTitle.textContent = text.title;
    hoverRoute.textContent = text.route;
    hoverDelay.textContent = text.delay;
    hoverDelay.style.setProperty('--delay-color', `rgb(${delayColor(display.delay).join(',')})`);
    hoverEvidence.textContent = text.evidence;
    tooltip.hidden = false;
    hoverWidth = tooltip.offsetWidth;
    hoverHeight = tooltip.offsetHeight;
  }
  const gutter = window.innerWidth < 640 ? 12 : 16;
  const x = Math.max(gutter, Math.min(hovered.x + 16, window.innerWidth - hoverWidth - gutter));
  const y = Math.max(gutter, Math.min(hovered.y + 16, window.innerHeight - hoverHeight - gutter));
  tooltip.style.transform = `translate(${x}px, ${y}px)`;
}

function hoverRun(run: RunFrame | null, x: number, y: number) {
  const changed = hovered?.id !== run?.run_id;
  hovered = run ? { id: run.run_id, x, y } : undefined;
  map.getCanvas().style.cursor = run ? 'pointer' : '';
  renderHover(changed);
}

const closeSearch = initSearch(document.querySelector<HTMLInputElement>('#search')!,
  document.querySelector<HTMLElement>('#search-results')!, () => store.runs, selectRun);
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') { closeSearch(); clearSelection(); hoverRun(null, 0, 0); }
});
map.on('move', () => hoverRun(null, 0, 0));
map.on('mousemove', event => {
  const { x, y } = event.point;
  hoverRun(map.isMoving() ? null : nearestRun(displays, drawnHeads, p => map.project(p), x, y), x, y);
});
map.on('mouseout', () => hoverRun(null, 0, 0));
map.on('click', event => {
  const { x, y } = event.point;
  const run = nearestRun(displays, drawnHeads, p => map.project(p), x, y);
  if (run) selectRun(run);
});

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
        highlight = chosen;
        if (panelRun !== chosen.run) { panelRun = chosen.run; void panel.show(chosen.run, simTime); }
      } else if (selected !== null) {
        highlight = undefined;
        const run = store.runs[selected];
        if (panelRun !== run) { panelRun = run; void panel.show(run, simTime); }
      }
      renderHover(true);
      redraw = true;
    }
    const zoom = map.getZoom();
    if (settling || tick - lastHeads >= 100) {
      const dt = tick - lastHeads;
      lastHeads = tick;
      const eps = 0.05 * 360 / (512 * 2 ** zoom); // 0.05 px in degrees
      settling = false;
      for (const d of displays) {
        const target = headAt(d, simTime);
        const head = easeHead(drawnHeads.get(d.run.run_id), target, dt, eps);
        if (head !== target) settling = true;
        drawnHeads.set(d.run.run_id, head);
      }
      headsVersion++;
      redraw = true;
    }
    if (redraw || zoom !== lastZoom || (highlight && tick - lastDraw >= 33)) {
      lastDraw = tick;
      lastZoom = zoom;
      redraw = false;
      const bucket = labelBucket(zoom);
      if (bucket !== labelZoom) {
        labelZoom = bucket;
        labels = stationLabelLayer(labelStations(network, bucket), bucket);
      }
      overlay.setProps({ layers: [background, ...(highlight ? selectionLayers(highlight, tick, simTime) : []),
        ...trainLayers(displays, trails, simTime, zoom, selected, drawnHeads, headsVersion),
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
      const sourceAge = (ts: string | null | undefined) => {
        const time = ts ? Date.parse(ts) : NaN;
        return Number.isFinite(time) ? Math.max(0, Math.floor((now - time) / 1000)) : null;
      };
      const frameAge = store.latest ? Math.max(0, Math.floor(now / 1000 - store.latest.ts)) : null;
      const live = sourceStatus(sourceAge(health?.source?.poll_ok_at), sourceAge(health?.source?.harvest_ok_at),
        frameAge, health?.dispatchers?.['rails-ris.makemytrip.com']?.state);
      status.textContent = live.text;
      status.dataset.freshness = live.freshness;
      status.title = live.title;
    }
    if (panelRun) panel.update(panelRun, simTime);
    renderHover(true);
  }
  animation = requestAnimationFrame(draw);
}
animation = requestAnimationFrame(draw);
window.addEventListener('pagehide', () => {
  clearInterval(healthInterval);
  disposeHud(); closeSearch(); panel.close(); disconnect(); cancelAnimationFrame(animation); map.remove();
}, { once: true });
