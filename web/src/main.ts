/// <reference types="vite/client" />
import { Map, NavigationControl } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { connectLive, FrameStore } from './frames';
import { displayRun, networkLayer, trainLayers, type Network } from './layers';
import { liveTime } from './timeline';
import { runPanel } from './panel';

const map = new Map({ container: 'map', center: [79, 22], zoom: 4,
  style: { version: 8, sources: {}, layers: [{ id: 'background', type: 'background',
    paint: { 'background-color': '#0c121a' } }] } });
map.addControl(new NavigationControl());
const overlay = new MapboxOverlay({ interleaved: false, layers: [] });
map.addControl(overlay);
const store = new FrameStore();
const disconnect = connectLive(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/live`, store);
const showRun = runPanel(document.querySelector<HTMLElement>('#panel')!);
const status = document.querySelector<HTMLElement>('#connection')!;
let network: Network | undefined;
let displays: NonNullable<ReturnType<typeof displayRun>>[] = [];
let generation = -1;
let lastTick = 0;
let animation: number;

async function loadNetwork() {
  try {
    const response = await fetch('/api/static/network');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.stations) || !Array.isArray(data.segments) || !Array.isArray(data.routes))
      throw new Error('Invalid network');
    network = data;
  } catch (error) {
    status.textContent = `Network unavailable: ${error instanceof Error ? error.message : 'request failed'}`;
  }
}
void loadNetwork();

function draw(tick: number) {
  const simTime = liveTime();
  if (network && (generation !== store.gen || tick - lastTick >= 100)) {
    generation = store.gen;
    lastTick = tick;
    displays = Object.values(store.runs).flatMap(run => {
      const display = displayRun(run, network!, simTime);
      return display ? [display] : [];
    });
    overlay.setProps({ layers: [networkLayer(network), ...trainLayers(displays, simTime,
      run => { void showRun(run, liveTime()); })] });
    const age = store.latest ? Math.max(0, Date.now() / 1000 - store.latest.ts) : null;
    status.textContent = age === null ? 'Waiting for live frames' :
      `${displays.length} trains · frame age ${Math.floor(age)} s`;
  }
  animation = requestAnimationFrame(draw);
}
animation = requestAnimationFrame(draw);
window.addEventListener('pagehide', () => { disconnect(); cancelAnimationFrame(animation); map.remove(); }, { once: true });
