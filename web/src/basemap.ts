import type { MapOptions, StyleSpecification } from 'maplibre-gl';

export const basemapStyle: StyleSpecification = {
  version: 8,
  sources: { india: { type: 'vector', url: 'pmtiles:///tiles/india.pmtiles', maxzoom: 10 } },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': '#05070d' } },
    { id: 'earth', type: 'fill', source: 'india', 'source-layer': 'earth',
      filter: ['==', '$type', 'Polygon'], paint: { 'fill-color': '#0b1119' } },
    { id: 'water', type: 'fill', source: 'india', 'source-layer': 'water',
      filter: ['==', '$type', 'Polygon'], paint: { 'fill-color': '#03050a' } },
    { id: 'country', type: 'line', source: 'india', 'source-layer': 'boundaries',
      filter: ['==', ['get', 'kind'], 'country'], paint: { 'line-color': '#2b3b4f', 'line-width': 1 } },
    { id: 'region', type: 'line', source: 'india', 'source-layer': 'boundaries',
      filter: ['==', ['get', 'kind'], 'region'],
      paint: { 'line-color': '#18222f', 'line-width': 0.6, 'line-dasharray': [3, 3] } },
    { id: 'roads', type: 'line', source: 'india', 'source-layer': 'roads', minzoom: 6,
      filter: ['in', ['get', 'kind'], ['literal', ['highway', 'major_road']]],
      paint: { 'line-color': '#111923' } },
    { id: 'rail', type: 'line', source: 'india', 'source-layer': 'transit', minzoom: 8,
      filter: ['==', ['get', 'kind'], 'rail'], paint: { 'line-color': '#0e2630' } },
  ],
};

export const mapOptions: Omit<MapOptions, 'container'> = {
  style: basemapStyle, renderWorldCopies: false, minZoom: 3.5, maxZoom: 13,
  maxBounds: [[60, 0], [106, 40]], bounds: [[68, 6.5], [97.5, 36]],
  fitBoundsOptions: { padding: 48, duration: 0 },
  attributionControl: { compact: true, customAttribution: '© OpenStreetMap · Protomaps' },
};
