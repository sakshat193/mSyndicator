import { expect, it } from 'vitest';
import { basemapStyle, mapOptions } from './basemap';

it('uses the local PMTiles extract, overzooms it and needs no glyph server', () => {
  expect(basemapStyle.sources.india).toMatchObject({ type: 'vector',
    url: 'pmtiles:///tiles/india.pmtiles', maxzoom: 10 });
  expect(basemapStyle.layers.some(layer => layer.type === 'symbol')).toBe(false);
  expect(basemapStyle.glyphs).toBeUndefined();
  expect(basemapStyle.layers.map(layer => layer.id)).toEqual([
    'background', 'earth', 'water', 'country', 'region', 'roads', 'rail',
  ]);
});

it('bounds the map to India without repeated worlds', () => {
  expect(mapOptions).toMatchObject({ renderWorldCopies: false, minZoom: 3.5, maxZoom: 13,
    maxBounds: [[60, 0], [106, 40]] });
});
