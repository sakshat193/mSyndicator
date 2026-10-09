import { defineConfig } from 'vite';

export default defineConfig({
  // MapLibre resolves its worker relative to import.meta.url; prebundling relocates it.
  optimizeDeps: { exclude: ['maplibre-gl'] },
  server: { proxy: { '/api': { target: 'http://127.0.0.1:8000', ws: true } } },
});
