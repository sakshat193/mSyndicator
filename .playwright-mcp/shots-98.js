async (page) => {
  // [port, network file or null for the API's, filename prefix]
  const [port, net, prefix, only] = [5173, null, 't100', ['gayajn', 'kangra', 'sirpur']];
  await page.unrouteAll();
  if (net) await page.route('**/api/static/network', route => route.fulfill({ path: `B:/m-syndicator/.playwright-mcp/net-${net}.json`, contentType: 'application/json' }));
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(`http://localhost:${port}/`);
  await page.waitForTimeout(10000);
  await page.evaluate(async () => {
    const url = performance.getEntriesByType('resource').map(e => e.name).find(n => n.includes('maplibre-gl.mjs') || n.includes('maplibre-gl.js'));
    const m = await import(url); const M = m.Map || m.default.Map;
    for (const k of ['getZoom', 'getCenter', 'getBounds', 'project', 'unproject']) {
      const o = M.prototype[k]; M.prototype[k] = function (...a) { window.__map = this; return o.apply(this, a); };
    }
    document.querySelector('canvas').dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 300, clientY: 300 }));
    await new Promise(r => setTimeout(r, 500));
  });
  const views = {
    bhokar: [[77.695, 19.27], 11.5], sirpur: [[79.56, 19.48], 12], gaya: [[84.26, 24.885], 11.5],
    gayajn: [[85.0, 24.80], 12.5], tripal: [[76.19, 32.015], 12.5], kangra: [[76.318, 32.096], 12.5],
  };
  for (const [name, [center, zoom]] of Object.entries(views).filter(([n]) => !only || only.includes(n))) {
    await page.evaluate(([center, zoom]) => window.__map.jumpTo({ center, zoom }), [center, zoom]);
    await page.waitForTimeout(4000);
    await page.screenshot({ path: `B:/m-syndicator/reports/${prefix}-${name}.png` });
  }
  return Object.keys(views);
}
