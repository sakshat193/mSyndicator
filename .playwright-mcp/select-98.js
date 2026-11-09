async (page) => {
  await page.unrouteAll();
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:5173/');
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
  const input = page.getByPlaceholder('Train number or name');
  let picked = null;
  for (const q of ['rajdhani', 'shatabdi', 'express', 'passenger']) {
    await input.fill(q); await page.waitForTimeout(1500);
    const n = await page.locator('[role=option]').count();
    for (let i = 0; i < n && !picked; i++) {
      const b = page.locator('[role=option]').nth(i), label = await b.textContent();
      await b.click(); await page.waitForTimeout(2500);
      if (!(await page.locator('[role=option]').count())) picked = label;
      else if (!(await b.textContent()).includes('Not running')) picked = label;
    }
    if (picked) break;
  }
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'B:/m-syndicator/reports/t98-select-overview.png' });
  const z = await page.evaluate(() => { const m = window.__map; m.jumpTo({ zoom: 11.5 }); return m.getZoom(); });
  await page.waitForTimeout(5000);
  await page.screenshot({ path: 'B:/m-syndicator/reports/t98-select-z11.png' });
  return { picked, z };
}
