async (page) => {
  const out = {};
  for (const [name, url] of [['before', 'http://localhost:5174/'], ['after', 'http://localhost:5173/'], ['before2', 'http://localhost:5174/'], ['after2', 'http://localhost:5173/']]) {
    await page.goto(url);
    await page.waitForTimeout(6000);
    await page.getByPlaceholder('Train number or name').fill('17631');
    await page.waitForTimeout(1500);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(4000);
    out[name] = await page.evaluate(async () => {
      const url = performance.getEntriesByType('resource').map(e => e.name).find(n => n.includes('maplibre-gl.mjs') || n.includes('maplibre-gl.js'));
      const m = await import(url);
      const M = m.Map || m.default.Map;
      for (const k of ['getZoom', 'getCenter', 'getBounds', 'project', 'unproject']) {
        const o = M.prototype[k];
        M.prototype[k] = function (...a) { window.__map = this; return o.apply(this, a); };
      }
      document.querySelector('canvas').dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 300, clientY: 300 }));
      await new Promise(r => setTimeout(r, 500));
      const map = window.__map;
      map.jumpTo({ center: [83, 22.01], zoom: 4.42 });
      await new Promise(r => setTimeout(r, 1500));
      const d = []; let last = performance.now(), on = true;
      const tick = t => { d.push(t - last); last = t; if (on) requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      for (let i = 0; i < 3; i++) {
        map.zoomTo(6, { duration: 2000 }); await new Promise(r => setTimeout(r, 2100));
        map.zoomTo(4.42, { duration: 2000 }); await new Promise(r => setTimeout(r, 2100));
      }
      on = false;
      d.shift(); d.sort((a, b) => a - b);
      const q = p => +d[Math.floor(p * (d.length - 1))].toFixed(1);
      return { n: d.length, p50: q(0.5), p90: q(0.9), p99: q(0.99), heapMB: Math.round(performance.memory.usedJSHeapSize / 1048576),
        selected: !!document.querySelector('[class*=panel]:not([hidden])') };
    });
  }
  return out;
}
