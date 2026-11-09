async (page) => {
  const out = {};
  const views = { delhi: [77.2, 28.62], mumbai: [72.86, 19.05], kolkata: [88.36, 22.6] };
  for (const name of ['15', '75', '15b']) {
    const file = `B:/m-syndicator/.playwright-mcp/net-${name.slice(0, 2)}.json`;
    await page.unrouteAll();
    await page.route('**/api/static/network', route => route.fulfill({ path: file, contentType: 'application/json' }));
    await page.goto('http://localhost:5173/');
    await page.waitForTimeout(9000);
    out[name] = await page.evaluate(async (views) => {
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
      const res = {};
      for (const [v, center] of Object.entries(views)) {
        map.jumpTo({ center, zoom: 11 });
        await new Promise(r => setTimeout(r, 2000));
        const d = []; let last = performance.now(), on = true;
        const tick = t => { d.push(t - last); last = t; if (on) requestAnimationFrame(tick); };
        requestAnimationFrame(tick);
        for (let i = 0; i < 2; i++) {
          map.easeTo({ center: [center[0] + 0.15, center[1] + 0.08], zoom: 12.5, duration: 2000 }); await new Promise(r => setTimeout(r, 2100));
          map.easeTo({ center, zoom: 11, duration: 2000 }); await new Promise(r => setTimeout(r, 2100));
        }
        on = false;
        d.shift(); d.sort((a, b) => a - b);
        const q = p => +d[Math.floor(p * (d.length - 1))].toFixed(1);
        res[v] = [d.length, q(0.5), q(0.9), q(0.99)];
      }
      res.heapMB = Math.round(performance.memory.usedJSHeapSize / 1048576);
      return res;
    }, views);
  }
  return out;
}
