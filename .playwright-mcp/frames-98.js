async (page) => {
  const out = {};
  const runs = [['old-75', 5173, '75'], ['new-15', 5174, '15'], ['old-75b', 5173, '75'], ['new-15b', 5174, '15']];
  for (const [name, port, net] of runs) {
    await page.unrouteAll();
    await page.route('**/api/static/network', route => route.fulfill({ path: `B:/m-syndicator/.playwright-mcp/net-${net}.json`, contentType: 'application/json' }));
    await page.goto(`http://localhost:${port}/`);
    await page.waitForTimeout(10000);
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
      const map = window.__map, wait = ms => new Promise(r => setTimeout(r, ms));
      const record = async body => {
        const d = []; let last = performance.now(), on = true;
        const tick = t => { d.push(t - last); last = t; if (on) requestAnimationFrame(tick); };
        requestAnimationFrame(tick);
        await body();
        on = false; d.shift();
        const max = Math.max(...d); d.sort((a, b) => a - b);
        const q = p => +d[Math.floor(p * (d.length - 1))].toFixed(1);
        return [d.length, q(0.5), q(0.9), q(0.99), +max.toFixed(1)];
      };
      const res = {};
      map.jumpTo({ center: [83, 22.01], zoom: 4.42 }); await wait(2000);
      res.india = await record(async () => { for (let i = 0; i < 3; i++) {
        map.zoomTo(6, { duration: 2000 }); await wait(2100); map.zoomTo(4.42, { duration: 2000 }); await wait(2100); } });
      for (const [v, c] of Object.entries({ delhi: [77.2, 28.62], mumbai: [72.86, 19.05], kolkata: [88.36, 22.6] })) {
        map.jumpTo({ center: c, zoom: 9 }); await wait(1500);
        res[v + 'Cross'] = (await record(async () => { map.zoomTo(11, { duration: 1000 }); await wait(1600); }))[4];
        await wait(1000);
        res[v] = await record(async () => { for (let i = 0; i < 2; i++) {
          map.easeTo({ center: [c[0] + 0.15, c[1] + 0.08], zoom: 12.5, duration: 2000 }); await wait(2100);
          map.easeTo({ center: c, zoom: 11, duration: 2000 }); await wait(2100); } });
      }
      res.heapMB = Math.round(performance.memory.usedJSHeapSize / 1048576);
      return res;
    });
  }
  return out;
}
