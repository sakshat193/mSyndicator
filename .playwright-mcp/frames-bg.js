async (page) => {
  // z11 <-> z12.5 frames with the 15 m file: normal vs the fine background swapped for the coarse one
  await page.unrouteAll();
  await page.route('**/api/static/network', route => route.fulfill({ path: 'B:/m-syndicator/.playwright-mcp/net-15.json', contentType: 'application/json' }));
  await page.goto('http://localhost:5173/');
  await page.waitForTimeout(9000);
  return page.evaluate(async () => {
    const url = performance.getEntriesByType('resource').map(e => e.name).find(n => n.includes('maplibre-gl.mjs') || n.includes('maplibre-gl.js'));
    const m = await import(url); const M = m.Map || m.default.Map;
    for (const k of ['getZoom', 'getCenter', 'getBounds', 'project', 'unproject']) {
      const o = M.prototype[k]; M.prototype[k] = function (...a) { window.__map = this; return o.apply(this, a); };
    }
    document.querySelector('canvas').dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 300, clientY: 300 }));
    const wait = ms => new Promise(r => setTimeout(r, ms));
    await wait(500);
    const map = window.__map, overlay = map._controls.find(c => c._deck);
    const setProps = overlay.setProps.bind(overlay);
    let swap = false;
    overlay.setProps = p => setProps(swap && p.layers ? { ...p, layers: p.layers.map(l =>
      l.id === 'network-fine' ? l.clone({ visible: false }) : l.id === 'network' ? l.clone({ visible: true }) : l) } : p);
    const res = {};
    for (const [v, center] of Object.entries({ delhi: [77.2, 28.62], mumbai: [72.86, 19.05], kolkata: [88.36, 22.6] })) {
      for (const mode of ['fine', 'coarseBg', 'fine2', 'coarseBg2']) {
        swap = mode.startsWith('coarse');
        map.jumpTo({ center, zoom: 11 }); await wait(2000);
        const d = []; let last = performance.now(), on = true;
        const tick = t => { d.push(t - last); last = t; if (on) requestAnimationFrame(tick); };
        requestAnimationFrame(tick);
        for (let i = 0; i < 2; i++) {
          map.easeTo({ center: [center[0] + 0.15, center[1] + 0.08], zoom: 12.5, duration: 2000 }); await wait(2100);
          map.easeTo({ center, zoom: 11, duration: 2000 }); await wait(2100);
        }
        on = false; d.shift(); d.sort((a, b) => a - b);
        const q = p => +d[Math.floor(p * (d.length - 1))].toFixed(1);
        res[v + ':' + mode] = [d.length, q(0.5), q(0.9), q(0.99)];
      }
    }
    return res;
  });
}
