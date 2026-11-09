async (page) => {
  const [port, net, mode] = [5173, null, 'cross'];
  await page.unrouteAll();
  if (net) await page.route('**/api/static/network', route => route.fulfill({ path: `B:/m-syndicator/.playwright-mcp/net-${net}.json`, contentType: 'application/json' }));
  await page.goto(`http://localhost:${port}/`);
  await page.waitForTimeout(10000);
  return page.evaluate(async mode => {
    const url = performance.getEntriesByType('resource').map(e => e.name).find(n => n.includes('maplibre-gl.mjs') || n.includes('maplibre-gl.js'));
    const m = await import(url); const M = m.Map || m.default.Map;
    for (const k of ['getZoom', 'getCenter', 'getBounds', 'project', 'unproject']) {
      const o = M.prototype[k]; M.prototype[k] = function (...a) { window.__map = this; return o.apply(this, a); };
    }
    document.querySelector('canvas').dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 300, clientY: 300 }));
    const wait = ms => new Promise(r => setTimeout(r, ms));
    await wait(500);
    const map = window.__map, overlay = map._controls.find(c => c._deck), deck = overlay._deck;
    let frame = {}; const frames = [];
    const lm = deck.layerManager, upd = lm._updateLayer.bind(lm);
    lm._updateLayer = layer => { const t = performance.now(); const r = upd(layer); const k = 'u:' + layer.id; frame[k] = (frame[k] || 0) + performance.now() - t; return r; };
    const setProps = overlay.setProps.bind(overlay);
    overlay.setProps = p => { const t = performance.now(); const r = setProps(p); frame.setProps = (frame.setProps || 0) + performance.now() - t; return r; };
    let last = performance.now(), on = false;
    const tick = t => { if (on) { frames.push({ dt: t - last, ...frame }); } frame = {}; last = t; requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    const out = {};
    const record = async body => {
      const d = []; let last = performance.now(), on = true;
      const tick = t => { d.push(t - last); last = t; if (on) requestAnimationFrame(tick); };
      requestAnimationFrame(tick); await body(); on = false; d.shift();
      const max = Math.max(...d); d.sort((a, b) => a - b);
      const q = p => +d[Math.floor(p * (d.length - 1))].toFixed(1);
      return [d.length, q(0.5), q(0.9), q(0.99), +max.toFixed(1)];
    };
    map.jumpTo({ center: [83, 22.01], zoom: 4.42 }); await wait(2000);
    out.india = await record(async () => { for (let i = 0; i < 3; i++) {
      map.zoomTo(6, { duration: 2000 }); await wait(2100); map.zoomTo(4.42, { duration: 2000 }); await wait(2100); } });
    if (mode === 'idle') {
      map.jumpTo({ center: [77.2, 28.62], zoom: 11 }); await wait(2000);
      frames.length = 0; on = true; await wait(65000); on = false;
      out.idle = frames.filter(f => f.dt > 60).map(f => +f.dt.toFixed(0));
      map.jumpTo({ center: [83, 22.01], zoom: 4.42 }); await wait(2000);
      frames.length = 0; on = true; await wait(65000); on = false;
      out.idleIndia = frames.filter(f => f.dt > 60).map(f => +f.dt.toFixed(0));
      return out;
    }
    for (const [name, c] of [['delhi', [77.2, 28.62]], ['mumbai', [72.86, 19.05]], ['kolkata', [88.36, 22.6]]]) {
      map.jumpTo({ center: c, zoom: 9 }); await wait(1500);
      frames.length = 0; on = true; window.__t = [];
      map.zoomTo(11, { duration: 1000 }); await wait(1600);
      on = false;
      out[name] = frames.slice().sort((a, b) => b.dt - a.dt).slice(0, 3).map(f =>
        Object.fromEntries(Object.entries(f).filter(([, v]) => v >= 2).map(([k, v]) => [k, +v.toFixed(0)])));
      await wait(500);
      out[name + 'T'] = window.__t.map(([k, v, ...r]) => [k, +v.toFixed(0), ...r]);
      frames.length = 0; on = true; window.__t = [];
      map.zoomTo(9, { duration: 1000 }); await wait(1600);
      on = false;
      out[name + 'Out'] = frames.slice().sort((a, b) => b.dt - a.dt).slice(0, 2).map(f =>
        Object.fromEntries(Object.entries(f).filter(([, v]) => v >= 2).map(([k, v]) => [k, +v.toFixed(0)])));
      out[name + 'OutT'] = window.__t.map(([k, v, ...r]) => [k, +v.toFixed(0), ...r]);
    }
    return out;
  }, mode);
}
