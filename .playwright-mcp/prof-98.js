async (page) => {
  await page.unrouteAll();
  await page.route('**/api/static/network', route => route.fulfill({ path: 'B:/m-syndicator/.playwright-mcp/net-15.json', contentType: 'application/json' }));
  await page.goto('http://localhost:5174/');
  await page.waitForTimeout(10000);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('HeapProfiler.enable');
  const heap = async () => { await cdp.send('HeapProfiler.collectGarbage'); return page.evaluate(() => Math.round(performance.memory.usedJSHeapSize / 1048576)); };
  await page.evaluate(async () => {
    const url = performance.getEntriesByType('resource').map(e => e.name).find(n => n.includes('maplibre-gl.mjs') || n.includes('maplibre-gl.js'));
    const m = await import(url); const M = m.Map || m.default.Map;
    for (const k of ['getZoom', 'getCenter', 'getBounds', 'project', 'unproject']) {
      const o = M.prototype[k]; M.prototype[k] = function (...a) { window.__map = this; return o.apply(this, a); };
    }
    document.querySelector('canvas').dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 300, clientY: 300 }));
    await new Promise(r => setTimeout(r, 500));
    window.__map.jumpTo({ center: [77.2, 28.62], zoom: 9 });
    await new Promise(r => setTimeout(r, 2000));
  });
  const out = { heapIndia: await heap() };
  await cdp.send('Profiler.enable');
  await cdp.send('Profiler.setSamplingInterval', { interval: 200 });
  const results = {};
  for (const [name, c] of [['delhi', [77.2, 28.62]], ['mumbai', [72.86, 19.05]]]) {
    await page.evaluate(c => { window.__map.jumpTo({ center: c, zoom: 9 }); }, c);
    await page.waitForTimeout(1500);
    await cdp.send('Profiler.start');
    await page.evaluate(() => { window.__map.zoomTo(11, { duration: 1000 }); });
    await page.waitForTimeout(1600);
    const { profile } = await cdp.send('Profiler.stop');
    const self = new Map(), byId = new Map(profile.nodes.map(n => [n.id, n]));
    const dt = profile.timeDeltas; const counts = new Map();
    profile.samples.forEach((id, i) => counts.set(id, (counts.get(id) || 0) + (dt[i] || 0)));
    for (const [id, us] of counts) {
      const n = byId.get(id), f = n.callFrame;
      const key = `${f.functionName || '(anon)'} ${f.url.split('/').pop().split('?')[0]}:${f.lineNumber + 1}`;
      self.set(key, (self.get(key) || 0) + us);
    }
    // inclusive time per function
    const parent = new Map(); for (const n of profile.nodes) for (const c of n.children || []) parent.set(c, n.id);
    const incl = new Map();
    for (const [id, us] of counts) {
      const seen = new Set(); let cur = id;
      while (cur !== undefined) {
        const f = byId.get(cur).callFrame;
        const key = `${f.functionName || '(anon)'} ${f.url.split('/').pop().split('?')[0]}:${f.lineNumber + 1}`;
        if (!seen.has(key)) { seen.add(key); incl.set(key, (incl.get(key) || 0) + us); }
        cur = parent.get(cur);
      }
    }
    // longest busy stretch (no (idle) sample) = the hitch frame
    let t = 0, best = [0, 0], start = 0, last = 0;
    const times = profile.timeDeltas.map(d => (t += d));
    profile.samples.forEach((id, i) => {
      const idle = byId.get(id).callFrame.functionName === '(idle)';
      if (idle) { if (times[i] - times[start] > best[1] - best[0]) best = [times[start], times[i]]; start = i + 1; }
    });
    const hitch = new Map();
    profile.samples.forEach((id, i) => {
      if (times[i] < best[0] || times[i] > best[1]) return;
      const seen = new Set(); let cur = id;
      while (cur !== undefined) {
        const f = byId.get(cur).callFrame;
        const key = `${f.functionName || '(anon)'} ${f.url.split('/').pop().split('?')[0]}:${f.lineNumber + 1}`;
        if (!seen.has(key)) { seen.add(key); hitch.set(key, (hitch.get(key) || 0) + (profile.timeDeltas[i + 1] || 0)); }
        cur = parent.get(cur);
      }
    });
    results[name + 'Hitch'] = [`${((best[1] - best[0]) / 1000).toFixed(0)}ms stretch`, ...[...hitch].sort((a, b) => b[1] - a[1]).slice(0, 45).map(([k, v]) => `${(v / 1000).toFixed(0)}ms ${k}`)];
    const top = m => [...m].sort((a, b) => b[1] - a[1]).slice(0, 25).map(([k, v]) => `${(v / 1000).toFixed(0)}ms ${k}`);

    out['heap_' + name] = await heap();
  }
  await page.evaluate(() => { window.__map.zoomTo(8, { duration: 500 }); });
  await page.waitForTimeout(1500);
  out.heapBack = await heap();
  out.results = results;
  return out;
}
