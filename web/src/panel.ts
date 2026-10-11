import type { RunFrame } from './frames';
import { delayColor, indexNetwork, stationName, type Network } from './layers';

type ScheduleStop = { idx: number; station: string; occ: number; arr_off: number | null; dep_off: number | null };
type Halt = { idx: number; station: string; occ: number; arr: (number | null)[]; dep: (number | null)[] };
type Revision = { station: string; occ: number; kind: string; actual_ts: string; received_at: string };
type RunDetail = { train: string; start_date: string; evidence_state: string;
  schedule: ScheduleStop[]; halts: Halt[]; revisions: Revision[] };

export function delayText(delay: number): string {
  if (Math.abs(delay) <= 2) return 'On time';
  return `${delay < 0 ? '−' : '+'}${Math.abs(Math.round(delay))} min`;
}

export function evidenceText(run: RunFrame, now: number): string {
  const last = (run.headline as { last_received?: number | null } | undefined)?.last_received;
  const age = typeof last === 'number' && Number.isFinite(last) ? Math.max(0, Math.floor((now - last) / 60)) : null;
  if (run.state === 'observed') return age === null ? 'Live · update time unavailable' : `Live · updated ${age} min ago`;
  if (run.state === 'stale') return age === null ? 'Stale · update time unavailable' : `Stale · ${age} min`;
  if (run.state === 'schedule_only') return 'Schedule only';
  return typeof run.state === 'string' ? run.state.replaceAll('_', ' ') : 'Evidence unavailable';
}

export function hoverText(run: RunFrame, network: Network, delay: number, now: number) {
  const route = indexNetwork(network).routes.get((run.versions as { route?: number } | undefined)?.route ?? -1);
  const stops = route?.stops ?? [];
  return { title: route ? `${route.train} · ${route.name}` : `Run ${run.run_id}`,
    name: route?.name ?? `Run ${run.run_id}`,
    route: stops.length ? `${stationName(network, stops[0].station)} → ${stationName(network, stops.at(-1)!.station)}` : '',
    delay: delayText(delay), evidence: evidenceText(run, now) };
}

export function haltRows(schedule: ScheduleStop[], halts: Halt[], revisions: Revision[], origin: number, simTime: number) {
  const predictions = new Map(halts.map(h => [h.idx, h]));
  const actuals = new Map<string, { time: number; received: number }>();
  for (const r of revisions) {
    const time = Date.parse(r.actual_ts) / 1000, received = Date.parse(r.received_at) / 1000;
    const key = `${r.station}:${r.occ}:${r.kind}`;
    if (received <= simTime && Number.isFinite(time) && received >= (actuals.get(key)?.received ?? -Infinity))
      actuals.set(key, { time, received });
  }
  let next = false;
  return schedule.map((s, i) => {
    const kind = i === 0 ? 'dep' : 'arr';
    const off = kind === 'dep' ? s.dep_off : s.arr_off;
    const scheduled = off === null ? null : origin + off * 60;
    const qs = predictions.get(s.idx)?.[kind];
    const quantiles = [1, 3, 5].map(q => qs?.[q] == null ? null : origin + qs[q]! * 60);
    const actual = actuals.get(`${s.station}:${s.occ}:${kind}`)?.time ?? null;
    const past = actual !== null || (quantiles[1] ?? scheduled ?? Infinity) < simTime;
    const status = past ? 'past' : next ? 'future' : 'next';
    if (!past) next = true;
    return { ...s, kind, scheduled, quantiles, actual, status,
      visit: s.occ > 1 ? '2nd visit' : null,
      delay: quantiles[1] !== null && scheduled !== null ? (quantiles[1] - scheduled) / 60 : null };
  });
}

const clock = new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata',
  hour: '2-digit', minute: '2-digit', hour12: false });
const day = new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short' });
const time = (ts: number | null) => ts === null ? '—' : clock.format(ts * 1000);
const range = (qs: (number | null)[]) => `${time(qs[0])}–${time(qs[2])}`;

function node<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = '') {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

export function delayChip(delay: number) {
  const chip = node('span', 'delay-chip', delayText(delay));
  chip.style.setProperty('--delay-color', `rgb(${delayColor(delay).join(',')})`);
  return chip;
}

function validDetail(value: RunDetail): boolean {
  const offset = (v: unknown) => v === null || typeof v === 'number' && Number.isFinite(v);
  const stop = (s: Pick<ScheduleStop, 'idx' | 'station' | 'occ'>) => s && Number.isInteger(s.idx) && typeof s.station === 'string'
    && Number.isInteger(s.occ) && s.occ > 0;
  return value && typeof value.train === 'string' && typeof value.start_date === 'string'
    && Array.isArray(value.schedule) && value.schedule.every(s => stop(s) && offset(s.arr_off) && offset(s.dep_off))
    && Array.isArray(value.halts) && value.halts.every(h => stop(h) && [h.arr, h.dep]
      .every(q => Array.isArray(q) && q.length === 7 && q.every(offset)))
    && Array.isArray(value.revisions) && value.revisions.every(r => r && typeof r.station === 'string'
      && Number.isInteger(r.occ) && typeof r.kind === 'string'
      && Number.isFinite(Date.parse(r.actual_ts)) && Number.isFinite(Date.parse(r.received_at)));
}

export function runPanel(element: HTMLElement, getNetwork: () => Network | undefined, onClose: () => void) {
  let request: AbortController | undefined;
  let run: RunFrame | undefined;
  let detail: RunDetail | undefined;
  let signature = '';
  const closeButton = element.querySelector<HTMLButtonElement>('[data-close]')!;
  const content = element.querySelector<HTMLElement>('[data-content]')!;
  closeButton.addEventListener('click', onClose);

  function render(simTime: number) {
    const network = getNetwork();
    if (!run || !detail || !network || typeof run.origin_ts !== 'number') return;
    const rows = haltRows(detail.schedule, detail.halts, detail.revisions, run.origin_ts, simTime);
    signature = JSON.stringify(rows);
    const next = rows.find(r => r.status === 'next');
    const summary = hoverText(run, network, next?.delay ?? 0, Date.now() / 1000);
    const header = node('header', 'panel-header');
    header.append(node('p', 'run-date', `${detail.train} · ${day.format(Date.parse(`${detail.start_date}T00:00:00+05:30`))}`),
      node('h1', 'train-name', summary.name),
      node('p', 'route-name', summary.route));
    const metadata = node('div', 'train-metadata');
    if (next?.delay !== null && next?.delay !== undefined) metadata.append(delayChip(next.delay));
    const evidence = node('span', 'evidence', summary.evidence);
    evidence.dataset.evidence = '';
    metadata.append(evidence);
    header.append(metadata);
    content.replaceChildren(header);
    if (next) {
      const card = node('section', 'next-stop');
      card.append(node('p', 'section-label', 'Next stop'), node('h2', '', stationName(network, next.station)));
      const times = node('div', 'next-times');
      times.append(node('strong', '', time(next.quantiles[1])));
      if (next.delay !== null && Math.abs(next.delay) >= 2)
        times.append(node('s', 'muted', time(next.scheduled)));
      card.append(times, node('p', 'prediction-range', `P10–P90 · ${range(next.quantiles)} IST`));
      content.append(card);
    }
    const timeline = node('ol', 'stop-timeline');
    timeline.setAttribute('aria-label', 'Stop timeline');
    for (const row of rows) {
      const item = node('li', `stop ${row.status}`);
      if (row.status === 'next') item.setAttribute('aria-current', 'step');
      const heading = node('div', 'stop-heading');
      const name = node('span', 'stop-name', stationName(network, row.station));
      name.append(node('span', 'station-code', ` ${row.station}`));
      heading.append(name, node('time', 'scheduled-time', time(row.scheduled)));
      item.append(heading);
      if (row.visit) item.append(node('span', 'visit', row.visit));
      const times = node('div', 'stop-times');
      if (row.status === 'past' && row.actual !== null)
        times.append(node('span', 'actual-time', `Actual ${time(row.actual)}`));
      else if (row.status !== 'past' && row.quantiles[1] !== null) {
        const prediction = node('span', 'predicted-time', time(row.quantiles[1]));
        if (row.delay !== null) prediction.style.color = `rgb(${delayColor(row.delay).join(',')})`;
        times.append(prediction, node('span', 'prediction-range', `P10–P90 ${range(row.quantiles)}`));
      }
      times.append(node('span', 'stop-kind', row.kind === 'dep' ? 'Departure' : 'Arrival'));
      item.append(times);
      timeline.append(item);
    }
    content.append(node('h2', 'timeline-title', 'Stops · IST'), timeline);
    if (!rows.length) content.append(node('p', 'muted', 'No scheduled stops available.'));
  }

  async function show(selected: RunFrame, simTime: number) {
    request?.abort();
    const current = new AbortController();
    request = current;
    if (run?.run_id !== selected.run_id) {
      detail = undefined;
      content.replaceChildren(node('p', 'panel-message', 'Loading train…'));
      element.scrollTop = 0;
    }
    run = selected;
    element.hidden = false;
    try {
      const response = await fetch(`/api/run/${selected.run_id}`, { signal: current.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const value = await response.json();
      if (request !== current || current.signal.aborted) return;
      if (!validDetail(value) || typeof selected.origin_ts !== 'number' || !Number.isFinite(selected.origin_ts))
        throw new Error('Invalid run detail');
      detail = value;
      render(simTime);
    } catch {
      if (current.signal.aborted || request !== current) return;
      detail = undefined;
      const message = node('p', 'panel-message', 'Train details unavailable.');
      const retry = node('button', 'retry', 'Retry');
      retry.type = 'button';
      retry.addEventListener('click', () => { if (run) void show(run, Date.now() / 1000 - 15); });
      content.replaceChildren(message, retry);
    }
  }

  return { show, close() { request?.abort(); run = undefined; detail = undefined; element.hidden = true; },
    update(selected: RunFrame, simTime: number) {
      run = selected;
      if (!detail || typeof selected.origin_ts !== 'number') return;
      const rows = haltRows(detail.schedule, detail.halts, detail.revisions, selected.origin_ts, simTime);
      if (JSON.stringify(rows) !== signature) render(simTime);
      else {
        const evidence = content.querySelector<HTMLElement>('[data-evidence]');
        if (evidence) evidence.textContent = evidenceText(selected, Date.now() / 1000);
      }
    } };
}
