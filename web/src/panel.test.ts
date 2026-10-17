import { afterEach, expect, it, vi } from 'vitest';
import { delayText, evidenceText, haltRows, hoverText, runPanel } from './panel';
import type { Network } from './layers';

afterEach(() => vi.unstubAllGlobals());

it('toggles train details and keeps the choice across closing and selecting another run', async () => {
  const node = (): any => ({ hidden: true, dataset: {}, attributes: {}, listeners: {}, children: [],
    textContent: '', scrollTop: 0,
    setAttribute(name: string, value: string) { this.attributes[name] = value; },
    addEventListener(name: string, callback: Function) { this.listeners[name] = callback; },
    append(...children: any[]) { this.children.push(...children); },
    replaceChildren(...children: any[]) { this.children = children; } });
  const element = node(), close = node(), toggle = node(), content = node();
  toggle.setAttribute('aria-expanded', 'true');
  toggle.setAttribute('aria-label', 'Hide train details');
  element.querySelector = (selector: string) =>
    ({ '[data-close]': close, '[data-toggle]': toggle, '[data-content]': content })[selector];
  vi.stubGlobal('document', { createElement: node });
  vi.stubGlobal('fetch', async () => ({ ok: true, json: async () => ({
    train: '12951', start_date: '2026-10-18', schedule: [], halts: [], revisions: [],
  }) }));
  const onClose = vi.fn(), onToggle = vi.fn();
  const panel = runPanel(element, () => network, onClose, onToggle);
  expect(element.dataset.collapsed).toBeUndefined();
  expect(onToggle).not.toHaveBeenCalled();
  toggle.listeners.click?.();
  expect(element.dataset.collapsed).toBe('');
  expect(toggle.attributes['aria-expanded']).toBe('false');
  expect(toggle.attributes['aria-label']).toBe('Show train details');
  expect(onToggle).toHaveBeenLastCalledWith(true);

  const first = panel.show({ run_id: 1, versions: { route: 7 }, origin_ts: 1000 }, 1200);
  expect(content.children[0].textContent).toBe('Loading train…');
  expect(element.dataset.collapsed).toBe('');
  await first;
  expect(content.children[0].children[0].textContent).toBe('12951 · 18 Oct');
  expect(content.children[0].children[1].textContent).toBe('Mumbai Rajdhani');
  close.listeners.click();
  expect(onClose).toHaveBeenCalledOnce();
  panel.close();
  expect(element.hidden).toBe(true);
  await panel.show({ run_id: 2, versions: { route: 7 }, origin_ts: 1000 }, 1200);
  expect(element.hidden).toBe(false);
  expect(element.dataset.collapsed).toBe('');
  expect(toggle.attributes['aria-expanded']).toBe('false');
  expect(toggle.attributes['aria-label']).toBe('Show train details');
  expect(onToggle).toHaveBeenCalledTimes(1);
  toggle.listeners.click?.();
  expect(element.dataset.collapsed).toBeUndefined();
  expect(toggle.attributes['aria-expanded']).toBe('true');
  expect(toggle.attributes['aria-label']).toBe('Hide train details');
  expect(onToggle.mock.calls).toEqual([[true], [false]]);
});

it.each([[-3, '−3 min'], [0, 'On time'], [1, 'On time'], [12, '+12 min'],
  [-2, 'On time'], [2, 'On time']])('formats a %s minute delay', (delay, text) => {
  expect(delayText(delay as number)).toBe(text);
});

it.each([
  ['observed', 'Live · updated 5 min ago'], ['stale', 'Stale · 5 min'],
  ['schedule_only', 'Schedule only'],
])('describes %s evidence using the last received time', (state, text) => {
  expect(evidenceText({ run_id: 1, state, headline: { last_received: 1000 } }, 1300)).toBe(text);
});

const network: Network = { stations: [
  { code: 'A', name: 'MUMBAI CENTRAL', lon: 70, lat: 20 },
  { code: 'B', name: 'NEW DELHI', lon: 72, lat: 22 }], segments: [],
  routes: [{ id: 7, train: '12951', name: 'Mumbai Rajdhani', class: 'RAJ', stops: [
    { idx: 0, station: 'A', halt: true, dist_km: 0, sched_off: 0 },
    { idx: 1, station: 'B', halt: true, dist_km: 100, sched_off: 10 }], segments: [] }] };

it.each([
  ['observed', 'Live · updated 5 min ago'], ['stale', 'Stale · 5 min'],
  ['schedule_only', 'Schedule only'],
])('shows identity, endpoints, delay and %s evidence on hover', (state, evidence) => {
  expect(hoverText({ run_id: 1, state, versions: { route: 7 },
    headline: { last_received: 1000 } }, network, 12, 1300)).toEqual({
    title: '12951 · Mumbai Rajdhani', name: 'Mumbai Rajdhani', route: 'Mumbai Central → New Delhi',
    delay: '+12 min', evidence,
  });
});

it('keeps a readable run title when route metadata is unavailable', () => {
  expect(hoverText({ run_id: 42, state: 'schedule_only' }, network, 0, 1300)).toMatchObject({
    title: 'Run 42', name: 'Run 42', route: '', evidence: 'Schedule only',
  });
});

it('merges halt events, uses origin departure, and joins actuals by station and occurrence', () => {
  const schedule = [
    { idx: 0, station: 'A', occ: 1, arr_off: null, dep_off: 0 },
    { idx: 1, station: 'B', occ: 1, arr_off: 10, dep_off: 12 },
    { idx: 2, station: 'A', occ: 2, arr_off: 30, dep_off: 32 },
    { idx: 3, station: 'C', occ: 1, arr_off: 40, dep_off: null },
  ];
  const halts = [
    { idx: 0, station: 'A', occ: 1, arr: Array(7).fill(null), dep: Array(7).fill(1) },
    { idx: 1, station: 'B', occ: 1, arr: [11, 12, 13, 14, 15, 16, 17], dep: Array(7).fill(18) },
    { idx: 2, station: 'A', occ: 2, arr: Array(7).fill(35), dep: Array(7).fill(37) },
  ];
  const revisions = [
    { station: 'A', occ: 1, kind: 'dep', actual_ts: new Date(1060 * 1000).toISOString(),
      received_at: new Date(1070 * 1000).toISOString() },
    { station: 'A', occ: 2, kind: 'arr', actual_ts: new Date(3100 * 1000).toISOString(),
      received_at: new Date(3200 * 1000).toISOString() },
  ];
  const rows = haltRows(schedule, halts, revisions, 1000, 1200);
  expect(rows).toHaveLength(4);
  expect(rows[0]).toMatchObject({ kind: 'dep', scheduled: 1000, quantiles: [1060, 1060, 1060],
    actual: 1060, status: 'past', visit: null });
  expect(rows[1]).toMatchObject({ kind: 'arr', scheduled: 1600, quantiles: [1720, 1840, 1960],
    actual: null, status: 'next', visit: null, delay: 4 });
  expect(rows[2]).toMatchObject({ station: 'A', occ: 2, actual: null, status: 'future', visit: '2nd visit' });
  expect(rows[3]).toMatchObject({ quantiles: [null, null, null], status: 'future' });
});
