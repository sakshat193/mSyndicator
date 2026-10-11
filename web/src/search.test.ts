import { expect, it } from 'vitest';
import { pickLiveRun } from './search';

it('picks the latest start date among runs in the live store, regardless of API order', () => {
  const live = { 10: { run_id: 10 }, 11: { run_id: 11 }, 12: { run_id: 12 } };
  const runs = [{ id: 11, start_date: '2026-10-11', lifecycle: 'active' },
    { id: 99, start_date: '2026-10-12', lifecycle: 'active' },
    { id: 10, start_date: '2026-10-10', lifecycle: 'active' },
    { id: 12, start_date: '2026-10-09', lifecycle: 'active' }];
  expect(pickLiveRun(runs, live)).toBe(live[11]);
});

it('reports Not running now when no returned run is in the live store', () => {
  expect(pickLiveRun([{ id: 99, start_date: '2026-10-11', lifecycle: 'active' }], {})).toBe('Not running now');
  expect(pickLiveRun([], { 10: { run_id: 10 } })).toBe('Not running now');
});
