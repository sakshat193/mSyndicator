import { expect, it } from 'vitest';
import { initHud, liveStatus, trainCount } from './hud';

it.each([
  [null, 'Connecting…', 'pending'],
  [0, 'updated 0 s ago', 'fresh'],
  [59, 'updated 59 s ago', 'fresh'],
  [60, 'updated 60 s ago', 'aging'],
  [179, 'updated 179 s ago', 'aging'],
  [180, 'updated 180 s ago', 'old'],
])('shows age %s as %s with freshness %s', (age, text, freshness) => {
  expect(liveStatus(age as number | null)).toEqual({ text, freshness });
});

it('counts drawn trains with singular/plural text and a live/timetable title', () => {
  expect(trainCount([])).toEqual({ text: '0 trains', title: '0 live · 0 by timetable' });
  expect(trainCount([{ run: { state: 'observed' } }]))
    .toEqual({ text: '1 train', title: '1 live · 0 by timetable' });
  const displays = Array.from({ length: 3171 }, (_, i) => ({
    run: { state: i < 4 ? 'observed' : i === 4 ? 'stale' : 'schedule_only' },
  }));
  expect(trainCount(displays)).toEqual({ text: '3,171 trains', title: '5 live · 3,166 by timetable' });
});

function setup(initial: string | null = null, blocked = false) {
  const listeners = new Map<string, EventListener>();
  const attributes = new Map<string, string>();
  const button = {
    setAttribute: (name: string, value: string) => { attributes.set(name, value); },
    addEventListener: (name: string, listener: EventListener) => { listeners.set(name, listener); },
    removeEventListener: (name: string) => { listeners.delete(name); },
  };
  const doc = {
    body: { dataset: { hud: undefined as string | undefined } },
    querySelector: (selector: string) => selector === '#hud-toggle' ? button : null,
    addEventListener: (name: string, listener: EventListener) => { listeners.set(name, listener); },
    removeEventListener: (name: string) => { listeners.delete(name); },
  };
  const values = new Map<string, string>();
  if (initial !== null) values.set('msynd.hud', initial);
  const storage = {
    getItem: (key: string) => {
      if (blocked) throw new Error('Storage blocked');
      return values.get(key) ?? null;
    },
    setItem: (key: string, value: string) => {
      if (blocked) throw new Error('Storage blocked');
      values.set(key, value);
    },
  };
  return { doc, storage, values, attributes, listeners,
    click: () => listeners.get('click')!(new Event('click')),
    key: (event: Partial<KeyboardEvent>) => listeners.get('keydown')!(event as KeyboardEvent) };
}

it.each([['hidden', 'hidden', 'true'], ['shown', 'shown', 'false'],
  [null, 'shown', 'false'], ['invalid', 'shown', 'false']])
('restores %s storage as %s with aria-pressed=%s', (initial, state, pressed) => {
  const { doc, storage, attributes } = setup(initial);
  initHud(doc, storage);
  expect(doc.body.dataset.hud).toBe(state);
  expect(attributes.get('aria-pressed')).toBe(pressed);
});

it('click and H toggle visibility, persistence and aria-pressed', () => {
  const { doc, storage, values, attributes, click, key } = setup();
  initHud(doc, storage);
  click();
  expect(doc.body.dataset.hud).toBe('hidden');
  expect(values.get('msynd.hud')).toBe('hidden');
  expect(attributes.get('aria-pressed')).toBe('true');
  key({ key: 'H' });
  expect(doc.body.dataset.hud).toBe('shown');
  expect(values.get('msynd.hud')).toBe('shown');
  expect(attributes.get('aria-pressed')).toBe('false');
  key({ key: 'h' });
  expect(doc.body.dataset.hud).toBe('hidden');
  expect(values.get('msynd.hud')).toBe('hidden');
  expect(attributes.get('aria-pressed')).toBe('true');
});

it.each([
  { key: 'h', target: { tagName: 'INPUT', id: 'search' } },
  { key: 'h', target: { tagName: 'TEXTAREA' } },
  { key: 'h', ctrlKey: true }, { key: 'h', metaKey: true },
  { key: 'h', altKey: true }, { key: 'h', shiftKey: true }, { key: 'Escape' },
])('ignores shortcut when typing, modified, or a different key: %j', event => {
  const { doc, storage, values, attributes, key } = setup();
  initHud(doc, storage);
  key(event as Partial<KeyboardEvent>);
  expect(doc.body.dataset.hud).toBe('shown');
  expect(values.has('msynd.hud')).toBe(false);
  expect(attributes.get('aria-pressed')).toBe('false');
});

it('defaults to shown and still toggles when storage read and write throw', () => {
  const { doc, storage, attributes, click, key } = setup('hidden', true);
  initHud(doc, storage);
  expect(doc.body.dataset.hud).toBe('shown');
  click();
  expect(doc.body.dataset.hud).toBe('hidden');
  expect(attributes.get('aria-pressed')).toBe('true');
  key({ key: 'h' });
  expect(doc.body.dataset.hud).toBe('shown');
  expect(attributes.get('aria-pressed')).toBe('false');
});

it('removes its listeners on dispose', () => {
  const { doc, storage, listeners } = setup();
  const dispose = initHud(doc, storage);
  dispose();
  expect(listeners.size).toBe(0);
});
