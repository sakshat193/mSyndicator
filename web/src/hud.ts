export function sourceStatus(pollAgeS: number | null, harvestAgeS: number | null,
  frameAgeS: number | null, dispatcher?: string) {
  const ageText = (age: number | null) => age === null ? 'unknown' :
    `${age < 120 ? `${Math.floor(age)} s` : `${Math.floor(age / 60)} min`} ago`;
  const suffix = dispatcher === 'breaker' ? ' · paused' : dispatcher === 'stopped' ? ' · stopped' : '';
  return {
    text: `live data ${ageText(pollAgeS)}${suffix}`,
    freshness: pollAgeS === null ? 'unknown' : pollAgeS < 300 ? 'fresh' : pollAgeS < 900 ? 'aging' : 'old',
    title: `frame ${ageText(frameAgeS)} · harvest ${ageText(harvestAgeS)}`,
  };
}

export function trainCount(displays: readonly { run: Record<string, unknown> }[]) {
  const total = displays.length;
  const timetable = displays.filter(d => d.run.state === 'schedule_only').length;
  return {
    text: `${total.toLocaleString('en-IN')} ${total === 1 ? 'train' : 'trains'}`,
    title: `${(total - timetable).toLocaleString('en-IN')} live · ${timetable.toLocaleString('en-IN')} by timetable`,
  };
}

type Listeners = {
  addEventListener(type: string, listener: EventListener): void;
  removeEventListener(type: string, listener: EventListener): void;
};
type HudDocument = Listeners & {
  body: { dataset: { hud?: string } };
  querySelector(selector: string): (Listeners & { setAttribute(name: string, value: string): void }) | null;
};

export function initHud(doc: HudDocument, storage: Pick<Storage, 'getItem' | 'setItem'>) {
  const button = doc.querySelector('#hud-toggle')!;
  let initial = 'shown';
  try { if (storage.getItem('msynd.hud') === 'hidden') initial = 'hidden'; } catch {}

  function show(state: string) {
    doc.body.dataset.hud = state;
    button.setAttribute('aria-pressed', String(state === 'hidden'));
  }
  function toggle() {
    const state = doc.body.dataset.hud === 'hidden' ? 'shown' : 'hidden';
    show(state);
    try { storage.setItem('msynd.hud', state); } catch {}
  }
  function onKey(event: Event) {
    const key = event as KeyboardEvent;
    const target = key.target as { tagName?: string } | null;
    if (key.key?.toLowerCase() !== 'h' || key.ctrlKey || key.metaKey || key.altKey || key.shiftKey
      || target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA') return;
    toggle();
  }
  show(initial);
  button.addEventListener('click', toggle);
  doc.addEventListener('keydown', onKey);
  return () => {
    button.removeEventListener('click', toggle);
    doc.removeEventListener('keydown', onKey);
  };
}
