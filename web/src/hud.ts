export function liveStatus(age: number | null) {
  return {
    text: age === null ? 'Connecting…' : `updated ${age} s ago`,
    freshness: age === null ? 'pending' : age < 60 ? 'fresh' : age < 180 ? 'aging' : 'old',
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
