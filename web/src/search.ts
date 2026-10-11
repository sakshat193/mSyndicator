import type { RunFrame } from './frames';

type Train = { number: string; name: string; class: string };
type Run = { id: number; start_date: string; lifecycle: string };

export function pickLiveRun(runs: Run[], live: Record<string, RunFrame>): RunFrame | 'Not running now' {
  const latest = runs.filter(r => live[r.id]).sort((a, b) => b.start_date.localeCompare(a.start_date))[0];
  return latest ? live[latest.id] : 'Not running now';
}

export function initSearch(input: HTMLInputElement, results: HTMLElement,
                           getLive: () => Record<string, RunFrame>, select: (run: RunFrame) => void) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let request: AbortController | undefined;
  let trains: Train[] = [];
  let active = -1;
  let buttons: HTMLButtonElement[] = [];

  function close() {
    clearTimeout(timer);
    request?.abort();
    results.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
    active = -1;
  }
  function highlight(index: number) {
    active = index;
    buttons.forEach((button, i) => button.setAttribute('aria-selected', String(i === active)));
    if (buttons[active]) {
      input.setAttribute('aria-activedescendant', buttons[active].id);
      buttons[active].scrollIntoView({ block: 'nearest' });
    }
  }
  function message(text: string) {
    const line = document.createElement('p');
    line.className = 'search-message';
    line.textContent = text;
    line.setAttribute('role', 'status');
    results.replaceChildren(line);
    results.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  }
  async function choose(index: number) {
    const train = trains[index], button = buttons[index];
    if (!train || !button) return;
    request?.abort();
    const current = new AbortController();
    request = current;
    button.textContent = `${train.number} · ${train.name} · Loading…`;
    try {
      const response = await fetch(`/api/train/${encodeURIComponent(train.number)}`, { signal: current.signal });
      if (current.signal.aborted) return;
      if (response.status === 404) { button.textContent = `${train.number} · ${train.name} · Not running now`; return; }
      if (!response.ok) throw new Error('Request failed');
      const data = await response.json();
      if (current.signal.aborted) return;
      if (!Array.isArray(data.runs) || !data.runs.every((r: Run) => r && Number.isSafeInteger(r.id)
        && typeof r.start_date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(r.start_date)))
        throw new Error('Invalid runs');
      const run = pickLiveRun(data.runs, getLive());
      if (run === 'Not running now') button.textContent = `${train.number} · ${train.name} · Not running now`;
      else { select(run); close(); }
    } catch {
      if (!current.signal.aborted) button.textContent = `${train.number} · ${train.name} · Unable to load. Try again`;
    }
  }
  async function search(query: string) {
    const current = new AbortController();
    request = current;
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`, { signal: current.signal });
      if (!response.ok) throw new Error('Request failed');
      const data = await response.json();
      if (current.signal.aborted) return;
      if (!Array.isArray(data.trains) || !data.trains.every((t: Train) => t && typeof t.number === 'string'
        && typeof t.name === 'string')) throw new Error('Invalid trains');
      trains = data.trains.slice(0, 8);
      active = -1;
      input.removeAttribute('aria-activedescendant');
      buttons = trains.map((train, i) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.id = `train-result-${i}`;
        button.setAttribute('role', 'option');
        button.setAttribute('aria-selected', 'false');
        button.textContent = `${train.number} · ${train.name}`;
        button.addEventListener('click', () => { void choose(i); });
        return button;
      });
      if (!buttons.length) message('No trains found');
      else { results.replaceChildren(...buttons); results.hidden = false; input.setAttribute('aria-expanded', 'true'); }
    } catch {
      if (!current.signal.aborted) message('Search unavailable. Try again');
    }
  }
  input.addEventListener('input', () => {
    close(); trains = []; buttons = [];
    const query = input.value.trim().slice(0, 100);
    if (query) timer = setTimeout(() => { message('Searching…'); void search(query); }, 250);
  });
  input.addEventListener('keydown', event => {
    if (event.key === 'Escape') { close(); return; }
    if (results.hidden || !buttons.length) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      highlight((active + (event.key === 'ArrowDown' ? 1 : active < 0 ? 0 : -1) + buttons.length) % buttons.length);
    } else if (event.key === 'Enter') { event.preventDefault(); void choose(active < 0 ? 0 : active); }
  });
  input.addEventListener('focus', () => {
    if (buttons.length) { results.hidden = false; input.setAttribute('aria-expanded', 'true'); }
  });
  results.addEventListener('focusout', event => {
    if (!results.contains(event.relatedTarget as Node | null) && event.relatedTarget !== input) close();
  });
  input.addEventListener('blur', event => { if (!results.contains(event.relatedTarget as Node | null)) close(); });
  return close;
}
