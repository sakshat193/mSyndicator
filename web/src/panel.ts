import type { RunFrame } from './frames';

type Halt = { station: string; occ: number; arr: (number | null)[]; dep: (number | null)[] };

export function haltRows(halts: Halt[], origin: number, simTime: number) {
  return halts.map(h => ({ station: h.station, occ: h.occ,
    arr: [1, 3, 5].map(i => h.arr[i] == null ? null : origin + h.arr[i]! * 60),
    dep: [1, 3, 5].map(i => h.dep[i] == null ? null : origin + h.dep[i]! * 60),
  })).filter(h => [...h.arr, ...h.dep].some(t => t !== null && t >= simTime));
}

const clock = new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata',
  day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false });

export function runPanel(element: HTMLElement) {
  let request: AbortController | undefined;
  element.querySelector('button')!.addEventListener('click', () => {
    request?.abort(); element.hidden = true;
  });
  const content = element.querySelector<HTMLElement>('[data-content]')!;
  return async (run: RunFrame, simTime: number) => {
    request?.abort();
    const current = new AbortController();
    request = current;
    element.hidden = false;
    content.textContent = `Loading run ${run.run_id}…`;
    try {
      const response = await fetch(`/api/run/${run.run_id}`, { signal: current.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const detail = await response.json();
      if (request !== current) return;
      if (!Array.isArray(detail.halts) || typeof run.origin_ts !== 'number')
        throw new Error('Invalid run detail');
      content.replaceChildren();
      const title = document.createElement('h2');
      title.textContent = `${detail.train ?? run.run_id} · ${detail.start_date ?? ''}`;
      content.append(title);
      const state = document.createElement('p');
      state.textContent = detail.evidence_state;
      content.append(state);
      const table = document.createElement('table');
      const header = table.createTHead().insertRow();
      for (const label of ['Halt / IST', 'P10', 'P50', 'P90']) {
        const th = document.createElement('th'); th.textContent = label; header.append(th);
      }
      const body = table.createTBody();
      const rows = haltRows(detail.halts, run.origin_ts, simTime);
      for (const halt of rows) for (const kind of ['arr', 'dep'] as const) {
        if (halt[kind].every(t => t === null)) continue;
        const row = body.insertRow();
        row.insertCell().textContent = `${halt.station} (${halt.occ}) ${kind}`;
        for (const t of halt[kind]) row.insertCell().textContent = t === null ? '—' : clock.format(t * 1000);
      }
      content.append(table);
      if (!rows.length) content.append('No remaining halt predictions.');
    } catch (error) {
      if (current.signal.aborted) return;
      content.textContent = `Run ${run.run_id} unavailable: ${error instanceof Error ? error.message : 'request failed'}`;
    }
  };
}
