import { decode } from '@msgpack/msgpack';

export interface RunFrame {
  run_id: number;
  paths?: Uint8Array;
  path_shape?: [number, number, number];
  base_ts?: number;
  [key: string]: unknown;
}

export interface Frame {
  gen: number;
  ts: number;
  is_full: boolean;
  versions: Record<string, unknown>;
  base_ts: number | null;
  runs: Record<string, RunFrame>;
  tombstones: number[];
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    && !(value instanceof Uint8Array);
}

function id(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function validate(value: unknown): asserts value is Frame {
  if (!record(value) || !id(value.gen) || !Number.isFinite(value.ts)
    || typeof value.is_full !== 'boolean' || !record(value.versions)
    || !(value.base_ts === null || Number.isFinite(value.base_ts))
    || !record(value.runs) || !Array.isArray(value.tombstones)
    || !value.tombstones.every(id)) throw new Error('Invalid frame');
  for (const [key, run] of Object.entries(value.runs)) {
    if (!record(run) || !id(run.run_id) || String(run.run_id) !== key)
      throw new Error('Invalid run');
    if ('paths' in run) {
      const shape = run.path_shape;
      if (!(run.paths instanceof Uint8Array) || !Array.isArray(shape)
        || shape.length !== 3 || !shape.every(id) || shape[2] !== 2
        || shape[0] * shape[1] * 4 !== run.paths.byteLength
        || !Number.isFinite(run.base_ts)) throw new Error('Invalid display paths');
    }
  }
}

export class FrameStore {
  gen = 0;
  runs: Record<string, RunFrame> = {};
  latest: Frame | null = null;

  apply(frame: Frame): 'ok' | 'gap' {
    validate(frame);
    if (!frame.is_full && frame.gen !== this.gen + 1) return 'gap';
    const runs = frame.is_full ? { ...frame.runs } : { ...this.runs, ...frame.runs };
    for (const key of frame.tombstones) delete runs[key];
    this.runs = runs;
    this.gen = frame.gen;
    this.latest = frame;
    return 'ok';
  }
}

/** Returns a disconnect function that also cancels pending reconnects. */
export function connectLive(url: string, store: FrameStore): () => void {
  let stopped = false;
  let socket: WebSocket;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let delay = 1000;

  function connect() {
    const ws = new WebSocket(url);
    socket = ws;
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      if (!stopped && socket === ws) ws.send(JSON.stringify({ gen: store.gen }));
    };
    ws.onmessage = event => {
      if (stopped || socket !== ws) return;
      try {
        if (!(event.data instanceof ArrayBuffer)) throw new Error('Expected binary frame');
        const frame = decode(new Uint8Array(event.data));
        validate(frame);
        if (store.apply(frame) === 'gap') ws.send(JSON.stringify({ gen: store.gen }));
        else delay = 1000;
      } catch {
        ws.close();
      }
    };
    ws.onerror = () => ws.close();
    ws.onclose = () => {
      ws.onopen = ws.onmessage = ws.onerror = null;
      if (stopped || socket !== ws || timer !== undefined) return;
      timer = setTimeout(() => { timer = undefined; connect(); }, delay);
      delay = Math.min(delay * 2, 30_000);
    };
  }
  connect();
  return () => {
    stopped = true;
    clearTimeout(timer);
    socket.close();
  };
}
