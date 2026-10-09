/** Wall clock milliseconds in, simulation UTC seconds out. */
export const liveTime = (now = Date.now()): number => now / 1000 - 15;

export function zone(simTime: number, now: number): 'past' | 'recent' | 'future' {
  if (simTime > now) return 'future';
  return simTime < now - 30 * 60 ? 'past' : 'recent';
}
