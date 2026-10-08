/** Port of msynd.pos.pos; all times are in the same run clock. */
export function pos(t: number, t_dep_a: number, anchor_t: number,
  anchor_km: number, t_arr_b: number, km_a: number, km_b: number,
  pattern: [number, number][] | null): number {
  if (t < t_dep_a) return km_a;
  if (t >= t_arr_b) return km_b;
  if (t <= anchor_t) return anchor_km;

  let fraction = (t - anchor_t) / (t_arr_b - anchor_t);
  if (pattern?.length) {
    let lo = 0, hi = pattern.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (fraction < pattern[mid][0]) hi = mid;
      else lo = mid + 1;
    }
    const [t0, k0] = lo ? pattern[lo - 1] : [0, 0];
    const [t1, k1] = lo < pattern.length ? pattern[lo] : [1, 1];
    fraction = t1 === t0 ? k1 : k0 + (k1 - k0) * (fraction - t0) / (t1 - t0);
  }
  return anchor_km + (km_b - anchor_km) * fraction;
}
