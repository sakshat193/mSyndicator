import { describe, expect, it } from 'vitest';
import cases from '../../tests/fixtures/pos_cases.json';
import { pos } from './position';

describe('Python position parity', () => {
  for (const { name, input, expected } of cases) {
    it(name, () => {
      const p = input;
      expect(Math.abs(pos(p.t, p.t_dep_a, p.anchor_t, p.anchor_km,
        p.t_arr_b, p.km_a, p.km_b, p.pattern as [number, number][] | null) - expected))
        .toBeLessThanOrEqual(1e-6);
    });
  }
});
