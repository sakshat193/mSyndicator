import math

from numba import njit
import numpy as np

from msynd.dist import _cdf, _eff_q, _sample, _window, level_weight
from msynd.rng import uniform


@njit
def eff_for(e, row, route, tables, v_max, resched):
    kind, sched, km, seg, halt, anchor, dest, departure = route
    r, rb, ra, rtau, bias, w, wb, wa, wtau, o, ob, oa, otau = tables
    if e == 0:
        return (o, ob, oa, True, otau[0], otau[1], 0, 0.0,
                sched[e], 1.0, -math.inf, float(resched), 1.0)
    p = anchor[e]
    j, weight = level_weight(float(row[p])-sched[p])
    if kind[e] == 2:  # intermediate departure: NONNEG dwell
        h = halt[e]
        return (w[h], wb[h], wa[h], False, 1.0, wtau[h], j, weight,
                float(row[p]), 1.0, 0.0, 0.0, 1.0)
    end, segment = dest[e], seg[e]
    sched_run = sched[end]-sched[departure[e]]
    # ponytail: floor avoids s=0 when a pass shares B's minute; upgrade: finer published pass times.
    sched_rem = max(0.1, sched[end]-sched[p])
    scale = sched_rem/sched_run
    a = min(1.0, max(0.0, (sched[e]-sched[p])/sched_rem)) if kind[e] == 1 else 1.0
    floor = 60*(km[end]-km[p])/v_max
    return (r[segment], rb[segment], ra[segment], True, rtau[segment, 0], rtau[segment, 1], j, weight,
            float(row[p]), a, floor, sched_rem+scale*bias[segment], scale)


@njit
def prior(eff, u):
    return _eff_q(eff, 1-u if u > .5 else u, u > .5)


@njit
def draw(times, start, stop, route, tables, v_max, resched, key, attempt=0):
    for e in range(start, stop+1):
        for i in range(len(times)):
            eff = eff_for(e, times[i], route, tables, v_max, resched)
            times[i, e] = prior(eff, uniform(key, i, e, attempt))


@njit
def survival(lag, table):
    grid, values, tau, eps = table
    if lag <= 0:
        return 1.0
    if lag >= grid[-1]:
        return eps+(values[-1]-eps)*math.exp(-(lag-grid[-1])/tau)
    j = np.searchsorted(grid, lag, side="right")-1
    return values[j]+(values[j+1]-values[j])*(lag-grid[j])/(grid[j+1]-grid[j])


@njit
def block(times, start, e, actual, minute, lag_table, route, tables, v_max, resched, key, attempt):
    draw(times, start, e-1, route, tables, v_max, resched, key, attempt)
    logw = np.full(len(times), -math.inf)
    pit = 0.0
    for i in range(len(times)):
        eff = eff_for(e, times[i], route, tables, v_max, resched)
        u = uniform(key, i, e, attempt)
        if actual:
            weight = _window(eff, minute, minute+1)
            pit += _cdf(eff, minute)[0]+.5*weight
            if weight > 0:
                value = np.float32(_sample(eff, minute, minute+1, u))
                # Storage must preserve the observation's half-open window too.
                if value >= minute+1:
                    value = np.nextafter(np.float32(minute+1), np.float32(-math.inf))
                if value < minute:
                    value = np.nextafter(np.float32(minute), np.float32(math.inf))
                times[i, e] = value
        else:
            value = prior(eff, u)
            times[i, e] = value
            weight = survival(minute-value, lag_table)
        if weight > 0:
            logw[i] = math.log(weight)
    return times, logw, pit/len(times)


@njit
def systematic(weights, n, u):
    indices = np.empty(n, dtype=np.int64)
    i, cumulative = 0, weights[0]
    for j in range(n):
        point = (u+j)/n
        while point >= cumulative and i < len(weights)-1:
            i += 1
            cumulative += weights[i]
        indices[j] = i
    return indices
