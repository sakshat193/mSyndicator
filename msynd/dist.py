from dataclasses import dataclass, field
import math

from numba import njit
import numpy as np

SIGNED_LO, SIGNED_HI, SIGNED_NB = -60.0, 240.0, 302
NONNEG_LO, NONNEG_HI, NONNEG_NB = 0.0, 180.0, 181
LEVELS = np.array([-30,-10,0,5,10,15,20,30,45,60,90,120,180,240,360,600], dtype=float)
QUANTILES = np.array([.02,.05,.1,.25,.5,.75,.9,.95,.98])


@njit
def level_weight(d):
    if d <= LEVELS[0]:
        return 0, 0.0
    if d >= LEVELS[-1]:
        return len(LEVELS)-1, 0.0
    j = np.searchsorted(LEVELS, d, side="right")-1
    return j, (d-LEVELS[j])/(LEVELS[j+1]-LEVELS[j])


def from_quantiles(q, signed):
    q = np.sort(np.asarray(q, dtype=float))
    if q.shape != QUANTILES.shape or not np.all(np.isfinite(q)):
        raise ValueError("require nine finite quantiles")
    lo = max(.5, (q[2]-q[0])/math.log(5))
    hi = max(.5, (q[-1]-q[-3])/math.log(5))
    edges = np.arange(-60, 241) if signed else np.arange(0, 181)
    cdf = np.interp(edges, q, QUANTILES)
    lower, upper = edges < q[0], edges >= q[-1]
    cdf[lower] = .02*np.exp((edges[lower]-q[0])/lo)
    survival = 1-cdf
    survival[upper] = .02*np.exp(-(edges[upper]-q[-1])/hi)
    cdf[upper] = 1-survival[upper]
    mass = np.empty(len(edges)+1 if signed else len(edges))
    start = 1 if signed else 0
    if signed:
        mass[0] = cdf[0]
    for i in range(len(edges)-1):
        mass[start+i] = survival[i]-survival[i+1] if cdf[i] >= .5 else cdf[i+1]-cdf[i]
    if not signed:
        mass[0] += cdf[0]  # Nonnegative W labels: left-tail approximation becomes a zero-floor atom bin.
    mass[-1] = survival[-1]
    return mass, lo, hi


@dataclass
class Eff:
    masses: np.ndarray
    signed: bool
    tau_lo: float
    tau_hi: float
    j: int
    w: float
    b: float
    a: float
    f: float
    c: float
    s: float
    args: tuple = field(init=False, repr=False)

    def __post_init__(self):
        m = np.atleast_2d(np.asarray(self.masses, dtype=float))
        nb = SIGNED_NB if self.signed else NONNEG_NB
        if m.shape[1] != nb or not np.all(np.isfinite(m)) or np.any(m < 0) or not np.allclose(m.sum(axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError("invalid probability masses")
        if not 0 <= self.j < len(m) or not 0 <= self.w <= 1 or (self.w and self.j+1 >= len(m)):
            raise ValueError("invalid delay interpolation")
        if any(not math.isfinite(v) for v in (self.tau_lo,self.tau_hi,self.b,self.a,self.c,self.s)) or min(self.tau_lo,self.tau_hi,self.a,self.s) <= 0 or math.isnan(self.f) or self.f == math.inf:
            raise ValueError("invalid effective parameters")
        below = np.cumsum(m, axis=1)
        above = np.cumsum(m[:, ::-1], axis=1)[:, ::-1].copy()
        self.masses = m
        j, w = (self.j+1, 0.0) if self.w == 1 else (self.j, self.w)
        self.args = (m,below,above,self.signed,self.tau_lo,self.tau_hi,j,w,self.b,self.a,self.f,self.c,self.s)


@njit
def _raw_q(m, below, above, signed, tau_lo, tau_hi, p, upper):
    low, high = (-60.0, 240.0) if signed else (0.0, 180.0)
    offset = 1 if signed else 0
    if upper:
        if m[-1] > 0 and p <= m[-1]:
            return high + tau_hi*(math.log(m[-1])-math.log(p)) if p > 0 else math.inf
        for i in range(len(m)-2, offset-1, -1):
            if m[i] > 0 and p < above[i]:
                return low+(i-offset)+1-(p-above[i+1])/m[i]
        u = 1-p
        return low+tau_lo*(math.log(u)-math.log(m[0])) if signed and u > 0 else -math.inf
    if signed and m[0] > 0 and p < m[0]:
        return low+tau_lo*(math.log(p)-math.log(m[0])) if p > 0 else -math.inf
    for i in range(offset, len(m)-1):
        if m[i] > 0 and p < below[i]:
            previous = below[i-1] if i else 0.0
            return low+(i-offset)+(p-previous)/m[i]
    v = 1-p
    return high+tau_hi*(math.log(m[-1])-math.log(v)) if m[-1] > 0 and v > 0 else math.inf


@njit
def _raw_interp(e, p, upper):
    m, below, above, signed, lo, hi, j, w, b, a, f, c, s = e
    q = _raw_q(m[j],below[j],above[j],signed,lo,hi,p,upper)
    if w == 0:
        return q
    return (1-w)*q+w*_raw_q(m[j+1],below[j+1],above[j+1],signed,lo,hi,p,upper)


@njit
def _eff_q(e, p, upper):
    return e[8]+e[9]*max(e[10],e[11]+e[12]*_raw_interp(e,p,upper))


@njit
def _cdf(e, x):
    if x <= e[8]+e[9]*e[10]:
        return 0.0, 1.0
    y = (x-e[8])/e[9]
    if y <= e[10]:
        return 0.0, 1.0
    target = (y-e[11])/e[12]
    if target <= _raw_interp(e,0.0,False):
        return 0.0, 1.0
    if target >= _raw_interp(e,0.0,True):
        return 1.0, 0.0
    m, below, above, signed, lo, hi, j, w, b, a, f, c, s = e
    other = j+1 if w > 0 else j
    # Joint exponential tails have an exact inverse in their stable coordinate.
    upper_mass = min(m[j,-1], m[other,-1])
    if upper_mass > 0 and target >= _raw_interp(e,upper_mass,True):
        anchor = (240.0 if signed else 180.0)+hi*((1-w)*math.log(m[j,-1])+w*math.log(m[other,-1]))
        survival = math.exp((anchor-target)/hi)
        return 1-survival, survival
    lower_mass = min(m[j,0],m[other,0]) if signed else 0.0
    if lower_mass > 0:
        anchor = -60-lo*((1-w)*math.log(m[j,0])+w*math.log(m[other,0]))
        # Use the tail's left limit, not the quantile after a possible support gap.
        if target <= anchor+lo*math.log(lower_mass):
            cdf = math.exp((target-anchor)/lo)
            return cdf, 1-cdf
    upper = target > _raw_interp(e,.5,False)
    left, right = 0.0, .5
    for _ in range(60):
        mid = (left+right)/2
        q = _raw_interp(e,mid,upper)
        if (q < target) != upper:
            left = mid
        else:
            right = mid
    p = (left+right)/2
    return (1-p,p) if upper else (p,1-p)


@njit
def _window(e, low, high):
    if high <= low:
        return 0.0
    fl, sl = _cdf(e,low)
    fh, sh = _cdf(e,high)
    return max(0.0, sl-sh if fl >= .5 else fh-fl)


@njit
def _sample(e, low, high, u):
    mass = _window(e,low,high)
    if mass <= 0:
        raise ValueError("zero-mass observation window")
    fl, sl = _cdf(e,low)
    x = _eff_q(e,sl-u*mass,True) if fl >= .5 else _eff_q(e,fl+u*mass,False)
    return min(max(low,x),np.nextafter(high,-math.inf))


def eff_quantile(e, u):
    if not 0 <= u <= 1:
        raise ValueError("uniform outside [0,1]")
    return _eff_q(e.args, 1-u if u > .5 else u, u > .5)


def eff_cdf_left(e, x):
    return _cdf(e.args, x)


def eff_window_mass(e, lo, hi):
    return _window(e.args, lo, hi)


def eff_sample_window(e, lo, hi, u):
    if not 0 <= u < 1:
        raise ValueError("conditional uniform outside [0,1)")
    return _sample(e.args, lo, hi, u)
