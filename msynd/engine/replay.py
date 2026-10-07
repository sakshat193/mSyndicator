from dataclasses import dataclass, field
import math

import numpy as np

from msynd.dist import Eff, LEVELS, QUANTILES
from msynd.engine import kernel
from msynd.engine.evidence import RunEvidence
from msynd.rng import RESAMPLE, uniform


@dataclass(frozen=True)
class ModelTables:
    r: np.ndarray
    r_tau: np.ndarray
    bias: np.ndarray
    w: np.ndarray
    w_tau: np.ndarray
    o: np.ndarray
    o_tau: np.ndarray
    cold_start: bool


@dataclass(frozen=True)
class Params:
    n: int = 10_000
    k: int = 32
    v_max: float = 160.0
    ess_min: float = .05
    # Survival grid includes eps; beyond it use eps + (last-eps)*exp(-excess/tau).
    s_tables: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.n, int) or not isinstance(self.k, int) or not 0 < self.k <= self.n or 4*self.n > RESAMPLE:
            raise ValueError("require integer 0 < k <= n within RNG counter range")
        if not math.isfinite(self.v_max) or self.v_max <= 0 or not 0 < self.ess_min <= 1:
            raise ValueError("invalid speed or ESS threshold")


@dataclass
class ReplayOut:
    halt_q: np.ndarray
    next_q: np.ndarray
    paths: np.ndarray
    min_ess: float
    surprise: bool
    flags: tuple[str, ...]
    pits: tuple[float, ...]


def _tables(models):
    r, rtau, bias, w, wtau, o, otau = (
        np.ascontiguousarray(x, dtype=np.float64) for x in
        (models.r, models.r_tau, models.bias, models.w, models.w_tau, models.o, models.o_tau))
    if r.ndim != 3 or r.shape[1:] != (len(LEVELS), 302) or w.ndim != 3 or w.shape[1:] != (len(LEVELS), 181):
        raise ValueError("invalid R/W model table shapes")
    if rtau.shape != (len(r), 2) or bias.shape != (len(r),) or wtau.shape != (len(w),) or o.shape != (302,) or otau.shape != (2,):
        raise ValueError("invalid model scales, bias or origin shape")
    if not np.isfinite(bias).all():
        raise ValueError("nonfinite bias")
    # Reuse Eff's probability and tail validation and cumulative tables.
    re = [Eff(m, True, *tau, 0, 0, 0, 1, 0, 0, 1) for m, tau in zip(r, rtau)]
    we = [Eff(m, False, 1, tau, 0, 0, 0, 1, 0, 0, 1) for m, tau in zip(w, wtau)]
    oe = Eff(o, True, *otau, 0, 0, 0, 1, -np.inf, 0, 1)
    return (r, np.array([e.args[1] for e in re]), np.array([e.args[2] for e in re]), rtau, bias,
            w, np.array([e.args[1] for e in we]), np.array([e.args[2] for e in we]), wtau,
            oe.args[0], oe.args[1], oe.args[2], otau)


def _route(ev, models):
    events = ev.events
    if not events or events[0].kind != "dep" or [e.idx for e in events] != list(range(len(events))):
        raise ValueError("require canonical events starting at origin departure")
    kind = np.array([{"arr": 0, "pass": 1, "dep": 2}[e.kind] for e in events], dtype=np.int64)
    sched, km = (np.array([getattr(e, attr) for e in events], dtype=np.float64) for attr in ("sched_off", "km"))
    if not np.isfinite(sched).all() or not np.isfinite(km).all() or np.any(np.diff(sched) < 0) or np.any(np.diff(km) < 0):
        raise ValueError("route times and distances must be finite and nondecreasing")
    seg = np.array([e.seg for e in events], dtype=np.int64)
    halt = np.array([-1 if e.halt_idx is None else e.halt_idx for e in events], dtype=np.int64)
    if np.any(halt >= len(models.w)) or np.any(halt < -1) or np.any((kind != 1) & (halt < 0)):
        raise ValueError("halt index outside W table")
    anchor = np.arange(len(events), dtype=np.int64)-1
    dest = np.zeros(len(events), dtype=np.int64)
    departure = np.zeros(len(events), dtype=np.int64)
    departures = {e.seg: e.idx for e in events if e.kind == "dep"}
    arrivals = {e.seg: e.idx for e in events if e.kind == "arr"}
    for e in events:
        if e.kind == "dep":
            continue
        if e.seg not in arrivals or e.seg not in departures or not 0 <= e.seg < len(models.r):
            raise ValueError("running event without segment model/endpoints")
        dest[e.idx], departure[e.idx] = arrivals[e.seg], departures[e.seg]
        if sched[dest[e.idx]] <= sched[departure[e.idx]]:
            raise ValueError("scheduled running time must be positive")
    return kind, sched, km, seg, halt, anchor, dest, departure


def _lag_table(table):
    grid, values = (np.asarray(table[name], dtype=np.float64) for name in ("lag", "survival"))
    tau, eps = float(table["tau"]), float(table["eps"])
    if (grid.ndim != 1 or len(grid) < 1 or values.shape != grid.shape or grid[0] != 0 or
        not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0) or not np.isfinite(values).all() or
        values[0] != 1 or np.any(np.diff(values) > 0) or np.any(values < eps) or np.any(values > 1) or
        not math.isfinite(tau) or tau <= 0 or not 0 <= eps <= 1):
        raise ValueError("invalid publication-lag survival table")
    return grid, values, tau, eps


def replay(ev: RunEvidence, models: ModelTables, params: Params, key: int) -> ReplayOut:
    if not 0 <= key <= np.iinfo(np.uint64).max:
        raise ValueError("key outside uint64")
    key = np.uint64(key)
    tables, route = _tables(models), _route(ev, models)
    stop = len(ev.events)-1 if ev.stop_event is None else ev.stop_event
    evidence = sorted(ev.evidence, key=lambda e: e.event)
    if not 0 <= stop < len(ev.events) or len({e.event for e in evidence}) != len(evidence):
        raise ValueError("invalid stop or duplicate evidence")
    if any(not 0 <= e.event <= stop or not math.isfinite(e.minute) for e in evidence):
        raise ValueError("evidence outside simulated route or nonfinite minute")
    lag_tables = {source: _lag_table(table) for source, table in params.s_tables.items()}
    unused_lag = (np.array([0.]), np.array([1.]), 1.0, 0.0)
    times = np.full((params.n, len(ev.events)), np.nan, dtype=np.float32)
    nxt, min_ess, invariant = 0, 1.0, False
    pits = []
    for observed in evidence:
        lag = unused_lag if observed.actual else lag_tables[observed.source]
        for attempt in (0, 1):
            n = params.n if attempt == 0 else 4*params.n
            proposal = times[np.arange(n) % params.n]
            proposal, logw, pit = kernel.block(proposal, nxt, observed.event, observed.actual,
                observed.minute, lag, route, tables, params.v_max, ev.resched_min, key, attempt)
            maximum = np.max(logw)
            weights = np.exp(logw-maximum) if np.isfinite(maximum) else np.zeros(n)
            total = weights.sum()
            ess = total**2/np.dot(weights, weights) if total else 0.0
            min_ess = min(min_ess, ess/params.n)
            if ess/params.n >= params.ess_min or attempt == 1:
                break
        if observed.actual:
            pits.append(float(pit))
        if total == 0:
            if not observed.actual:
                raise ValueError("zero-weight pending evidence")
            proposal[:, observed.event] = observed.minute+.5
            times = proposal[:params.n]
            invariant = True
        else:
            indices = kernel.systematic(weights/total, params.n, uniform(key, RESAMPLE, observed.event, attempt))
            times = proposal[indices]
        nxt = observed.event+1
    kernel.draw(times, nxt, stop, route, tables, params.v_max, ev.resched_min, key)
    last_actual = max((e.event for e in evidence if e.actual), default=-1)
    halt_q = np.full((len(models.w), 2, 7), np.nan)
    next_q = np.full(7, np.nan)
    for e in ev.events:
        if last_actual < e.idx <= stop and e.kind != "pass":
            q = np.quantile(times[:, e.idx], QUANTILES[1:-1])
            halt_q[e.halt_idx, 0 if e.kind == "arr" else 1] = q
            if e.kind == "arr" and np.isnan(next_q).all():
                next_q = q.copy()
    flags = tuple(name for name, enabled in (("data_invariant", invariant), ("cold_start", models.cold_start)) if enabled)
    return ReplayOut(halt_q, next_q, times[np.arange(params.k)*params.n//params.k], min_ess,
                     any(p < .001 or p > .999 for p in pits), flags, tuple(pits))
