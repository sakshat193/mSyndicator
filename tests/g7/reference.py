"""Small G7 models and independent exact / iid likelihood-weighting oracles.

The oracles use analytic bin CDFs and NumPy's RNG, never the engine probability
functions or resampling. Fixtures have delay-independent tables; this keeps
enumeration exact while Task 17 separately pins delay interpolation.
"""
from dataclasses import dataclass, replace
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from msynd.engine.evidence import Evidence, RouteEvent, RunEvidence
from msynd.engine.replay import ModelTables, Params, replay
from msynd.rng import run_key, v_hash

CASE_NAMES = (
    "a_upstream", "b_bridge", "c_pass", "d_soft_lag", "d_epsilon_upper", "d_tail_x2",
    "d_measured_eps_upper", "d_measured_tail_x2",
    "e_infeasible", "f_three_blocks", "g_degenerate", "h_rare", "i_floor_atom",
    "j_mixed_ancestors", "k_repeated_pending", "l_negative_origin", "m_rescheduled",
)


@dataclass(frozen=True)
class Case:
    name: str
    ev: RunEvidence
    models: ModelTables
    params: Params
    grid: tuple[float, ...]
    origin_cut: float
    target: int
    exact: bool = False
    independent_pass: bool = False


def _mass(values, probabilities, signed=True):
    masses = np.zeros(302 if signed else 181)
    for value, probability in zip(values, probabilities):
        masses[value+61 if signed else value] = probability
    return masses


def make_case(name):
    if name not in CASE_NAMES:
        raise ValueError(name)
    segments = 3 if name in ("b_bridge", "f_three_blocks") else 2
    origin, run, floor, resched = 0, 70, 1, 0
    o_values, o_probs = [0, 10], [.5, .5]
    r_values, first_probs = [-60, -50], [.5, .5]
    evidence, passes = [], []
    grid = (29, 32, 42, 52, 56)
    exact = name in ("a_upstream", "h_rare", "l_negative_origin", "m_rescheduled")
    table = {"lag": [0, 10], "survival": [1, .5], "tau": 80, "eps": .05}
    if name == "a_upstream":
        # Synthetic hard likelihood, with the threshold in a support gap.
        table = {"lag": [0, 1], "survival": [1, 0], "tau": 1, "eps": 0}
        evidence = [Evidence(1, False, 15, "ntes_status")]
    elif name == "b_bridge":
        evidence = [Evidence(3, True, 33, "ntes_status")]
        grid = (42, 44, 48, 54, 58)
    elif name == "c_pass":
        passes, first_probs, floor = [35], [.4, .6], 12
        evidence = [Evidence(1, True, 16, "ntes_status")]
        grid = (23, 24.75, 25.5, 26.75, 29)
    elif name.startswith("d_"):
        evidence = [Evidence(1, False, 100, "ntes_status")]
        if name == "d_epsilon_upper":
            # Synthetic pilot: 5/100 never published, 95% Wilson upper bound.
            p, z, count = .05, 1.96, 100
            table["eps"] = (p+z*z/(2*count)+z*math.sqrt(p*(1-p)/count+z*z/(4*count*count)))/(1+z*z/count)
        if name == "d_tail_x2":
            table["tau"] *= 2
        if name in ("d_measured_eps_upper", "d_measured_tail_x2"):
            measured = json.loads((Path(__file__).parents[2] / "data/params.json").read_text())["S"]["mmt_status"]
            table = {key: measured[key] for key in ("lag", "survival", "tau", "eps")}
            if name == "d_measured_eps_upper":
                table["eps"] = measured["eps_ci"][1]
            else:
                table["tau"] *= 2
    elif name in ("e_infeasible", "i_floor_atom"):
        run, floor, r_values = 10, 5, [-10, -5]
        minute = 1 if name == "e_infeasible" else 5
        evidence = [Evidence(1, True, minute, "ntes_status")]
        grid = (6, 7, 7.5, 8.5, 10) if name == "e_infeasible" else (9, 10.5, 11.5, 12.5, 15)
    elif name == "f_three_blocks":
        evidence = [Evidence(0, True, 0, "ntes_status"), Evidence(1, False, 15, "ntes_status"),
                    Evidence(3, True, 32, "ntes_status")]
        grid = (41, 43, 47, 53, 57)
    elif name in ("g_degenerate", "j_mixed_ancestors"):
        evidence = [Evidence(1, True, 10, "ntes_status")]
        grid = (19, 21, 23, 31, 34)
        if name == "g_degenerate":
            o_probs = [.01, .99]
    elif name == "h_rare":
        o_probs = [5e-7, 1-5e-7]
        evidence = [Evidence(0, True, 0, "ntes_status")]
        grid = (19, 22, 32, 42, 46)
    elif name == "k_repeated_pending":
        o_values = [0, 1]
        table = {"lag": [0], "survival": [1], "tau": 1, "eps": 0}
        evidence = [Evidence(1, False, 30, "ntes_status")]
        grid = (31, 32, 33, 41, 44)
    elif name in ("l_negative_origin", "m_rescheduled"):
        origin = 100
        resched, o_values = (60, [-11, -10]) if name == "l_negative_origin" else (20, [9, 10])
        center = 150 if name == "l_negative_origin" else 130
        grid = tuple(center+x for x in (-1.75, -.75, 0, .75, 1.75))

    events = [RouteEvent(0, "A", 1, "dep", 0, 0, origin, 0, 0)]
    distance = floor*160/60
    clock = origin
    for seg in range(segments):
        if seg == 0:
            for number, minute in enumerate(passes, 1):
                events.append(RouteEvent(len(events), f"P{number}", 1, "pass", len(events), None,
                    origin+minute, distance*.3, seg))
        clock += run
        events.append(RouteEvent(len(events), chr(66+seg), 1, "arr", len(events), seg+1,
                                 clock, (seg+1)*distance, seg))
        if seg < segments-1:
            clock += 1
            events.append(RouteEvent(len(events), chr(66+seg), 1, "dep", len(events)-1, seg+1,
                                     clock, (seg+1)*distance, seg+1))
    r = np.stack([np.tile(_mass(r_values, first_probs if i == 0 else [.4, .6]), (16, 1)) for i in range(segments)])
    w = np.tile(_mass([0, 1], [.4, .6], False), (segments+1, 16, 1))
    models = ModelTables(r, np.full((segments, 2), 7.), np.zeros(segments), w,
                         np.full(segments+1, 7.), _mass(o_values, o_probs), np.array([4., 7.]), False)
    ev = RunEvidence(1, {}, tuple(events), tuple(evidence), resched, None)
    target = 2 if name == "c_pass" else 0 if name in ("l_negative_origin", "m_rescheduled") else len(events)-1
    cut = origin+resched+(1 if name == "k_repeated_pending" else
                         -10 if name == "l_negative_origin" else 10 if name == "m_rescheduled" else 5)
    return Case(name, ev, models, Params(s_tables={"ntes_status": table}), grid, cut, target, exact)


def _survival(lag, table):
    grid, values = np.asarray(table["lag"]), np.asarray(table["survival"])
    tail = table["eps"]+(values[-1]-table["eps"])*np.exp(-np.maximum(0, lag-grid[-1])/table["tau"])
    return np.where(lag <= 0, 1, np.where(lag >= grid[-1], tail, np.interp(lag, grid, values)))


def _raw_cdf(x, masses, signed, tau, upper=False):
    low, high, offset = (-60, 240, 1) if signed else (0, 180, 0)
    below = np.concatenate(([0.], np.cumsum(masses)))
    above = np.concatenate((np.cumsum(masses[::-1])[::-1], [0.]))
    index = np.clip(np.floor(x-low).astype(np.int64)+offset, offset, len(masses)-2)
    fraction = np.clip(x-(low+index-offset), 0, 1)
    if upper:
        result = above[index+1]+masses[index]*(1-fraction)
        result = np.where(x >= high, masses[-1]*np.exp(-np.maximum(0, x-high)/tau[1]), result)
        left = masses[0]*np.exp(np.minimum(0, x-low)/tau[0]) if signed else np.zeros_like(x)
        return np.where(x < low, 1-left, result)
    result = below[index]+masses[index]*fraction
    left = masses[0]*np.exp(np.minimum(0, x-low)/tau[0]) if signed else np.zeros_like(x)
    result = np.where(x < low, left, result)
    return np.where(x >= high, 1-masses[-1]*np.exp(-np.maximum(0, x-high)/tau[1]), result)


def _raw_quantile(u, masses, signed, tau):
    low, high, offset = (-60, 240, 1) if signed else (0, 180, 0)
    cumulative = np.cumsum(masses)
    index = np.minimum(np.searchsorted(cumulative, u, side="right"), len(masses)-1)
    before = np.concatenate(([0.], cumulative[:-1]))
    # The selected bin has positive mass; zero-mass bins are skipped by searchsorted.
    value = low+index-offset+(u-before[index])/masses[index]
    if signed and masses[0] > 0:
        mask = index == 0
        value[mask] = low+tau[0]*np.log(u[mask]/masses[0])
    if masses[-1] > 0:
        mask = index == len(masses)-1
        value[mask] = high-tau[1]*np.log((1-u[mask])/masses[-1])
    return value


def _law(case, e, times):
    events, models = case.ev.events, case.models
    if e == 0:
        return np.full(len(times), events[0].sched_off), 1., -np.inf, float(case.ev.resched_min), 1., models.o, True, models.o_tau
    event, anchor = events[e], events[e-1]
    if event.kind == "dep":
        return times[:, e-1], 1., 0., 0., 1., models.w[event.halt_idx, 0], False, (1., models.w_tau[event.halt_idx])
    end = next(s for s in events[e:] if s.kind == "arr" and s.seg == event.seg)
    start = next(s for s in events if s.kind == "dep" and s.seg == event.seg)
    full = end.sched_off-start.sched_off
    remaining = max(.1, end.sched_off-anchor.sched_off)
    scale = remaining/full
    a = np.clip((event.sched_off-anchor.sched_off)/remaining, 0, 1) if event.kind == "pass" else 1.
    floor = 60*(end.km-anchor.km)/case.params.v_max
    if case.independent_pass and (event.kind == "pass" or anchor.kind == "pass"):
        # Independent proportional sub-segment model for the A7 comparison only.
        remaining = event.sched_off-anchor.sched_off
        scale, a = remaining/full, 1.
        floor = 60*(event.km-anchor.km)/case.params.v_max
    return times[:, e-1], float(a), floor, remaining+scale*models.bias[event.seg], scale, models.r[event.seg, 0], True, models.r_tau[event.seg]


def _effective_cdf(x, law, upper=False):
    b, a, floor, c, s, masses, signed, tau = law
    if a == 0:
        return (x <= b).astype(float) if upper else (x > b).astype(float)
    raw = ((x-b)/a-c)/s
    result = _raw_cdf(raw, masses, signed, tau, upper)
    return np.where(x <= b+a*floor, 1. if upper else 0., result)


def _effective_quantile(u, law):
    b, a, floor, c, s, masses, signed, tau = law
    if a == 0:
        return b.copy()
    return b+a*np.maximum(floor, c+s*_raw_quantile(u, masses, signed, tau))


def _summary(case, times, weights):
    norm = weights/weights.sum()
    estimates, errors = {}, {}

    def mean(name, values):
        active = values[weights > 0]
        if np.all(active == active[0]):
            estimates[name], errors[name] = float(active[0]), 0.
            return
        value = float(np.dot(norm, values))
        estimates[name] = value
        errors[name] = float(np.sqrt(np.sum(norm**2*(values-value)**2)))

    target = times[:, case.target]
    for i, x in enumerate(case.grid):
        mean(f"cdf_{i}", target < x)
    mean("origin_hi", times[:, 0] >= case.origin_cut)
    if case.name == "i_floor_atom":
        mean("floor_atom", np.abs(times[:, 1]-times[:, 0]-5) < 1e-5)
    windows = np.ones(len(times), dtype=bool)
    for observed in case.ev.evidence:
        if observed.actual:
            windows &= (times[:, observed.event] >= observed.minute) & (times[:, observed.event] < observed.minute+1)
    mean("_window_fraction", windows)
    active = weights > 0
    order = np.argsort(target[active])
    sorted_times = target[active][order]
    cumulative = np.cumsum(norm[active][order])
    prefix = np.concatenate(([0.], cumulative))
    for name, p in (("q10", .1), ("q50", .5), ("q90", .9)):
        value = float(np.interp(p, cumulative, sorted_times))
        estimates[name] = value
        width = max(1e-4, .005*(sorted_times[-1]-sorted_times[0]))
        left, right = np.searchsorted(sorted_times, [value-width, value+width], side="left")
        density = (prefix[right]-prefix[left])/(2*width)
        if density <= 0:
            raise ValueError("reference quantile has zero local density")
        errors[name] = float(np.sqrt(np.sum(norm**2*((target <= value)-p)**2))/density)
    return estimates, errors


def lw_reference(case, n=2_000_000, seed=1234):
    if n < 2_000_000:
        raise ValueError("G7 likelihood reference requires at least 2M iid samples")
    # ponytail: fixtures have identical delay levels; add independent quantile inversion for varying levels.
    for tables in (case.models.r, case.models.w):
        if not np.all(tables == tables[:, :1]):
            raise ValueError("reference fixture requires delay-independent tables")
    rng = np.random.default_rng(seed)
    times = np.empty((n, len(case.ev.events)), dtype=float)
    weights, invariant = np.ones(n), False
    observations = {o.event: o for o in case.ev.evidence}
    for e in range(times.shape[1]):
        law = _law(case, e, times)
        u = rng.random(n)
        observed = observations.get(e)
        if observed and observed.actual:
            fl = _effective_cdf(observed.minute, law)
            fh = _effective_cdf(observed.minute+1, law)
            sl = _effective_cdf(observed.minute, law, True)
            sh = _effective_cdf(observed.minute+1, law, True)
            mass = np.maximum(0, np.where(fl >= .5, sl-sh, fh-fl))
            if not np.any(mass > 0):
                times[:, e] = observed.minute+.5
                invariant = True
                continue
            p = np.where(mass > 0, fl+u*mass, u)
            p = np.minimum(np.nextafter(1., 0.), p)
            times[:, e] = _effective_quantile(p, law)
            times[:, e] = np.where(mass > 0, np.minimum(times[:, e], np.nextafter(observed.minute+1., -np.inf)), times[:, e])
            weights *= mass
        else:
            times[:, e] = _effective_quantile(u, law)
            if observed:
                weights *= _survival(observed.minute-times[:, e], case.params.s_tables[observed.source])
    if not weights.sum() > 0:
        raise ValueError("empty likelihood-weighted reference")
    estimates, errors = _summary(case, times, weights)
    estimates["_invariant_rate"], errors["_invariant_rate"] = float(invariant), 0.
    return estimates, errors


def _uniform_sum_cdf(x, count):
    if x <= 0:
        return 0.
    if x >= count:
        return 1.
    return sum((-1)**k*math.comb(count, k)*max(x-k, 0)**count for k in range(count+1))/math.factorial(count)


def enumerate_posterior(case):
    if case.name == "k_repeated_pending":
        # Independent uniform jitters cancel in the ratio of exponential likelihoods.
        return {"origin_hi": 1/(1+math.exp(-1))}
    if not case.exact:
        raise ValueError("use likelihood weighting for non-bin-aligned evidence")
    choices = []
    events, models = case.ev.events, case.models
    for e in events:
        if e.idx == 0:
            masses, shift, signed = models.o, e.sched_off+case.ev.resched_min, True
        elif e.kind == "dep":
            masses, shift, signed = models.w[e.halt_idx, 0], 0, False
        else:
            start = next(s for s in events if s.kind == "dep" and s.seg == e.seg)
            masses, shift, signed = models.r[e.seg, 0], e.sched_off-start.sched_off+models.bias[e.seg], True
        choices.append([(i-61+shift if signed else i+shift, float(masses[i])) for i in np.flatnonzero(masses)])
    components = []
    for chosen in product(*choices):
        bases = np.cumsum([value for value, _ in chosen])
        weight = math.prod(p for _, p in chosen)
        for observed in case.ev.evidence:
            lo, hi = bases[observed.event], bases[observed.event]+observed.event+1
            if observed.actual:
                if hi <= observed.minute or lo >= observed.minute+1:
                    weight = 0
                elif not observed.minute <= lo < hi <= observed.minute+1:
                    raise ValueError("observation cuts a uniform cell; use LW")
            else:
                table = case.params.s_tables[observed.source]
                first, last = (_survival(np.array([observed.minute-x]), table)[0] for x in (lo, hi))
                if first != last:
                    raise ValueError("pending likelihood varies inside cell; use LW")
                weight *= first
        if weight:
            components.append((bases, weight))
    total = sum(weight for _, weight in components)
    if not total:
        raise ValueError("empty exact posterior")

    def cdf(x, event):
        return sum(weight*_uniform_sum_cdf(x-bases[event], event+1) for bases, weight in components)/total

    result = {f"cdf_{i}": cdf(x, case.target) for i, x in enumerate(case.grid)}
    result["origin_hi"] = 1-cdf(case.origin_cut, 0)
    for name, p in (("q10", .1), ("q50", .5), ("q90", .9)):
        lo = min(bases[case.target] for bases, _ in components)
        hi = max(bases[case.target]+case.target+1 for bases, _ in components)
        for _ in range(64):
            middle = (lo+hi)/2
            if cdf(middle, case.target) < p:
                lo = middle
            else:
                hi = middle
        result[name] = (lo+hi)/2
    result["_invariant_rate"], result["_window_fraction"] = 0., 1.
    return result


def engine_replicates(case, n, reps=20):
    if reps < 20:
        raise ValueError("G7 requires at least 20 salted replicates")
    version = v_hash({**case.ev.versions, "case": case.name, "N": n})
    estimates = []
    for replicate in range(reps):
        key = run_key(version, case.ev.run_id, replicate)
        out = replay(case.ev, case.models, replace(case.params, n=n, k=n), key)
        if not np.isfinite(out.paths).all():
            raise AssertionError("nonfinite replay paths")
        times = out.paths.astype(float)
        values = {f"cdf_{i}": float(np.mean(times[:, case.target] < x)) for i, x in enumerate(case.grid)}
        values["origin_hi"] = float(np.mean(times[:, 0] >= case.origin_cut))
        if case.name == "i_floor_atom":
            values["floor_atom"] = float(np.mean(np.abs(times[:, 1]-times[:, 0]-5) < 1e-5))
        for name, p in (("q10", .1), ("q50", .5), ("q90", .9)):
            values[name] = float(np.quantile(times[:, case.target], p))
        windows = [(times[:, o.event] >= o.minute) & (times[:, o.event] < o.minute+1) for o in case.ev.evidence if o.actual]
        values["_window_fraction"] = float(np.mean(np.logical_and.reduce(windows))) if windows else 1.
        values["_invariant_rate"] = float("data_invariant" in out.flags)
        values["_fallback_rate"] = float(out.min_ess < case.params.ess_min)
        estimates.append(values)
    mean = {key: float(np.mean([e[key] for e in estimates])) for key in estimates[0]}
    se = {key: float(np.std([e[key] for e in estimates], ddof=1)/math.sqrt(reps)) for key in mean}
    return mean, se
