"""MMT status admission; clocks and T_max are in seconds."""
from dataclasses import dataclass
from datetime import date, datetime
import math


@dataclass(frozen=True)
class PollRun:
    id: int
    train: str
    start_date: date
    last_fetch: datetime
    priority: float | None = None


@dataclass(frozen=True)
class AdmissionState:
    runs: tuple[PollRun, ...]
    host_state: str = 'ok'
    rate: float = .2
    h: float = 1
    T_max: float | None = None
    # ponytail: CP-3 has not fixed T_max; replace this provisional factor after its decision.
    t_max_factor: float = 1.25
    recovery_at: datetime | None = None

    def __post_init__(self):
        values = (self.rate, self.h, self.t_max_factor)
        if self.T_max is not None:
            values += (self.T_max,)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('admission budgets must be finite and positive')
        if self.host_state not in ('ok', 'breaker', 'stopped'):
            raise ValueError('unknown dispatcher state')


@dataclass(frozen=True)
class Job:
    run: PollRun
    reason: str


def next_job(state: AdmissionState, now: datetime) -> Job | None:
    if state.host_state == 'stopped' or not state.runs:
        return None
    oldest = min(state.runs, key=lambda run: (run.last_fetch, run.id))
    if state.host_state == 'breaker':
        if state.recovery_at is None or now < state.recovery_at:
            return None
        # MMT has no session endpoint: the next status request is the recovery probe.
        return Job(oldest, 'recovery')
    interval = state.h / state.rate
    cycle = len(state.runs) * interval
    ceiling = state.T_max if state.T_max is not None else state.t_max_factor * cycle
    if ceiling < cycle:
        raise ValueError('T_max is below the available polling capacity')
    # Reserve a full cycle, including the dispatcher's next-start wait, before the oldest deadline.
    if (now - oldest.last_fetch).total_seconds() >= ceiling - cycle:
        return Job(oldest, 'overdue')
    run = max(state.runs, key=lambda run: (
        run.priority if run.priority is not None else (now - run.last_fetch).total_seconds() / ceiling,
        -run.last_fetch.timestamp(), -run.id))
    return Job(run, 'targeted' if run.priority is not None else 'age')
