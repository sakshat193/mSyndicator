from dataclasses import dataclass
from datetime import date, datetime


class InvalidResponse(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    station: str
    occ: int
    kind: str
    actual_ts: datetime


@dataclass(frozen=True)
class Pending:
    station: str
    occ: int
    kind: str


@dataclass(frozen=True)
class PassingStation:
    station: str
    occ: int
    dist_km: float
    sched_off: float | None
    after: tuple[str, int]
    before: tuple[str, int]
    fraction: float


def position_passing(rows) -> tuple[PassingStation, ...]:
    """Anchor passes in source route order, dropping unbounded rows."""
    halts = [(idx, row) for idx, row in enumerate(rows) if row[2]]
    passing = []
    for (lo, after), (hi, before) in zip(halts, halts[1:]):
        distance = before[3] - after[3]
        for station, occ, _, km, off in rows[lo + 1:hi]:
            fraction = (km - after[3]) / distance if distance else 0
            passing.append(PassingStation(station, occ, km, off, after[:2], before[:2], fraction))
    return tuple(passing)


@dataclass(frozen=True)
class Eta:
    station: str
    occ: int
    kind: str
    eta_ts: datetime


@dataclass(frozen=True)
class ParsedStatus:
    train: str
    start_date: date
    status: str
    last_updated_at: datetime
    events: tuple[Event, ...]
    pending: tuple[Pending, ...]
    passing_stations: tuple[PassingStation, ...]
    exceptions: tuple[dict, ...]
    flags: tuple[str, ...]
    etas: tuple[Eta, ...] = ()
