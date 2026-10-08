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
