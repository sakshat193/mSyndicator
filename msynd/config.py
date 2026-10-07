from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://msynd:msynd@localhost:55432/msynd"
    ntes_rate: float = 1.0
    etrain_rate: float = 0.2
    live: bool = False
    n_particles: int = 10000
    k_paths: int = 32
    f_minutes: float = 60

    def __post_init__(self):
        for value in (self.ntes_rate, self.etrain_rate, self.f_minutes):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("rates and freshness must be finite and positive")
        if self.n_particles <= 0 or not 0 < self.k_paths <= self.n_particles:
            raise ValueError("require 0 < k_paths <= n_particles")

    @classmethod
    def from_env(cls):
        return cls(
            database_url=os.environ.get("DATABASE_URL", cls.database_url),
            ntes_rate=float(os.environ.get("NTES_RATE", "1.0")),
            etrain_rate=float(os.environ.get("ETRAIN_RATE", "0.2")),
            live=os.environ.get("MSYND_LIVE") == "1",
            n_particles=int(os.environ.get("N_PARTICLES", "10000")),
            k_paths=int(os.environ.get("K_PATHS", "32")),
            f_minutes=float(os.environ.get("F_MINUTES", "60")),
        )


settings = Settings.from_env()
