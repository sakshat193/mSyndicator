import pytest


@pytest.mark.db
def test_db(db):
    assert db.execute("select 1").fetchone() == (1,)


def test_settings_defaults(monkeypatch):
    from msynd.config import Settings

    for key in ("NTES_RATE", "ETRAIN_RATE", "MSYND_LIVE", "N_PARTICLES", "K_PATHS", "F_MINUTES"):
        monkeypatch.delenv(key, raising=False)
    s = Settings.from_env()
    assert (s.ntes_rate, s.etrain_rate, s.live, s.n_particles, s.k_paths, s.f_minutes) == (1.0, 0.2, False, 10000, 32, 60)


def test_settings_reject_invalid_budget(monkeypatch):
    from msynd.config import Settings

    monkeypatch.setenv("NTES_RATE", "0")
    with pytest.raises(ValueError):
        Settings.from_env()
