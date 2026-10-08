from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace


def test_main_wires_configured_database_and_model_source(monkeypatch):
    from msynd import db
    from msynd.engine import __main__ as engine, cache
    from msynd.engine.replay import replay
    from msynd.models import registry

    conn, models, tables = object(), object(), object()
    calls = []
    url = 'postgresql://test:test@127.0.0.1:55432/test_only'
    monkeypatch.setattr(engine, 'settings', replace(engine.settings, database_url=url))

    @contextmanager
    def connect(database_url):
        assert database_url == url
        yield conn
        calls.append('closed')

    def active(connection):
        assert connection is conn
        return models

    ev = SimpleNamespace(versions={'bias_gen': 7})

    def tables_for(connection, evidence, active_models, bias_gen):
        assert (connection, evidence, active_models, bias_gen) == (conn, ev, models, 7)
        return tables

    def run_loop(connection, replay_fn, model_source):
        assert connection is conn
        assert replay_fn is replay
        assert model_source(conn, ev) is tables
        calls.append('loop')

    monkeypatch.setattr(db, 'connect', connect)
    monkeypatch.setattr(registry, 'active', active)
    monkeypatch.setattr(cache, 'tables_for', tables_for)
    monkeypatch.setattr(engine, 'run_loop', run_loop)
    engine.main()
    assert calls == ['loop', 'closed']
