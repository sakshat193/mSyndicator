from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from msynd.engine import __main__ as engine
from msynd.engine.evidence import RunEvidence
from msynd.engine.replay import Params
from msynd.engine.worker import _job


def broken_replay(*args):
    raise ValueError('bad evidence')


def test_worker_error_payload_identifies_run_and_preserves_versions():
    ev = RunEvidence(42, {'route': 7}, (), (), 0, None)
    run_id, payload, cached = _job(({'evidence': ev}, None, Params(), broken_replay))

    assert run_id == 42
    assert payload == {'run_id': 42, 'state': 'error', 'error': 'bad evidence',
                       'versions': {'route': 7}}
    assert cached is None


@pytest.mark.parametrize('failure', ['evidence', 'models'])
def test_prepare_error_payload_identifies_run(monkeypatch, failure):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (datetime(2025, 10, 8, tzinfo=timezone.utc),)
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = {
        'id': 42, 'lifecycle': 'running',
    }
    ev = RunEvidence(42, {'route': 7}, (), (), 0, None)
    monkeypatch.setattr(engine, 'load_run_evidence',
                        broken_replay if failure == 'evidence' else lambda *args: ev)

    jobs, runs, tombstones = engine._prepare(conn, [42], broken_replay, Params(), broken_replay)

    assert jobs == []
    assert tombstones == []
    assert runs[42]['run_id'] == 42
    assert runs[42]['state'] == 'error'
    assert runs[42]['error'] == 'bad evidence'
