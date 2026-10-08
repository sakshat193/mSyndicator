import hashlib
import json
import subprocess
import sys
import os
from datetime import datetime

import httpx
import pytest

from msynd.collect import capture
from msynd.collect.dispatcher import Dispatcher
from msynd.db import connect
from tests.collect.test_dispatcher import FakeClock


def write_plan(tmp_path, entries):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.mark.db
def test_capture_files_headers_and_host_spacing(db_url, tmp_path, monkeypatch):
    clock = FakeClock()
    starts = {}
    requests = []

    def handle(request):
        starts.setdefault(request.url.host, []).append(clock.t)
        requests.append((request.method, request.content))
        return httpx.Response(200, content=b"captured\x00body", headers={
            "Content-Type": "application/json", "Set-Cookie": "session=secret",
            "Authorization": "secret-auth", "X-Api-Token": "secret-token",
            "X-CSRF-Value": "secret-csrf", "X-AuTh-Info": "secret-info",
        })

    monkeypatch.setattr(capture, "real_transport", lambda: httpx.MockTransport(handle))
    monkeypatch.setattr(capture, "connect", lambda: connect(db_url))
    monkeypatch.setattr(capture, "Dispatcher", lambda *a, **kw: Dispatcher(*a, **kw, clock=clock))
    entries = [
        {"name": "first", "purpose": "token", "method": "GET", "url": "https://capture-a.test/token"},
        {"name": "other", "purpose": "status", "method": "GET", "url": "https://capture-b.test/status"},
        {"name": "last", "purpose": "status", "method": "POST", "url": "https://capture-a.test/status", "json": {"train": "12951"}},
    ]
    plan = write_plan(tmp_path, entries)
    out = tmp_path / "out"
    assert capture.main([str(plan), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert [row["name"] for row in manifest] == ["first", "other", "last"]
    for row in manifest:
        assert (out / (row["name"] + ".body")).read_bytes() == b"captured\x00body"
        assert row["status"] == 200 and row["outcome"] == "ok"
        assert row["sha256"] == hashlib.sha256(b"captured\x00body").hexdigest()
        assert datetime.fromisoformat(row["received_at"]).utcoffset().total_seconds() == 0
        assert row["headers"]["content-type"] == "application/json"
        assert "set-cookie" not in row["headers"]
        assert all(row["headers"][h] == "[REDACTED]" for h in (
            "authorization", "x-api-token", "x-csrf-value", "x-auth-info"))
    assert "secret" not in (out / "manifest.json").read_text()
    assert starts["capture-a.test"][1] - starts["capture-a.test"][0] >= 5
    assert requests[-1][0] == "POST" and json.loads(requests[-1][1]) == {"train": "12951"}
    with connect(db_url) as conn:
        assert conn.execute("select count(*) from obs.fetch where host in ('capture-a.test','capture-b.test')").fetchone()[0] == 3
        assert conn.execute("select pg_try_advisory_lock(hashtext('http:capture-a.test'))").fetchone()[0]


def test_cli_refuses_without_live_before_any_request(tmp_path):
    plan = write_plan(tmp_path, [{"name": "status", "purpose": "status", "method": "GET", "url": "https://capture.test/status"}])
    env = dict(os.environ)
    env.pop("MSYND_LIVE", None)
    env["DATABASE_URL"] = "postgresql://invalid:invalid@127.0.0.1:1/invalid"
    result = subprocess.run([sys.executable, "-m", "msynd.collect.capture", str(plan), "--out", str(tmp_path / "out")], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "MSYND_LIVE=1" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("entry", [
    {"name": "../escape", "purpose": "status", "method": "GET", "url": "https://capture.test/x"},
    {"name": "ok", "purpose": "status", "method": "GET", "url": "file:///etc/passwd"},
    {"name": "ok", "purpose": "status", "method": "GET", "url": "https://user:password@capture.test/x"},
    {"name": "ok", "method": "GET", "url": "https://capture.test/x"},
])
def test_invalid_plan_rejected_before_transport(entry, tmp_path, monkeypatch):
    def forbidden():
        pytest.fail("invalid plan reached transport construction")
    monkeypatch.setattr(capture, "real_transport", forbidden)
    plan = write_plan(tmp_path, [entry])
    with pytest.raises(ValueError):
        capture.main([str(plan), "--out", str(tmp_path / "out")])


def test_duplicate_names_rejected(tmp_path, monkeypatch):
    entry = {"name": "same", "purpose": "status", "method": "GET", "url": "https://capture.test/x"}
    plan = write_plan(tmp_path, [entry, entry])
    monkeypatch.setattr(capture, "real_transport", lambda: pytest.fail("duplicate plan reached transport"))
    with pytest.raises(ValueError):
        capture.main([str(plan), "--out", str(tmp_path / "out")])
