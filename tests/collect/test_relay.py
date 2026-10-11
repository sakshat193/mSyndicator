import asyncio
from contextlib import asynccontextmanager, suppress
import gzip
import logging
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from msynd.collect import relay


@asynccontextmanager
async def local_relay(relay, monkeypatch):
    requests, handled = [], asyncio.Event()
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')

    async def stub(reader, writer):
        try:
            head = await reader.readuntil(b'\r\n\r\n')
            lines = head.decode('latin-1').split('\r\n')
            headers = dict((name.lower(), value.strip()) for name, value in
                           (line.split(':', 1) for line in lines[1:-2]))
            body = await reader.readexactly(int(headers.get('content-length', '0')))
            requests.append((lines[0], headers, body))
            payload = gzip.compress(b'{"reply":"private-response"}')
            writer.write(b'HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n'
                         b'Retry-After: 7\r\nContent-Encoding: gzip\r\nX-Private: hidden\r\n'
                         b'Connection: close\r\nContent-Length: ' + str(len(payload)).encode()
                         + b'\r\n\r\n' + payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    start_server = asyncio.start_server
    upstream = await start_server(stub, '127.0.0.1', 0)
    port = upstream.sockets[0].getsockname()[1]
    monkeypatch.setattr(relay, 'UPSTREAM', f'http://127.0.0.1:{port}', raising=False)
    open_connection = asyncio.open_connection

    async def loopback_only(host, port, **kwargs):
        assert host == '127.0.0.1', 'test attempted a non-loopback upstream'
        return await open_connection(host, port, **kwargs)

    ready = asyncio.get_running_loop().create_future()

    async def start(handler, host, port, **kwargs):
        assert host == '127.0.0.1'

        async def handle(reader, writer):
            try:
                await handler(reader, writer)
            finally:
                handled.set()

        server = await start_server(handle, host, port, **kwargs)
        ready.set_result(server)
        return server

    monkeypatch.setattr(relay.asyncio, 'open_connection', loopback_only)
    monkeypatch.setattr(relay.asyncio, 'start_server', start)
    task = asyncio.create_task(relay.serve(0))
    async with upstream:
        server = await ready
        reader, writer = await open_connection(*server.sockets[0].getsockname()[:2])
        try:
            yield reader, writer, requests, handled
        finally:
            writer.close()
            await writer.wait_closed()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.parametrize('host', ['rails-ris.makemytrip.com', 'RAILS-RIS.MAKEMYTRIP.COM:443'])
def test_forwards_post_and_decoded_response_without_private_logs(monkeypatch, caplog, host):
    caplog.set_level(logging.INFO)

    async def run():
        async with local_relay(relay, monkeypatch) as (reader, writer, requests, handled):
            body = b'{"request":"private-payload"}'
            writer.write(f'POST /api/status?secret=private-query&x=%2f HTTP/1.1\r\nhOsT: {host}\r\n'.encode()
                         + b'Content-Type: application/json\r\nUser-Agent: private-agent\r\n'
                         b'Accept-Encoding: private-encoding\r\nConnection: close, X-Hop\r\n'
                         b'X-Hop: private-hop\r\nKeep-Alive: timeout=9\r\nProxy-Authorization: private-auth\r\n'
                         b'TE: trailers\r\nTrailer: X-Trailer\r\nUpgrade: websocket\r\n'
                         b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
            await writer.drain()
            response = await reader.read()
            assert response == (b'HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n'
                                b'Retry-After: 7\r\nContent-Length: 28\r\nConnection: close\r\n\r\n'
                                b'{"reply":"private-response"}')
            await handled.wait()
            assert len(requests) == 1
            line, headers, received = requests[0]
            assert line == 'POST /api/status?secret=private-query&x=%2f HTTP/1.1'
            assert received == body
            assert headers['content-type'] == 'application/json'
            assert headers['user-agent'] == 'private-agent'
            assert headers['host'].startswith('127.0.0.1:')
            assert headers['content-length'] == str(len(body))
            assert headers.get('accept-encoding') != 'private-encoding'
            assert headers.get('connection') != 'close, X-Hop'
            assert not {'x-hop', 'keep-alive', 'proxy-authorization', 'te', 'trailer', 'upgrade'} & headers.keys()
            records = [record for record in caplog.records if record.name == relay.__name__]
            assert len(records) == 1 and records[0].levelno == logging.INFO
            assert 'POST /api/status' in records[0].message and '429' in records[0].message
            assert 'bytes_in=29' in records[0].message and 'bytes_out=28' in records[0].message
            assert 'duration=' in records[0].message
            assert 'private-' not in caplog.text
    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize('head', [
    b'GET /?private-query HTTP/1.1\r\nHost: other.test\r\n\r\n',
    b'GET / HTTP/1.1\r\nHost: rails-ris.makemytrip.com:80\r\n\r\n',
    b'GET http://rails-ris.makemytrip.com/ HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\n\r\n',
    b'CONNECT rails-ris.makemytrip.com:443 HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\n\r\n',
    b'GET / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nX: ' + b'x' * 8192 + b'\r\n\r\n',
    b'GET / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nX: ' + b'x' * 8192,
    b'POST / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nContent-Length: 65537\r\n\r\n',
    b'POST / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nContent-Length: no\r\n\r\n',
    b'POST / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nContent-Length: -1\r\n\r\n',
    b'POST / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n',
    b'GET / HTTP/1.0\r\nHost: rails-ris.makemytrip.com\r\n\r\n',
    b'get / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\n\r\n',
    b'GET / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nMalformed\r\n\r\n',
    b'GET / HTTP/1.1\r\n\r\n',
    b'GET / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nHost: other.test\r\n\r\n',
    b'POST / HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n',
], ids=['host', 'port', 'absolute', 'connect', 'head-size', 'unterminated', 'body-size',
        'length-nan', 'length-negative', 'chunked', 'version', 'method', 'header',
        'no-host', 'duplicate-host', 'duplicate-length'])
def test_refuses_invalid_requests_without_upstream(monkeypatch, caplog, head):
    caplog.set_level(logging.INFO, logger=relay.__name__)

    async def run():
        async with local_relay(relay, monkeypatch) as (reader, writer, requests, handled):
            writer.write(head)
            await writer.drain()
            assert (await reader.read()).startswith(b'HTTP/1.1 403 Forbidden\r\n')
            await handled.wait()
            assert requests == []
            records = [record for record in caplog.records if record.name == relay.__name__]
            assert [record.levelno for record in records] == [logging.WARNING, logging.INFO]
            assert records[0].message in ('ValueError', 'LimitOverrunError', 'IncompleteReadError')
            assert 'private-query' not in caplog.text and 'x' * 100 not in caplog.text
    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize('body', [b'', b'x' * 65536], ids=['no-length', 'body-limit'])
def test_forwarding_body_boundaries_and_one_request_only(monkeypatch, body):
    async def run():
        async with local_relay(relay, monkeypatch) as (reader, writer, requests, handled):
            length = b'Content-Length: 65536\r\n' if body else b''
            writer.write(b'POST //another.test/x?y=1 HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\n'
                         + length + b'\r\n' + body
                         + b'GET /second HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\n\r\n')
            await writer.drain()
            assert (await reader.read()).startswith(b'HTTP/1.1 429 Too Many Requests\r\n')
            await handled.wait()
            assert len(requests) == 1
            assert requests[0][0] == 'POST //another.test/x?y=1 HTTP/1.1'
            assert requests[0][2] == body
    asyncio.run(asyncio.wait_for(run(), 5))


def test_unreachable_upstream_returns_502_without_private_logs(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=relay.__name__)

    async def run():
        unused = await asyncio.start_server(lambda r, w: w.close(), '127.0.0.1', 0)
        port = unused.sockets[0].getsockname()[1]
        unused.close()
        await unused.wait_closed()
        async with local_relay(relay, monkeypatch) as (reader, writer, requests, handled):
            monkeypatch.setattr(relay, 'UPSTREAM', f'http://127.0.0.1:{port}', raising=False)
            writer.write(b'GET /?private-query HTTP/1.1\r\nHost: rails-ris.makemytrip.com\r\n\r\n')
            await writer.drain()
            assert (await reader.read()).startswith(b'HTTP/1.1 502 Bad Gateway\r\n')
            await handled.wait()
            assert requests == []
            records = [record for record in caplog.records if record.name == relay.__name__]
            assert [record.levelno for record in records] == [logging.WARNING, logging.INFO]
            assert records[0].message == 'ConnectError'
            assert 'private-query' not in caplog.text
    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize('identity', [None, 'C:/keys/msynd_oci'])
def test_ssh_argv(identity):
    expected = ['ssh', '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
                '-R', '127.0.0.1:18081:127.0.0.1:18081']
    if identity:
        expected += ['-i', identity]
    assert relay.ssh_argv('user@vm', 18081, identity) == expected + ['user@vm']


def test_supervisor_restarts_after_each_exit(caplog):
    caplog.set_level(logging.INFO, logger=relay.__name__)

    async def run():
        task = asyncio.create_task(relay.supervise([sys.executable, '-c', 'raise SystemExit(3)'], delay=0))
        try:
            while sum('code 3' in record.getMessage() for record in caplog.records) < 3:
                await asyncio.sleep(.01)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    asyncio.run(asyncio.wait_for(run(), 10))


@pytest.fixture
def windows_process_api():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows job objects')
@pytest.mark.parametrize('use_job', [True, False], ids=['job-kills-child', 'control-orphans-child'])
def test_relay_child_lifetime_after_forced_parent_exit(tmp_path, windows_process_api, use_job):
    api = windows_process_api
    pid_file = tmp_path / 'relay-test-pids'
    child_file = tmp_path / 'relay-test-child'
    source = r'''
import asyncio
import os
from pathlib import Path
import sys

pids = Path(sys.argv[2])
pids.write_text(str(os.getpid()) + '\n')
from msynd.collect import relay
spawn = asyncio.create_subprocess_exec

async def tracked_spawn(*args, **kwargs):
    child = await spawn(*args, **kwargs, stdout=-3, stderr=-3)  # DEVNULL; no inherited readiness pipe.
    with pids.open('a') as output:
        output.write(str(child.pid) + '\n')
    wait = child.wait
    async def ready_wait():
        print(os.getpid(), child.pid, flush=True)
        return await wait()
    child.wait = ready_wait
    return child

async def sleeping_relay(*args):
    # Self-registration covers a parent killed between spawning and recording the child.
    child_source = ('import os, sys, time; from pathlib import Path; '
                    'p = Path(sys.argv[1]); t = p.with_suffix(".tmp"); '
                    't.write_text(str(os.getpid())); t.replace(p); time.sleep(60)')
    await relay.supervise([sys._base_executable, '-c', child_source, sys.argv[3]])

asyncio.create_subprocess_exec = tracked_spawn
relay.serve = sleeping_relay
if sys.argv[1] == 'job':
    relay.main([])
else:
    asyncio.run(sleeping_relay())
'''
    parent = subprocess.Popen([sys.executable, '-c', source, 'job' if use_job else 'control',
                               str(pid_file), str(child_file)],
                              cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    reader = ThreadPoolExecutor(max_workers=1)
    handles = []
    cleanup_errors = []

    def stop(pid):
        handle = api.OpenProcess(0x00100000 | 0x0001, False, pid)
        if handle:
            try:
                if api.WaitForSingleObject(handle, 0) == 258 and not api.TerminateProcess(handle, 1):
                    cleanup_errors.append(f'could not terminate test process {pid}')
                if api.WaitForSingleObject(handle, 5000) != 0:
                    cleanup_errors.append(f'test process {pid} did not exit')
            finally:
                api.CloseHandle(handle)
    try:
        line = reader.submit(parent.stdout.readline).result(timeout=15)
        assert line, 'test parent exited before reporting its child'
        parent_pid, child_pid = map(int, line.split())
        parent_handle = api.OpenProcess(0x00100000 | 0x0001, False, parent_pid)  # SYNCHRONIZE | TERMINATE
        assert parent_handle, 'cannot open the test parent'
        handles.append(parent_handle)
        child_handle = api.OpenProcess(0x00100000, False, child_pid)  # SYNCHRONIZE
        assert child_handle, 'cannot open the test child'
        handles.append(child_handle)
        assert api.TerminateProcess(parent_handle, 1)
        assert api.WaitForSingleObject(parent_handle, 5000) == 0
        result = api.WaitForSingleObject(child_handle, 5000)
        expected = 0 if use_job else 258  # WAIT_OBJECT_0 / WAIT_TIMEOUT
        assert result == expected, f'child wait returned {result}, expected {expected}'
    finally:
        try:
            # Stop the real interpreter before reading its final list of children.
            if pid_file.exists():
                stop(int(pid_file.read_text().splitlines()[0]))
        finally:
            try:
                if parent.poll() is None:
                    parent.kill()
                parent.wait(timeout=5)
            finally:
                try:
                    pids = pid_file.read_text().splitlines() if pid_file.exists() else []
                    if len(pids) < 2:
                        deadline = time.monotonic() + 5
                        while not child_file.exists() and time.monotonic() < deadline:
                            time.sleep(.01)
                    if child_file.exists():
                        pids.append(child_file.read_text())
                    for pid in set(map(int, pids)):
                        stop(pid)
                finally:
                    reader.shutdown(wait=True)
                    for handle in handles:
                        api.CloseHandle(handle)
                    parent.stdout.close()
                    parent.stderr.close()
                    pid_file.unlink(missing_ok=True)
                    child_file.unlink(missing_ok=True)
                    child_file.with_suffix('.tmp').unlink(missing_ok=True)
        assert not cleanup_errors, cleanup_errors


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows job objects')
def test_job_setup_failure_warns_and_keeps_serving(monkeypatch, caplog):
    import ctypes

    served = []
    async def serve(*args):
        served.append(args)

    def denied(*args, **kwargs):
        raise OSError('test job setup denied')

    monkeypatch.setattr(ctypes, 'WinDLL', denied)
    monkeypatch.setattr(relay, 'serve', serve)
    relay.main([])
    assert served == [(18080, None, None)]
    assert any(record.levelno == logging.WARNING and 'job' in record.getMessage().lower()
               for record in caplog.records)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows PowerShell parser')
def test_relay_task_parses_with_windows_powershell():
    script = Path(__file__).resolve().parents[2] / 'relay-task.ps1'
    assert script.is_file()
    command = ("$tokens = $null; $errors = $null; "
               "[System.Management.Automation.Language.Parser]::ParseFile("
               "'" + str(script).replace("'", "''") + "', [ref]$tokens, [ref]$errors) | Out-Null; "
               "$errors | ForEach-Object { Write-Output $_ }; if ($errors.Count) { exit 1 }")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
