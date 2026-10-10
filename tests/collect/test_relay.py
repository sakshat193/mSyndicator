import asyncio
from contextlib import asynccontextmanager, suppress
import gzip
import logging
import sys

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
