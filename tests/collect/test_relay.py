import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import sys

import pytest

from msynd.collect import relay


@asynccontextmanager
async def local_relay(relay, monkeypatch):
    connected, closed, handled = [], asyncio.Event(), asyncio.Event()

    async def echo(reader, writer):
        connected.append(True)
        try:
            writer.write(b'ready')
            await writer.drain()
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            closed.set()

    async def handle(reader, writer):
        try:
            await relay.handle(reader, writer)
        finally:
            handled.set()

    upstream = await asyncio.start_server(echo, '127.0.0.1', 0)
    target = upstream.sockets[0].getsockname()[:2]
    monkeypatch.setattr(relay, 'ALLOW', {target})
    open_connection = asyncio.open_connection

    async def loopback_only(host, port, **kwargs):
        assert host == '127.0.0.1', 'test attempted a non-loopback upstream'
        return await open_connection(host, port, **kwargs)

    monkeypatch.setattr(relay.asyncio, 'open_connection', loopback_only)
    server = await asyncio.start_server(handle, '127.0.0.1', 0, limit=8192)
    async with upstream, server:
        reader, writer = await open_connection(*server.sockets[0].getsockname()[:2])
        try:
            yield reader, writer, target, connected, closed, handled
        finally:
            writer.close()
            await writer.wait_closed()


def test_allowlisted_connect_tunnels_both_directions(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=relay.__name__)

    async def run():
        async with local_relay(relay, monkeypatch) as (reader, writer, target, connected, closed, handled):
            writer.write(f'CONNECT {target[0]}:{target[1]} HTTP/1.1\r\nHost: ignored\r\n\r\n'.encode()
                         + b'private-payload')
            await writer.drain()
            assert await reader.readuntil(b'\r\n\r\n') == b'HTTP/1.1 200 Connection established\r\n\r\n'
            assert await reader.readexactly(20) == b'readyprivate-payload'
            writer.write(b'next')
            await writer.drain()
            assert await reader.readexactly(4) == b'next'
            writer.close()
            await writer.wait_closed()
            await closed.wait()
            await handled.wait()
            assert connected == [True]
            records = [record for record in caplog.records if record.name == relay.__name__]
            assert len(records) == 1 and records[0].levelno == logging.INFO
            assert f'{target[0]}:{target[1]}' in records[0].message
            assert 'bytes_up=19' in records[0].message and 'bytes_down=24' in records[0].message
            assert 'duration=' in records[0].message
            assert 'private-payload' not in records[0].message
    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize('kind', ['host', 'port', 'get', 'oversized', 'unterminated', 'version', 'malformed'])
def test_refuses_invalid_connect_without_upstream(monkeypatch, caplog, kind):
    async def run():
        async with local_relay(relay, monkeypatch) as (reader, writer, target, connected, closed, handled):
            host, port = target
            head = f'CONNECT {host}:{port} HTTP/1.1\r\n\r\n'.encode()
            if kind == 'host':
                head = f'CONNECT localhost:{port} HTTP/1.1\r\n\r\n'.encode()
            elif kind == 'port':
                head = f'CONNECT {host}:0 HTTP/1.1\r\n\r\n'.encode()
            elif kind == 'get':
                head = f'GET {host}:{port} HTTP/1.1\r\n\r\n'.encode()
            elif kind in ('oversized', 'unterminated'):
                head = head[:-2] + b'X: ' + b'x' * 8192
                if kind == 'oversized':
                    head += b'\r\n\r\n'
            elif kind == 'version':
                head = head.replace(b'HTTP/1.1', b'HTTP/2.0')
            else:
                head = b'CONNECT bad:port HTTP/1.1\r\n\r\n'
            writer.write(head)
            await writer.drain()
            assert (await reader.read()).startswith(b'HTTP/1.1 403 Forbidden\r\n')
            await handled.wait()
            assert connected == []
            records = [record for record in caplog.records if record.name == relay.__name__]
            assert len(records) == 1 and records[0].levelno == logging.WARNING
            assert 'x' * 100 not in records[0].message
    asyncio.run(asyncio.wait_for(run(), 5))


def test_unreachable_allowlisted_upstream_returns_502(monkeypatch):
    async def run():
        async with local_relay(relay, monkeypatch) as (reader, writer, target, connected, closed, handled):
            unused = await asyncio.start_server(lambda r, w: w.close(), '127.0.0.1', 0)
            port = unused.sockets[0].getsockname()[1]
            unused.close()
            await unused.wait_closed()
            monkeypatch.setattr(relay, 'ALLOW', {('127.0.0.1', port)})
            writer.write(f'CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n'.encode())
            await writer.drain()
            assert (await reader.read()).startswith(b'HTTP/1.1 502 Bad Gateway\r\n')
            await handled.wait()
            assert connected == []
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
