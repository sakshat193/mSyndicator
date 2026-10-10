"""Loopback CONNECT relay with optional supervised reverse SSH forwarding."""

import argparse
import asyncio
from contextlib import suppress
import logging
import re
import subprocess
import sys

from msynd.collect.mmt import HOST


# ponytail: single-host allowlist; add a host only with a new approved source.
ALLOW = {(HOST, 443)}
HEAD_LIMIT = 8192
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
log = logging.getLogger(__name__)


async def handle(reader, writer):
    upstream_writer, pipes = None, []
    counts = [0, 0]
    started = asyncio.get_running_loop().time()

    async def pipe(source, destination, direction):
        while data := await source.read(65536):
            destination.write(data)
            counts[direction] += len(data)
            await destination.drain()

    try:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            if len(head) > HEAD_LIMIT:
                raise ValueError("oversized head")
            match = re.fullmatch(br"CONNECT ([^:\s]+):([0-9]+) HTTP/1\.[0-9]", head.split(b"\r\n", 1)[0])
            if match is None:
                raise ValueError("invalid request line")
            host, port = match[1].decode("ascii").lower(), int(match[2])
            if (host, port) not in ALLOW:
                raise ValueError("target is not allowlisted")
        except (ValueError, asyncio.LimitOverrunError, asyncio.IncompleteReadError):
            log.warning("refused invalid or non-allowlisted CONNECT request")
            writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await writer.drain()
            return
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(asyncio.open_connection(host, port), 30)
        except (OSError, TimeoutError):
            log.warning("upstream connection failed for %s:%s", host, port)
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()
            return
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        pipes = [asyncio.create_task(pipe(reader, upstream_writer, 0)),
                 asyncio.create_task(pipe(upstream_reader, writer, 1))]
        await asyncio.wait(pipes, return_when=asyncio.FIRST_COMPLETED)
    except OSError:
        pass
    finally:
        for task in pipes:
            task.cancel()
        await asyncio.gather(*pipes, return_exceptions=True)
        writers = [writer] if upstream_writer is None else [writer, upstream_writer]
        for stream in writers:
            stream.close()
        await asyncio.gather(*(stream.wait_closed() for stream in writers), return_exceptions=True)
        if upstream_writer is not None:
            log.info("tunnel %s:%s bytes_up=%s bytes_down=%s duration=%.3fs",
                     host, port, *counts, asyncio.get_running_loop().time() - started)


def ssh_argv(user_host, port, identity=None):
    argv = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-R", f"127.0.0.1:{port}:127.0.0.1:{port}"]
    if identity:
        argv += ["-i", identity]
    return argv + [user_host]


async def supervise(argv, delay=10):
    while True:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            code = await process.wait()
            log.warning("ssh exited with code %s; restarting in %s s", code, delay)
        except OSError as exc:
            log.warning("could not start ssh: %s; restarting in %s s", exc, delay)
        finally:
            if process is not None and process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                await process.wait()
        await asyncio.sleep(delay)


async def serve(port, ssh=None, identity=None):
    async with await asyncio.start_server(handle, "127.0.0.1", port, limit=HEAD_LIMIT) as server:
        if ssh:
            await asyncio.gather(server.serve_forever(), supervise(ssh_argv(ssh, port, identity)))
        else:
            await server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--ssh", metavar="USER@HOST")
    parser.add_argument("--identity", metavar="PATH")
    args = parser.parse_args(argv)
    execution_state = None
    if sys.platform == "win32":
        import ctypes

        execution_state = ctypes.windll.kernel32.SetThreadExecutionState
        execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        asyncio.run(serve(args.port, args.ssh, args.identity))
    except KeyboardInterrupt:
        pass
    finally:
        if execution_state is not None:
            execution_state(ES_CONTINUOUS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
