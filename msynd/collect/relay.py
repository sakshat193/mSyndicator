"""Loopback HTTP relay with optional supervised reverse SSH forwarding."""

import argparse
import asyncio
from contextlib import suppress
import logging
import re
import subprocess
import sys

import httpx

from msynd.collect.mmt import HOST


# ponytail: single-host allowlist; add a host only with a new approved source.
ALLOW = {(HOST, 443)}
UPSTREAM = "https://" + HOST
HEAD_LIMIT = 8192
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
log = logging.getLogger(__name__)
_job_handle = None


def setup_job():
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits),
                    ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    global _job_handle
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                    ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        # Never close this non-inheritable handle: process exit must kill the job's children.
        _job_handle = kernel32.CreateJobObjectW(None, None)
        if not _job_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(_job_handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(_job_handle, wintypes.HANDLE(-1)):
            raise ctypes.WinError(ctypes.get_last_error())
    except (OSError, AttributeError) as exc:
        log.warning("could not set up relay job object: %s", exc)


async def handle(reader, writer, client):
    method, path = "-", "-"
    status, reason, body = 403, "Forbidden", b""
    response_headers = []
    bytes_in = 0
    started = asyncio.get_running_loop().time()
    try:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            if len(head) > HEAD_LIMIT:
                raise ValueError("oversized head")
            lines = head[:-4].split(b"\r\n")
            match = re.fullmatch(br"([A-Z]+) (/[^\x00-\x20\x7f-\xff#]*) HTTP/1\.1", lines[0])
            if match is None or match[1] == b"CONNECT":
                raise ValueError("invalid request line")
            method, target = match[1].decode("ascii"), match[2].decode("ascii")
            path = target.partition("?")[0]
            fields = []
            for line in lines[1:]:
                field = re.fullmatch(br"([!#$%&'*+.^_`|~0-9A-Za-z-]+):[ \t]*([\t\x20-\x7e\x80-\xff]*)", line)
                if field is None:
                    raise ValueError("invalid header")
                fields.append((field[1], field[2].strip(b" \t")))
            headers = httpx.Headers(fields)
            if (len(headers.get_list("host")) != 1
                    or (headers["host"].lower().removesuffix(":443"), 443) not in ALLOW):
                raise ValueError("host is not allowlisted")
            length = headers.get("content-length", "0")
            if ("transfer-encoding" in headers or len(headers.get_list("content-length")) > 1
                    or re.fullmatch(r"[0-9]+", length) is None or int(length) > 65536):
                raise ValueError("invalid body length")
            body = await reader.readexactly(int(length))
            bytes_in = len(body)
        except (ValueError, asyncio.LimitOverrunError, asyncio.IncompleteReadError) as exc:
            log.warning("%s", type(exc).__name__)
            body = b""
        else:
            dropped = {"connection", "keep-alive", "te", "trailer", "transfer-encoding", "upgrade",
                       "host", "content-length", "accept-encoding"}
            dropped.update(name.strip().lower() for name in headers.get("connection", "").split(","))
            forwarded = [(name, value) for name, value in headers.raw
                         if name.decode("ascii").lower() not in dropped and not name.lower().startswith(b"proxy-")]
            url = httpx.URL(UPSTREAM + target)
            try:
                response = await client.request(method, url, headers=forwarded, content=body)
                status, reason, body = response.status_code, response.reason_phrase, response.content
                response_headers = [(b"Content-Type" if name.lower() == b"content-type" else b"Retry-After", value)
                                    for name, value in response.headers.raw
                                    if name.lower() in (b"content-type", b"retry-after")]
            except httpx.HTTPError as exc:
                log.warning("%s", type(exc).__name__)
                status, reason, body = 502, "Bad Gateway", b""
        writer.write(f"HTTP/1.1 {status} {reason}\r\n".encode("ascii"))
        for name, value in response_headers:
            writer.write(name + b": " + value + b"\r\n")
        writer.write(f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii") + body)
        await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()
        with suppress(OSError):
            await writer.wait_closed()
        log.info("%s %s status=%s bytes_in=%s bytes_out=%s duration=%.3fs",
                 method, path, status, bytes_in, len(body), asyncio.get_running_loop().time() - started)


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
    # Library request/debug logs include URLs or headers; the relay logs only safe fields.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    async with httpx.AsyncClient(follow_redirects=False, timeout=25) as client:
        async with await asyncio.start_server(lambda r, w: handle(r, w, client), "127.0.0.1", port,
                                             limit=HEAD_LIMIT) as server:
            if ssh:
                await asyncio.gather(server.serve_forever(), supervise(ssh_argv(ssh, port, identity)))
            else:
                await server.serve_forever()


def main(argv=None):
    setup_job()
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
