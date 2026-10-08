"""Capture approved discovery requests through the host-locked dispatcher."""

import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
import math
from pathlib import Path
import re

import httpx

from msynd.collect.dispatcher import Dispatcher, real_transport
from msynd.db import connect


def load_plan(path):
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("plan must be a nonempty list")
    names = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("plan entries must be objects")
        name = entry.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name.casefold() in names:
            raise ValueError("capture names must be unique filename-safe names")
        names.add(name.casefold())
        if any(not isinstance(entry.get(key), str) or not entry[key] for key in ("purpose", "method", "url")):
            raise ValueError("each entry requires purpose, method and url")
        url = httpx.URL(entry["url"])
        if url.scheme not in ("http", "https") or not url.host or url.userinfo:
            raise ValueError("capture URLs must be HTTP(S), without embedded credentials")
    return entries


async def capture(entries, out, rate):
    manifest, dispatchers, headers = [], {}, {}
    async with AsyncExitStack() as stack:
        for entry in entries:
            host = httpx.URL(entry["url"]).host
            if host not in dispatchers:
                dispatcher = Dispatcher(host, rate, connect, real_transport())
                stack.push_async_callback(dispatcher.close)
                headers[host] = {}

                async def on_request(request, host=host):
                    headers[host].clear()

                async def on_response(response, host=host):
                    headers[host].update({
                        key: "[REDACTED]" if any(word in key.lower() for word in ("auth", "token", "csrf")) else value
                        for key, value in response.headers.items() if key.lower() != "set-cookie"
                    })

                dispatcher.client.event_hooks = {"request": [on_request], "response": [on_response]}
                await dispatcher.start()
                dispatchers[host] = dispatcher
            result = await dispatchers[host].request(
                entry["purpose"], entry["method"], entry["url"],
                **({"json": entry["json"]} if "json" in entry else {}),
            )
            out.mkdir(parents=True, exist_ok=True)
            (out / (entry["name"] + ".body")).write_bytes(result.body)
            manifest.append({
                "name": entry["name"], "purpose": entry["purpose"],
                "status": result.status, "headers": dict(headers[host]),
                "sha256": hashlib.sha256(result.body).hexdigest(),
                "received_at": result.received_at.isoformat(), "outcome": result.outcome,
            })
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=0.2)
    args = parser.parse_args(argv)
    if not math.isfinite(args.rate) or args.rate <= 0:
        raise ValueError("rate must be finite and positive")
    entries = load_plan(args.plan)
    try:
        asyncio.run(capture(entries, args.out, args.rate))
    except RuntimeError as exc:
        parser.exit(1, str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
