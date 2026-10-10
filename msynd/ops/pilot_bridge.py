"""Replay stored pilot status bodies into a separate, migrated view database."""
import argparse
from contextlib import contextmanager
from datetime import date
from time import sleep

import psycopg
import zstandard

from msynd.collect.dispatcher import FetchResult
from msynd.collect.lifecycle import ensure_run
from msynd.collect.parse_mmt import parse_status
from msynd.collect.status import InvalidResponse
from msynd.collect.store import store_status


@contextmanager
def open_source(dsn):
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        conn.read_only = True
        yield conn


def apply_status(dst, row) -> str:
    """Apply one bridge row; the caller owns the evidence transaction."""
    host, purpose, key, started, received, outcome, status, compressed = row
    try:
        train, day = key.split('/')
        start = date.fromisoformat(day)
        if received is None or compressed is None:
            raise InvalidResponse('missing receipt or body')
        payload = zstandard.decompress(compressed)
        page = parse_status(payload, train, start)
    except (InvalidResponse, ValueError, TypeError, AttributeError, zstandard.ZstdError):
        return 'invalid'
    run_id = ensure_run(dst, train, start)
    if run_id is None:
        return 'no_version'
    copied = dst.execute('''INSERT INTO obs.fetch
        (host,purpose,key,run_id,started_at,received_at,outcome,http_status)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
        (host, purpose, key, run_id, started, received, outcome, status)).fetchone()[0]
    store_status(dst, run_id, FetchResult(copied, outcome, status, payload, received), page)
    return 'applied'


def bridge_once(source, target) -> int:
    applied = invalid = no_version = 0
    skipped = []
    with open_source(source) as src, psycopg.connect(target, connect_timeout=5) as dst:
        # ponytail: one pilot source per target; use source-keyed cursors before bridging multiple pilots.
        with dst.transaction():
            dst.execute('INSERT INTO obs.pilot_bridge_cursor DEFAULT VALUES ON CONFLICT DO NOTHING')
            last = dst.execute('SELECT last_fetch_id FROM obs.pilot_bridge_cursor').fetchone()[0]
        rows = src.execute('''SELECT id,host,purpose,key,started_at,received_at,outcome,http_status,body_zstd
            FROM obs.fetch WHERE purpose='status' AND outcome='ok' AND id>%s ORDER BY id''', (last,))
        for row in rows:
            fetch_id, *status_row = row
            with dst.transaction():
                last = dst.execute('SELECT last_fetch_id FROM obs.pilot_bridge_cursor FOR UPDATE').fetchone()[0]
                if fetch_id <= last:
                    continue
                result = apply_status(dst, status_row)
                if result == 'invalid':
                    invalid += 1
                elif result == 'no_version':
                    no_version += 1
                    key = status_row[2]
                    if key not in skipped and len(skipped) < 20:
                        skipped.append(key)
                else:
                    applied += 1
                dst.execute('UPDATE obs.pilot_bridge_cursor SET last_fetch_id=%s', (fetch_id,))
    print(f'applied={applied} no_version={no_version} invalid={invalid} skipped={",".join(skipped)}')
    return applied


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args(argv)
    while True:
        bridge_once(args.source, args.target)
        if not args.follow:
            return
        sleep(60)


if __name__ == '__main__':
    main()
