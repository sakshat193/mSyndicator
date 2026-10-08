from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from msynd.ops.nightly import backup, disk_check, prune_bodies, prune_rows
from tests.helpers import insert_event, insert_fetch, make_run

NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)


@pytest.mark.db
def test_old_referenced_fetch_pruned_event_kept(db):
    run = make_run(db)
    old = insert_fetch(db, started_at=NOW - timedelta(days=181))
    edge = insert_fetch(db, started_at=NOW - timedelta(days=180))
    event = insert_event(db, run, 'BRC', 1, 'arr', NOW, NOW, old)
    for days in (181, 180):
        db.execute("INSERT INTO obs.pending_rev VALUES (%s,'BRC',1,'dep','ntes_status',%s,%s)",
                   (run, old, NOW - timedelta(days=days)))
    for days in (121, 120):
        db.execute("INSERT INTO sim.forecast_issue VALUES (%s,%s,'{}','{}','{}',false,'{}')",
                   (run, NOW - timedelta(days=days)))
    prune_rows(db, NOW)
    assert db.execute('SELECT fetch_id FROM obs.event_rev WHERE id=%s', (event,)).fetchone() == (None,)
    assert db.execute('SELECT id FROM obs.fetch WHERE id=%s', (edge,)).fetchone() == (edge,)
    assert db.execute('SELECT received_at,fetch_id FROM obs.pending_rev WHERE run_id=%s', (run,)).fetchall() == [(NOW - timedelta(days=180), None)]
    assert db.execute('SELECT issued_at FROM sim.forecast_issue WHERE run_id=%s', (run,)).fetchall() == [(NOW - timedelta(days=120),)]


@pytest.mark.db
def test_eta_35_days(db):
    run = make_run(db)
    for days in (36, 35, 34):
        db.execute("INSERT INTO obs.ntes_eta_rev VALUES (%s,'BRC',1,'arr',%s,NULL,%s)",
                   (run, NOW, NOW - timedelta(days=days)))
    prune_rows(db, NOW)
    assert db.execute('SELECT received_at FROM obs.ntes_eta_rev WHERE run_id=%s ORDER BY received_at', (run,)).fetchall() == [(NOW - timedelta(days=d),) for d in (35, 34)]


@pytest.mark.db
def test_body_retention_drops_to_2h_at_80pct(db, caplog):
    pct, hours = disk_check('.', conn=db, usage=lambda _: (100, 80, 20))
    assert (pct, hours) == (80, 2)
    assert 'disk' in caplog.text
    assert db.execute("SELECT value FROM sim.metric WHERE name='disk_pct'").fetchone() == (80,)
    ids = [insert_fetch(db, started_at=NOW - timedelta(hours=h), body_zstd=b'body') for h in (1, 2, 3, 24)]
    # Avoid the stable 1% sample in this test.
    db.execute('UPDATE obs.fetch SET outcome=\'ok\' WHERE id=ANY(%s)', (ids,))
    prune_bodies(db, NOW, hours)
    assert db.execute('SELECT id,body_zstd FROM obs.fetch WHERE id=ANY(%s) ORDER BY id', (ids,)).fetchall() == [(i, b'body' if h <= 2 or i % 100 == 0 else None) for i, h in zip(ids, (1, 2, 3, 24))]


@pytest.mark.db
def test_sample_kept_14_days(db):
    db.execute('DELETE FROM obs.fetch')
    rows = [(100, 'ok', 13), (200, 'ok', 14), (300, 'ok', 15),
            (101, 'ok', 2), (102, 'challenge', 13), (103, 'challenge', 15)]
    for ident, outcome, days in rows:
        db.execute("INSERT INTO obs.fetch(id,host,purpose,started_at,outcome,body_zstd) VALUES (%s,'h','status',%s,%s,%s)", (ident, NOW - timedelta(days=days), outcome, b'body'))
    prune_bodies(db, NOW, 24)
    assert db.execute('SELECT id FROM obs.fetch WHERE body_zstd IS NOT NULL ORDER BY id').fetchall() == [(100,), (102,), (200,)]


@pytest.mark.parametrize('used,hours,alarm', [(69, 24, False), (70, 24, True), (79, 24, True), (80, 2, True)])
def test_disk_thresholds(used, hours, alarm, caplog):
    class Connection:
        def execute(self, query, args):
            assert args == (used,)
    assert disk_check('.', conn=Connection(), usage=lambda _: (100, used, 100-used)) == (used, hours)
    assert bool(caplog.text) == alarm


def runner(calls, objects=(), fail=None):
    objects = list(objects)
    def run(args, **kwargs):
        calls.append(args)
        assert kwargs['check'] is True
        if fail and fail(args):
            raise subprocess.CalledProcessError(1, args)
        if args[0] == 'pg_dump':
            Path(args[args.index('--file') + 1]).write_bytes(b'dump')
        if 'put' in args:
            objects.append({'name': args[args.index('--name') + 1], 'size': 4})
        return subprocess.CompletedProcess(args, 0, json.dumps({'data': list(objects)}) if 'list' in args else '')
    return run


def test_backup_contents_rotation_and_alarm(tmp_path, caplog):
    calls = []
    for date in ('2026-09-01', '2026-09-02'):
        (tmp_path / f'local-{date}.dump').write_bytes(b'old')
    unrelated = tmp_path / 'keep.txt'
    unrelated.write_text('keep')
    objects = [{'name': f'msynd/daily/2026-09-{i:02}.dump', 'size': 2_000_000_000} for i in range(1, 10)]
    objects += [{'name': f'msynd/weekly/2026-08-{i:02}.dump', 'size': 1} for i in range(1, 6)]
    objects += [{'name': 'unrelated', 'size': 1}]
    backup(database_url='test-dsn', bucket='test-bucket', directory=tmp_path, now=NOW, run=runner(calls, objects))
    dumps = [a for a in calls if a[0] == 'pg_dump']
    assert len(dumps) == 2
    assert '--format=custom' in dumps[0]
    assert '--schema=obs' in dumps[0] and '--schema=obs' not in dumps[1]
    for args in dumps:
        assert '--schema=ref' in args and '--schema=ml' in args and '--schema=sim' in args
        assert '--exclude-table=sim.frame' in args and '--exclude-table=sim.metric' in args
    uploads = [a for a in calls if 'put' in a]
    assert len(uploads) == 1
    assert 'offsite-' in uploads[0][uploads[0].index('--file') + 1]
    assert len(list(tmp_path.glob('local-*.dump'))) == 2
    assert not list(tmp_path.glob('offsite-*'))
    assert unrelated.exists()
    deleted = [a[a.index('--name') + 1] for a in calls if 'delete' in a]
    assert len(deleted) == 4 and 'unrelated' not in deleted
    assert '15 GB' in caplog.text


def test_weekly_backup(tmp_path):
    calls = []
    backup(database_url='test', bucket='test', directory=tmp_path,
           now=NOW + timedelta(days=3), run=runner(calls))
    assert len([a for a in calls if 'put' in a]) == 2
    assert any('msynd/weekly/2026-10-11.dump' in a for a in calls)


@pytest.mark.parametrize('failure', ['pg_dump', 'put'])
def test_failed_backup_preserves_two_valid_copies(tmp_path, failure):
    for date in ('2026-09-01', '2026-09-02'):
        (tmp_path / f'local-{date}.dump').write_bytes(b'old')
    calls = []
    with pytest.raises(subprocess.CalledProcessError):
        backup(database_url='test', bucket='test', directory=tmp_path, now=NOW,
               run=runner(calls, fail=lambda a: failure in a))
    assert len(list(tmp_path.glob('local-*.dump'))) == 2
    assert not any('delete' in a for a in calls)


def test_backup_requires_bucket_before_dump(tmp_path):
    with pytest.raises(ValueError, match='bucket'):
        backup(database_url='test', bucket='', directory=tmp_path, now=NOW, run=lambda *a, **k: pytest.fail('command started'))
