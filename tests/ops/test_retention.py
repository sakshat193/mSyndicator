from datetime import datetime, timedelta, timezone
import json
import logging
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
        if args[0] == 'pg_dump':
            Path(args[args.index('--file') + 1]).write_bytes(b'dump')
        if fail and fail(args):
            raise subprocess.CalledProcessError(1, args)
        if args[:2] == ['rclone', 'copyto']:
            objects.append({'Path': args[3].removeprefix('gdrive:msynd-backups/'), 'Size': 4})
        if args[:2] == ['rclone', 'lsjson']:
            assert kwargs == {'check': True, 'capture_output': True, 'text': True}
        return subprocess.CompletedProcess(args, 0, json.dumps(objects) if 'lsjson' in args else '')
    return run


def test_backup_contents_rotation_and_alarm(tmp_path, caplog):
    calls = []
    for date in ('2026-09-01', '2026-09-02'):
        (tmp_path / f'local-{date}.dump').write_bytes(b'old')
    unrelated = tmp_path / 'keep.txt'
    unrelated.write_text('keep')
    objects = [{'Path': f'daily/2026-09-{i:02}.dump', 'Size': 2_000_000_000} for i in range(9, 0, -1)]
    objects += [{'Path': f'weekly/2026-08-{i:02}.dump', 'Size': 1} for i in range(5, 0, -1)]
    objects += [{'Path': name, 'Size': 1} for name in (
        'unrelated', 'msynd/daily/2026-01-01.dump', 'daily/2026-01-01.dump.partial',
        'daily/nested/2026-01-01.dump', 'daily/not-a-date.dump')]
    backup(database_url='test-dsn', remote='gdrive:msynd-backups', directory=tmp_path, now=NOW, run=runner(calls, objects))
    dumps = [a for a in calls if a[0] == 'pg_dump']
    assert len(dumps) == 2
    assert '--format=custom' in dumps[0]
    assert '--schema=obs' in dumps[0] and '--schema=obs' not in dumps[1]
    for args in dumps:
        assert '--schema=ref' in args and '--schema=ml' in args and '--schema=sim' in args
        assert '--exclude-table=sim.frame' in args and '--exclude-table=sim.metric' in args
    assert [a for a in calls if 'copyto' in a] == [[
        'rclone', 'copyto', str(tmp_path / 'offsite-2026-10-08.dump'),
        'gdrive:msynd-backups/daily/2026-10-08.dump']]
    assert [a for a in calls if 'lsjson' in a] == [
        ['rclone', 'lsjson', 'gdrive:msynd-backups', '--recursive', '--files-only']]
    assert len(list(tmp_path.glob('local-*.dump'))) == 2
    assert not list(tmp_path.glob('offsite-*'))
    assert unrelated.exists()
    assert [a for a in calls if 'deletefile' in a] == [
        ['rclone', 'deletefile', '--drive-use-trash=false', 'gdrive:msynd-backups/daily/2026-09-01.dump'],
        ['rclone', 'deletefile', '--drive-use-trash=false', 'gdrive:msynd-backups/daily/2026-09-02.dump'],
        ['rclone', 'deletefile', '--drive-use-trash=false', 'gdrive:msynd-backups/daily/2026-09-03.dump'],
        ['rclone', 'deletefile', '--drive-use-trash=false', 'gdrive:msynd-backups/weekly/2026-08-01.dump']]
    assert '15 GB' in caplog.text


def test_weekly_backup(tmp_path, monkeypatch):
    monkeypatch.setenv('BACKUP_REMOTE', 'gdrive:msynd-backups')
    calls = []
    backup(database_url='test', directory=tmp_path,
           now=datetime(2026, 10, 10, 21, tzinfo=timezone.utc), run=runner(calls))
    assert [a for a in calls if a[0] == 'rclone'] == [
        ['rclone', 'copyto', str(tmp_path / 'offsite-2026-10-11.dump'),
         'gdrive:msynd-backups/daily/2026-10-11.dump'],
        ['rclone', 'copyto', str(tmp_path / 'offsite-2026-10-11.dump'),
         'gdrive:msynd-backups/weekly/2026-10-11.dump'],
        ['rclone', 'lsjson', 'gdrive:msynd-backups', '--recursive', '--files-only']]


@pytest.mark.parametrize('size,alarm', [(14_999_999_996, False), (14_999_999_997, True)])
def test_backup_size_alarm(tmp_path, caplog, size, alarm):
    objects = [{'Path': 'daily/2026-10-07.dump', 'Size': size},
               {'Path': 'unrelated', 'Size': 99_000_000_000}]
    backup(database_url='test', remote='gdrive:msynd-backups', directory=tmp_path, now=NOW,
           run=runner([], objects))
    assert ('off-site backup size alarm: exceeds 15 GB' in caplog.text) == alarm


@pytest.mark.parametrize('remote,failure', [
    ('gdrive:msynd-backups', 'pg_dump'), ('gdrive:msynd-backups', 'copyto'), ('', 'pg_dump')])
def test_failed_backup_preserves_two_valid_copies(tmp_path, caplog, remote, failure):
    for date in ('2026-09-01', '2026-09-02'):
        (tmp_path / f'local-{date}.dump').write_bytes(b'old')
    calls = []
    with pytest.raises(subprocess.CalledProcessError):
        backup(database_url='test', remote=remote, directory=tmp_path, now=NOW,
               run=runner(calls, fail=lambda a: failure in a))
    copies = sorted(tmp_path.glob('local-*.dump'))
    assert [p.read_bytes() for p in copies] == ([b'old', b'dump'] if failure == 'copyto' else [b'old', b'old'])
    assert not list(tmp_path.glob('*.partial')) and not list(tmp_path.glob('offsite-*'))
    assert not any('deletefile' in a for a in calls)
    assert 'backup failed' in caplog.text


@pytest.mark.parametrize('remote,env_remote', [('', 'gdrive:msynd-backups'), (None, ''), (None, None)])
def test_backup_empty_remote_keeps_two_local_copies(tmp_path, monkeypatch, caplog, remote, env_remote):
    if env_remote is None:
        monkeypatch.delenv('BACKUP_REMOTE', raising=False)
    else:
        monkeypatch.setenv('BACKUP_REMOTE', env_remote)
    for date in ('2026-09-01', '2026-09-02'):
        (tmp_path / f'local-{date}.dump').write_bytes(b'old')
    unrelated = tmp_path / 'keep.txt'
    unrelated.write_text('keep')
    calls = []

    local = backup(database_url='test-dsn', remote=remote, directory=tmp_path, now=NOW,
                   run=runner(calls))

    assert local == tmp_path / 'local-2026-10-08.dump'
    assert local.read_bytes() == b'dump'
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'keep.txt', 'local-2026-09-02.dump', 'local-2026-10-08.dump']
    assert unrelated.read_text() == 'keep'
    assert len(calls) == 1
    assert calls[0][0] == 'pg_dump'
    assert '--schema=obs' in calls[0]
    assert caplog.record_tuples == [
        ('msynd.ops.nightly', logging.WARNING, 'off-site backup disabled: BACKUP_REMOTE empty')]
