from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from tests.helpers import make_run, insert_event, insert_fetch

pytestmark = pytest.mark.db
IST = timezone(timedelta(hours=5, minutes=30))


def ist(value):
    return datetime.fromisoformat(value).replace(tzinfo=IST)


def test_event_asof_filters_before_latest(db):
    r = make_run(db)
    insert_event(db, r, "BRC", 1, "arr", ist("2026-10-07 09:50"), ist("2026-10-07 10:00"))
    insert_event(db, r, "BRC", 1, "arr", ist("2026-10-07 09:55"), ist("2026-10-07 10:30"))
    assert db.execute("select actual_ts from obs.event_asof(%s) where run_id=%s", (ist("2026-10-07 10:15"), r)).fetchone()[0] == ist("2026-10-07 09:50")
    assert db.execute("select actual_ts from obs.event where run_id=%s", (r,)).fetchone()[0] == ist("2026-10-07 09:55")


def test_fetch_retention_sets_null_never_cascades(db):
    now = datetime.now(timezone.utc)
    f = insert_fetch(db, started_at=now - timedelta(days=181))
    r = make_run(db)
    e = insert_event(db, r, "BRC", 1, "arr", now, now, f)
    db.execute("insert into obs.pending_rev values (%s,'BRC',1,'dep','ntes_status',%s,%s)", (r, f, now))
    db.execute("insert into obs.ntes_eta_rev values (%s,'BRC',1,'arr',%s,%s,%s)", (r, now, f, now))
    db.execute("insert into obs.exception_rev values ('12951','2026-10-07','{}',%s,%s)", (f, now))
    db.execute("insert into obs.history_daily values ('12951','2026-10-07','BRC',1,0,%s)", (f,))
    db.execute("delete from obs.fetch where id=%s", (f,))
    assert db.execute("select fetch_id from obs.event_rev where id=%s", (e,)).fetchone() == (None,)
    for table in ("pending_rev", "ntes_eta_rev", "exception_rev", "history_daily"):
        assert db.execute(f"select fetch_id from obs.{table}").fetchall() == [(None,)]


def test_mark_dirty_bumps_ver(db):
    from msynd.db.dirty import mark_dirty

    r = make_run(db)
    mark_dirty(db, [r])
    first = db.execute("select ver,since from obs.dirty_run where run_id=%s", (r,)).fetchone()
    mark_dirty(db, [r])
    second = db.execute("select ver,since from obs.dirty_run where run_id=%s", (r,)).fetchone()
    assert second[0] > first[0] and second[1] == first[1]
    db.execute("delete from obs.dirty_run where run_id=%s and ver=%s", (r, first[0]))
    assert db.execute("select ver from obs.dirty_run where run_id=%s", (r,)).fetchone() == (second[0],)


def test_event_kind_order(db):
    assert db.execute("select 'arr'::obs.event_kind < 'pass' and 'pass'::obs.event_kind < 'dep'").fetchone() == (True,)


def test_one_active_model_per_role(db):
    query = "insert into ml.model (role,kind,feature_ver,trained_at,train_window,path,tail,metrics,active) values ('run','empirical',1,now(),'[2026-09-01,2026-10-01)','model','{}','{}',true)"
    db.execute(query)
    with pytest.raises(psycopg.errors.UniqueViolation), db.transaction():
        db.execute(query)


def test_outcome_check(db):
    with pytest.raises(psycopg.errors.CheckViolation), db.transaction():
        insert_fetch(db, outcome="weird")


def test_fetch_mode_check(db):
    fetch = insert_fetch(db)
    assert db.execute('SELECT mode,flags FROM obs.fetch WHERE id=%s', (fetch,)).fetchone() == ('poll', [])
    for mode in ('poll', 'harvest', 'probe', 'backfill'):
        insert_fetch(db, mode=mode)
    with pytest.raises(psycopg.errors.CheckViolation), db.transaction():
        insert_fetch(db, mode='weird')


def test_migrations_are_idempotent(db_url):
    from msynd.db import connect, migrate

    migrate.apply(db_url)
    migrate.apply(db_url)
    with connect(db_url) as conn:
        assert conn.execute("select version from schema_migrations order by version").fetchall() == [
            ("0001_init.sql",), ("0002_pending_source_key.sql",), ("0003_history_seq.sql",),
            ("0004_mmt_source.sql",), ("0005_pilot_bridge.sql",), ("0006_fetch_provenance.sql",)
        ]
