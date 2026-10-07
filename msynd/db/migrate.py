from pathlib import Path

from msynd.db import connect


def apply(url=None):
    with connect(url) as conn:
        conn.execute("select pg_advisory_lock(hashtext('msynd:migrate'))")
        conn.execute("create table if not exists public.schema_migrations (version text primary key, applied_at timestamptz not null default now())")
        conn.commit()
        for path in sorted(Path(__file__).with_name("migrations").glob("[0-9][0-9][0-9][0-9]_*.sql")):
            with conn.transaction():
                if conn.execute("select 1 from public.schema_migrations where version=%s", (path.name,)).fetchone():
                    continue
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("insert into public.schema_migrations(version) values (%s)", (path.name,))


if __name__ == "__main__":
    apply()
