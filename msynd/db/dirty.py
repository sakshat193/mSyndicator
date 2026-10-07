def mark_dirty(conn, run_ids):
    if conn.autocommit:
        raise ValueError("mark_dirty must share the evidence transaction")
    if run_ids:
        conn.execute("""
            INSERT INTO obs.dirty_run(run_id)
            SELECT DISTINCT unnest(%s::bigint[])
            ON CONFLICT (run_id) DO UPDATE
            SET ver = nextval('obs.dirty_seq'),
                since = LEAST(obs.dirty_run.since, excluded.since)
        """, (run_ids,))
        conn.execute("NOTIFY dirty")
