from datetime import timedelta


def b2_eta(conn, run_id, station, occ, kind, t):
    row = conn.execute('''SELECT expected_ts,received_at FROM obs.ntes_eta_rev
        WHERE run_id=%s AND station=%s AND occ=%s AND kind=%s AND received_at<=%s
        ORDER BY received_at DESC LIMIT 1''', (run_id, station, occ, kind, t)).fetchone()
    return row[0] if row is not None and t - row[1] <= timedelta(minutes=30) else None
