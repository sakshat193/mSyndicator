"""Read-only acceptance measurements; live and restore gates remain external."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from msynd.db import connect


def a1(conn, since, until):
    rows = conn.execute("""SELECT ts, name, value FROM sim.metric
        WHERE ts >= %s AND ts <= %s AND name = ANY(%s) ORDER BY ts, name""",
        (since, until, ['a1_' + name for name in
         ('active', 'discovered', 'fresh', 'active_1h', 'discovered_1h', 'cold')])).fetchall()
    minutes = {}
    for ts, name, value in rows:
        minutes.setdefault(ts, {})[name[3:]] = value
    result = {'minutes': len(minutes),
              'complete_minutes': sum(len(row) == 6 and all(v is not None for v in row.values())
                                      for row in minutes.values())}
    for name, denominator in (('discovered', 'active'), ('fresh', 'active'),
                              ('discovered_1h', 'active_1h')):
        ratios = [row[name] / row[denominator] for row in minutes.values()
                  if row.get(name) is not None and row.get(denominator, 0)
                  and row[denominator] > 0]
        result[name] = {'mean': mean(ratios) if ratios else None,
                        'min': min(ratios) if ratios else None, 'minutes': len(ratios)}
    minimum = result['discovered_1h']['min']
    result['target_passes'] = minimum >= .95 if minimum is not None else None
    cold = [row['cold'] > 0 for row in minutes.values() if row.get('cold') is not None]
    result['cold_share'] = mean(cold) if cold else None
    return result


def a2(conn, since, until):
    """Snapshot at until, including older runs; since does not truncate last evidence."""
    rows = conn.execute("""WITH ages AS (
        SELECT t.class, s.zone, e.received_at,
               extract(epoch FROM (%s::timestamptz-e.received_at))/60 AS age
        FROM obs.run r JOIN ref.train t ON t.number=r.train
        JOIN LATERAL (SELECT station FROM ref.schedule_stop
            WHERE version_id=r.schedule_version_id ORDER BY idx LIMIT 1) origin ON true
        LEFT JOIN LATERAL (SELECT station, received_at FROM obs.event_rev
            WHERE run_id=r.id AND received_at <= %s
            ORDER BY received_at DESC, id DESC LIMIT 1) e ON true
        LEFT JOIN ref.station s ON s.code=coalesce(e.station, origin.station)
        WHERE r.start_date <= (%s::timestamptz AT TIME ZONE 'Asia/Kolkata')::date)
        SELECT class, zone, count(*), count(*) FILTER (WHERE received_at IS NULL),
               percentile_cont(.5) WITHIN GROUP (ORDER BY age),
               percentile_cont(.9) WITHIN GROUP (ORDER BY age)
        FROM ages GROUP BY class, zone ORDER BY class, zone""", (until, until, until)).fetchall()
    return [dict(zip(('class', 'zone', 'runs', 'unseen', 'p50_min', 'p90_min'), row)) for row in rows]


def a5(conn, since, until):
    # PostgreSQL timestamps have microsecond precision: (end-60s, end] has at most 60 paced starts.
    rows = conn.execute("""WITH starts AS (
        SELECT host, started_at, 1 AS weight FROM obs.fetch
        WHERE started_at > %s::timestamptz - INTERVAL '60 seconds' AND started_at <= %s),
        endpoints AS (
        SELECT * FROM starts UNION ALL SELECT DISTINCT host, %s::timestamptz, 0 FROM starts),
        windows AS (
        SELECT host, started_at, sum(weight) OVER (PARTITION BY host ORDER BY started_at
            RANGE BETWEEN INTERVAL '59.999999 seconds' PRECEDING AND CURRENT ROW) AS starts
        FROM endpoints)
        SELECT host, max(starts) FROM windows WHERE started_at >= %s
        GROUP BY host ORDER BY host""", (since, until, since, since)).fetchall()
    return [{'host': host, 'max_starts': count, 'passes': count <= 60} for host, count in rows]


def a8(conn, since, until):
    """Completed cohort = first receipt of actual destination arrival in the interval."""
    rows = conn.execute("""WITH completed AS (
        SELECT r.id, r.schedule_version_id, t.class
        FROM obs.run r JOIN ref.train t ON t.number=r.train
        JOIN LATERAL (SELECT station, occ FROM ref.schedule_stop
            WHERE version_id=r.schedule_version_id ORDER BY idx DESC LIMIT 1) dest ON true
        JOIN obs.event_rev e ON e.run_id=r.id AND e.station=dest.station
            AND e.occ=dest.occ AND e.kind='arr'
        WHERE e.received_at <= %s AND e.actual_ts <= %s
        GROUP BY r.id, t.class HAVING min(e.received_at) >= %s),
        rates AS (
        SELECT r.id, r.class, count(*) AS expected,
               count(*) FILTER (WHERE e.id IS NULL) AS missing
        FROM completed r JOIN ref.schedule_stop s ON s.version_id=r.schedule_version_id
        CROSS JOIN LATERAL (VALUES ('arr',s.arr_off), ('dep',s.dep_off)) k(kind,off)
        LEFT JOIN obs.event_asof(%s) e ON e.run_id=r.id AND e.station=s.station
            AND e.occ=s.occ AND e.kind::text=k.kind
        WHERE k.off IS NOT NULL
        GROUP BY r.id, r.class)
        SELECT class, count(*), sum(expected)::bigint, sum(missing)::bigint,
               avg(missing::double precision/expected)
        FROM rates GROUP BY class ORDER BY class""", (until, until, since, until)).fetchall()
    return [dict(zip(('class', 'runs', 'expected', 'missing', 'mean_missing_rate'), row)) for row in rows]


def invalid(conn, since, until):
    total, count = conn.execute("""SELECT count(*), count(*) FILTER (WHERE outcome='invalid')
        FROM obs.fetch WHERE started_at >= %s AND started_at <= %s
          AND received_at <= %s AND outcome <> 'inflight'""", (since, until, until)).fetchone()
    share = count / total if total else None
    return {'total': total, 'invalid': count, 'share': share,
            'passes': share < .01 if share is not None else None}


def _value(value):
    if value is None:
        return 'unavailable'
    if isinstance(value, bool):
        return 'pass' if value else 'fail'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render(name, measurements, since, until):
    lines = [f'# {name} report', '', f'Interval (UTC): {since.isoformat()} .. {until.isoformat()} (inclusive).',
             'Actual evidence is filtered by receipt time before selecting revisions.', '']
    for section, data in measurements.items():
        lines += [f'## {section.upper() if section != "invalid" else section}', '']
        if section == 'a1':
            lines += ['Engine per-minute coverage ratios; zero denominators are unavailable.',
                      'Target: minimum discovered_1h / active_1h >= 0.95 over eligible minutes.', '']
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f'- {key}: ' + ', '.join(f'{k}={_value(v)}' for k, v in value.items()))
                else:
                    lines.append(f'- {key}: {_value(value)}')
        elif isinstance(data, list):
            if section == 'a2':
                lines += ['Snapshot at until of runs started by that date, including older runs.',
                          'Ages are minutes; unseen runs have no finite age. Zone is last event station, origin before any event.', '']
            elif section == 'a5':
                lines += ['Maximum starts in a half-open 60-second window ending in the interval, by host; limit 60.', '']
            elif section == 'a8':
                lines += ['Runs whose actual destination arrival was first received in the interval.',
                          'Required halt arrivals/departures only; mean of per-run missing-event fractions.', '']
            if data:
                keys = list(data[0])
                lines += ['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join('---' for _ in keys) + ' |']
                lines += ['| ' + ' | '.join(_value(row[k]) for k in keys) + ' |' for row in data]
            else:
                lines.append('unavailable (no eligible data)')
        else:
            lines += ['Completed requests started in the interval and received by until; invalid target < 0.01.', '']
            lines += [f'- {key}: {_value(value)}' for key, value in data.items()]
        lines.append('')
    if name == 'phase2':
        lines += ['## External exit evidence', '',
                  '- Continuous live collection: unavailable (monitor must verify 7 continuous days).',
                  '- Restore drill: unavailable (monitor must supply the measured drill result).',
                  '- Phase 2: unavailable; measurements alone cannot establish these external gates.', '']
    return '\n'.join(lines)


def _timestamp(value):
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected an ISO timestamp with timezone') from exc
    if result.utcoffset() is None:
        raise argparse.ArgumentTypeError('timestamp requires an explicit timezone')
    return result.astimezone(timezone.utc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', choices=('a1', 'a2', 'a5', 'a8', 'invalid', 'phase2'))
    parser.add_argument('--since', required=True, type=_timestamp)
    parser.add_argument('--until', required=True, type=_timestamp)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.since >= args.until:
        parser.error('--since must be earlier than --until')
    names = ('a2', 'a5', 'a8', 'invalid') if args.report == 'phase2' else (args.report,)
    with connect() as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        measurements = {name: globals()[name](conn, args.since, args.until) for name in names}
    text = render(args.report, measurements, args.since, args.until)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
