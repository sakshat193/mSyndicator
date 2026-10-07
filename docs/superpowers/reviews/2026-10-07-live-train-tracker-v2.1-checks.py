"""Design witnesses for v2.1; not tests of an implemented train engine.

Run: python docs/superpowers/reviews/2026-10-07-live-train-tracker-v2.1-checks.py
Only the standard library is needed. SQL witness uses SQLite's matching
default foreign-key NO ACTION behavior, not a PostgreSQL integration test.
"""

from itertools import product
from math import exp, floor, isclose, log
import sqlite3


def main():
    results = []

    # The weighted proposal resolves the original upstream-conditioning witness.
    surviving = [(x, d) for x, d in product((0, 10), (10, 20)) if x + d > 15]
    posterior = sum(x == 10 for x, _ in surviving) / len(surviving)
    assert isclose(posterior, 2 / 3)
    results.append("Resolved: weighted upstream posterior is 2/3")

    # Filter receipt times before selecting the latest eligible revision.
    revisions = [(600, 590), (630, 595)]  # received minute, actual minute
    eligible = [row for row in revisions if row[0] <= 615]
    assert max(eligible)[1] == 590 and max(revisions)[1] == 595
    results.append("Resolved: as-of reference returns 09:50; current truth is 09:55")

    scheduled, rescheduled, actual = 100, 20, 130
    residual = actual - (scheduled + rescheduled)
    assert residual == 10 and scheduled + rescheduled + residual == actual
    results.append("Resolved: rescheduling counted once gives departure 130")

    scheduled, rescheduled, actual = 100, 60, 150
    residual = actual - (scheduled + rescheduled)
    assert residual == -10 and scheduled + rescheduled > actual
    results.append("Blocker: O=-10 is permitted by its label but absent from O's bins")

    # A rare observation interval has positive mass and a uniform conditional law.
    window_mass, tau, lo, hi = 5e-7, 1, 0, 1
    assert 0 < window_mass < 1e-6
    fallback = min(lo + tau * -log(1 - 0.9), hi)
    endpoint_mass = exp(-(hi - lo) / tau)
    fallback_mean = 1 - exp(-1)  # E[min(Exp(1), 1)]
    assert fallback == hi and endpoint_mass > 0.36
    assert not isclose(fallback_mean, 0.5)
    results.append(
        f"Blocker: rare-window fallback puts {endpoint_mass:.2%} at excluded hi; "
        f"mean {fallback_mean:.6f}, correct mean 0.5"
    )

    # F must describe the effective prior, including physical-floor atoms.
    raw_window_mass = (6 - 5) / 10  # D uniform on [0,10]
    floored_window_mass = 6 / 10  # max(5,D) in [5,6)
    assert isclose(raw_window_mass, 0.1) and isclose(floored_window_mass, 0.6)
    results.append("G7 addition: physical-floor window mass is 0.6, not raw-CDF 0.1")

    eta_seen, latest_ok_fetch, issue = 540, 600, 610
    # Latest ok fetch contains another ETA, but does not contain this event key.
    run_confirmation_eligible = issue - latest_ok_fetch <= 30
    event_confirmation_eligible = issue - eta_seen <= 30
    assert run_confirmation_eligible and not event_confirmation_eligible
    results.append("Blocker: run-wide ok fetch makes a 70-minute-old missing ETA fresh")

    requested_rate = 40 * 2 / 120
    exclusive_max_cohort = floor(0.3 * 120 / 2)
    assert requested_rate > 0.3 and exclusive_max_cohort == 18
    results.append(
        f"G2 clarification: 40 concurrent runs need {requested_rate:.3f} req/s; "
        "at most 18 fit before other traffic"
    )

    # A zero-failure sample cannot establish a zero-probability tail.
    assert (1 - 0.01) ** 100 > 0.36
    results.append("G2 clarification: 100 observations miss a 1% tail with 36.6% probability")

    # One persistent publication delay: later non-report subsumes earlier non-report.
    candidates, checks = (0, 1), (2, 3)
    latest_weights = [exp(-(max(checks) - t)) for t in candidates]
    repeated_weights = [exp(-sum(r - t for r in checks)) for t in candidates]
    correct = latest_weights[1] / sum(latest_weights)
    double_counted = repeated_weights[1] / sum(repeated_weights)
    assert isclose(correct, 0.7310585786300049)
    assert isclose(double_counted, 0.8807970779778824)
    results.append(
        f"G7 addition: repeated pending gives {double_counted:.6f}, "
        f"latest cumulative marker gives {correct:.6f}"
    )

    with sqlite3.connect(":memory:") as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("CREATE TABLE fetch (id INTEGER PRIMARY KEY)")
        db.execute("CREATE TABLE event_rev (fetch_id INTEGER REFERENCES fetch(id))")
        db.execute("INSERT INTO fetch VALUES (1)")
        db.execute("INSERT INTO event_rev VALUES (1)")
        try:
            db.execute("DELETE FROM fetch WHERE id = 1")
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("Retained evidence must block the default FK deletion")
    results.append("Retention defect: retained event revision blocks deletion of its fetch row")

    print(f"{len(results)} focused design checks passed:")
    for result in results:
        print(f"- {result}")


if __name__ == "__main__":
    main()
