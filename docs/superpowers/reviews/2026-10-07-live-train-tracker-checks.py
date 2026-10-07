"""Reproduce design counterexamples; these are not tests of an existing engine.

Run from the repository: python docs/superpowers/reviews/2026-10-07-live-train-tracker-checks.py
Only the Python standard library is needed.
"""

from math import exp, isclose
from random import Random
from struct import calcsize


def main():
    findings = []

    # Logical poll rate assumes one HTTP request per poll; observed NTES flow uses two.
    targeted_minutes = 2500 / 0.7 / 60
    assert targeted_minutes > 40
    findings.append(f"70% targeted budget: {targeted_minutes:.2f} min minimum sweep")
    fallback_minutes = 2500 / 0.9 / 60
    assert fallback_minutes > 40
    findings.append(f"90% targeted fallback: {fallback_minutes:.2f} min minimum sweep")
    assert 2500 * 2 / 0.9 / 60 > 90
    findings.append("Fresh CSRF + enquiry: 92.59 min at 90% targeted budget")
    assert isclose(400 * 2 / 0.3 / 60, 44.44444444444444)
    findings.append("400 boards, 30% budget, two requests/action: 44.44 min sweep")

    # A standard initially-full token bucket can burst, then refill at its steady rate.
    starts = [0.0] * 3 + list(range(1, 60))
    requests_in_window = sum(0 <= t < 60 for t in starts)
    assert requests_in_window == 62 and requests_in_window > 60
    findings.append("Rate 1/s, capacity 3 permits 62 requests in [0,60): S4 fails")
    assert isclose(3900 * 2 / 0.2 / 3600, 10.833333333333334)
    findings.append("Two etrain requests/train: 10.83 h backfill, before retries")

    # Same run and leg, different arrival/departure training targets.
    scheduled_dwell = 10
    arrival_delay_a, departure_delay_a, arrival_delay_b = 0, 15, 20
    history_delta = arrival_delay_b - arrival_delay_a
    live_delta = arrival_delay_b - departure_delay_a
    actual_dwell = scheduled_dwell + departure_delay_a - arrival_delay_a
    assert history_delta == 20 and live_delta == 5 and actual_dwell == 25
    findings.append("One leg: history target +20 min, live target +5 min; dwell 25 min")
    minimum_hours = 100 / 100
    minimum_minutes = 60 * 100 / 100
    assert minimum_hours == 1 and minimum_minutes == 60
    findings.append("100 km at 100 km/h: floor is 60 min, not 1 min")

    # Event time and information-availability time differ.
    actual_time, received_time, forecast_cut = 10, 20, 15
    assert actual_time < forecast_cut < received_time
    findings.append("actual_ts<t accepts an event not received until after t")
    forecast_cut, eta_fetch_times = 15.6, [14.0, 16.0]
    nearest = min(eta_fetch_times, key=lambda t: abs(t - forecast_cut))
    prior = max(t for t in eta_fetch_times if t <= forecast_cut)
    assert nearest > forecast_cut and prior <= forecast_cut
    findings.append("Closest NTES ETA snapshot can be from the future")

    # The proposed event-time cursor cannot recover older arrivals or revisions.
    checkpoint_time, late_event_time = 20, 10
    assert not late_event_time > checkpoint_time
    findings.append("Late/corrected t=10 event is discarded by an event-time t=20 cursor")
    old_route = {1: "A", 2: "B"}
    new_route = {1: "A", 2: "PASS", 3: "B"}
    persisted_event_seq = 2
    assert old_route[persisted_event_seq] != new_route[persisted_event_seq]
    findings.append("Inserting a passing stop changes the meaning of stored seq=2")

    # Stable log-weight normalization does not underflow for a very surprising event.
    log_weights = [-1000.0, -1001.0]
    shifted = [exp(w - max(log_weights)) for w in log_weights]
    assert sum(shifted) > 0 and all(w > 0 for w in shifted)
    findings.append("Stable log normalization never activates the all-weights-underflow reset")
    entered, elapsed, sampled_duration = 0, 30, 20
    assert entered + sampled_duration < entered + elapsed
    findings.append("Roughening/rebuilding can move a leg's end into already-observed past")

    # Mean-field segment assignment can invent occupancy between two distinct modes.
    positions = [0, 200]
    mean_position = sum(positions) / len(positions)
    assert mean_position == 100 and mean_position not in positions
    findings.append("Bimodal 0/200 km posterior gives phantom occupancy at 100 km")
    residual = 10
    per_segment_biases = [residual] * 3
    assert sum(per_segment_biases) == 3 * residual
    findings.append("Assigning a whole-leg residual to each segment triples an additive correction")

    # A forecast delta is not recoverable after sim.latest overwrites an earlier delta.
    client_state = {"A": 0, "B": 0}
    missed_delta, latest_delta = {"A": 1}, {"B": 1}
    client_state.update(latest_delta)
    assert client_state["A"] != missed_delta["A"]
    findings.append("Missed frame delta cannot be recovered from the latest delta alone")

    # Batch-number seeds explicitly change when the same observations are regrouped.
    batch_rng = Random(1)
    one_batch = [batch_rng.random(), batch_rng.random()]
    two_batches = [Random(1).random(), Random(2).random()]
    assert one_batch != two_batches
    findings.append("Batch-dependent seeds do not establish observation-stream determinism")
    record_bytes = calcsize("<hff")
    assert record_bytes == 10 and 2500 * 32 * 8 * record_bytes == 6_400_000
    findings.append("int16 leg + two float32 times: 6.4 MB forecast, before headers")

    print(f"{len(findings)} design counterexamples verified:")
    for finding in findings:
        print(f"- {finding}")


if __name__ == "__main__":
    main()
