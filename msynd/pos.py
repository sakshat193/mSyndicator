from bisect import bisect_right


def pos(t, t_dep_a, anchor_t, anchor_km, t_arr_b, km_a, km_b,
        pattern: list[tuple[float, float]] | None) -> float:
    """Interpolate route km from the last anchor using scheduled passing points."""
    if t < t_dep_a:
        return km_a
    if t >= t_arr_b:
        return km_b
    if t <= anchor_t:
        return anchor_km

    fraction = (t-anchor_t)/(t_arr_b-anchor_t)
    if pattern:
        i = bisect_right(pattern, fraction, key=lambda point: point[0])
        t0, k0 = pattern[i-1] if i else (0, 0)
        t1, k1 = pattern[i] if i < len(pattern) else (1, 1)
        fraction = k1 if t1 == t0 else k0 + (k1-k0)*(fraction-t0)/(t1-t0)
    return anchor_km + (km_b-anchor_km)*fraction
