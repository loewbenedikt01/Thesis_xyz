"""
Test suite for the nowcast-and-persistence change-point regime detector.

Test 4 (causality audit) is the most important test in the module: it is
the guard against lookahead bias creeping into the sequential detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimes.vix.detector import detect
from regimes.vix.state import compute_state

MIN_SEG = 20


# ─────────────────────────────────────────────────────────────────────────
# 1. Null calibration
# ─────────────────────────────────────────────────────────────────────────
def test_null_calibration_mean_run_length_order_of_magnitude(test_thresholds, test_arl0):
    """
    Simulated i.i.d. normal paths, long enough that most fire before the
    series ends (path length = 3x the calibration ARL0). Mean time to first
    false alarm should be the same order of magnitude as ARL0 — not exactly
    equal (the estimator is noisy and truncation at t_max biases it upward),
    but an order of magnitude off indicates a bug in the calibration or the
    detector, not sampling noise.
    """
    rng = np.random.default_rng(7)
    n_paths = 40
    path_len = test_arl0 * 3
    startup = 21

    first_alarm_times = []
    for i in range(n_paths):
        x = rng.standard_normal(path_len)
        idx = pd.bdate_range("2000-01-01", periods=path_len)
        s = pd.Series(x, index=idx)
        out = detect(s, test_thresholds, min_seg=MIN_SEG, startup=startup)
        hits = out.index[out["detected"]]
        if len(hits):
            first_pos = out.index.get_loc(hits[0]) + startup
            first_alarm_times.append(first_pos)
        else:
            # right-censored: treat as having survived to the end of the path
            first_alarm_times.append(path_len)

    mean_time = float(np.mean(first_alarm_times))
    print(f"\n  mean time to first false alarm: {mean_time:.0f}  (target ARL0={test_arl0})")

    # Order-of-magnitude check: within a factor of 5 either way.
    assert test_arl0 / 5 <= mean_time <= test_arl0 * 5, (
        f"mean run length {mean_time:.0f} is not within an order of magnitude "
        f"of ARL0={test_arl0} — likely a calibration or detector bug"
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. Known change point
# ─────────────────────────────────────────────────────────────────────────
def test_known_change_point_is_detected_near_true_location(test_thresholds):
    """
    500 days of N(0, 0.01^2) followed by 500 days of N(0, 0.03^2) (a 3x
    vol increase). The detector must fire, and its change-point estimate
    tau_hat must land within ~40 trading days of the true break at index 500.
    """
    rng = np.random.default_rng(42)
    n1, n2 = 500, 500
    true_cp = n1

    x = np.concatenate(
        [rng.normal(0, 0.01, n1), rng.normal(0, 0.03, n2)]
    )
    idx = pd.bdate_range("2005-01-01", periods=n1 + n2)
    s = pd.Series(x, index=idx)

    out = detect(s, test_thresholds, min_seg=MIN_SEG, startup=21)
    hits = out[out["detected"]]
    assert len(hits) > 0, "no change point detected in a series with an obvious 3x vol jump"

    # At this reduced-scale ARL0, a spurious alarm inside the 500-day null
    # stretch before the true break is possible and not itself a bug (the
    # detector restarts and keeps monitoring). What matters is that SOME
    # detection's tau_hat lands near the true break.
    tau_hat_positions = [s.index.get_loc(d) for d in hits["tau_hat"]]
    closest_pos = min(tau_hat_positions, key=lambda p: abs(p - true_cp))
    delay_from_true = abs(closest_pos - true_cp)

    print(f"\n  true_cp={true_cp}  all tau_hat positions={tau_hat_positions}")
    print(f"  closest tau_hat={closest_pos}  |diff|={delay_from_true}")
    assert delay_from_true <= 40, (
        f"closest tau_hat at position {closest_pos} is {delay_from_true} days from "
        f"the true change point at {true_cp} — expected within 40"
    )


# ─────────────────────────────────────────────────────────────────────────
# 3. Fat-tail robustness
# ─────────────────────────────────────────────────────────────────────────
def test_fat_tail_false_alarm_rate_matches_gaussian_null(test_thresholds):
    """
    The Mood statistic is rank-based, so its null distribution is the same
    under any continuous F — thresholds calibrated on Gaussian innovations
    must remain valid (not over-trigger) under fat-tailed t(4) innovations
    at CONSTANT scale (no true change point present). This is the test that
    justifies using a nonparametric statistic instead of a Gaussian F-test.
    """
    rng = np.random.default_rng(99)
    n_paths = 60
    path_len = 1000
    startup = 21

    # t(4) has variance 4/(4-2) = 2; normalise to unit variance so the ONLY
    # difference from the Gaussian-null paths is tail shape, not scale.
    t4_scale = np.sqrt(4.0 / (4.0 - 2.0))

    def false_alarm_rate(sampler) -> float:
        n_fired = 0
        for _ in range(n_paths):
            x = sampler()
            idx = pd.bdate_range("2010-01-01", periods=path_len)
            s = pd.Series(x, index=idx)
            out = detect(s, test_thresholds, min_seg=MIN_SEG, startup=startup)
            if out["detected"].any():
                n_fired += 1
        return n_fired / n_paths

    p_gauss = false_alarm_rate(lambda: rng.standard_normal(path_len))
    p_fat = false_alarm_rate(lambda: rng.standard_t(4, size=path_len) / t4_scale)

    print(f"\n  false-alarm rate: gaussian={p_gauss:.3f}  t(4)={p_fat:.3f}")

    # Fat tails must not blow up the false-alarm rate relative to the
    # Gaussian-null baseline the thresholds were calibrated against.
    assert p_fat <= p_gauss + 0.25, (
        f"t(4) false-alarm rate ({p_fat:.3f}) is far above the Gaussian-null "
        f"rate ({p_gauss:.3f}) — the rank-based statistic should be robust to "
        f"tail shape"
    )


# ─────────────────────────────────────────────────────────────────────────
# 4. Causality audit — the most important test in the module
# ─────────────────────────────────────────────────────────────────────────
def test_causality_state_at_T_depends_only_on_data_up_to_T(test_thresholds, vix_log_returns):
    """
    For a set of truncation dates T, recompute detect() + compute_state()
    using only returns[:T] and assert the resulting state at T is IDENTICAL
    to the value taken from the full-sample run. This must pass exactly —
    any mismatch means a future observation leaked into a past decision.
    """
    r = vix_log_returns.iloc[:3000]  # VIX returns as the detected series
    full_out = detect(r, test_thresholds, min_seg=MIN_SEG, startup=21)

    check_positions = [100, 500, 1200, 2000, 2900]
    for pos in check_positions:
        T = full_out.index[pos]
        truncated_returns = r.loc[:T]
        truncated_out = detect(truncated_returns, test_thresholds, min_seg=MIN_SEG, startup=21)

        full_row = full_out.loc[T]
        trunc_row = truncated_out.iloc[-1]

        assert bool(full_row["detected"]) == bool(trunc_row["detected"]), T
        assert int(full_row["segment_age"]) == int(trunc_row["segment_age"]), T
        if full_row["detected"]:
            assert full_row["tau_hat"] == trunc_row["tau_hat"], T
            assert full_row["detection_delay"] == trunc_row["detection_delay"], T


def test_causality_state_module_depends_only_on_data_up_to_T(
    test_thresholds, vix_log_returns, spx_log_returns
):
    """Same causality audit, one level up: through state.compute_state()."""
    overlap_start = max(vix_log_returns.index.min(), spx_log_returns.index.min())
    r_detect = vix_log_returns.loc[overlap_start:].iloc[:2000]
    r_vol = spx_log_returns.reindex(r_detect.index).ffill()

    full_det = detect(r_detect, test_thresholds, min_seg=MIN_SEG, startup=21)
    full_state = compute_state(r_vol, full_det, mode="frozen")

    pos = 1500
    T = full_det.index[pos]
    trunc_det = detect(r_detect.loc[:T], test_thresholds, min_seg=MIN_SEG, startup=21)
    trunc_state = compute_state(r_vol.loc[:T].reindex(trunc_det.index), trunc_det, mode="frozen")

    assert np.isclose(full_state.loc[T, "sigma_ann"], trunc_state.iloc[-1]["sigma_ann"])
    assert np.isclose(full_state.loc[T, "exposure"], trunc_state.iloc[-1]["exposure"])
    assert full_state.loc[T, "regime_id"] == trunc_state.iloc[-1]["regime_id"]


# ─────────────────────────────────────────────────────────────────────────
# 5. Crisis sanity
# ─────────────────────────────────────────────────────────────────────────
def test_crisis_sanity_detections_near_2008_and_2020(test_thresholds, vix_log_returns):
    """
    Sanity check against known history: a change point should be flagged
    inside the Sep-Oct 2008 (Lehman) window and the Feb-Mar 2020 (COVID)
    window when running the detector on VIX log-returns over the full
    available history.
    """
    out = detect(vix_log_returns, test_thresholds, min_seg=MIN_SEG, startup=21)
    hits = out[out["detected"]]

    def any_detection_in(start, end):
        window = hits.loc[start:end]
        return window

    w2008 = any_detection_in("2008-09-01", "2008-10-31")
    w2020 = any_detection_in("2020-02-01", "2020-03-31")

    print(f"\n  2008 window detections:\n{w2008[['tau_hat', 'detection_delay']]}")
    print(f"  2020 window detections:\n{w2020[['tau_hat', 'detection_delay']]}")

    assert len(w2008) > 0, "no detection flagged in the Sep-Oct 2008 (Lehman) window"
    assert len(w2020) > 0, "no detection flagged in the Feb-Mar 2020 (COVID) window"
