"""
detector.py — sequential change-point detection with restart.

Non-negotiable constraint: the state emitted for date T is a function of
returns[:T] only (r_1 .. r_T). This module is written as a strictly forward
loop for exactly that reason — do not vectorise it in a way that touches
future observations.

Detection lag is real and is not engineered away: tau_hat < t always (you
learn about the change only at t, the detection date). The output state
changes at t, not at tau_hat — back-dating it to tau_hat would be lookahead
bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mood import mood_max


def detect(
    returns: pd.Series,
    thresholds: np.ndarray,
    min_seg: int = 20,
    startup: int = 21,
) -> pd.DataFrame:
    """
    Sequential Mood-test change-point detection with restart-after-detection.

    Parameters
    ----------
    returns    : daily log-returns, indexed by date, sorted ascending, with no
                 missing values (a gap would silently corrupt the segment —
                 this is asserted).
    thresholds : h[t] array from `thresholds.calibrate_thresholds` /
                 `load_or_generate_thresholds`. h[t] is applied when the
                 current monitored segment holds exactly t observations.
                 Must have been calibrated with the same `min_seg`.
    min_seg    : minimum sub-sample size for the Mood statistic — must match
                 the value used to calibrate `thresholds`.
    startup    : number of initial observations that seed the first segment
                 before monitoring begins (paper: 21, one month). Monitoring
                 (i.e. the first date a detection can be flagged) starts at
                 returns.index[startup].

    Returns
    -------
    pd.DataFrame indexed by returns.index[startup:] with columns:
        detected          bool   — True on the day a change point was flagged
        tau_hat           object — date of the estimated change point
                                    (NaT on non-detection rows)
        detection_delay   float  — trading days between tau_hat and the
                                    detection date (NaN on non-detection rows)
        segment_age       int    — trading days since the start of the
                                    segment currently being monitored (0 on
                                    the first day of a fresh segment; on a
                                    detection row this reflects the NEW
                                    segment, i.e. equals detection_delay)
    """
    if returns.isna().any():
        bad = returns[returns.isna()].index[:5].tolist()
        raise ValueError(f"returns contains missing values, e.g. at {bad} — fill or drop first")

    values = returns.to_numpy(dtype=float)
    dates = returns.index
    n = len(values)

    if n <= startup:
        raise ValueError(f"returns series (len={n}) must be longer than startup={startup}")

    max_h_idx = len(thresholds) - 1

    records = []
    segment_start_pos = 0  # index into `values` of the first obs of the current segment

    for t in range(startup, n):
        segment = values[segment_start_pos : t + 1]  # causal: only data up to & including t
        seg_len = len(segment)

        detected = False
        tau_hat_date = pd.NaT
        detection_delay = np.nan

        if seg_len >= 2 * min_seg:
            h_t = thresholds[seg_len] if seg_len <= max_h_idx else thresholds[max_h_idx]
            D, tau_local = mood_max(segment, min_seg=min_seg)

            if D > h_t:
                detected = True
                tau_hat_pos = segment_start_pos + tau_local
                tau_hat_date = dates[tau_hat_pos]
                detection_delay = float(t - tau_hat_pos)

                # Restart: discard pre-change data, KEEP the post-change
                # observations retained between tau_hat and t — these are
                # exactly what state.py uses to seed the new regime's
                # volatility estimate.
                segment_start_pos = tau_hat_pos

        segment_age = t - segment_start_pos

        records.append(
            {
                "date": dates[t],
                "detected": detected,
                "tau_hat": tau_hat_date,
                "detection_delay": detection_delay,
                "segment_age": segment_age,
            }
        )

    out = pd.DataFrame.from_records(records).set_index("date")
    out["detected"] = out["detected"].astype(bool)
    out["segment_age"] = out["segment_age"].astype(int)
    return out
