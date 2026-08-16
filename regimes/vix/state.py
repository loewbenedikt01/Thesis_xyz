"""
state.py — EWMA volatility estimate and exposure mapping.

    EWMA_t    = lambda * EWMA_{t-1} + (1 - lambda) * r_t^2      (lambda = 0.95)
    sigma_ann = sqrt(252 * EWMA_t)

lambda = 0.95 gives an effective memory of ~20 trading days.

Seeding at detection: on detection at t with change point tau_hat, the EWMA
is re-initialised from the realised variance of the retained post-change
observations, mean(r[tau_hat:t+1]^2), then the standard recursion runs
forward from there. If fewer than `reseed_min_obs` post-change observations
are available, that direct estimate is too noisy (imagine 3 observations) —
instead the previous regime's EWMA value is kept as-is and left to decay
via the ordinary recursion on subsequent days.

Two variants:
  - 'frozen'  (primary, faithful to the paper): sigma_hat is fixed at the
    value computed at detection and held constant until the next detection.
  - 'rolling' (robustness check): the EWMA keeps updating daily and the
    rebalance always reads its current value.

Exposure mapping (paper Figure 5). The 20% switching threshold is an
assumption about long-run average equity volatility and is deliberately NOT
tuned — report sensitivity to it as a robustness table instead of optimising
it, which would introduce backtesting bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDA_DEFAULT = 0.95
TRADING_DAYS_PER_YEAR = 252


def _exposure_from_vol(sigma_ann: np.ndarray, rule: str, vol_threshold: float) -> np.ndarray:
    if rule == "linear_long_only":
        e = 1.5 - 5.0 * sigma_ann
        return np.clip(e, 0.0, 1.0)
    if rule == "linear_long_short":
        e = 1.5 - 5.0 * sigma_ann
        return np.clip(e, -0.5, 1.5)
    if rule == "switching":
        return np.where(sigma_ann < vol_threshold, 1.0, 0.0).astype(float)
    raise ValueError(f"unknown exposure_rule: {rule!r}")


def compute_state(
    vol_returns: pd.Series,
    detection: pd.DataFrame,
    lam: float = LAMBDA_DEFAULT,
    mode: str = "frozen",
    exposure_rule: str = "switching",
    vol_threshold: float = 0.20,
    reseed_min_obs: int = 5,
) -> pd.DataFrame:
    """
    Turn detector output into a daily volatility estimate and exposure.

    Parameters
    ----------
    vol_returns    : daily log-returns of the series used to ESTIMATE
                      volatility (e.g. S&P 500 returns — this can differ from
                      the series the detector was run on, e.g. VIX). Must be
                      defined (no NaNs) over `detection.index`.
    detection       : output of `detector.detect()`.
    lam             : EWMA decay, default 0.95 (~20 trading day memory).
    mode            : 'frozen' (primary) or 'rolling' (robustness check).
    exposure_rule   : 'linear_long_only', 'linear_long_short', or 'switching'.
    vol_threshold   : threshold used by the 'switching' rule. Do not tune
                      this by backtest performance — report a sensitivity
                      table across a small grid instead.
    reseed_min_obs  : minimum post-change observations required to seed the
                      EWMA directly from their realised variance; below this,
                      keep the previous regime's value and let it decay.

    Returns
    -------
    pd.DataFrame indexed like `detection` with columns:
        sigma_ann, exposure, regime_id, segment_age, detected
    """
    if mode not in ("frozen", "rolling"):
        raise ValueError("mode must be 'frozen' or 'rolling'")

    idx = detection.index
    r = vol_returns.reindex(idx)
    if r.isna().any():
        bad = r[r.isna()].index[:5].tolist()
        raise ValueError(f"vol_returns missing inside monitored window, e.g. at {bad}")

    r2 = r.to_numpy(dtype=float) ** 2
    n = len(idx)
    seg_age = detection["segment_age"].to_numpy()
    detected = detection["detected"].to_numpy()

    ewma_running = np.empty(n)
    ewma_frozen = np.empty(n)
    regime_id = np.empty(n, dtype=int)

    running = float(r2[0])
    frozen_val = running
    rid = 0

    for i in range(n):
        if detected[i]:
            rid += 1
            delay = int(seg_age[i])
            local_start = i - delay  # position (local to this array) of tau_hat
            window = r2[local_start : i + 1]
            if len(window) >= reseed_min_obs:
                running = float(window.mean())
            # else: keep `running` as the previous regime's value — it decays
            # via the ordinary recursion on subsequent (non-detection) days.
            frozen_val = running
        else:
            running = lam * running + (1.0 - lam) * r2[i]
            # frozen_val intentionally unchanged until the next detection

        ewma_running[i] = running
        ewma_frozen[i] = frozen_val
        regime_id[i] = rid

    sigma_running = np.sqrt(TRADING_DAYS_PER_YEAR * ewma_running)
    sigma_frozen = np.sqrt(TRADING_DAYS_PER_YEAR * ewma_frozen)
    sigma_ann = sigma_frozen if mode == "frozen" else sigma_running

    exposure = _exposure_from_vol(sigma_ann, rule=exposure_rule, vol_threshold=vol_threshold)

    return pd.DataFrame(
        {
            "sigma_ann": sigma_ann,
            "exposure": exposure,
            "regime_id": regime_id,
            "segment_age": seg_age,
            "detected": detected,
        },
        index=idx,
    )
