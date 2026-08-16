"""
integrate.py — applying the regime signal to portfolio weights.

Usage from a model script (e.g. models/hrp.py), sketch:

    from regimes.vix import detector, state, integrate, thresholds

    h = thresholds.load_or_generate_thresholds()
    det = detector.detect(vix_log_returns, h)
    st  = state.compute_state(spx_log_returns, det)   # vol estimated from SPX

    exposure_by_effective_date = integrate.sample_exposure_at_rebalance(
        st, rebalance_dates
    )
    ...
    w_final, cash = integrate.apply_exposure_to_weights(w_model, exposure_today)

Regime scaling is applied AFTER the model's own [w_min, w_max] clipping — the
weight bounds are a relative constraint within the equity sleeve and should
not themselves be rescaled by the regime exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_exposure_at_rebalance(
    state_df: pd.DataFrame,
    rebalance_dates,
    implementation_delay_days: int = 1,
) -> pd.Series:
    """
    For each rebalance date T, read exposure[T] from `state_df` (the most
    recent available observation at or before T — causal) and apply it
    effective `implementation_delay_days` trading days later (paper: 1 day).

    Returns
    -------
    pd.Series indexed by the EFFECTIVE date (not T), giving the exposure to
    hold from that date until the next rebalance's effective date.
    """
    idx = state_df.index
    out = {}
    for T in rebalance_dates:
        hits = idx[idx <= pd.Timestamp(T)]
        if hits.empty:
            continue
        obs_date = hits[-1]
        exposure_val = state_df.loc[obs_date, "exposure"]

        future = idx[idx > obs_date]
        if len(future) < implementation_delay_days:
            continue
        effective_date = future[implementation_delay_days - 1]
        out[effective_date] = float(exposure_val)

    return pd.Series(out, dtype=float).sort_index()


def apply_exposure_to_weights(w_model: pd.Series, exposure: float) -> tuple[pd.Series, float]:
    """
    Scale already-clipped model weights by the regime exposure.

    Returns (w_final, cash): w_final sums to `exposure`, cash = 1 - exposure
    (zero interest on cash, matching the paper).
    """
    w_final = w_model * exposure
    cash = 1.0 - exposure
    return w_final, cash


def realised_mean_exposure(exposure_series: pd.Series) -> float:
    """Mean exposure over the full backtest (pass a DAILY exposure series)."""
    return float(exposure_series.mean())


def build_matched_exposure_benchmark(
    w_model_by_date: dict[pd.Timestamp, pd.Series],
    mean_exposure: float,
) -> dict[pd.Timestamp, pd.Series]:
    """
    Static comparator required alongside every dynamic-exposure result:
    hold `mean_exposure` (the dynamic strategy's own realised average, e.g.
    0.64) constantly in the model's UNSCALED weights, rebalanced on the same
    schedule. Without this, a volatility reduction from merely holding less
    equity on average could be mistaken for regime-timing skill.
    """
    return {d: w * mean_exposure for d, w in w_model_by_date.items()}


def breakeven_transaction_cost(
    dynamic_log_returns: pd.Series,
    dynamic_turnover: pd.Series,
    comparator_log_returns: pd.Series,
    lo_bps: float = 0.0,
    hi_bps: float = 2000.0,
    n_iter: int = 60,
) -> float:
    """
    Solve for the one-way transaction cost (bps) applied to the dynamic
    strategy's turnover that equates its cumulative return to the
    (cost-free) comparator's — e.g. the matched-exposure static portfolio,
    or the raw index.

    `dynamic_turnover` is a per-rebalance one-way turnover series (same
    convention as the existing TC_BPS machinery in models/*.py: cost drag on
    a rebalance day = turnover * tc_bps / 10_000, applied multiplicatively).
    Reindexed onto `dynamic_log_returns.index` with 0 turnover on non-
    rebalance days.

    Returns the break-even cost in bps via bisection. Raises if the root is
    not bracketed in [lo_bps, hi_bps] — cumulative return must be monotone
    decreasing in cost, which holds as long as turnover >= 0.
    """
    comparator_cum = float(np.exp(comparator_log_returns.sum()))
    turnover_aligned = dynamic_turnover.reindex(dynamic_log_returns.index).fillna(0.0)

    def dynamic_cum_at_cost(tc_bps: float) -> float:
        drag = 1.0 - turnover_aligned * tc_bps / 10_000.0
        value = np.exp(dynamic_log_returns.cumsum()) * drag.cumprod()
        return float(value.iloc[-1])

    f_lo = dynamic_cum_at_cost(lo_bps) - comparator_cum
    f_hi = dynamic_cum_at_cost(hi_bps) - comparator_cum
    if f_lo * f_hi > 0:
        raise ValueError(
            f"breakeven cost not bracketed in [{lo_bps}, {hi_bps}] bps "
            f"(f_lo={f_lo:.4f}, f_hi={f_hi:.4f}) — widen the search range"
        )

    lo, hi = lo_bps, hi_bps
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f_mid = dynamic_cum_at_cost(mid) - comparator_cum
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)
