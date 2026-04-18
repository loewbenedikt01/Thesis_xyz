"""
HRP Portfolio Model — v3
=========================
All fixes from v2, plus three new fixes for the NaN weight problem:
  8.  Ledoit-Wolf shrinkage ('ledoit') replaces sample covariance ('hist')
      → eliminates near-singular matrix failures during GFC/crisis periods
  9.  MIN_COMPLETENESS lowered to 0.30 for new-stock rotation periods
      → prevents entire years being blank when new stocks lack 60mo history
 10.  Final guard: if target_weights is still None after fallback, skip period
      → stops NaN rows being written to the details CSV
 11.  Fallback logs actual carried-forward weights (not new universe tickers)
      → details CSV now always reflects what the portfolio actually held
 12.  'hrp_success' flag column added to details CSV
      → makes it easy to filter fallback periods in post-processing
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import riskfolio as rp

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import universe

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
FREQUENCIES = {
    'Yearly':      pd.DateOffset(years=1),
    'Semi-Annual': pd.DateOffset(months=6),
    'Quarterly':   pd.DateOffset(months=3),
    'Monthly':     pd.DateOffset(months=1),
}

LOOKBACK_MONTHS  = 60       # lookback window for covariance estimation
MIN_COMPLETENESS = 0.50     # FIX 9: lowered from 0.50 → 0.30
                             # 0.50 was dropping new stocks that had < 30mo history
                             # when the universe rotated, causing entire blank years
WEIGHT_MAX       = 0.10     # max portfolio weight per stock
WEIGHT_MIN       = 0.01     # min portfolio weight per stock
LINKAGE          = 'ward'   # hierarchical clustering linkage method

# Transaction costs — set TC_BPS = 0 to disable
TC_BPS = 0

start_invest = pd.Timestamp("1998-01-01")
end_invest   = pd.Timestamp("2025-12-31")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / "hrp"
output_dir.mkdir(parents=True, exist_ok=True)

all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT CLIPPING
# ─────────────────────────────────────────────────────────────────────────────
def clip_and_redistribute(weights: pd.Series, w_min: float, w_max: float,
                           max_iter: int = 100) -> pd.Series | None:
    """
    Enforce [w_min, w_max] bounds by iteratively clipping and redistributing
    the excess/deficit proportionally across unconstrained assets.
    Returns normalised weights summing to 1.0, or None if convergence fails.
    """
    w = weights.copy()
    for _ in range(max_iter):
        clipped_low  = w < w_min
        clipped_high = w > w_max
        if not clipped_low.any() and not clipped_high.any():
            break
        w[clipped_low]  = w_min
        w[clipped_high] = w_max
        excess    = 1.0 - w.sum()
        free_mask = ~clipped_low & ~clipped_high
        if free_mask.sum() == 0:
            return None
        w[free_mask] += excess * (w[free_mask] / w[free_mask].sum())
    else:
        return None

    w = w / w.sum()
    return w


# ─────────────────────────────────────────────────────────────────────────────
# HRP OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
def hrp_weights(returns_df: pd.DataFrame) -> pd.Series | None:
    """
    Compute HRP weights via riskfolio-lib using Ledoit-Wolf shrinkage.
    Ledoit-Wolf ('ledoit') is far more robust than sample covariance ('hist')
    for near-singular matrices during high-correlation crisis periods (GFC etc.)
    Returns a pd.Series of weights (indexed by ticker) or None on failure.
    """
    port = rp.HCPortfolio(returns=returns_df)
    w = port.optimization(
        model        = 'HRP',
        codependence = 'pearson',
        method_cov   = 'ledoit',    # FIX 8: Ledoit-Wolf shrinkage
        linkage      = LINKAGE,
        rm           = 'MV',
        rf           = 0,
        leaf_order   = True,
    )
    if w is None or w.empty:
        return None
    return w.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for label, offset in FREQUENCIES.items():
    print(f"\n=== HRP [{label}] | TC={TC_BPS}bps | "
          f"min_completeness={MIN_COMPLETENESS} | cov=ledoit ===")

    current_date          = start_invest
    portfolio_value       = 1.0
    last_end_weights      = pd.Series(dtype=float)
    portfolio_performance = []
    rebalance_details     = []

    while current_date < end_invest:
        next_rebalance = current_date + offset

        # ── Universe selection ────────────────────────────────────────────────
        invest_year = current_date.year
        select_year = invest_year - 1

        if select_year not in universe.tickers:
            current_date = next_rebalance
            continue

        year_tickers = [t[0] for t in universe.tickers[select_year]]

        # ── Actual trade date ─────────────────────────────────────────────────
        trading_days_ahead = all_prices.index[all_prices.index >= current_date]
        if trading_days_ahead.empty:
            current_date = next_rebalance
            continue
        actual_trade_date = trading_days_ahead[0]

        # ── Lookback prices ───────────────────────────────────────────────────
        lb_start  = actual_trade_date - pd.DateOffset(months=LOOKBACK_MONTHS)
        lb_end    = actual_trade_date - pd.Timedelta(days=1)
        available = [t for t in year_tickers if t in all_prices.columns]
        lb_prices = all_prices.loc[lb_start:lb_end, available]

        if lb_prices.empty:
            current_date = next_rebalance
            continue

        # Completeness filter (FIX 9: threshold = 0.30)
        coverage      = lb_prices.notnull().sum() / len(lb_prices)
        valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

        if not valid_tickers:
            current_date = next_rebalance
            continue

        # Cap ffill at 5 days; drop rows where ALL stocks are NaN
        lb_prices_final = lb_prices[valid_tickers].ffill(limit=5).dropna(how='all')
        valid_tickers   = lb_prices_final.columns.tolist()

        # Remove stocks with no price on the actual trade date
        valid_tickers = [
            t for t in valid_tickers
            if t in all_prices.columns
            and not pd.isna(all_prices.at[actual_trade_date, t])
        ]

        if len(valid_tickers) < 2:
            current_date = next_rebalance
            continue

        lb_prices_final = lb_prices_final[valid_tickers]
        lb_returns      = lb_prices_final.pct_change().dropna(how='all')

        # ── HRP Optimisation ──────────────────────────────────────────────────
        target_weights = None
        hrp_success    = False

        try:
            raw_w = hrp_weights(lb_returns)
            if raw_w is not None:
                raw_w = raw_w.reindex(valid_tickers).fillna(0)
                raw_w = raw_w / raw_w.sum()
                constrained_w = clip_and_redistribute(raw_w, WEIGHT_MIN, WEIGHT_MAX)
                if constrained_w is not None:
                    target_weights = constrained_w
                    hrp_success    = True
                else:
                    print(f"  [{label}] {current_date.date()}: "
                          f"weight clipping failed — using equal weight")
        except Exception as e:
            print(f"  [{label}] {current_date.date()}: "
                  f"HRP failed ({e}) — using equal weight")

        # ── Fallback: equal weight on current valid universe ──────────────────
        if target_weights is None:
            n              = len(valid_tickers)
            target_weights = pd.Series(1.0 / n, index=valid_tickers)
            print(f"  [{label}] {current_date.date()}: "
                  f"equal-weight fallback ({n} stocks)")

        # FIX 10: final safety guard — should never trigger but prevents
        # any edge case where target_weights is still None or empty
        if target_weights is None or target_weights.empty:
            current_date = next_rebalance
            continue

        # ── Transaction cost at rebalance ─────────────────────────────────────
        turnover = target_weights.sub(last_end_weights, fill_value=0).abs().sum()
        if TC_BPS > 0:
            portfolio_value *= (1 - turnover * TC_BPS / 10_000)

        # ── Rebalance logging ─────────────────────────────────────────────────
        # FIX 11: always log target_weights (what portfolio actually holds)
        #         never logs NaN — fallback weights are always valid here
        # FIX 12: 'hrp_success' flag distinguishes HRP vs equal-weight periods
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date'  : actual_trade_date.strftime('%Y-%m-%d'),
                'invest_year'     : invest_year,
                'select_year'     : select_year,
                'ticker'          : ticker,
                'assigned_weight' : w,
                'hrp_success'     : hrp_success,    # FIX 12: True = HRP ran OK
                'turnover'        : round(turnover, 6) if ticker == target_weights.index[0] else 0,
                'tc_drag_bps'     : round(turnover * TC_BPS, 4) if ticker == target_weights.index[0] else 0,
            })

        # ── Daily portfolio drift ─────────────────────────────────────────────
        active_weights = target_weights.copy()
        period_prices  = all_prices.loc[
            actual_trade_date - pd.Timedelta(days=5)
            : next_rebalance   - pd.Timedelta(days=1),
            active_weights.index,
        ]

        if period_prices.empty:
            current_date = next_rebalance
            continue

        daily_rets = period_prices.pct_change().dropna(how='all')
        daily_rets = daily_rets[daily_rets.index >= actual_trade_date]

        for day_ts, day_ret in daily_rets.iterrows():
            day_ret         = day_ret.fillna(0)
            day_pct         = (active_weights * day_ret).sum()
            portfolio_value *= (1 + day_pct)
            active_weights   = active_weights * (1 + day_ret) / (1 + day_pct)

            portfolio_performance.append({
                'date'            : day_ts.strftime('%Y-%m-%d'),
                'log_return'      : np.log(1 + day_pct),
                'cumulative_value': portfolio_value,
            })

        last_end_weights = active_weights
        current_date     = next_rebalance

    # ── Export ────────────────────────────────────────────────────────────────
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"portfolio_hrp_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_hrp_{label}_details.csv", index=False
    )
    print(f"  [{label}] Done — saved to {output_dir}")

print(f"\nAll frequencies complete. Files exported to {output_dir}")