import pandas as pd
import numpy as np
import sys
from pathlib import Path
import riskfolio as rp

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import universe


# Configuration
FREQUENCIES = {
    'Yearly': pd.DateOffset(years=1),
    'Semi-Annual': pd.DateOffset(months=6),
    'Quarterly': pd.DateOffset(months=3),
    'Monthly': pd.DateOffset(months=1)
}

LOOKBACK_MONTHS = 60        # Lookback window in months (try 36, 60, etc.)
MIN_COMPLETENESS = 0.50     # Min fraction of history required, otherwise drop stock
WEIGHT_MAX = 0.10           # Max 10% per stock
WEIGHT_MIN = 0.01           # Min 1% per stock
LINKAGE = 'ward'            # Hierarchical clustering linkage method

start_invest = pd.Timestamp("1998-01-01")
end_invest = pd.Timestamp("2025-12-31")

# Directory
DATA_PATH = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir = DATA_PATH / "results" / "data" / "hrp"
output_dir.mkdir(parents=True, exist_ok=True)

# Load Data
all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)


def clip_and_redistribute(weights: pd.Series, w_min: float, w_max: float, max_iter: int = 100) -> pd.Series:
    """
    Enforce [w_min, w_max] bounds by iteratively clipping and redistributing
    the excess/deficit proportionally across unconstrained assets.
    Returns normalised weights that sum to 1.0, or None if convergence fails.
    """
    w = weights.copy()
    for _ in range(max_iter):
        clipped_low  = w < w_min
        clipped_high = w > w_max
        if not clipped_low.any() and not clipped_high.any():
            break

        # Fix the out-of-bounds weights
        w[clipped_low]  = w_min
        w[clipped_high] = w_max

        # How much weight needs to be redistributed
        excess = 1.0 - w.sum()
        free_mask = ~clipped_low & ~clipped_high
        if free_mask.sum() == 0:
            return None   # No free assets left — cannot satisfy constraints
        w[free_mask] += excess * (w[free_mask] / w[free_mask].sum())
    else:
        return None   # Did not converge

    w = w / w.sum()   # Final normalisation for floating-point safety
    return w


def hrp_weights(returns_df: pd.DataFrame) -> pd.Series | None:
    """
    Compute HRP weights using riskfolio-lib with sample covariance and ward linkage.
    Returns a pd.Series of weights (indexed by ticker) or None on failure.
    """
    port = rp.HCPortfolio(returns=returns_df)
    w = port.optimization(
        model='HRP',
        codependence='pearson',
        method_cov='hist',          # Sample covariance
        linkage=LINKAGE,
        rm='MV',
        rf=0,
        leaf_order=True,
    )
    if w is None or w.empty:
        return None
    return w.squeeze()              # DataFrame → Series


# --- Main Frequency Loop ---
for label, offset in FREQUENCIES.items():
    print(f"Processing: {label}")

    current_date = start_invest
    portfolio_value = 1.0
    last_end_weights = pd.Series(dtype=float)

    portfolio_performance = []  # → portfolio_{label}.csv
    rebalance_details = []      # → portfolio_{label}_details.csv

    while current_date < end_invest:
        # 1. Universe Selection
        target_year = current_date.year - 1
        if target_year not in universe.tickers:
            current_date += offset
            continue

        year_tickers = [t[0] for t in universe.tickers[target_year]]
        next_rebalance = current_date + offset

        # 2. Lookback data — filter by completeness
        lb_start = current_date - pd.DateOffset(months=LOOKBACK_MONTHS)
        lb_end = current_date - pd.Timedelta(days=1)
        available = [t for t in year_tickers if t in all_prices.columns]
        lb_prices = all_prices.loc[lb_start:lb_end, available]

        if lb_prices.empty:
            current_date = next_rebalance
            continue

        coverage = lb_prices.notnull().sum() / len(lb_prices)
        valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

        if not valid_tickers:
            current_date = next_rebalance
            continue

        lb_prices_final = lb_prices[valid_tickers].ffill().dropna(axis=1)
        valid_tickers = lb_prices_final.columns.tolist()

        # Find the actual trade date (first available trading day >= current_date)
        trading_days_ahead = all_prices.index[all_prices.index >= current_date]
        if trading_days_ahead.empty:
            current_date = next_rebalance
            continue
        actual_trade_date = trading_days_ahead[0]
        valid_tickers = [t for t in valid_tickers if not pd.isna(all_prices.at[actual_trade_date, t])]

        if not valid_tickers:
            current_date = next_rebalance
            continue

        lb_prices_final = lb_prices_final[valid_tickers]
        lb_returns = lb_prices_final.pct_change().dropna(how='all')

        # 3. HRP Optimisation
        target_weights = None
        try:
            raw_w = hrp_weights(lb_returns)
            if raw_w is not None:
                raw_w = raw_w.reindex(valid_tickers).fillna(0)
                raw_w = raw_w / raw_w.sum()
                constrained_w = clip_and_redistribute(raw_w, WEIGHT_MIN, WEIGHT_MAX)
                if constrained_w is not None:
                    target_weights = constrained_w
        except Exception as e:
            print(f"  HRP failed at {current_date}: {e}")

        # Fall back to equal weight if HRP couldn't produce valid weights
        if target_weights is None:
            print(f"  Falling back to equal weight at {current_date}")
            n = len(valid_tickers)
            target_weights = pd.Series(1.0 / n, index=valid_tickers)

        # 4. Log rebalance details
        turnover = (target_weights.sub(last_end_weights, fill_value=0)).abs().sum()
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date': current_date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'assigned_weight': w,
                'turnover_at_rebalance': turnover if ticker == target_weights.index[0] else 0
            })

        active_weights = target_weights.copy()

        # 5. Daily Performance Calculation (drift period)
        period_prices = all_prices.loc[
            current_date - pd.Timedelta(days=5):next_rebalance - pd.Timedelta(days=1),
            active_weights.index
        ]
        if period_prices.empty:
            current_date = next_rebalance
            continue

        daily_stock_rets = period_prices.pct_change().dropna(how='all')
        daily_stock_rets = daily_stock_rets[daily_stock_rets.index >= current_date]

        for day_timestamp, returns in daily_stock_rets.iterrows():
            returns = returns.fillna(0)
            day_pct_return = (active_weights * returns).sum()
            day_log_return = np.log(1 + day_pct_return)
            portfolio_value *= (1 + day_pct_return)

            # Weight drift
            active_weights = (active_weights * (1 + returns)) / (1 + day_pct_return)

            portfolio_performance.append({
                'date': day_timestamp.strftime('%Y-%m-%d'),
                'log_return': day_log_return,
                'cumulative_value': portfolio_value
            })

        last_end_weights = active_weights
        current_date = next_rebalance

    # --- Export Files ---
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"portfolio_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_{label}_details.csv", index=False
    )

print(f"\nFiles exported to {output_dir}")
