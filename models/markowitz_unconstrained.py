"""
Markowitz Mean-Variance Portfolio — v2
========================================
Fixes vs original:
  1.  current_date + offset (not +=) — safe with all DateOffset types
  2.  ffill(limit=5) + dropna(how='all') — stops over-dropping stocks
  3.  Bare except: replaced with except Exception — safer error handling
  4.  actual_trade_date found and used throughout (logging + period prices)
  5.  period_prices window uses actual_trade_date not current_date
  6.  returns.fillna(0) added in daily drift loop
  7.  len(valid_tickers) < 2 guard before optimisation
  8.  invest_year / select_year added to rebalance_details CSV
  9.  TC_BPS transaction cost parameter added (0 = disabled)
 10.  LOOKBACK_MONTHS increased from 24 → 60 for thesis consistency
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from pypfopt import EfficientFrontier, risk_models, expected_returns

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

LOOKBACK_MONTHS  = 60       # FIX 10: increased from 24 → 60 months
                             # 60mo × ~21 days = ~1,260 obs for a 20×20 cov matrix
                             # matches HRP v2 and is consistent across all models
MIN_COMPLETENESS = 0.50     # min fraction of non-NaN rows required per stock
WEIGHT_MAX       = 1.00     # max portfolio weight per stock
WEIGHT_MIN       = 0.00     # min portfolio weight per stock

# FIX 9: Transaction costs — set TC_BPS = 0 to disable
# Applied as: portfolio_value *= (1 - turnover * TC_BPS / 10_000)
TC_BPS = 0

start_invest = pd.Timestamp("1998-01-01")
end_invest   = pd.Timestamp("2025-12-31")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / "markowitz_unconstrained"
output_dir.mkdir(parents=True, exist_ok=True)

all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for label, offset in FREQUENCIES.items():
    print(f"\n=== Markowitz [{label}] | lookback={LOOKBACK_MONTHS}mo | TC={TC_BPS}bps ===")

    current_date          = start_invest
    portfolio_value       = 1.0
    last_end_weights      = pd.Series(dtype=float)
    portfolio_performance = []
    rebalance_details     = []

    while current_date < end_invest:
        # FIX 1: always use + not += with DateOffset
        next_rebalance = current_date + offset

        # ── Universe selection ────────────────────────────────────────────────
        invest_year = current_date.year
        select_year = invest_year - 1       # tickers[1997] → invest in 1998, etc.

        if select_year not in universe.tickers:
            current_date = next_rebalance   # FIX 1
            continue

        year_tickers = [t[0] for t in universe.tickers[select_year]]

        # ── FIX 4: find actual trade date ─────────────────────────────────────
        trading_days_ahead = all_prices.index[all_prices.index >= current_date]
        if trading_days_ahead.empty:
            current_date = next_rebalance
            continue
        actual_trade_date = trading_days_ahead[0]

        # ── Lookback prices ───────────────────────────────────────────────────
        lb_start  = actual_trade_date - pd.DateOffset(months=LOOKBACK_MONTHS)
        lb_end    = actual_trade_date - pd.Timedelta(days=1)
        available = [t for t in year_tickers if t in all_prices.columns]
        lb_data   = all_prices.loc[lb_start:lb_end, available]

        if lb_data.empty:
            current_date = next_rebalance
            continue

        coverage      = lb_data.notnull().sum() / len(lb_data)
        valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

        # FIX 7: guard — pypfopt needs at least 2 stocks
        if len(valid_tickers) < 2:
            current_date = next_rebalance
            continue

        # FIX 2: cap ffill at 5 days; drop rows where ALL stocks are NaN
        lb_prices_final = lb_data[valid_tickers].ffill(limit=5).dropna(how='all')
        valid_tickers   = lb_prices_final.columns.tolist()

        if len(valid_tickers) < 2:
            current_date = next_rebalance
            continue

        # ── Mean-Variance Optimisation ────────────────────────────────────────
        target_weights = None
        opt_method     = 'none'

        try:
            mu = expected_returns.mean_historical_return(lb_prices_final)
            S  = risk_models.sample_cov(lb_prices_final)

            # Primary: maximum Sharpe ratio
            try:
                ef      = EfficientFrontier(mu, S, weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
                weights = ef.max_sharpe()
                opt_method = 'max_sharpe'
            except Exception:   # FIX 3: specific exception, not bare except
                # Fallback: minimum volatility (more numerically stable)
                ef      = EfficientFrontier(mu, S, weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
                weights = ef.min_volatility()
                opt_method = 'min_volatility'
                print(f"  [{label}] {current_date.date()}: "
                      f"max_sharpe failed — using min_volatility")

            target_weights = pd.Series(ef.clean_weights())
            # Remove zero-weight stocks (pypfopt clean_weights may leave them)
            target_weights = target_weights[target_weights > 0]

        except Exception as e:
            print(f"  [{label}] {current_date.date()}: optimisation failed ({e}) "
                  f"— skipping period")
            current_date = next_rebalance
            continue

        if target_weights is None or target_weights.empty:
            current_date = next_rebalance
            continue

        # ── FIX 9: Transaction cost at rebalance ──────────────────────────────
        turnover = target_weights.sub(last_end_weights, fill_value=0).abs().sum()
        if TC_BPS > 0:
            portfolio_value *= (1 - turnover * TC_BPS / 10_000)

        # ── Rebalance logging ─────────────────────────────────────────────────
        for ticker, w in target_weights.items():
            rebalance_details.append({
                # FIX 4: log actual_trade_date not current_date
                'rebalance_date'  : actual_trade_date.strftime('%Y-%m-%d'),
                'invest_year'     : invest_year,        # FIX 8
                'select_year'     : select_year,        # FIX 8
                'ticker'          : ticker,
                'assigned_weight' : w,
                'opt_method'      : opt_method,
                'turnover'        : round(turnover, 6) if ticker == target_weights.index[0] else 0,
                'tc_drag_bps'     : round(turnover * TC_BPS, 4) if ticker == target_weights.index[0] else 0,
            })

        # ── Daily portfolio drift ─────────────────────────────────────────────
        active_weights = target_weights.copy()

        # FIX 5: use actual_trade_date for period prices window
        period_prices = all_prices.loc[
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
            day_ret         = day_ret.fillna(0)         # FIX 6: fillna before sum
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
        output_dir / f"portfolio_markowitz_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_markowitz_{label}_details.csv", index=False
    )
    print(f"  [{label}] Done — saved to {output_dir}")

print(f"\nAll frequencies complete. Files exported to {output_dir}")