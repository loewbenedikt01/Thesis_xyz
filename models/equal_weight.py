"""
Equal-Weight Portfolio — v2
=============================
Fixes vs original:
  1.  current_date += offset replaced with current_date + offset everywhere
  2.  rebalance_date logs actual_trade_date (not current_date)
  3.  period_prices window uses actual_trade_date (not current_date)
  4.  invest_year / select_year added to rebalance_details CSV
  5.  TC_BPS transaction cost parameter added (0 = disabled)
  6.  Column name standardised: turnover_at_rebalance → turnover + tc_drag_bps
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

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

# FIX 5: Transaction costs — set TC_BPS = 0 to disable
# Applied as: portfolio_value *= (1 - turnover * TC_BPS / 10_000)
TC_BPS = 0

start_invest = pd.Timestamp("1998-01-01")
end_invest   = pd.Timestamp("2025-12-31")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / "equal_weight"
output_dir.mkdir(parents=True, exist_ok=True)

all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for label, offset in FREQUENCIES.items():
    print(f"\n=== Equal-Weight [{label}] | TC={TC_BPS}bps ===")

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

        year_tickers      = [t[0] for t in universe.tickers[select_year]]
        available_tickers = [t for t in year_tickers if t in all_prices.columns]

        # ── Actual trade date ─────────────────────────────────────────────────
        trading_days_ahead = all_prices.index[all_prices.index >= current_date]
        if trading_days_ahead.empty:
            current_date = next_rebalance   # FIX 1
            continue
        actual_trade_date = trading_days_ahead[0]

        # Drop tickers with no price on the actual trade date
        valid_tickers = [
            t for t in available_tickers
            if not pd.isna(all_prices.at[actual_trade_date, t])
        ]

        if not valid_tickers:
            current_date = next_rebalance
            continue

        # ── Equal-weight allocation (1/N) ─────────────────────────────────────
        n_assets       = len(valid_tickers)
        target_weights = pd.Series(1.0 / n_assets, index=valid_tickers)

        # ── FIX 5: Transaction cost at rebalance ──────────────────────────────
        turnover = target_weights.sub(last_end_weights, fill_value=0).abs().sum()
        if TC_BPS > 0:
            portfolio_value *= (1 - turnover * TC_BPS / 10_000)

        # ── Rebalance logging ─────────────────────────────────────────────────
        for ticker, w in target_weights.items():
            rebalance_details.append({
                # FIX 2: log actual_trade_date not current_date
                'rebalance_date'  : actual_trade_date.strftime('%Y-%m-%d'),
                'invest_year'     : invest_year,        # FIX 4
                'select_year'     : select_year,        # FIX 4
                'ticker'          : ticker,
                'assigned_weight' : w,
                'n_stocks'        : n_assets,
                'turnover'        : round(turnover, 6) if ticker == target_weights.index[0] else 0,
                'tc_drag_bps'     : round(turnover * TC_BPS, 4) if ticker == target_weights.index[0] else 0,
            })

        # ── Daily portfolio drift ─────────────────────────────────────────────
        active_weights = target_weights.copy()

        # FIX 3: use actual_trade_date for period prices window
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
        output_dir / f"portfolio_ew_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_ew_{label}_details.csv", index=False
    )
    print(f"  [{label}] Done — saved to {output_dir}")

print(f"\nAll frequencies complete. Files exported to {output_dir}")