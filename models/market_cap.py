"""
Market-Cap Weighted Portfolio — v2
====================================
Fixes vs original:
  1.  current_date + offset (not +=) — safe with all DateOffset types
  2.  invest_year / select_year added to rebalance_details CSV
  3.  TC_BPS transaction cost parameter added (0 = disabled)
  4.  Comment added explaining year_start_trade_date fallback behaviour
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

# FIX 3: Transaction costs — set TC_BPS = 0 to disable
# Applied as: portfolio_value *= (1 - turnover * TC_BPS / 10_000)
TC_BPS = 0

start_invest = pd.Timestamp("1998-01-01")
end_invest   = pd.Timestamp("2025-12-31")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / "market_cap_v2"
output_dir.mkdir(parents=True, exist_ok=True)

all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for label, offset in FREQUENCIES.items():
    print(f"\n=== Market-Cap [{label}] | TC={TC_BPS}bps ===")

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
        mktcap_dict  = {t: mc for t, mc in universe.tickers[select_year]}

        # ── Actual trade date ─────────────────────────────────────────────────
        trading_days_ahead = all_prices.index[all_prices.index >= current_date]
        if trading_days_ahead.empty:
            current_date = next_rebalance
            continue
        actual_trade_date = trading_days_ahead[0]

        # ── Year-start reference price for market cap updating ────────────────
        # All rebalances within the same calendar year use the same year_start
        # reference so that updated_mktcap = initial_mktcap × (price_now / price_jan1)
        # consistently reflects price appreciation since the snapshot date.
        # Note: if a stock has no price on year_start_trade_date (e.g. newly listed),
        # we fall back to the raw initial_mktcap from the universe snapshot — this is
        # conservative and documented in the methodology.
        year_start            = pd.Timestamp(f"{current_date.year}-01-01")
        year_start_days       = all_prices.index[all_prices.index >= year_start]
        if year_start_days.empty:
            current_date = next_rebalance
            continue
        year_start_trade_date = year_start_days[0]

        # ── Filter to stocks with a valid price on the trade date ─────────────
        available_tickers = [t for t in year_tickers if t in all_prices.columns]
        valid_tickers     = [
            t for t in available_tickers
            if not pd.isna(all_prices.at[actual_trade_date, t])
        ]

        if not valid_tickers:
            current_date = next_rebalance
            continue

        # ── Updated market caps ───────────────────────────────────────────────
        updated_mktcaps = {}
        for ticker in valid_tickers:
            initial_mc = mktcap_dict.get(ticker, np.nan)
            if np.isnan(initial_mc):
                continue

            price_start = (
                all_prices.at[year_start_trade_date, ticker]
                if year_start_trade_date in all_prices.index else np.nan
            )
            price_now = all_prices.at[actual_trade_date, ticker]

            if pd.isna(price_start) or pd.isna(price_now) or price_start == 0:
                # Fallback: use raw snapshot mktcap — conservative and documented
                updated_mktcaps[ticker] = initial_mc
            else:
                updated_mktcaps[ticker] = initial_mc * (price_now / price_start)

        if not updated_mktcaps:
            current_date = next_rebalance
            continue

        mktcap_series  = pd.Series(updated_mktcaps)
        target_weights = mktcap_series / mktcap_series.sum()

        # ── FIX 3: Transaction cost at rebalance ──────────────────────────────
        turnover = target_weights.sub(last_end_weights, fill_value=0).abs().sum()
        if TC_BPS > 0:
            portfolio_value *= (1 - turnover * TC_BPS / 10_000)

        # ── Rebalance logging ─────────────────────────────────────────────────
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date'     : actual_trade_date.strftime('%Y-%m-%d'),
                'invest_year'        : invest_year,     # FIX 2
                'select_year'        : select_year,     # FIX 2
                'ticker'             : ticker,
                'assigned_weight'    : w,
                'updated_mktcap_bn'  : updated_mktcaps[ticker],
                'turnover'           : round(turnover, 6) if ticker == target_weights.index[0] else 0,
                'tc_drag_bps'        : round(turnover * TC_BPS, 4) if ticker == target_weights.index[0] else 0,
            })

        # ── Daily portfolio drift ─────────────────────────────────────────────
        active_weights = target_weights.copy()
        period_prices  = all_prices.loc[
            actual_trade_date - pd.Timedelta(days=5)
            : next_rebalance   - pd.Timedelta(days=1),
            active_weights.index,
        ]

        if not period_prices.empty:
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
        output_dir / f"portfolio_mktcap_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_mktcap_{label}_details.csv", index=False
    )
    print(f"  [{label}] Done — saved to {output_dir}")

print(f"\nAll frequencies complete. Files exported to {output_dir}")