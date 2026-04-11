import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import universe


# Configuration
FREQUENCIES = {
    'Yearly':      pd.DateOffset(years=1),
    'Semi-Annual': pd.DateOffset(months=6),
    'Quarterly':   pd.DateOffset(months=3),
    'Monthly':     pd.DateOffset(months=1),
}

start_invest = pd.Timestamp("1998-01-01")
end_invest   = pd.Timestamp("2025-12-31")

# Directory
DATA_PATH   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / "market_cap"
output_dir.mkdir(parents=True, exist_ok=True)

# Load Data
all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)


# --- Main Frequency Loop ---
for label, offset in FREQUENCIES.items():
    print(f"Processing: {label}")

    current_date      = start_invest
    portfolio_value   = 1.0
    last_end_weights  = pd.Series(dtype=float)
    portfolio_performance = []
    rebalance_details     = []

    while current_date < end_invest:
        # 1. Universe selection
        target_year = current_date.year - 1
        if target_year not in universe.tickers:
            current_date += offset
            continue

        year_tickers  = [t[0] for t in universe.tickers[target_year]]
        mktcap_dict   = {t: mc for t, mc in universe.tickers[target_year]}
        next_rebalance = current_date + offset

        # Find actual trade date (first trading day >= current_date)
        trading_days_ahead = all_prices.index[all_prices.index >= current_date]
        if trading_days_ahead.empty:
            current_date = next_rebalance
            continue
        actual_trade_date = trading_days_ahead[0]

        # Find first trading day of the current year (reference for mktcap update)
        year_start = pd.Timestamp(f"{current_date.year}-01-01")
        year_start_days = all_prices.index[all_prices.index >= year_start]
        if year_start_days.empty:
            current_date = next_rebalance
            continue
        year_start_trade_date = year_start_days[0]

        # Drop tickers with no price on trade date
        available_tickers = [t for t in year_tickers if t in all_prices.columns]
        valid_tickers = [
            t for t in available_tickers
            if not pd.isna(all_prices.at[actual_trade_date, t])
        ]

        if not valid_tickers:
            current_date = next_rebalance
            continue

        # 2. Market-cap weights
        # Updated mktcap = initial_mktcap × (price_now / price_year_start)
        # For the first rebalance of the year the ratio is ~1 so we use raw mktcaps.
        updated_mktcaps = {}
        for ticker in valid_tickers:
            initial_mc = mktcap_dict.get(ticker, np.nan)
            if np.isnan(initial_mc):
                continue
            price_start = all_prices.at[year_start_trade_date, ticker] \
                if year_start_trade_date in all_prices.index else np.nan
            price_now   = all_prices.at[actual_trade_date, ticker]

            if pd.isna(price_start) or pd.isna(price_now) or price_start == 0:
                updated_mktcaps[ticker] = initial_mc      # fallback: use raw mktcap
            else:
                updated_mktcaps[ticker] = initial_mc * (price_now / price_start)

        if not updated_mktcaps:
            current_date = next_rebalance
            continue

        mktcap_series  = pd.Series(updated_mktcaps)
        target_weights = mktcap_series / mktcap_series.sum()

        # 3. Turnover
        turnover = (target_weights.sub(last_end_weights, fill_value=0)).abs().sum()

        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date':        actual_trade_date.strftime('%Y-%m-%d'),
                'ticker':                ticker,
                'assigned_weight':       w,
                'updated_mktcap_bn':     updated_mktcaps[ticker],
                'turnover_at_rebalance': turnover if ticker == target_weights.index[0] else 0,
            })

        # 4. Daily drift
        active_weights = target_weights.copy()
        period_prices  = all_prices.loc[
            actual_trade_date - pd.Timedelta(days=5):next_rebalance - pd.Timedelta(days=1),
            active_weights.index,
        ]

        if not period_prices.empty:
            daily_rets = period_prices.pct_change().dropna(how='all')
            daily_rets = daily_rets[daily_rets.index >= actual_trade_date]

            for day_timestamp, returns in daily_rets.iterrows():
                returns        = returns.fillna(0)
                day_pct_return = (active_weights * returns).sum()
                portfolio_value *= (1 + day_pct_return)
                active_weights  = (active_weights * (1 + returns)) / (1 + day_pct_return)

                portfolio_performance.append({
                    'date':             day_timestamp.strftime('%Y-%m-%d'),
                    'log_return':       np.log(1 + day_pct_return),
                    'cumulative_value': portfolio_value,
                })

        last_end_weights = active_weights
        current_date     = next_rebalance

    # --- Export ---
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"portfolio_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_{label}_details.csv", index=False
    )

print(f"\nFiles exported to {output_dir}")
