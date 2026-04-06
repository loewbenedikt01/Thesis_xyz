



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
    'Yearly': pd.DateOffset(years=1),
    'Semi-Annual': pd.DateOffset(months=6),
    'Quarterly': pd.DateOffset(months=3),
    'Monthly': pd.DateOffset(months=1)
}

start_data = "1992-01-01"
start_invest = pd.Timestamp("1998-01-01")
end_invest = pd.Timestamp("2025-12-31")

# Directory
DATA_PATH = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir = DATA_PATH / "results" / "data" / "equal_weight"
output_dir.mkdir(parents=True, exist_ok=True)

# Load Data
all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index) .tz_localize(None)

# --- Main Frequency Loop ---
for label, offset in FREQUENCIES.items():
    print(f"Processing: {label}")

    current_date = start_invest
    portfolio_value = 1.0
    last_end_weights = pd.Series(dtype=float)
    
    portfolio_performance = [] # For portfolio_{label}.csv
    rebalance_details = []    # For portfolio_{label}_details.csv

    while current_date < end_invest:
        # 1. Universe Selection
        target_year = current_date.year - 1
        if target_year not in universe.tickers:
            current_date += offset
            continue
        
        year_tickers = [t[0] for t in universe.tickers[target_year]]
        next_rebalance = current_date + offset
        available_tickers = [t for t in year_tickers if t in all_prices.columns]
        trading_days_ahead = all_prices.index[all_prices.index >= current_date]
        if trading_days_ahead.empty:
            current_date += offset
            continue
            
        actual_trade_date = trading_days_ahead[0]
        # Drop tickers that have a NaN price on the actual trade date (current_date may be a holiday)
        valid_tickers = [t for t in available_tickers if not pd.isna(all_prices.at[actual_trade_date, t])]

        if not valid_tickers:
            current_date = next_rebalance
            continue

        # --- PURE EQUAL WEIGHT LOGIC (1/N) ---
        n_assets = len(valid_tickers)
        target_weights = pd.Series(1.0 / n_assets, index=valid_tickers)

        # Calculate Turnover (Change from last period's ending drifted weights)
        turnover = (target_weights.sub(last_end_weights, fill_value=0)).abs().sum()
        
        # Log Rebalance Details
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date': current_date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'assigned_weight': w,
                'turnover_at_rebalance': turnover if ticker == target_weights.index[0] else 0
            })

        active_weights = target_weights.copy()

        # 3. Daily Performance Calculation (The drift period)
        period_prices = all_prices.loc[current_date - pd.Timedelta(days=5):next_rebalance - pd.Timedelta(days=1), active_weights.index]
        if period_prices.empty:
            current_date = next_rebalance
            continue

        daily_stock_rets = period_prices.pct_change().dropna(how='all')
        daily_stock_rets = daily_stock_rets[daily_stock_rets.index >= current_date]

        for day_timestamp, returns in daily_stock_rets.iterrows():
            returns = returns.fillna(0)
            day_pct_return = (active_weights * returns).sum()
            
            # Log Return calculation: ln(1 + r)
            day_log_return = np.log(1 + day_pct_return)
            portfolio_value *= (1 + day_pct_return)

            # Update weights for the next day (drift)
            active_weights = (active_weights * (1 + returns)) / (1 + day_pct_return)

            # --- PERFORMANCE LOGGING (Daily Portfolio Level) ---
            portfolio_performance.append({
                'date': day_timestamp.strftime('%Y-%m-%d'),
                'log_return': day_log_return,
                'cumulative_value': portfolio_value
            })

        last_end_weights = active_weights
        current_date = next_rebalance

    # --- Export Files ---
    # 1. Daily Portfolio Performance
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"portfolio_{label}.csv", index=False
    )
    
    # 2. Rebalance Snapshot Details
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_{label}_details.csv", index=False
    )

print(f"\nFiles exported to {output_dir}")