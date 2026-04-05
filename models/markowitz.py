



import pandas as pd
import numpy as np
import sys
from pathlib import Path
from pypfopt import EfficientFrontier, risk_models, expected_returns

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

LOOKBACK_MONTHS = 24        # 24 Month Lookback
MIN_COMPLETENESS = 0.50     # Min 50% of Stock History given, otherwise drop stock
WEIGHT_MAX = 0.10           # Max 10% of Stock
WEIGHT_MIN = 0.01           # Min 1% of Stock

start_data = "1992-01-01"
start_invest = pd.Timestamp("1998-01-01")
end_invest = pd.Timestamp("2025-12-31")

# Directory
DATA_PATH = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir = DATA_PATH / "results" / "data" / "markowitz"
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

        # 2. Optimization at Rebalance Date
        lb_start = current_date - pd.DateOffset(months=LOOKBACK_MONTHS)
        lb_end = current_date - pd.Timedelta(days=1)
        available = [t for t in year_tickers if t in all_prices.columns]
        lb_data = all_prices.loc[lb_start:lb_end, available]
        
        coverage = lb_data.notnull().sum() / len(lb_data) if not lb_data.empty else 0
        valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

        try:
            lb_prices_final = lb_data[valid_tickers].ffill().dropna(axis=1)
            mu = expected_returns.mean_historical_return(lb_prices_final)
            S = risk_models.sample_cov(lb_prices_final)
            
            ef = EfficientFrontier(mu, S, weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
            try:
                weights = ef.max_sharpe()
            except:
                ef = EfficientFrontier(mu, S, weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
                weights = ef.min_volatility()
            
            target_weights = pd.Series(ef.clean_weights())
            
            # --- DETAILED LOGGING (Only on Rebalance Date) ---
            turnover = (target_weights.sub(last_end_weights, fill_value=0)).abs().sum()
            
            for ticker, w in target_weights.items():
                rebalance_details.append({
                    'rebalance_date': current_date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'assigned_weight': w,
                    'turnover_at_rebalance': turnover if ticker == target_weights.index[0] else 0
                })

            active_weights = target_weights.copy()

        except Exception as e:
            print(f"Error at {current_date}: {e}")
            current_date = next_rebalance
            continue

        # 3. Daily Performance Calculation (The drift period)
        # Fetch 5 extra days before current_date so pct_change() has a reference row (handles weekends/holidays)
        period_prices = all_prices.loc[current_date - pd.Timedelta(days=5):next_rebalance - pd.Timedelta(days=1), active_weights.index]
        if period_prices.empty:
            current_date = next_rebalance
            continue

        daily_stock_rets = period_prices.pct_change().dropna(how='all')
        daily_stock_rets = daily_stock_rets[daily_stock_rets.index >= current_date]

        for day_timestamp, returns in daily_stock_rets.iterrows():
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