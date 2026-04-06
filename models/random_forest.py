import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- CONFIGURATION (Global Controls) ---
TRAIN_MONTHS = 60           # 5 Year Training
VAL_MONTHS = 12             # 1 Year Validation
MIN_COMPLETENESS = 0.50     # 50% data history required
WEIGHT_MAX = 0.10           # Max 10% per stock
WEIGHT_MIN = 0.01           # Min 1% per stock
N_ESTIMATORS = 135          # Number of trees
MAX_DEPTH = 20              # Max tree depth
N_RUNS = 1                  # How many independent runs per frequency (files get _1, _2, ...)
RANDOM_SEED = 42            # Fixed seed — same result every run

FREQUENCIES = {
    'Yearly': pd.DateOffset(years=1),
    'Semi-Annual': pd.DateOffset(months=6),
    'Quarterly': pd.DateOffset(months=3),
    'Monthly': pd.DateOffset(months=1)
}

start_invest = pd.Timestamp("1998-01-01")
end_invest = pd.Timestamp("2025-12-31")

# --- Directory Setup ---
DATA_PATH = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir = DATA_PATH / "results" / "data" / "random_forest"
output_dir.mkdir(parents=True, exist_ok=True)

# Load Data
all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)

import universe

def create_features(prices_df):
    """Generates technical and risk features for the RF model."""
    if len(prices_df) < 252: return pd.DataFrame()
    
    features = pd.DataFrame(index=prices_df.columns)
    returns = prices_df.pct_change()

    # 1. Momentum & Trend
    features['mom_1m'] = prices_df.pct_change(21).iloc[-1]
    features['mom_6m'] = prices_df.pct_change(126).iloc[-1]
    features['mom_12m'] = prices_df.pct_change(252).iloc[-1]
    
    # 2. Moving Average Ratio (50 / 200)
    ma50 = prices_df.tail(50).mean()
    ma200 = prices_df.tail(200).mean()
    features['ma_ratio'] = ma50 / ma200
    
    # 3. Distance from 52-Week High
    high_52w = prices_df.tail(252).max()
    features['dist_52w_high'] = prices_df.iloc[-1] / high_52w
    
    # 4. Max Drawdown (Last 12 Months)
    window_1y = prices_df.tail(252)
    rolling_max = window_1y.cummax()
    drawdown = (window_1y - rolling_max) / rolling_max
    features['max_dd_12m'] = drawdown.min()
    
    # 5. Volatility (6m & 12m)
    features['vol_6m'] = returns.tail(126).std() * np.sqrt(252)
    features['vol_12m'] = returns.tail(252).std() * np.sqrt(252)
    
    # 6. RSI (Standard 14-day)
    delta = returns.tail(15) # Need 14 changes
    gain = delta.clip(lower=0).mean()
    loss = -delta.clip(upper=0).mean()
    rs = gain / (loss + 1e-9) # Avoid division by zero
    features['rsi'] = 100 - (100 / (1 + rs))

    return features.dropna()

def allocate_weights(predictions, w_min, w_max):
    """Allocates weights proportionally to predictions within bounds."""
    # Ensure all predictions are treated as relative scores
    scores = predictions - predictions.min() + 0.0001
    weights = scores / scores.sum()
    
    for _ in range(15):
        weights = weights.clip(w_min, w_max)
        weights = weights / weights.sum()
        if weights.max() <= w_max + 1e-5 and weights.min() >= w_min - 1e-5:
            break
    return weights

# --- Backtest Loop ---
for run in range(1, N_RUNS + 1):
    run_seed = RANDOM_SEED  # same seed every run — change RANDOM_SEED at the top to vary
    print(f"\n=== Run {run}/{N_RUNS} (seed={run_seed}) ===")

    for label, offset in FREQUENCIES.items():
        print(f"  Processing Random Forest ({label}) | min coverage: {MIN_COMPLETENESS:.0%}")

        current_date = start_invest
        portfolio_value = 1.0
        last_end_weights = pd.Series(dtype=float)
        portfolio_performance = []
        rebalance_details = []
        model_stats = []

        while current_date < end_invest:
            next_rebalance = current_date + offset

            valid_days = all_prices.index[all_prices.index >= current_date]
            if valid_days.empty: break
            actual_trade_date = valid_days[0]

            target_weights = None  # will be set by model or fallback

            target_year = current_date.year - 1
            if target_year in universe.tickers:
                year_tickers = [t[0] for t in universe.tickers[target_year]]

                lb_start = actual_trade_date - pd.DateOffset(months=TRAIN_MONTHS + VAL_MONTHS + 12)
                lb_end   = actual_trade_date - pd.Timedelta(days=1)

                available     = [t for t in year_tickers if t in all_prices.columns]
                hist_prices   = all_prices.loc[lb_start:lb_end, available]

                if not hist_prices.empty:
                    coverage      = hist_prices.notnull().sum() / len(hist_prices)
                    valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

                    if len(valid_tickers) >= 2:
                        X_train, y_train = [], []
                        train_prices = hist_prices[valid_tickers].ffill()

                        for i in range(252, len(train_prices) - 21, 21):
                            feat = create_features(train_prices.iloc[:i])
                            if feat.empty: continue
                            fwd_ret = train_prices.iloc[i+21] / train_prices.iloc[i] - 1
                            common  = feat.index.intersection(fwd_ret.dropna().index)
                            X_train.append(feat.loc[common])
                            y_train.append(fwd_ret.loc[common])

                        if X_train:
                            model = RandomForestRegressor(
                                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                n_jobs=-1, random_state=run_seed,
                            )
                            model.fit(pd.concat(X_train), pd.concat(y_train))

                            current_feat = create_features(train_prices)
                            preds        = model.predict(current_feat)
                            # All tickers that passed MIN_COMPLETENESS — no TOP_N cutoff
                            pred_series  = pd.Series(preds, index=current_feat.index).sort_values(ascending=False)
                            target_weights = allocate_weights(pred_series, WEIGHT_MIN, WEIGHT_MAX)

            # Fallback: carry forward last weights, or equal-weight if first period
            if target_weights is None:
                if not last_end_weights.empty:
                    live = [t for t in last_end_weights.index if t in all_prices.columns]
                    if live:
                        tw = last_end_weights.loc[live]
                        target_weights = tw / tw.sum()
                        print(f"  [{label}] {current_date.date()}: model skipped — carrying forward {len(live)} tickers.")
                if target_weights is None:
                    year_tickers_fb = []
                    if target_year in universe.tickers:
                        year_tickers_fb = [t[0] for t in universe.tickers[target_year]
                                           if t[0] in all_prices.columns]
                    if not year_tickers_fb:
                        current_date = next_rebalance
                        continue
                    n_fb = len(year_tickers_fb)
                    target_weights = pd.Series(1.0 / n_fb, index=year_tickers_fb)
                    print(f"  [{label}] {current_date.date()}: first period fallback — equal weight {n_fb} tickers.")

            # Logging & Daily Drift
            turnover = (target_weights.sub(last_end_weights, fill_value=0)).abs().sum()
            for ticker, w in target_weights.items():
                rebalance_details.append({
                    'rebalance_date': actual_trade_date.strftime('%Y-%m-%d'),
                    'ticker': ticker, 'assigned_weight': w,
                    'turnover': turnover if ticker == target_weights.index[0] else 0
                })

            active_weights = target_weights.copy()
            # Fetch 5 extra days before actual_trade_date so pct_change() has a reference row
            period_prices = all_prices.loc[
                actual_trade_date - pd.Timedelta(days=5):next_rebalance - pd.Timedelta(days=1),
                active_weights.index
            ]

            if not period_prices.empty:
                daily_rets = period_prices.pct_change().dropna(how='all')
                daily_rets = daily_rets[daily_rets.index >= actual_trade_date]
                for day_timestamp, returns in daily_rets.iterrows():
                    returns = returns.fillna(0)
                    day_pct_return = (active_weights * returns).sum()
                    portfolio_value *= (1 + day_pct_return)
                    active_weights = (active_weights * (1 + returns)) / (1 + day_pct_return)

                    portfolio_performance.append({
                        'date': day_timestamp.strftime('%Y-%m-%d'),
                        'log_return': np.log(1 + day_pct_return),
                        'cumulative_value': portfolio_value
                    })

                # ── Model evaluation: predicted vs actual realized return ──────
                # Actual return over the period for each selected stock
                first_price = period_prices.iloc[0]
                last_price  = period_prices.iloc[-1]
                actual_ret  = (last_price / first_price - 1).dropna()
                common_idx  = pred_series.index.intersection(actual_ret.index)
                if len(common_idx) >= 2:
                    y_pred = pred_series.loc[common_idx].values
                    y_true = actual_ret.loc[common_idx].values
                    # Direction arrays (1 = up, 0 = down)
                    pred_dir = (y_pred >= 0).astype(int)
                    true_dir = (y_true >= 0).astype(int)
                    tp = np.sum((pred_dir == 1) & (true_dir == 1))
                    tn = np.sum((pred_dir == 0) & (true_dir == 0))
                    fp = np.sum((pred_dir == 1) & (true_dir == 0))
                    fn = np.sum((pred_dir == 0) & (true_dir == 1))
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                    geo_score   = np.sqrt(sensitivity * specificity) if not (np.isnan(sensitivity) or np.isnan(specificity)) else np.nan
                    dir_acc     = np.mean(pred_dir == true_dir)
                    mse  = np.mean((y_pred - y_true) ** 2)
                    rmse = np.sqrt(mse)
                    mae  = np.mean(np.abs(y_pred - y_true))
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                    r2   = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    # MAPE: skip zeros in denominator
                    nonzero = y_true != 0
                    mape = np.mean(np.abs((y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero])) if nonzero.any() else np.nan
                    model_stats.append({
                        'rebalance_date'      : actual_trade_date.strftime('%Y-%m-%d'),
                        'n_stocks'            : len(common_idx),
                        'RMSE'                : rmse,
                        'MSE'                 : mse,
                        'MAE'                 : mae,
                        'R_squared'           : r2,
                        'MAPE'                : mape,
                        'Directional_Accuracy': dir_acc,
                        'Geometric_Score'     : geo_score,
                    })

            last_end_weights = active_weights
            current_date = next_rebalance

        pd.DataFrame(portfolio_performance).to_csv(
            output_dir / f"portfolio_rf_{label}_{run}.csv", index=False
        )
        pd.DataFrame(rebalance_details).to_csv(
            output_dir / f"portfolio_rf_{label}_{run}_details.csv", index=False
        )
        pd.DataFrame(model_stats).to_csv(
            output_dir / f"portfolio_rf_{label}_{run}_statistics.csv", index=False
        )
        print(f"  [{label}] Run {run} exported.")