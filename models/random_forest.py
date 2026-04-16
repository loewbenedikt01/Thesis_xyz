import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from itertools import product

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- CONFIGURATION (Global Controls) ---
TRAIN_MONTHS    = 60            # 5 Year Training
VAL_MONTHS      = 12            # 1 Year Validation
MIN_COMPLETENESS = 0.50         # 50% data history required
WEIGHT_MAX      = 0.10          # Max 10% per stock
WEIGHT_MIN      = 0.01          # Min 1% per stock
N_RUNS          = 10            # Number of seeds to average into one prediction
BASE_SEED       = 41            # Seeds will be BASE_SEED+0, BASE_SEED+1, ..., BASE_SEED+9

# --- Hyperparameter Search Grid ---
# All combinations are evaluated on the validation set each period;
# the best combination (highest val R²) is used for that period's prediction.
PARAM_GRID = {
    'max_depth'       : [1, 2, 3, 4, 5],       # 5 depths tried simultaneously
    'n_estimators'    : [100, 135, 200],             # tree count
    'min_samples_leaf': [2, 3, 4, 5],                 # leaf size regularisation
}

FREQUENCIES = {
    'Yearly':      (pd.DateOffset(years=1),  252),
    'Semi-Annual': (pd.DateOffset(months=6), 126),
    'Quarterly':   (pd.DateOffset(months=3),  63),
    'Monthly':     (pd.DateOffset(months=1),  21),
}

start_invest = pd.Timestamp("1998-01-01")
end_invest   = pd.Timestamp("2025-12-31")

# --- Directory Setup ---
DATA_PATH  = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / "rf_tuned_ensemble"
output_dir.mkdir(parents=True, exist_ok=True)

# Load Data
all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)

import universe


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def create_features(prices_df):
    """Generates technical and risk features for the RF model."""
    if len(prices_df) < 260:
        return pd.DataFrame()

    features = pd.DataFrame(index=prices_df.columns)
    returns  = prices_df.pct_change()

    # 1. Momentum
    features['mom_1w']   = prices_df.pct_change(5).iloc[-1]
    features['mom_1m']   = prices_df.pct_change(21).iloc[-1]
    features['mom_3m']   = prices_df.pct_change(63).iloc[-1]
    features['mom_6m']   = prices_df.pct_change(126).iloc[-1]
    features['mom_12_1'] = prices_df.iloc[-22] / prices_df.iloc[-253] - 1

    # 2. Trend
    ma50  = prices_df.tail(50).mean()
    ma200 = prices_df.tail(200).mean()
    features['ma_ratio'] = ma50 / ma200

    # 3. Distance from 52-week high
    high_52w = prices_df.tail(252).max()
    features['dist_52w_high'] = prices_df.iloc[-1] / high_52w

    # 4. Max drawdown (last 12 months)
    window_1y   = prices_df.tail(252)
    rolling_max = window_1y.cummax()
    features['max_dd_12m'] = ((window_1y - rolling_max) / rolling_max).min()

    # 5. Volatility + regime ratio
    vol_1m  = returns.tail(21).std()  * np.sqrt(252)
    vol_6m  = returns.tail(126).std() * np.sqrt(252)
    vol_12m = returns.tail(252).std() * np.sqrt(252)
    features['vol_1m']    = vol_1m
    features['vol_6m']    = vol_6m
    features['vol_12m']   = vol_12m
    features['vol_ratio'] = vol_1m / (vol_12m + 1e-9)

    # 6. Bollinger Band position
    p20 = prices_df.tail(20)
    features['bb_position'] = (prices_df.iloc[-1] - p20.mean()) / (2 * p20.std() + 1e-9)

    # 7. RSI — Wilder's EMA
    delta    = returns.tail(252)
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1]
    rs = avg_gain / (avg_loss + 1e-9)
    features['rsi'] = 100 - (100 / (1 + rs))

    # 8. Sharpe-like score
    features['sharpe_6m'] = features['mom_6m'] / (features['vol_6m'] + 1e-9)

    # Cross-sectional rank normalization → [0, 1]
    features = features.rank(pct=True)

    return features.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Weight Allocation
# ─────────────────────────────────────────────────────────────────────────────

def allocate_weights(predictions, w_min, w_max):
    """Allocates weights proportionally to predictions within bounds."""
    scores  = predictions - predictions.min() + 0.0001
    weights = scores / scores.sum()
    for _ in range(15):
        weights = weights.clip(w_min, w_max)
        weights = weights / weights.sum()
        if weights.max() <= w_max + 1e-5 and weights.min() >= w_min - 1e-5:
            break
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning via Validation Set
# ─────────────────────────────────────────────────────────────────────────────

def build_param_combinations(grid):
    """Returns a list of param dicts from a grid dict."""
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def tune_and_predict(X_train_all, y_train_all, X_val_all, y_val_all,
                     current_feat, param_grid, seeds):
    """
    For every combination in param_grid:
      - Train on train set (averaged over `seeds`)
      - Evaluate val R² on val set
    Pick the best combination, retrain on train+val, then predict.

    Returns
    -------
    pred_series   : pd.Series  — averaged ensemble prediction (index = stock tickers)
    best_params   : dict       — winning hyperparameters this period
    best_val_r2   : float      — validation R² of winning combo
    """
    X_train = pd.concat(X_train_all)
    y_train = pd.concat(y_train_all)
    X_val   = pd.concat(X_val_all)
    y_val   = pd.concat(y_val_all)
    X_all   = pd.concat([X_train, X_val])
    y_all   = pd.concat([y_train, y_val])

    param_combos  = build_param_combinations(param_grid)
    best_val_r2   = -np.inf
    best_params   = param_combos[0]

    # ── Step 1: Grid search on validation set ────────────────────────────────
    for params in param_combos:
        fold_preds = []

        for seed in seeds:
            model = RandomForestRegressor(
                n_estimators     = params['n_estimators'],
                max_depth        = params['max_depth'],
                min_samples_leaf = params['min_samples_leaf'],
                max_features     = 'sqrt',
                n_jobs           = -1,
                random_state     = seed,
            )
            model.fit(X_train, y_train)
            fold_preds.append(model.predict(X_val))

        # Average predictions across seeds, then compute val R²
        avg_pred = np.mean(fold_preds, axis=0)
        ss_res   = np.sum((y_val.values - avg_pred) ** 2)
        ss_tot   = np.sum((y_val.values - y_val.values.mean()) ** 2)
        val_r2   = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        if not np.isnan(val_r2) and val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_params = params

    # ── Step 2: Retrain best config on train + val, predict with seed ensemble ─
    final_preds = []
    for seed in seeds:
        model = RandomForestRegressor(
            n_estimators     = best_params['n_estimators'],
            max_depth        = best_params['max_depth'],
            min_samples_leaf = best_params['min_samples_leaf'],
            max_features     = 'sqrt',
            n_jobs           = -1,
            random_state     = seed,
        )
        model.fit(X_all, y_all)
        final_preds.append(model.predict(current_feat))

    # Average seed predictions → single ensemble prediction
    avg_final    = np.mean(final_preds, axis=0)
    pred_series  = pd.Series(avg_final, index=current_feat.index).sort_values(ascending=False)

    return pred_series, best_params, best_val_r2


# ─────────────────────────────────────────────────────────────────────────────
# Main Backtest Loop
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [BASE_SEED + i for i in range(N_RUNS)]  # [41, 42, ..., 50]

for label, (offset, horizon) in FREQUENCIES.items():
    print(f"\n=== Random Forest — {label} | horizon={horizon}d | "
          f"grid={len(build_param_combinations(PARAM_GRID))} combos | "
          f"seeds={SEEDS} ===")

    current_date          = start_invest
    portfolio_value       = 1.0
    last_end_weights      = pd.Series(dtype=float)
    portfolio_performance = []
    rebalance_details     = []
    model_stats           = []

    while current_date < end_invest:
        next_rebalance = current_date + offset

        valid_days = all_prices.index[all_prices.index >= current_date]
        if valid_days.empty:
            break
        actual_trade_date = valid_days[0]

        target_weights = None
        pred_series    = None

        target_year = current_date.year - 1
        if target_year in universe.tickers:
            year_tickers = [t[0] for t in universe.tickers[target_year]]

            lb_start = actual_trade_date - pd.DateOffset(months=TRAIN_MONTHS + VAL_MONTHS + 12)
            lb_end   = actual_trade_date - pd.Timedelta(days=1)

            available   = [t for t in year_tickers if t in all_prices.columns]
            hist_prices = all_prices.loc[lb_start:lb_end, available]

            if not hist_prices.empty:
                coverage      = hist_prices.notnull().sum() / len(hist_prices)
                valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

                if len(valid_tickers) >= 2:
                    train_prices = hist_prices[valid_tickers].ffill()

                    val_start_date = actual_trade_date - pd.DateOffset(months=VAL_MONTHS)
                    val_split      = int(np.searchsorted(train_prices.index, val_start_date))

                    X_train, y_train = [], []
                    X_val,   y_val   = [], []

                    # Training samples
                    for i in range(260, val_split - horizon, 21):
                        feat = create_features(train_prices.iloc[:i])
                        if feat.empty:
                            continue
                        fwd_ret = train_prices.iloc[i + horizon] / train_prices.iloc[i] - 1
                        fwd_ret = fwd_ret.dropna()
                        common  = feat.index.intersection(fwd_ret.index)
                        if len(common) < 2:
                            continue
                        y_ranked = fwd_ret.loc[common].rank(pct=True)
                        X_train.append(feat.loc[common])
                        y_train.append(y_ranked)

                    # Validation samples
                    for i in range(val_split, len(train_prices) - horizon, 21):
                        feat = create_features(train_prices.iloc[:i])
                        if feat.empty:
                            continue
                        fwd_ret = train_prices.iloc[i + horizon] / train_prices.iloc[i] - 1
                        fwd_ret = fwd_ret.dropna()
                        common  = feat.index.intersection(fwd_ret.index)
                        if len(common) < 2:
                            continue
                        y_ranked = fwd_ret.loc[common].rank(pct=True)
                        X_val.append(feat.loc[common])
                        y_val.append(y_ranked)

                    if X_train and X_val:
                        current_feat = create_features(train_prices)

                        if not current_feat.empty:
                            pred_series, best_params, best_val_r2 = tune_and_predict(
                                X_train, y_train,
                                X_val,   y_val,
                                current_feat,
                                PARAM_GRID,
                                SEEDS,
                            )
                            target_weights = allocate_weights(pred_series, WEIGHT_MIN, WEIGHT_MAX)

                            print(
                                f"  [{label}] {current_date.date()} — "
                                f"best_params={best_params} | "
                                f"val_R²={best_val_r2:.3f}"
                            )

        # ── Fallback logic ────────────────────────────────────────────────────
        if target_weights is None:
            if not last_end_weights.empty:
                live = [t for t in last_end_weights.index if t in all_prices.columns]
                if live:
                    tw = last_end_weights.loc[live]
                    target_weights = tw / tw.sum()
                    print(f"  [{label}] {current_date.date()}: model skipped — "
                          f"carrying forward {len(live)} tickers.")
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
                print(f"  [{label}] {current_date.date()}: first period fallback — "
                      f"equal weight {n_fb} tickers.")

        # ── Logging & Daily Drift ─────────────────────────────────────────────
        turnover = (target_weights.sub(last_end_weights, fill_value=0)).abs().sum()
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date' : actual_trade_date.strftime('%Y-%m-%d'),
                'ticker'         : ticker,
                'assigned_weight': w,
                'turnover'       : turnover if ticker == target_weights.index[0] else 0,
                'best_depth'     : best_params.get('max_depth', np.nan) if pred_series is not None else np.nan,
                'best_n_est'     : best_params.get('n_estimators', np.nan) if pred_series is not None else np.nan,
                'best_leaf'      : best_params.get('min_samples_leaf', np.nan) if pred_series is not None else np.nan,
            })

        active_weights = target_weights.copy()
        period_prices  = all_prices.loc[
            actual_trade_date - pd.Timedelta(days=5):next_rebalance - pd.Timedelta(days=1),
            active_weights.index
        ]

        if not period_prices.empty:
            daily_rets = period_prices.pct_change().dropna(how='all')
            daily_rets = daily_rets[daily_rets.index >= actual_trade_date]

            for day_timestamp, day_returns in daily_rets.iterrows():
                day_returns    = day_returns.fillna(0)
                day_pct_return = (active_weights * day_returns).sum()
                portfolio_value *= (1 + day_pct_return)
                active_weights  = (active_weights * (1 + day_returns)) / (1 + day_pct_return)

                portfolio_performance.append({
                    'date'            : day_timestamp.strftime('%Y-%m-%d'),
                    'log_return'      : np.log(1 + day_pct_return),
                    'cumulative_value': portfolio_value,
                })

            # ── Out-of-sample model evaluation ───────────────────────────────
            hold_prices = period_prices.loc[actual_trade_date:]
            first_price = hold_prices.iloc[0]
            last_price  = hold_prices.iloc[-1]
            actual_ret  = (last_price / first_price - 1).dropna()
            common_idx  = pred_series.index.intersection(actual_ret.index) if pred_series is not None else []

            if len(common_idx) >= 2:
                y_pred      = pred_series.loc[common_idx].values
                y_true      = actual_ret.loc[common_idx].values
                y_true_rank = pd.Series(y_true).rank(pct=True).values

                pred_dir    = (y_pred >= 0.5).astype(int)
                true_dir    = (y_true >= 0).astype(int)
                tp = np.sum((pred_dir == 1) & (true_dir == 1))
                tn = np.sum((pred_dir == 0) & (true_dir == 0))
                fp = np.sum((pred_dir == 1) & (true_dir == 0))
                fn = np.sum((pred_dir == 0) & (true_dir == 1))
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                geo_score   = (np.sqrt(sensitivity * specificity)
                               if not (np.isnan(sensitivity) or np.isnan(specificity)) else np.nan)

                ss_res   = np.sum((y_true_rank - y_pred) ** 2)
                ss_tot   = np.sum((y_true_rank - y_true_rank.mean()) ** 2)
                r2       = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                spearman = pd.Series(y_pred).corr(pd.Series(y_true_rank), method='spearman')

                model_stats.append({
                    'rebalance_date'      : actual_trade_date.strftime('%Y-%m-%d'),
                    'n_stocks'            : len(common_idx),
                    'RMSE'                : np.sqrt(np.mean((y_pred - y_true_rank) ** 2)),
                    'MSE'                 : np.mean((y_pred - y_true_rank) ** 2),
                    'MAE'                 : np.mean(np.abs(y_pred - y_true_rank)),
                    'R_squared'           : r2,
                    'Spearman'            : spearman,
                    'Directional_Accuracy': np.mean(pred_dir == true_dir),
                    'Geometric_Score'     : geo_score,
                    # ── Tuning diagnostics ────────────────────────────────────
                    'best_depth'          : best_params.get('max_depth')        if pred_series is not None else np.nan,
                    'best_n_estimators'   : best_params.get('n_estimators')     if pred_series is not None else np.nan,
                    'best_min_leaf'       : best_params.get('min_samples_leaf') if pred_series is not None else np.nan,
                    'val_R2_selected'     : best_val_r2                         if pred_series is not None else np.nan,
                })

        last_end_weights = active_weights
        current_date     = next_rebalance

    # ── Export results ────────────────────────────────────────────────────────
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"portfolio_rf_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_rf_{label}_details.csv", index=False
    )
    pd.DataFrame(model_stats).to_csv(
        output_dir / f"portfolio_rf_{label}_statistics.csv", index=False
    )
    print(f"\n  [{label}] Complete. Files saved to {output_dir}")