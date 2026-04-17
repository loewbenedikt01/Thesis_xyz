"""
Random Forest Portfolio Model — v3
===================================
Fixes applied vs v2:
  1. best_params / best_val_r2 initialised at the top of each loop iteration
     → no NameError when the model block is skipped (fallback path)
  2. Forward-return target uses date-based lookup instead of iloc position
     → survives gaps / halted stocks without silently landing on the wrong date
  3. Transaction costs: optional per-trade cost applied at each rebalance
     → set TC_BPS = 0 to disable; increase to model realistic friction
  4. Validation window extended to 24 months (VAL_MONTHS = 24)
  5. ffill capped at 5 consecutive days — avoids stale prices for thin stocks
  6. Universe logic: use tickers[year-1] to predict for year `year`
     e.g. at start of 1998, train on 1997 universe; at start of 1999 use 1998, etc.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from itertools import product

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_MONTHS     = 60       # training lookback in months
VAL_MONTHS       = 24       # FIX 4: extended from 12 → 24 months
MIN_COMPLETENESS = 0.50     # min fraction of non-NaN rows required per ticker
WEIGHT_MAX       = 0.10     # max portfolio weight per stock
WEIGHT_MIN       = 0.01     # min portfolio weight per stock
N_RUNS           = 10       # seeds averaged into one ensemble prediction
BASE_SEED        = 41       # seeds: 41, 42, …, 50

# FIX 3: Transaction costs
# Set TC_BPS = 0 to run cost-free. Increase to model friction.
# Cost is applied as:  return_after_cost = return - turnover * TC_BPS/10000
# Example: TC_BPS = 10  →  10 basis points per unit of one-way turnover
TC_BPS = 0   # <-- change this value; 0 = no costs, 10 = 10bps, 30 = 30bps

# Hyperparameter search grid — all combinations evaluated each period
PARAM_GRID = {
    'max_depth'       : [1, 2, 3, 4, 5],
    'n_estimators'    : [100, 135, 200],
    'min_samples_leaf': [2, 3, 4, 5],
}

FREQUENCIES = {
    'Yearly':      (pd.DateOffset(years=1),  252),
    'Semi-Annual': (pd.DateOffset(months=6), 126),
    'Quarterly':   (pd.DateOffset(months=3),  63),
    'Monthly':     (pd.DateOffset(months=1),  21),
}

start_invest = pd.Timestamp("1998-01-01")
end_invest   = pd.Timestamp("2025-12-31")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / "random_forest"
output_dir.mkdir(parents=True, exist_ok=True)

all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)

import universe

SEEDS = [BASE_SEED + i for i in range(N_RUNS)]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def create_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectionally rank-normalised features from a price matrix.
    prices_df : rows = trading days (ascending), columns = tickers
    Returns   : DataFrame  shape (n_tickers, n_features), or empty if too short
    """
    if len(prices_df) < 260:
        return pd.DataFrame()

    features = pd.DataFrame(index=prices_df.columns)
    returns  = prices_df.pct_change()

    # --- Momentum ---
    features['mom_1w']   = prices_df.pct_change(5).iloc[-1]
    features['mom_1m']   = prices_df.pct_change(21).iloc[-1]
    features['mom_3m']   = prices_df.pct_change(63).iloc[-1]
    features['mom_6m']   = prices_df.pct_change(126).iloc[-1]
    # Skip-month momentum: return from t-13m to t-2m (avoids 1-month reversal)
    features['mom_12_1'] = prices_df.iloc[-22] / prices_df.iloc[-253] - 1

    # --- Trend ---
    features['ma_ratio'] = prices_df.tail(50).mean() / prices_df.tail(200).mean()

    # --- Distance from 52-week high ---
    features['dist_52w_high'] = prices_df.iloc[-1] / prices_df.tail(252).max()

    # --- Max drawdown over last 12 months ---
    w1y = prices_df.tail(252)
    features['max_dd_12m'] = ((w1y - w1y.cummax()) / w1y.cummax()).min()

    # --- Volatility + regime ratio ---
    vol_1m  = returns.tail(21).std()  * np.sqrt(252)
    vol_6m  = returns.tail(126).std() * np.sqrt(252)
    vol_12m = returns.tail(252).std() * np.sqrt(252)
    features['vol_1m']    = vol_1m
    features['vol_6m']    = vol_6m
    features['vol_12m']   = vol_12m
    features['vol_ratio'] = vol_1m / (vol_12m + 1e-9)

    # --- Bollinger Band position ---
    p20 = prices_df.tail(20)
    features['bb_position'] = (
        (prices_df.iloc[-1] - p20.mean()) / (2 * p20.std() + 1e-9)
    )

    # --- RSI (Wilder's EMA, 14-period) ---
    delta    = returns.tail(252)
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14,
                                       adjust=False).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14,
                                          adjust=False).mean().iloc[-1]
    rs = avg_gain / (avg_loss + 1e-9)
    features['rsi'] = 100 - (100 / (1 + rs))

    # --- Risk-adjusted momentum ---
    features['sharpe_6m'] = features['mom_6m'] / (features['vol_6m'] + 1e-9)

    # Cross-sectional rank → [0, 1] per feature
    features = features.rank(pct=True)
    return features.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: DATE-BASED FORWARD RETURN
# ─────────────────────────────────────────────────────────────────────────────
def forward_return(prices_df: pd.DataFrame, anchor_iloc: int,
                   horizon_days: int) -> pd.Series:
    """
    Compute forward return starting at prices_df.index[anchor_iloc],
    landing on the first available trading day >= anchor_date + horizon_days.
    Uses date arithmetic — immune to price gaps and halted stocks.
    """
    anchor_date  = prices_df.index[anchor_iloc]
    target_date  = anchor_date + pd.Timedelta(days=int(horizon_days * 1.5))
    future_iloc  = prices_df.index.searchsorted(target_date)

    # Guard: if target lands beyond the DataFrame, skip this sample
    if future_iloc >= len(prices_df):
        return pd.Series(dtype=float)

    p_now    = prices_df.iloc[anchor_iloc]
    p_future = prices_df.iloc[future_iloc]
    return (p_future / p_now - 1).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────
def allocate_weights(predictions: pd.Series, w_min: float,
                     w_max: float) -> pd.Series:
    """Proportional allocation bounded to [w_min, w_max], iterative clipping."""
    scores  = predictions - predictions.min() + 1e-4
    weights = scores / scores.sum()
    for _ in range(20):
        weights = weights.clip(w_min, w_max)
        weights = weights / weights.sum()
        if weights.max() <= w_max + 1e-5 and weights.min() >= w_min - 1e-5:
            break
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER GRID HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_param_combinations(grid: dict) -> list:
    keys, values = list(grid.keys()), list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ─────────────────────────────────────────────────────────────────────────────
# TUNE + ENSEMBLE PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def tune_and_predict(X_train_list, y_train_list,
                     X_val_list,   y_val_list,
                     current_feat: pd.DataFrame,
                     param_grid:   dict,
                     seeds:        list):
    """
    1. Grid-search all param combos on the validation set (seed-averaged).
    2. Retrain the winner on train + val combined.
    3. Return seed-averaged predictions on current_feat.

    Returns
    -------
    pred_series  : pd.Series  (index = tickers)
    best_params  : dict
    best_val_r2  : float
    """
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    X_val   = pd.concat(X_val_list)
    y_val   = pd.concat(y_val_list)
    X_all   = pd.concat([X_train, X_val])
    y_all   = pd.concat([y_train, y_val])

    param_combos = build_param_combinations(param_grid)
    best_val_r2  = -np.inf
    best_params  = param_combos[0]           # safe default

    # ── Grid search ──────────────────────────────────────────────────────────
    for params in param_combos:
        seed_preds = []
        for seed in seeds:
            m = RandomForestRegressor(
                n_estimators     = params['n_estimators'],
                max_depth        = params['max_depth'],
                min_samples_leaf = params['min_samples_leaf'],
                max_features     = 'sqrt',
                n_jobs           = -1,
                random_state     = seed,
            )
            m.fit(X_train, y_train)
            seed_preds.append(m.predict(X_val))

        avg_pred = np.mean(seed_preds, axis=0)
        ss_res   = np.sum((y_val.values - avg_pred) ** 2)
        ss_tot   = np.sum((y_val.values - y_val.values.mean()) ** 2)
        val_r2   = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        if not np.isnan(val_r2) and val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_params = params

    # ── Final model: retrain best config on train + val, seed ensemble ───────
    final_preds = []
    for seed in seeds:
        m = RandomForestRegressor(
            n_estimators     = best_params['n_estimators'],
            max_depth        = best_params['max_depth'],
            min_samples_leaf = best_params['min_samples_leaf'],
            max_features     = 'sqrt',
            n_jobs           = -1,
            random_state     = seed,
        )
        m.fit(X_all, y_all)
        final_preds.append(m.predict(current_feat))

    avg_final   = np.mean(final_preds, axis=0)
    pred_series = pd.Series(avg_final, index=current_feat.index).sort_values(
        ascending=False
    )
    return pred_series, best_params, best_val_r2


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────
n_combos = len(build_param_combinations(PARAM_GRID))

for label, (offset, horizon) in FREQUENCIES.items():
    print(
        f"\n=== RF [{label}] | horizon={horizon}d | "
        f"{n_combos} param combos | {len(SEEDS)} seeds | "
        f"TC={TC_BPS}bps ==="
    )

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

        # ── FIX 1: always initialise before any conditional block ─────────────
        target_weights = None
        pred_series    = None
        best_params    = {}          # ← guaranteed to exist on fallback path
        best_val_r2    = np.nan      # ← guaranteed to exist on fallback path

        # ── Universe for this period ──────────────────────────────────────────
        # FIX 6 (universe logic):
        #   At start of year Y we invest using the universe selected at end of Y-1.
        #   universe.tickers[1997] → used during 1998, etc.
        invest_year  = current_date.year          # the calendar year we're in
        select_year  = invest_year - 1            # the year whose snapshot we use

        if select_year in universe.tickers:
            year_tickers = [t[0] for t in universe.tickers[select_year]]

            # Lookback window: need TRAIN + VAL months of history before today
            lb_start = actual_trade_date - pd.DateOffset(
                months=TRAIN_MONTHS + VAL_MONTHS + 12
            )
            lb_end   = actual_trade_date - pd.Timedelta(days=1)  # strictly before today

            available   = [t for t in year_tickers if t in all_prices.columns]
            hist_prices = all_prices.loc[lb_start:lb_end, available]

            if not hist_prices.empty:
                coverage      = hist_prices.notnull().sum() / len(hist_prices)
                valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

                if len(valid_tickers) >= 2:
                    # FIX 5: cap ffill at 5 days — avoids stale prices for thin stocks
                    train_prices = hist_prices[valid_tickers].ffill(limit=5)

                    # Temporal split index
                    val_start_date = actual_trade_date - pd.DateOffset(months=VAL_MONTHS)
                    val_split      = int(np.searchsorted(train_prices.index,
                                                         val_start_date))

                    X_train_list, y_train_list = [], []
                    X_val_list,   y_val_list   = [], []

                    # Training samples (step every 21 days ≈ monthly)
                    for i in range(260, val_split - horizon, 21):
                        feat    = create_features(train_prices.iloc[:i])
                        if feat.empty:
                            continue
                        # FIX 2: date-based forward return
                        fwd_ret = forward_return(train_prices, i, horizon)
                        if fwd_ret.empty:
                            continue
                        common = feat.index.intersection(fwd_ret.index)
                        if len(common) < 2:
                            continue
                        y_ranked = fwd_ret.loc[common].rank(pct=True)
                        X_train_list.append(feat.loc[common])
                        y_train_list.append(y_ranked)

                    # Validation samples
                    for i in range(val_split, len(train_prices) - horizon, 21):
                        feat    = create_features(train_prices.iloc[:i])
                        if feat.empty:
                            continue
                        fwd_ret = forward_return(train_prices, i, horizon)
                        if fwd_ret.empty:
                            continue
                        common = feat.index.intersection(fwd_ret.index)
                        if len(common) < 2:
                            continue
                        y_ranked = fwd_ret.loc[common].rank(pct=True)
                        X_val_list.append(feat.loc[common])
                        y_val_list.append(y_ranked)

                    if X_train_list and X_val_list:
                        current_feat = create_features(train_prices)

                        if not current_feat.empty:
                            pred_series, best_params, best_val_r2 = tune_and_predict(
                                X_train_list, y_train_list,
                                X_val_list,   y_val_list,
                                current_feat, PARAM_GRID, SEEDS,
                            )
                            target_weights = allocate_weights(
                                pred_series, WEIGHT_MIN, WEIGHT_MAX
                            )
                            print(
                                f"  [{label}] {current_date.date()} | "
                                f"universe={select_year} ({len(valid_tickers)} stocks) | "
                                f"best={best_params} | val_R²={best_val_r2:.4f}"
                            )

        # ── Fallback: carry forward or equal-weight ───────────────────────────
        if target_weights is None:
            if not last_end_weights.empty:
                live = [t for t in last_end_weights.index if t in all_prices.columns]
                if live:
                    tw             = last_end_weights.loc[live]
                    target_weights = tw / tw.sum()
                    print(f"  [{label}] {current_date.date()}: "
                          f"fallback — carrying forward {len(live)} tickers.")
            if target_weights is None:
                fb_tickers = (
                    [t[0] for t in universe.tickers[select_year]
                     if t[0] in all_prices.columns]
                    if select_year in universe.tickers else []
                )
                if not fb_tickers:
                    current_date = next_rebalance
                    continue
                target_weights = pd.Series(1.0 / len(fb_tickers), index=fb_tickers)
                print(f"  [{label}] {current_date.date()}: "
                      f"fallback — equal-weight {len(fb_tickers)} tickers.")

        # ── FIX 3: transaction cost at rebalance ──────────────────────────────
        # One-way turnover: sum of absolute weight changes
        turnover = target_weights.sub(last_end_weights, fill_value=0).abs().sum()
        if TC_BPS > 0:
            tc_drag         = turnover * TC_BPS / 10_000
            portfolio_value *= (1 - tc_drag)   # deduct cost from portfolio value

        # ── Logging rebalance ─────────────────────────────────────────────────
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date'  : actual_trade_date.strftime('%Y-%m-%d'),
                'invest_year'     : invest_year,
                'select_year'     : select_year,
                'ticker'          : ticker,
                'assigned_weight' : w,
                'turnover'        : round(turnover, 6) if ticker == target_weights.index[0] else 0,
                'tc_drag_bps'     : round(turnover * TC_BPS, 4) if ticker == target_weights.index[0] else 0,
                'best_depth'      : best_params.get('max_depth',        np.nan),
                'best_n_est'      : best_params.get('n_estimators',     np.nan),
                'best_leaf'       : best_params.get('min_samples_leaf', np.nan),
                'val_R2_selected' : round(best_val_r2, 6) if not np.isnan(best_val_r2) else np.nan,
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
                day_ret        = day_ret.fillna(0)
                day_pct        = (active_weights * day_ret).sum()
                portfolio_value *= (1 + day_pct)
                active_weights  = (
                    active_weights * (1 + day_ret) / (1 + day_pct)
                )
                portfolio_performance.append({
                    'date'            : day_ts.strftime('%Y-%m-%d'),
                    'log_return'      : np.log(1 + day_pct),
                    'cumulative_value': portfolio_value,
                })

            # ── Out-of-sample model evaluation ───────────────────────────────
            hold_prices = period_prices.loc[actual_trade_date:]
            if len(hold_prices) >= 2:
                actual_ret = (hold_prices.iloc[-1] / hold_prices.iloc[0] - 1).dropna()
                common_idx = (
                    pred_series.index.intersection(actual_ret.index)
                    if pred_series is not None else pd.Index([])
                )

                if len(common_idx) >= 2:
                    y_pred      = pred_series.loc[common_idx].values
                    y_true      = actual_ret.loc[common_idx].values
                    y_true_rank = pd.Series(y_true).rank(pct=True).values

                    pred_dir = (y_pred >= 0.5).astype(int)
                    true_dir = (y_true >= 0).astype(int)
                    tp = np.sum((pred_dir == 1) & (true_dir == 1))
                    tn = np.sum((pred_dir == 0) & (true_dir == 0))
                    fp = np.sum((pred_dir == 1) & (true_dir == 0))
                    fn = np.sum((pred_dir == 0) & (true_dir == 1))
                    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                    geo  = (np.sqrt(sens * spec)
                            if not (np.isnan(sens) or np.isnan(spec)) else np.nan)

                    ss_res   = np.sum((y_true_rank - y_pred) ** 2)
                    ss_tot   = np.sum((y_true_rank - y_true_rank.mean()) ** 2)
                    r2_rank  = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

                    # Raw R²oos vs zero benchmark (comparable to paper)
                    ss_res_raw = np.sum((y_true - y_pred) ** 2)
                    ss_tot_raw = np.sum(y_true ** 2)          # benchmark = 0
                    r2_raw     = 1 - ss_res_raw / ss_tot_raw if ss_tot_raw != 0 else np.nan

                    spearman = pd.Series(y_pred).corr(
                        pd.Series(y_true_rank), method='spearman'
                    )

                    model_stats.append({
                        'rebalance_date'      : actual_trade_date.strftime('%Y-%m-%d'),
                        'invest_year'         : invest_year,
                        'select_year'         : select_year,
                        'n_stocks'            : len(common_idx),
                        'RMSE'                : float(np.sqrt(np.mean((y_pred - y_true_rank)**2))),
                        'MSE'                 : float(np.mean((y_pred - y_true_rank)**2)),
                        'MAE'                 : float(np.mean(np.abs(y_pred - y_true_rank))),
                        'R2_rank'             : r2_rank,
                        'R2_raw_vs_zero'      : r2_raw,
                        'Spearman'            : spearman,
                        'Directional_Accuracy': float(np.mean(pred_dir == true_dir)),
                        'Geometric_Score'     : geo,
                        'best_depth'          : best_params.get('max_depth',        np.nan),
                        'best_n_estimators'   : best_params.get('n_estimators',     np.nan),
                        'best_min_leaf'       : best_params.get('min_samples_leaf', np.nan),
                        'val_R2_selected'     : best_val_r2,
                        'turnover'            : turnover,
                        'tc_drag_bps'         : turnover * TC_BPS,
                    })

        last_end_weights = active_weights
        current_date     = next_rebalance

    # ── Export ────────────────────────────────────────────────────────────────
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"portfolio_rf_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_rf_{label}_details.csv", index=False
    )
    pd.DataFrame(model_stats).to_csv(
        output_dir / f"portfolio_rf_{label}_statistics.csv", index=False
    )
    print(f"\n  [{label}] Done — saved to {output_dir}")