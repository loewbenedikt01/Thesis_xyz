"""
XGBoost Portfolio Model — v2
==============================
Key improvements vs original:
  1.  best_params / best_val_r2 initialised at top of every loop iteration
      → no NameError on fallback path
  2.  Forward-return target uses date-based lookup (not iloc)
      → immune to price gaps and halted stocks
  3.  Transaction costs: optional TC_BPS param (0 = disabled)
  4.  Validation window = 24 months (matches RF v3)
  5.  ffill capped at 5 days
  6.  Universe logic: tickers[year-1] used to invest in year `year`
  7.  GridSearchCV replaced with manual holdout-based tuning
      → GridSearchCV uses k-fold CV which breaks temporal ordering;
         we instead evaluate every combo on the true holdout val set,
         exactly mirroring what RF v3 does
  8.  Seed ensemble: 10 seeds averaged into one prediction per period
      (was: 10 separate independent runs → now: one averaged output)
  9.  Huber loss objective added ('reg:pseudohubererror')
      → better than MSE for fat-tailed financial returns (from paper)
 10.  subsample + colsample_bytree added to grid
      → XGBoost-specific regularisation the paper's GBRT equivalent uses
 11.  early_stopping_rounds added during final fit on train+val
      → prevents overfitting the combined dataset
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from xgboost import XGBRegressor
from itertools import product

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import universe

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_MONTHS_MONTHLY     = 9       # training lookback in months
VAL_MONTHS_MONTHLY       = 3       # holdout validation window in months
TRAIN_MONTHS_OTHERS     = 36       # training lookback in months
VAL_MONTHS_OTHERS       = 24       # holdout validation window in months
MIN_COMPLETENESS = 0.50     # min fraction of non-NaN rows per ticker
WEIGHT_MAX       = 0.10     # max portfolio weight per stock
WEIGHT_MIN       = 0.01     # min portfolio weight per stock
N_SEEDS          = 10       # seeds averaged into one ensemble prediction
BASE_SEED        = 41       # seeds: 41, 42, …, 50

# Transaction costs — set TC_BPS = 0 to disable
# Applied as: portfolio_value *= (1 - turnover * TC_BPS / 10_000)
TC_BPS = 0

# Early stopping: stop boosting if val loss doesn't improve for N rounds
EARLY_STOPPING_ROUNDS = 10

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER GRID
# 4 × 4 × 3 × 3 × 2 × 2 = 576 combos — evaluated on TRUE holdout val set
# (not k-fold CV, which would break temporal ordering)
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    'learning_rate'     : [0.05, 0.10, 0.3],          # step shrinkage (ν in paper)
    'max_depth'         : [1, 2, 5],                  # shallow = regularised
    'min_child_weight'  : [1, 5, 10],
    'gamma'             : [0, 0.5],
    'n_estimators'      : [100, 135, 200],            # boosting rounds (B in paper)
    'reg_lambda'        : [1, 3, 5],                  # L2 on leaf weights
    'subsample'         : [0.8, 1.0],                 # row sampling per tree
    'colsample_bytree'  : [0.8, 1.0],                 # feature sampling per tree
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
#DATA_PATH   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
script_dir = Path(__file__).resolve().parent
DATA_PATH   = script_dir.parent
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / "xgboost"
output_dir.mkdir(parents=True, exist_ok=True)

all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)

SEEDS = [BASE_SEED + i for i in range(N_SEEDS)]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (identical to RF v3)
# ─────────────────────────────────────────────────────────────────────────────
def create_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    if len(prices_df) < 260:
        return pd.DataFrame()

    features = pd.DataFrame(index=prices_df.columns)
    returns  = prices_df.pct_change()

    features['mom_1w']   = prices_df.pct_change(5).iloc[-1]
    features['mom_1m']   = prices_df.pct_change(21).iloc[-1]
    features['mom_3m']   = prices_df.pct_change(63).iloc[-1]
    features['mom_6m']   = prices_df.pct_change(126).iloc[-1]
    features['mom_12_1'] = prices_df.iloc[-22] / prices_df.iloc[-253] - 1

    features['ma_ratio']      = prices_df.tail(50).mean() / prices_df.tail(200).mean()
    features['dist_52w_high'] = prices_df.iloc[-1] / prices_df.tail(252).max()

    w1y = prices_df.tail(252)
    features['max_dd_12m'] = ((w1y - w1y.cummax()) / w1y.cummax()).min()

    vol_1m  = returns.tail(21).std()  * np.sqrt(252)
    vol_6m  = returns.tail(126).std() * np.sqrt(252)
    vol_12m = returns.tail(252).std() * np.sqrt(252)
    features['vol_1m']    = vol_1m
    features['vol_6m']    = vol_6m
    features['vol_12m']   = vol_12m
    features['vol_ratio'] = vol_1m / (vol_12m + 1e-9)

    p20 = prices_df.tail(20)
    features['bb_position'] = (
        (prices_df.iloc[-1] - p20.mean()) / (2 * p20.std() + 1e-9)
    )

    delta    = returns.tail(252)
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14,
                                       adjust=False).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14,
                                          adjust=False).mean().iloc[-1]
    rs = avg_gain / (avg_loss + 1e-9)
    features['rsi'] = 100 - (100 / (1 + rs))

    features['sharpe_6m'] = features['mom_6m'] / (features['vol_6m'] + 1e-9)

    features = features.rank(pct=True)
    return features.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: DATE-BASED FORWARD RETURN  (identical to RF v3)
# ─────────────────────────────────────────────────────────────────────────────
def forward_return(prices_df: pd.DataFrame, anchor_iloc: int,
                   horizon_days: int) -> pd.Series:
    anchor_date = prices_df.index[anchor_iloc]
    target_date = anchor_date + pd.Timedelta(days=int(horizon_days * 1.5))
    future_iloc = prices_df.index.searchsorted(target_date)
    if future_iloc >= len(prices_df):
        return pd.Series(dtype=float)
    return (prices_df.iloc[future_iloc] / prices_df.iloc[anchor_iloc] - 1).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT ALLOCATION  (identical to RF v3)
# ─────────────────────────────────────────────────────────────────────────────
def allocate_weights(predictions: pd.Series, w_min: float,
                     w_max: float) -> pd.Series:
    scores  = predictions - predictions.min() + 1e-4
    weights = scores / scores.sum()
    for _ in range(20):
        weights = weights.clip(w_min, w_max)
        weights = weights / weights.sum()
        if weights.max() <= w_max + 1e-5 and weights.min() >= w_min - 1e-5:
            break
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# GRID HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_param_combinations(grid: dict) -> list:
    keys, values = list(grid.keys()), list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ─────────────────────────────────────────────────────────────────────────────
# TUNE + ENSEMBLE PREDICT  (XGBoost version of RF v3's tune_and_predict)
# ─────────────────────────────────────────────────────────────────────────────
def tune_and_predict(X_train_list, y_train_list,
                     X_val_list,   y_val_list,
                     current_feat: pd.DataFrame,
                     param_grid:   dict,
                     seeds:        list,
                     early_stopping_rounds: int = 10):
    """
    1. Evaluate all param combos on the TRUE holdout val set (seed-averaged).
       This preserves temporal ordering — no k-fold CV across time.
    2. Retrain best combo on train + val with early stopping.
    3. Return seed-averaged predictions on current_feat.
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

    # ── Step 1: grid search on holdout val set ────────────────────────────────
    for params in param_combos:
        seed_preds = []
        for seed in seeds:
            m = XGBRegressor(
                objective         = 'reg:pseudohubererror',  # FIX 9: Huber loss
                n_jobs            = -1,
                random_state      = seed,
                verbosity         = 0,
                learning_rate     = params['learning_rate'],
                max_depth         = params['max_depth'],
                min_child_weight  = params['min_child_weight'],
                gamma             = params['gamma'],
                n_estimators      = params['n_estimators'],
                reg_lambda        = params['reg_lambda'],
                subsample         = params['subsample'],
                colsample_bytree  = params['colsample_bytree'],
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

    # ── Step 2: retrain best config on train+val, with early stopping ─────────
    # Use last 20% of training data as internal early-stopping monitor
    # (this is separate from, and inside, the already-held-out test period)
    n_es_val  = max(1, int(len(X_all) * 0.20))
    X_es_tr   = X_all.iloc[:-n_es_val]
    y_es_tr   = y_all.iloc[:-n_es_val]
    X_es_val  = X_all.iloc[-n_es_val:]
    y_es_val  = y_all.iloc[-n_es_val:]

    final_preds = []
    fi_accum    = []
    for seed in seeds:
        m = XGBRegressor(
            objective             = 'reg:pseudohubererror',
            n_jobs                = -1,
            random_state          = seed,
            verbosity             = 0,
            learning_rate         = best_params['learning_rate'],
            max_depth             = best_params['max_depth'],
            min_child_weight      = best_params['min_child_weight'],
            gamma                 = best_params['gamma'],
            n_estimators          = best_params['n_estimators'],
            reg_lambda            = best_params['reg_lambda'],
            subsample             = best_params['subsample'],
            colsample_bytree      = best_params['colsample_bytree'],
            early_stopping_rounds = early_stopping_rounds,
        )
        m.fit(
            X_es_tr, y_es_tr,
            eval_set = [(X_es_val, y_es_val)],
            verbose  = False,
        )
        final_preds.append(m.predict(current_feat))
        fi_accum.append(m.feature_importances_)

    avg_final   = np.mean(final_preds, axis=0)
    pred_series = pd.Series(
        avg_final, index=current_feat.index
    ).sort_values(ascending=False)

    mean_fi    = np.mean(fi_accum, axis=0)
    feat_names = list(current_feat.columns)

    return pred_series, best_params, best_val_r2, mean_fi, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────
n_combos = len(build_param_combinations(PARAM_GRID))

for label, (offset, horizon) in FREQUENCIES.items():
    if label == 'Monthly':
        TRAIN_MONTHS = TRAIN_MONTHS_MONTHLY
        VAL_MONTHS   = VAL_MONTHS_MONTHLY
    else:
        TRAIN_MONTHS = TRAIN_MONTHS_OTHERS
        VAL_MONTHS   = VAL_MONTHS_OTHERS

    print(
        f"\n=== XGBoost [{label}] | horizon={horizon}d | "
        f"train={TRAIN_MONTHS}mo val={VAL_MONTHS}mo | "
        f"{n_combos} param combos | {len(SEEDS)} seeds | "
        f"TC={TC_BPS}bps ==="
    )

    current_date              = start_invest
    portfolio_value           = 1.0
    last_end_weights          = pd.Series(dtype=float)
    portfolio_performance     = []
    rebalance_details         = []
    model_stats               = []
    feature_importance_records = []

    while current_date < end_invest:
        next_rebalance = current_date + offset

        valid_days = all_prices.index[all_prices.index >= current_date]
        if valid_days.empty:
            break
        actual_trade_date = valid_days[0]

        # FIX 1: always initialise before any conditional block
        target_weights = None
        pred_series    = None
        best_params    = {}
        best_val_r2    = np.nan

        # FIX 6: universe logic — invest_year uses tickers[invest_year - 1]
        invest_year = current_date.year
        select_year = invest_year - 1

        if select_year in universe.tickers:
            year_tickers = [t[0] for t in universe.tickers[select_year]]

            lb_start = actual_trade_date - pd.DateOffset(
                months=TRAIN_MONTHS + VAL_MONTHS + 12
            )
            lb_end   = actual_trade_date - pd.Timedelta(days=1)

            available   = [t for t in year_tickers if t in all_prices.columns]
            hist_prices = all_prices.loc[lb_start:lb_end, available]

            if not hist_prices.empty:
                coverage      = hist_prices.notnull().sum() / len(hist_prices)
                valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

                if len(valid_tickers) >= 2:
                    # FIX 5: cap ffill at 5 days
                    train_prices   = hist_prices[valid_tickers].ffill(limit=5)
                    val_start_date   = actual_trade_date - pd.DateOffset(months=VAL_MONTHS)
                    train_start_date = actual_trade_date - pd.DateOffset(months=TRAIN_MONTHS + VAL_MONTHS)
                    val_split        = int(np.searchsorted(train_prices.index, val_start_date))
                    train_start_idx  = max(260, int(np.searchsorted(train_prices.index, train_start_date)))

                    X_train_list, y_train_list = [], []
                    X_val_list,   y_val_list   = [], []

                    # Training samples — rolling window: only [train_start_idx, val_split)
                    for i in range(train_start_idx, val_split - horizon, 21):
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

                    # Validation samples (true holdout — never used in grid search)
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
                            pred_series, best_params, best_val_r2, fi_values, fi_names = tune_and_predict(
                                X_train_list, y_train_list,
                                X_val_list,   y_val_list,
                                current_feat, PARAM_GRID, SEEDS,
                                EARLY_STOPPING_ROUNDS,
                            )
                            fi_row = {'rebalance_date': actual_trade_date.strftime('%Y-%m-%d')}
                            fi_row.update(dict(zip(fi_names, fi_values)))
                            feature_importance_records.append(fi_row)
                            target_weights = allocate_weights(
                                pred_series, WEIGHT_MIN, WEIGHT_MAX
                            )
                            print(
                                f"  [{label}] {current_date.date()} | "
                                f"universe={select_year} ({len(valid_tickers)} stocks) | "
                                f"lr={best_params.get('learning_rate')} "
                                f"depth={best_params.get('max_depth')} "
                                f"rounds={best_params.get('n_estimators')} "
                                f"lambda={best_params.get('reg_lambda')} "
                                f"sub={best_params.get('subsample')} "
                                f"col={best_params.get('colsample_bytree')} | "
                                f"val_R2={best_val_r2:.4f}"
                            )

        # ── Fallback ──────────────────────────────────────────────────────────
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

        # ── FIX 3: transaction cost ───────────────────────────────────────────
        turnover = target_weights.sub(last_end_weights, fill_value=0).abs().sum()
        if TC_BPS > 0:
            portfolio_value *= (1 - turnover * TC_BPS / 10_000)

        # ── Rebalance logging ─────────────────────────────────────────────────
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date'     : actual_trade_date.strftime('%Y-%m-%d'),
                'invest_year'        : invest_year,
                'select_year'        : select_year,
                'ticker'             : ticker,
                'assigned_weight'    : w,
                'turnover'           : round(turnover, 6) if ticker == target_weights.index[0] else 0,
                'tc_drag_bps'        : round(turnover * TC_BPS, 4) if ticker == target_weights.index[0] else 0,
                'best_learning_rate' : best_params.get('learning_rate',     np.nan),
                'best_max_depth'     : best_params.get('max_depth',         np.nan),
                'best_n_estimators'  : best_params.get('n_estimators',      np.nan),
                'best_reg_lambda'    : best_params.get('reg_lambda',        np.nan),
                'best_subsample'     : best_params.get('subsample',         np.nan),
                'best_colsample'     : best_params.get('colsample_bytree',  np.nan),
                'val_R2_selected'    : round(best_val_r2, 6) if not np.isnan(best_val_r2) else np.nan,
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

                    ss_res       = np.sum((y_true_rank - y_pred) ** 2)
                    ss_tot       = np.sum((y_true_rank - y_true_rank.mean()) ** 2)
                    r2_rank      = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

                    # Raw R²oos vs zero (paper-comparable metric)
                    ss_res_raw   = np.sum((y_true - y_pred) ** 2)
                    ss_tot_raw   = np.sum(y_true ** 2)
                    r2_raw       = 1 - ss_res_raw / ss_tot_raw if ss_tot_raw != 0 else np.nan

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
                        'best_learning_rate'  : best_params.get('learning_rate',    np.nan),
                        'best_max_depth'      : best_params.get('max_depth',        np.nan),
                        'best_n_estimators'   : best_params.get('n_estimators',     np.nan),
                        'best_reg_lambda'     : best_params.get('reg_lambda',       np.nan),
                        'best_subsample'      : best_params.get('subsample',        np.nan),
                        'best_colsample'      : best_params.get('colsample_bytree', np.nan),
                        'val_R2_selected'     : best_val_r2,
                        'turnover'            : turnover,
                        'tc_drag_bps'         : turnover * TC_BPS,
                    })

        last_end_weights = active_weights
        current_date     = next_rebalance

    # ── Export ────────────────────────────────────────────────────────────────
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"portfolio_xgb_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_xgb_{label}_details.csv", index=False
    )
    pd.DataFrame(model_stats).to_csv(
        output_dir / f"portfolio_xgb_{label}_statistics.csv", index=False
    )
    pd.DataFrame(feature_importance_records).to_csv(
        output_dir / f"portfolio_xgb_{label}_feature_importance.csv", index=False
    )
    print(f"\n  [{label}] Done — saved to {output_dir}")