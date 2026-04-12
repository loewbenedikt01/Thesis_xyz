import gc
import sys
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import universe

# --- CONFIGURATION (Global Controls) ---
TRAIN_MONTHS     = 60           # 5 Year Training
VAL_MONTHS       = 12           # 1 Year Validation
MIN_COMPLETENESS = 0.50         # 50% data history required
WEIGHT_MAX       = 0.10         # Max 10% per stock
WEIGHT_MIN       = 0.01         # Min 1% per stock
N_RUNS           = 1            # How many independent runs per frequency
RANDOM_SEED      = 41           # Base seed — each run uses RANDOM_SEED + run
SEQ_LEN          = 12           # Monthly snapshots per LSTM sequence

# Grid search hyperparameters
NODES            = [25]
DROPOUTS         = [0.1, 0.3]
GRID_EPOCHS      = 15           # Max epochs during grid search
GRID_PATIENCE    = 3            # Early stopping patience during grid search

# Final model training
FULL_EPOCHS      = 50          # Max epochs for final model
PATIENCE         = 7           # Early stopping patience for final model
BATCH_SIZE       = 64
LEARNING_RATE    = 0.01

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
output_dir  = DATA_PATH / "results" / "data" / "lstm"
output_dir.mkdir(parents=True, exist_ok=True)

# Load Data
all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)


# ---------------------------------------------------------------------------
# Feature Engineering  (identical to Random Forest)
# ---------------------------------------------------------------------------
def create_features(prices_df):
    """Generates technical and risk features; returns rank-normalised DataFrame."""
    if len(prices_df) < 260:
        return pd.DataFrame()

    features = pd.DataFrame(index=prices_df.columns)
    returns  = prices_df.pct_change()

    # 1. Momentum
    features['mom_1w']   = prices_df.pct_change(5).iloc[-1]
    features['mom_1m']   = prices_df.pct_change(21).iloc[-1]
    features['mom_3m']   = prices_df.pct_change(63).iloc[-1]
    features['mom_6m']   = prices_df.pct_change(126).iloc[-1]
    features['mom_12_1'] = prices_df.iloc[-22] / prices_df.iloc[-253] - 1  # skip-month

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

    # 7. RSI — Wilder's EMA (14-period)
    delta    = returns.tail(252)
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1]
    rs = avg_gain / (avg_loss + 1e-9)
    features['rsi'] = 100 - (100 / (1 + rs))

    # 8. Sharpe-like score
    features['sharpe_6m'] = features['mom_6m'] / (features['vol_6m'] + 1e-9)

    # Cross-sectional rank normalisation
    features = features.rank(pct=True)
    return features.dropna()


# ---------------------------------------------------------------------------
# Weight Allocation  (identical to Random Forest)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# LSTM Model Builder
# ---------------------------------------------------------------------------
def build_lstm(node, dropout_rate, seq_len, n_features):
    model = Sequential([
        LSTM(node, input_shape=(seq_len, n_features), return_sequences=False),
        Dropout(dropout_rate),
        Dense(max(node // 2, 1), activation='relu'),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae'],
    )
    return model


# ---------------------------------------------------------------------------
# Main Backtest Loop
# ---------------------------------------------------------------------------
for run in range(1, N_RUNS + 1):
    run_seed = RANDOM_SEED + run
    tf.random.set_seed(run_seed)
    np.random.seed(run_seed)
    print(f"\n=== Run {run}/{N_RUNS} (seed={run_seed}) ===")

    for label, (offset, horizon) in FREQUENCIES.items():
        print(f"  Processing LSTM ({label}) | horizon={horizon}d | min coverage: {MIN_COMPLETENESS:.0%}")

        current_date        = start_invest
        portfolio_value     = 1.0
        last_end_weights    = pd.Series(dtype=float)
        portfolio_performance = []
        rebalance_details   = []
        model_stats         = []

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

                lb_start  = actual_trade_date - pd.DateOffset(months=TRAIN_MONTHS + VAL_MONTHS + 12)
                lb_end    = actual_trade_date - pd.Timedelta(days=1)

                available   = [t for t in year_tickers if t in all_prices.columns]
                hist_prices = all_prices.loc[lb_start:lb_end, available]

                if not hist_prices.empty:
                    coverage      = hist_prices.notnull().sum() / len(hist_prices)
                    valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

                    X_train_seqs, y_train_list = [], []
                    X_val_seqs,   y_val_list   = [], []

                    if len(valid_tickers) >= 2:
                        train_prices  = hist_prices[valid_tickers].ffill()
                        val_start_date = actual_trade_date - pd.DateOffset(months=VAL_MONTHS)
                        val_split      = int(np.searchsorted(train_prices.index, val_start_date))

                        # ── Build monthly snapshots ──────────────────────────────
                        # Sample every ~21 trading days to get monthly feature snapshots
                        train_snapshots = []
                        for i in range(260, val_split, 21):
                            feat = create_features(train_prices.iloc[:i])
                            if feat.empty:
                                continue
                            train_snapshots.append((i, feat))

                        val_snapshots = []
                        for i in range(val_split, len(train_prices), 21):
                            feat = create_features(train_prices.iloc[:i])
                            if feat.empty:
                                continue
                            val_snapshots.append((i, feat))

                        # ── Build LSTM sequences with matched forward returns ────
                        for snap_list, target_X, target_y in [
                            (train_snapshots, X_train_seqs, y_train_list),
                            (val_snapshots,   X_val_seqs,   y_val_list),
                        ]:
                            for end_idx in range(SEQ_LEN - 1, len(snap_list) - 1):
                                seq_snaps = snap_list[end_idx - SEQ_LEN + 1 : end_idx + 1]
                                next_snap_idx = snap_list[end_idx + 1][0]

                                # Ensure forward target index is within bounds
                                if next_snap_idx + horizon >= len(train_prices):
                                    continue

                                common = set(seq_snaps[0][1].index)
                                for _, fd in seq_snaps[1:]:
                                    common &= set(fd.index)
                                common = sorted(common)
                                if len(common) < 2:
                                    continue

                                # (n_stocks, seq_len, n_features)
                                seq_arr = np.stack(
                                    [fd.loc[common].values for _, fd in seq_snaps],
                                    axis=1
                                )

                                fwd_ret = (
                                    train_prices.iloc[next_snap_idx + horizon]
                                    / train_prices.iloc[next_snap_idx] - 1
                                ).loc[common].dropna()

                                common2 = [t for t in common if t in fwd_ret.index]
                                if len(common2) < 2:
                                    continue

                                idx_map   = {t: i for i, t in enumerate(common)}
                                kept_idx  = [idx_map[t] for t in common2]
                                seq_arr   = seq_arr[kept_idx]            # (n_kept, seq_len, n_feat)
                                y_ranked  = fwd_ret.loc[common2].rank(pct=True).values

                                target_X.append(seq_arr)
                                target_y.append(y_ranked)

                        if X_train_seqs:
                            X_tr = np.concatenate(X_train_seqs, axis=0)  # (N_tr, seq_len, n_feat)
                            y_tr = np.concatenate(y_train_list, axis=0)
                            n_features = X_tr.shape[2]

                            X_vl, y_vl = None, None
                            has_val = bool(X_val_seqs)
                            if has_val:
                                X_vl = np.concatenate(X_val_seqs, axis=0)
                                y_vl = np.concatenate(y_val_list, axis=0)

                            # ── Grid Search ──────────────────────────────────────
                            tf.keras.backend.clear_session()
                            best_val_loss  = np.inf
                            best_nodes     = NODES[0]
                            best_dropout   = DROPOUTS[0]

                            for g_node in NODES:
                                for g_drop in DROPOUTS:
                                    g_model = build_lstm(g_node, g_drop, SEQ_LEN, n_features)
                                    g_es    = EarlyStopping(
                                        monitor='val_loss' if has_val else 'loss',
                                        patience=GRID_PATIENCE,
                                        restore_best_weights=True,
                                    )
                                    val_data = (X_vl, y_vl) if has_val else None
                                    g_hist   = g_model.fit(
                                        X_tr, y_tr,
                                        validation_data=val_data,
                                        epochs=GRID_EPOCHS,
                                        batch_size=BATCH_SIZE,
                                        callbacks=[g_es],
                                        verbose=0,
                                    )
                                    if has_val:
                                        g_loss = min(g_hist.history.get('val_loss', [np.inf]))
                                    else:
                                        g_loss = min(g_hist.history.get('loss', [np.inf]))

                                    if g_loss < best_val_loss:
                                        best_val_loss = g_loss
                                        best_nodes    = g_node
                                        best_dropout  = g_drop

                                    del g_model
                                    gc.collect()

                            print(f"  [{label}] {current_date.date()} — Best config: nodes={best_nodes}, dropout={best_dropout}, val_loss={best_val_loss:.5f}")

                            # ── Final Model (train + val combined) ───────────────
                            tf.keras.backend.clear_session()
                            if has_val:
                                X_full = np.concatenate([X_tr, X_vl], axis=0)
                                y_full = np.concatenate([y_tr, y_vl], axis=0)
                            else:
                                X_full = X_tr
                                y_full = y_tr

                            final_model = build_lstm(best_nodes, best_dropout, SEQ_LEN, n_features)
                            final_es    = EarlyStopping(
                                monitor='loss',
                                patience=PATIENCE,
                                restore_best_weights=True,
                            )
                            final_model.fit(
                                X_full, y_full,
                                epochs=FULL_EPOCHS,
                                batch_size=BATCH_SIZE,
                                callbacks=[final_es],
                                verbose=0,
                            )

                            # ── Prediction at current date ───────────────────────
                            # Build SEQ_LEN snapshots ending at the most recent data
                            pred_snapshots = []
                            for i in range(
                                max(260, len(train_prices) - SEQ_LEN * 21 - 21),
                                len(train_prices),
                                21
                            ):
                                feat = create_features(train_prices.iloc[:i])
                                if feat.empty:
                                    continue
                                pred_snapshots.append(feat)

                            # Always append the most recent snapshot (current state)
                            current_feat = create_features(train_prices)
                            if not current_feat.empty:
                                pred_snapshots.append(current_feat)

                            pred_snapshots = pred_snapshots[-SEQ_LEN:]

                            if len(pred_snapshots) == SEQ_LEN:
                                common_pred = set(pred_snapshots[0].index)
                                for pf in pred_snapshots[1:]:
                                    common_pred &= set(pf.index)
                                common_pred = sorted(common_pred)

                                if len(common_pred) >= 2:
                                    X_pred = np.stack(
                                        [pf.loc[common_pred].values for pf in pred_snapshots],
                                        axis=1
                                    )  # (n_stocks, seq_len, n_feat)

                                    preds      = final_model.predict(X_pred, verbose=0).flatten()
                                    pred_series = pd.Series(preds, index=common_pred).sort_values(ascending=False)
                                    target_weights = allocate_weights(pred_series, WEIGHT_MIN, WEIGHT_MAX)

                            # ── Cleanup ──────────────────────────────────────────
                            del final_model, X_tr, y_tr, X_full, y_full
                            if has_val:
                                del X_vl, y_vl
                            del train_snapshots, val_snapshots, X_train_seqs, y_train_list
                            del X_val_seqs, y_val_list
                            gc.collect()

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

            # ── Logging & Daily Drift ────────────────────────────────────────
            turnover = (target_weights.sub(last_end_weights, fill_value=0)).abs().sum()
            for ticker, w in target_weights.items():
                rebalance_details.append({
                    'rebalance_date'  : actual_trade_date.strftime('%Y-%m-%d'),
                    'ticker'          : ticker,
                    'assigned_weight' : w,
                    'turnover'        : turnover if ticker == target_weights.index[0] else 0,
                })

            active_weights = target_weights.copy()
            period_prices  = all_prices.loc[
                actual_trade_date - pd.Timedelta(days=5) : next_rebalance - pd.Timedelta(days=1),
                active_weights.index,
            ]

            if not period_prices.empty:
                daily_rets = period_prices.pct_change().dropna(how='all')
                daily_rets = daily_rets[daily_rets.index >= actual_trade_date]
                for day_timestamp, day_ret in daily_rets.iterrows():
                    day_ret = day_ret.fillna(0)
                    day_pct_return  = (active_weights * day_ret).sum()
                    portfolio_value *= (1 + day_pct_return)
                    active_weights   = (active_weights * (1 + day_ret)) / (1 + day_pct_return)

                    portfolio_performance.append({
                        'date'             : day_timestamp.strftime('%Y-%m-%d'),
                        'log_return'       : np.log(1 + day_pct_return),
                        'cumulative_value' : portfolio_value,
                    })

                # ── Model evaluation: predicted vs actual realized return ────
                hold_prices = period_prices.loc[actual_trade_date:]
                if len(hold_prices) >= 2 and pred_series is not None:
                    first_price = hold_prices.iloc[0]
                    last_price  = hold_prices.iloc[-1]
                    actual_ret  = (last_price / first_price - 1).dropna()
                    common_idx  = pred_series.index.intersection(actual_ret.index)

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
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                        geo_score   = (np.sqrt(sensitivity * specificity)
                                       if not (np.isnan(sensitivity) or np.isnan(specificity))
                                       else np.nan)
                        dir_acc     = np.mean(pred_dir == true_dir)

                        mse      = np.mean((y_pred - y_true_rank) ** 2)
                        rmse     = np.sqrt(mse)
                        mae      = np.mean(np.abs(y_pred - y_true_rank))
                        ss_res   = np.sum((y_true_rank - y_pred) ** 2)
                        ss_tot   = np.sum((y_true_rank - y_true_rank.mean()) ** 2)
                        r2       = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                        spearman = pd.Series(y_pred).corr(pd.Series(y_true_rank), method='spearman')

                        model_stats.append({
                            'rebalance_date'      : actual_trade_date.strftime('%Y-%m-%d'),
                            'n_stocks'            : len(common_idx),
                            'best_nodes'          : best_nodes,
                            'best_dropout'        : best_dropout,
                            'best_val_loss'       : best_val_loss,
                            'RMSE'                : rmse,
                            'MSE'                 : mse,
                            'MAE'                 : mae,
                            'R_squared'           : r2,
                            'Spearman'            : spearman,
                            'Directional_Accuracy': dir_acc,
                            'Geometric_Score'     : geo_score,
                        })

            last_end_weights = active_weights
            current_date     = next_rebalance

        pd.DataFrame(portfolio_performance).to_csv(
            output_dir / f"portfolio_lstm_{label}_{run}.csv", index=False
        )
        pd.DataFrame(rebalance_details).to_csv(
            output_dir / f"portfolio_lstm_{label}_{run}_details.csv", index=False
        )
        pd.DataFrame(model_stats).to_csv(
            output_dir / f"portfolio_lstm_{label}_{run}_statistics.csv", index=False
        )
        print(f"  [{label}] Run {run} exported.")

    # ── Full memory reset after each run ────────────────────────────────────
    tf.keras.backend.clear_session()
    gc.collect()
    print(f"=== Run {run} complete — session cleared ===")
