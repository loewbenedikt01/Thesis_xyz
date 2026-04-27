"""
LSTM Portfolio Model — v2
==========================
Fixes vs original:
  1.  best_nodes/best_dropout/best_val_loss initialised at top of every iteration
  2.  forward_return() uses date-based lookup (not iloc position)
  3.  ffill capped at 5 days
  4.  VAL_MONTHS = 24
  5.  Universe logic: select_year = invest_year - 1, both logged to CSV
  6.  Grid expanded: 3 node sizes × 3 dropouts × 2 LRs = 18 combos
  7.  Single run design: RUN_NUMBER config added to seed; N_RUNS always = 1
  8.  Huber loss (tf.keras.losses.Huber) replacing MSE
  9.  L2 weight regularisation on LSTM and Dense layers
 10.  Batch normalisation between LSTM and Dense
 11.  ReduceLROnPlateau learning rate scheduler
 12.  R2_raw_vs_zero metric added (paper-comparable)
 13.  invest_year / select_year logged to all CSVs
 14.  TC_BPS transaction cost parameter added

Architecture (matches paper's NN3 spirit):
  LSTM(units) → BatchNorm → Dropout → Dense(units//2) → Dense(units//4) → Dense(1)
  Three trainable hidden transformations, progressively narrowing (geometric pyramid rule)
"""

import gc
import sys
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import universe

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION Fischer and Krauss 2018 Methodology
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_MONTHS     = 36       # training lookback in months as fischer and krauss
VAL_MONTHS       = 12       # put to 1 year, as fischer and frauss
MIN_COMPLETENESS = 0.50     # min fraction of non-NaN rows per ticker
WEIGHT_MAX       = 0.10     # max portfolio weight per stock
WEIGHT_MIN       = 0.01     # min portfolio weight per stock
SEQ_LEN          = 12       # monthly snapshots per LSTM sequence

# FIX 7: Single-run seed design
# Change RUN_NUMBER each time you run the script (1, 2, 3, …)
# The effective seed = BASE_SEED + RUN_NUMBER
# This gives reproducible but distinct results per run without looping
BASE_SEED   = 41
RUN_NUMBER  = 4            # <── change this per execution (1, 2, 3, ...)
RANDOM_SEED = BASE_SEED + RUN_NUMBER

# FIX 14: Transaction costs — set TC_BPS = 0 to disable
# Applied as: portfolio_value *= (1 - turnover * TC_BPS / 10_000)
TC_BPS = 0

# L2 regularisation strength (applied to LSTM and Dense kernel weights)
L2_LAMBDA = 1e-4

# ─────────────────────────────────────────────────────────────────────────────
# FIX 6: Expanded hyperparameter grid4``
# 3 node sizes × 3 dropout rates × 2 learning rates = 18 combinations
# ─────────────────────────────────────────────────────────────────────────────
GRID_NODES    = [16, 32]            # LSTM hidden units
GRID_LR       = [0.001, 0.01]       # Adam initial learning rate
BATCH_SIZE    = [32, 64]            # smaller batch = better gradient estimates for small N


RECURRENT_DROPOUT = 0.1
GRID_DROPOUTS = 0.2                 # dropout rate after LSTM
GRID_EPOCHS   = 10                  # max epochs per grid combo
GRID_PATIENCE = 3                   # early stopping patience during grid search
FULL_EPOCHS   = 50                  # max epochs for final model
PATIENCE      = 8                   # early stopping patience for final model

FREQUENCIES = {
    #'Yearly':      (pd.DateOffset(years=1),  252),
    'Semi-Annual': (pd.DateOffset(months=6), 126),
    #'Quarterly':   (pd.DateOffset(months=3),  63),
    #'Monthly':     (pd.DateOffset(months=1),  21),
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
output_dir  = DATA_PATH / "results" / "data" / "lstm"
output_dir.mkdir(parents=True, exist_ok=True)

all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)

# Set global seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (identical to RF v3 / XGB v2)
# ─────────────────────────────────────────────────────────────────────────────
def create_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    if len(prices_df) < 260:
        return pd.DataFrame()

    features = pd.DataFrame(index=prices_df.columns)
    returns  = prices_df.pct_change()

    features['mom_1w']        = prices_df.pct_change(5).iloc[-1]
    features['mom_1m']        = prices_df.pct_change(21).iloc[-1]
    features['mom_3m']        = prices_df.pct_change(63).iloc[-1]
    features['mom_6m']        = prices_df.pct_change(126).iloc[-1]
    features['mom_12_1']      = prices_df.iloc[-22] / prices_df.iloc[-253] - 1
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
    features['rsi']       = 100 - (100 / (1 + rs))
    features['sharpe_6m'] = features['mom_6m'] / (features['vol_6m'] + 1e-9)

    features = features.rank(pct=True)
    return features.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: DATE-BASED FORWARD RETURN
# ─────────────────────────────────────────────────────────────────────────────
def forward_return(prices_df: pd.DataFrame, anchor_iloc: int,
                   horizon_days: int) -> pd.Series:
    """
    Return from prices_df.index[anchor_iloc] to the first available
    trading day >= anchor_date + horizon_days * 1.5 calendar days.
    Immune to price gaps and halted stocks.
    """
    anchor_date = prices_df.index[anchor_iloc]
    target_date = anchor_date + pd.Timedelta(days=int(horizon_days * 1.5))
    future_iloc = prices_df.index.searchsorted(target_date)
    if future_iloc >= len(prices_df):
        return pd.Series(dtype=float)
    return (prices_df.iloc[future_iloc] / prices_df.iloc[anchor_iloc] - 1).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT ALLOCATION  (identical to RF v3 / XGB v2)
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
# FIX 8/9/10/11: LSTM MODEL BUILDER
# Architecture: LSTM → BatchNorm → Dropout → Dense → Dense → Dense(1)
# Three hidden transformations (matches paper's NN3 spirit)
# Geometric pyramid rule: units → units//2 → units//4
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm(units: int, dropout_rate: float, lr: float,
               seq_len: int, n_features: int,
               recurrent_dropout: float = 0.0) -> tf.keras.Model:
    """
    units       : LSTM hidden units (e.g. 16, 32, 64)
    dropout_rate: fraction of units dropped after LSTM
    lr          : Adam initial learning rate
    seq_len     : number of time steps in each sequence
    n_features  : number of input features per time step
    """
    d1 = max(units // 2, 4)    # first dense layer width
    d2 = max(units // 4, 2)    # second dense layer width (geometric pyramid)

    model = Sequential([
        # ── Recurrent layer ───────────────────────────────────────────────
        LSTM(
            units,
            input_shape           = (seq_len, n_features),
            return_sequences      = False,
            kernel_regularizer    = l2(L2_LAMBDA),
            recurrent_regularizer = l2(L2_LAMBDA),
            recurrent_dropout     = recurrent_dropout,
        ),
        # FIX 10: Batch normalisation — stabilises activations across stocks
        BatchNormalization(),
        # Dropout for stochastic regularisation
        Dropout(dropout_rate),

        # ── Hidden layer 1 ────────────────────────────────────────────────
        Dense(
            d1,
            activation         = 'relu',
            kernel_regularizer = l2(L2_LAMBDA),      # FIX 9
        ),

        # ── Hidden layer 2 ────────────────────────────────────────────────
        Dense(
            d2,
            activation         = 'relu',
            kernel_regularizer = l2(L2_LAMBDA),      # FIX 9
        ),

        # ── Output ────────────────────────────────────────────────────────
        Dense(1),
    ])

    model.compile(
        optimizer = Adam(learning_rate=lr),
        loss      = tf.keras.losses.Huber(delta=1.0),  # FIX 8: Huber loss
        metrics   = ['mae'],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== LSTM | seed={RANDOM_SEED} (base={BASE_SEED} + run={RUN_NUMBER}) ===")
n_grid = len(GRID_NODES) * len(GRID_LR) * len(BATCH_SIZE)
print(f"    Grid: {len(GRID_NODES)} nodes × {len(GRID_LR)} lr × {len(BATCH_SIZE)} batch = "
      f"{n_grid} combos | dropout={GRID_DROPOUTS} rec_drop={RECURRENT_DROPOUT} | "
      f"TC={TC_BPS}bps | VAL={VAL_MONTHS}mo\n")

for label, (offset, horizon) in FREQUENCIES.items():
    print(f"\n=== [{label}] horizon={horizon}d ===")

    current_date               = start_invest
    portfolio_value            = 1.0
    last_end_weights           = pd.Series(dtype=float)
    portfolio_performance      = []
    rebalance_details          = []
    model_stats                = []
    feature_importance_records = []

    while current_date < end_invest:
        next_rebalance = current_date + offset

        valid_days = all_prices.index[all_prices.index >= current_date]
        if valid_days.empty:
            break
        actual_trade_date = valid_days[0]

        # Clear TF session at the start of every rebalance to prevent state accumulation
        tf.keras.backend.clear_session()
        gc.collect()

        # FIX 1: always initialise before any conditional block
        target_weights = None
        pred_series    = None
        best_nodes     = GRID_NODES[0]
        best_lr        = GRID_LR[0]
        best_batch     = BATCH_SIZE[0]
        best_val_loss  = np.nan

        # FIX 5: explicit universe year logic
        invest_year = current_date.year
        select_year = invest_year - 1        # tickers[1997] → invest in 1998, etc.

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
                    # FIX 3: cap ffill at 5 consecutive days
                    train_prices     = hist_prices[valid_tickers].ffill(limit=5)
                    val_start_date   = actual_trade_date - pd.DateOffset(months=VAL_MONTHS)
                    train_start_date = actual_trade_date - pd.DateOffset(months=TRAIN_MONTHS + VAL_MONTHS)
                    val_split        = int(np.searchsorted(train_prices.index, val_start_date))
                    train_start_idx  = max(260, int(np.searchsorted(train_prices.index, train_start_date)))

                    # ── Build monthly snapshots ───────────────────────────────
                    train_snapshots, val_snapshots = [], []

                    for i in range(train_start_idx, val_split, 21):
                        feat = create_features(train_prices.iloc[:i])
                        if not feat.empty:
                            train_snapshots.append((i, feat))

                    for i in range(val_split, len(train_prices), 21):
                        feat = create_features(train_prices.iloc[:i])
                        if not feat.empty:
                            val_snapshots.append((i, feat))

                    # ── Build LSTM sequences with forward returns ─────────────
                    X_train_seqs, y_train_list = [], []
                    X_val_seqs,   y_val_list   = [], []

                    for snap_list, target_X, target_y in [
                        (train_snapshots, X_train_seqs, y_train_list),
                        (val_snapshots,   X_val_seqs,   y_val_list),
                    ]:
                        for end_idx in range(SEQ_LEN - 1, len(snap_list) - 1):
                            seq_snaps     = snap_list[end_idx - SEQ_LEN + 1: end_idx + 1]
                            next_snap_iloc = snap_list[end_idx + 1][0]

                            # FIX 2: date-based forward return
                            fwd_ret = forward_return(train_prices, next_snap_iloc, horizon)
                            if fwd_ret.empty:
                                continue

                            common = set(seq_snaps[0][1].index)
                            for _, fd in seq_snaps[1:]:
                                common &= set(fd.index)
                            common = sorted(common & set(fwd_ret.index))
                            if len(common) < 2:
                                continue

                            seq_arr  = np.stack(
                                [fd.loc[common].values for _, fd in seq_snaps], axis=1
                            )                                    # (n_stocks, seq_len, n_feat)
                            y_ranked = fwd_ret.loc[common].rank(pct=True).values

                            target_X.append(seq_arr)
                            target_y.append(y_ranked)

                    if X_train_seqs:
                        X_tr = np.concatenate(X_train_seqs, axis=0)
                        y_tr = np.concatenate(y_train_list, axis=0)
                        n_features = X_tr.shape[2]

                        has_val = bool(X_val_seqs)
                        X_vl = np.concatenate(X_val_seqs, axis=0) if has_val else None
                        y_vl = np.concatenate(y_val_list, axis=0) if has_val else None

                        # ── FIX 6: Grid search over 18 combos ────────────────
                        tf.keras.backend.clear_session()
                        best_val_loss_grid = np.inf

                        for g_node in GRID_NODES:
                            for g_lr in GRID_LR:
                                for g_batch in BATCH_SIZE:
                                    g_model = build_lstm(
                                        g_node, GRID_DROPOUTS, g_lr, SEQ_LEN, n_features,
                                        recurrent_dropout=RECURRENT_DROPOUT,
                                    )
                                    callbacks = [
                                        EarlyStopping(
                                            monitor='val_loss' if has_val else 'loss',
                                            patience=GRID_PATIENCE,
                                            restore_best_weights=True,
                                        ),
                                        ReduceLROnPlateau(
                                            monitor='val_loss' if has_val else 'loss',
                                            factor=0.5, patience=2, min_lr=1e-5,
                                        ),
                                    ]
                                    g_hist = g_model.fit(
                                        X_tr, y_tr,
                                        validation_data = (X_vl, y_vl) if has_val else None,
                                        epochs          = GRID_EPOCHS,
                                        batch_size      = g_batch,
                                        callbacks       = callbacks,
                                        verbose         = 0,
                                    )
                                    monitor_key = 'val_loss' if has_val else 'loss'
                                    g_loss = min(g_hist.history.get(monitor_key, [np.inf]))

                                    if g_loss < best_val_loss_grid:
                                        best_val_loss_grid = g_loss
                                        best_nodes    = g_node
                                        best_lr       = g_lr
                                        best_batch    = g_batch
                                        best_val_loss = g_loss

                                    del g_model
                                    gc.collect()

                        print(
                            f"  [{label}] {current_date.date()} | "
                            f"universe={select_year} ({len(valid_tickers)} stocks) | "
                            f"best: nodes={best_nodes} batch={best_batch} "
                            f"lr={best_lr} | val_loss={best_val_loss:.5f}"
                        )

                        # ── Final model: train + val combined ─────────────────
                        tf.keras.backend.clear_session()
                        X_full = np.concatenate([X_tr, X_vl], axis=0) if has_val else X_tr
                        y_full = np.concatenate([y_tr, y_vl], axis=0) if has_val else y_tr

                        final_model = build_lstm(
                            best_nodes, GRID_DROPOUTS, best_lr, SEQ_LEN, n_features,
                            recurrent_dropout=RECURRENT_DROPOUT,
                        )
                        final_callbacks = [
                            EarlyStopping(
                                monitor='loss',
                                patience=PATIENCE,
                                restore_best_weights=True,
                            ),
                            # FIX 11: LR scheduler on final model
                            ReduceLROnPlateau(
                                monitor='loss',
                                factor=0.5, patience=3, min_lr=1e-6,
                            ),
                        ]
                        final_model.fit(
                            X_full, y_full,
                            epochs     = FULL_EPOCHS,
                            batch_size = best_batch,
                            callbacks  = final_callbacks,
                            verbose    = 0,
                        )

                        # ── Prediction at current date ────────────────────────
                        pred_snapshots = []
                        for i in range(
                            max(260, len(train_prices) - SEQ_LEN * 21 - 21),
                            len(train_prices),
                            21
                        ):
                            feat = create_features(train_prices.iloc[:i])
                            if not feat.empty:
                                pred_snapshots.append(feat)

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
                                )                                # (n_stocks, seq_len, n_feat)
                                preds       = final_model.predict(X_pred, verbose=0).flatten()
                                pred_series = pd.Series(
                                    preds, index=common_pred
                                ).sort_values(ascending=False)
                                target_weights = allocate_weights(
                                    pred_series, WEIGHT_MIN, WEIGHT_MAX
                                )

                                # gradient-based feature attribution (|∂output/∂input|)
                                X_tensor = tf.constant(X_pred, dtype=tf.float32)
                                with tf.GradientTape() as tape:
                                    tape.watch(X_tensor)
                                    out = final_model(X_tensor, training=False)
                                grads   = tape.gradient(out, X_tensor).numpy()
                                fi_vals = np.abs(grads).mean(axis=(0, 1))
                                fi_vals = fi_vals / (fi_vals.sum() + 1e-12)
                                fi_row  = {'rebalance_date': actual_trade_date.strftime('%Y-%m-%d')}
                                fi_row.update(dict(zip(pred_snapshots[-1].columns.tolist(), fi_vals)))
                                feature_importance_records.append(fi_row)

                        # ── Cleanup ───────────────────────────────────────────
                        del final_model, X_tr, y_tr, X_full, y_full
                        if has_val:
                            del X_vl, y_vl
                        del train_snapshots, val_snapshots
                        del X_train_seqs, y_train_list, X_val_seqs, y_val_list
                        gc.collect()
                        tf.keras.backend.clear_session()

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

        # ── FIX 14: Transaction cost at rebalance ─────────────────────────────
        turnover = target_weights.sub(last_end_weights, fill_value=0).abs().sum()
        if TC_BPS > 0:
            portfolio_value *= (1 - turnover * TC_BPS / 10_000)

        # ── Rebalance logging ─────────────────────────────────────────────────
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date' : actual_trade_date.strftime('%Y-%m-%d'),
                'invest_year'    : invest_year,           # FIX 13
                'select_year'    : select_year,           # FIX 13
                'ticker'         : ticker,
                'assigned_weight': w,
                'turnover'       : round(turnover, 6) if ticker == target_weights.index[0] else 0,
                'tc_drag_bps'    : round(turnover * TC_BPS, 4) if ticker == target_weights.index[0] else 0,
                'best_nodes'     : best_nodes,
                'best_batch'     : best_batch,
                'best_lr'        : best_lr,
                'best_val_loss'  : round(best_val_loss, 6) if not np.isnan(best_val_loss) else np.nan,
                'run_number'     : RUN_NUMBER,
                'seed'           : RANDOM_SEED,
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
            if len(hold_prices) >= 2 and pred_series is not None:
                actual_ret = (hold_prices.iloc[-1] / hold_prices.iloc[0] - 1).dropna()
                common_idx = pred_series.index.intersection(actual_ret.index)

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

                    ss_res  = np.sum((y_true_rank - y_pred) ** 2)
                    ss_tot  = np.sum((y_true_rank - y_true_rank.mean()) ** 2)
                    r2_rank = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

                    # FIX 12: Raw R²oos vs zero benchmark (paper-comparable)
                    ss_res_raw = np.sum((y_true - y_pred) ** 2)
                    ss_tot_raw = np.sum(y_true ** 2)
                    r2_raw     = 1 - ss_res_raw / ss_tot_raw if ss_tot_raw != 0 else np.nan

                    spearman = pd.Series(y_pred).corr(
                        pd.Series(y_true_rank), method='spearman'
                    )

                    model_stats.append({
                        'rebalance_date'      : actual_trade_date.strftime('%Y-%m-%d'),
                        'invest_year'         : invest_year,      # FIX 13
                        'select_year'         : select_year,      # FIX 13
                        'run_number'          : RUN_NUMBER,
                        'seed'                : RANDOM_SEED,
                        'n_stocks'            : len(common_idx),
                        'best_nodes'          : best_nodes,
                        'best_batch'          : best_batch,
                        'best_lr'             : best_lr,
                        'best_val_loss'       : best_val_loss,
                        'RMSE'                : float(np.sqrt(np.mean((y_pred - y_true_rank)**2))),
                        'MSE'                 : float(np.mean((y_pred - y_true_rank)**2)),
                        'MAE'                 : float(np.mean(np.abs(y_pred - y_true_rank))),
                        'R2_rank'             : r2_rank,
                        'R2_raw_vs_zero'      : r2_raw,           # FIX 12
                        'Spearman'            : spearman,
                        'Directional_Accuracy': float(np.mean(pred_dir == true_dir)),
                        'Geometric_Score'     : geo,
                        'turnover'            : turnover,
                        'tc_drag_bps'         : turnover * TC_BPS,
                    })

        last_end_weights = active_weights
        current_date     = next_rebalance

    # ── Export — one file per frequency, named with run number ───────────────
    tag = f"{label}_run{RUN_NUMBER}"
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"portfolio_lstm_{tag}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"portfolio_lstm_{tag}_details.csv", index=False
    )
    pd.DataFrame(model_stats).to_csv(
        output_dir / f"portfolio_lstm_{tag}_statistics.csv", index=False
    )
    pd.DataFrame(feature_importance_records).to_csv(
        output_dir / f"portfolio_lstm_{tag}_feature_importance.csv", index=False
    )
    print(f"\n  [{label}] Run {RUN_NUMBER} done — saved to {output_dir}")

    # Hard reset between frequencies to prevent TF state accumulation
    tf.keras.backend.clear_session()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

# ── Final cleanup ─────────────────────────────────────────────────────────────
tf.keras.backend.clear_session()
gc.collect()
print(f"\n=== LSTM v2 complete | run={RUN_NUMBER} seed={RANDOM_SEED} ===")