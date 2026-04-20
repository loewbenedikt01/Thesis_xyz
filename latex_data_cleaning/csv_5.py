import os
import numpy as np
import pandas as pd

BASE     = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "data", "latex")

FREQUENCIES = {
    "Monthly":     "monthly",
    "Quarterly":   "quarterly",
    "Semi-Annual": "semi_annual",
    "Yearly":      "annual",
}

MODELS = [
    ("xgboost",       "xgboost",                "portfolio_xgb_{freq}.csv"),
    ("random_forest", "random_forest",           "portfolio_rf_{freq}.csv"),
    ("lstm",          "lstm",                    "portfolio_lstm_{freq}.csv"),
    ("equal_weight",  "equal_weight",            "portfolio_ew_{freq}.csv"),
    ("hrp",           "hrp",                     "portfolio_hrp_{freq}.csv"),
    ("market_cap",    "market_cap",              "portfolio_mktcap_{freq}.csv"),
    ("markowitz",     "markowitz",               "portfolio_markowitz_{freq}.csv"),
    ("markowitz_unc", "markowitz_unconstrained", "portfolio_markowitz_{freq}.csv"),
]
MODEL_KEYS = [m[0] for m in MODELS]

METRICS = ["sharpe", "sortino", "calmar", "cum_ret", "ann_ret", "var", "cvar"]

# +1 = higher is better, -1 = lower is better (for Borda direction)
METRIC_DIRECTION = {
    "sharpe":  1,
    "sortino": 1,
    "calmar":  1,
    "cum_ret": 1,
    "ann_ret": 1,
    "var":    -1,   # lower VaR = less tail risk = better
    "cvar":   -1,   # lower CVaR = less expected shortfall = better
}

TRADING_DAYS = 252
VAR_LEVEL    = 0.05   # 95% confidence


# ── Data loading ──────────────────────────────────────────────────────────────

def load_returns(data_subdir, file_tpl, freq):
    path = os.path.join(DATA_DIR, data_subdir, file_tpl.replace("{freq}", freq))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["log_return"]


# ── Metric calculations ───────────────────────────────────────────────────────

def calc_metrics(r):
    nan_row = {m: np.nan for m in METRICS}
    if r is None or len(r) < 5:
        return nan_row

    n       = len(r)
    cum_ret = float(np.exp(r.sum()) - 1)
    ann_ret = float((1 + cum_ret) ** (TRADING_DAYS / n) - 1)
    mean_r  = r.mean()
    std_r   = r.std()

    sharpe  = float(np.sqrt(TRADING_DAYS) * mean_r / std_r) if std_r > 0 else np.nan

    down    = r[r < 0]
    d_std   = float(down.std()) if len(down) > 1 else np.nan
    sortino = float(np.sqrt(TRADING_DAYS) * mean_r / d_std) if (d_std and d_std > 0) else np.nan

    cv     = np.exp(r.cumsum())
    rm     = cv.cummax()
    dd     = (cv - rm) / rm
    max_dd = float(dd.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan

    # VaR and CVaR at VAR_LEVEL (reported as positive loss magnitudes)
    var_threshold = float(np.percentile(r, VAR_LEVEL * 100))
    var  = -var_threshold
    cvar = float(-r[r <= var_threshold].mean()) if (r <= var_threshold).any() else np.nan

    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "cum_ret": cum_ret, "ann_ret": ann_ret, "var": var, "cvar": cvar}


# ── Z-scores (winsorized ±3 SD) ───────────────────────────────────────────────

def winsorized_zscore_row(values: dict) -> dict:
    """Z-score a dict {model: value} across models for one metric."""
    arr  = np.array(list(values.values()), dtype=float)
    keys = list(values.keys())
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return {k: np.nan for k in keys}
    mu, sigma = valid.mean(), valid.std()
    if sigma == 0:
        return {k: 0.0 for k in keys}
    z = {k: float(np.clip((arr[i] - mu) / sigma, -3, 3)) for i, k in enumerate(keys)}
    return z


# ── Borda count ───────────────────────────────────────────────────────────────

def borda_rank(zscore_rows: dict) -> dict:
    """
    zscore_rows: {metric: {model: z_score}}
    Returns Borda points per model (higher = better overall).
    """
    n      = len(MODEL_KEYS)
    scores = {k: 0.0 for k in MODEL_KEYS}

    for metric, model_zscores in zscore_rows.items():
        direction = METRIC_DIRECTION[metric]
        # Apply direction so higher always = better before ranking
        adjusted = {m: v * direction for m, v in model_zscores.items()}
        sorted_models = sorted(adjusted, key=lambda m: (
            adjusted[m] if not np.isnan(adjusted[m]) else -np.inf
        ), reverse=True)
        for rank_0, model in enumerate(sorted_models):
            scores[model] += n - 1 - rank_0   # best gets n-1 points

    return scores


# ── Per-frequency build ───────────────────────────────────────────────────────

def build_freq_csv(freq_label, out_name):
    # Column order: date, then for each metric all model suffixes
    metric_model_cols = [f"{metric}_{key}" for metric in METRICS for key in MODEL_KEYS]
    all_cols = ["date"] + metric_model_cols

    # Load returns for all models
    model_returns = {
        key: load_returns(subdir, tpl, freq_label)
        for key, subdir, tpl in MODELS
    }

    # ── Row 1: raw metrics ────────────────────────────────────────────────────
    raw_metrics = {key: calc_metrics(model_returns[key]) for key in MODEL_KEYS}
    raw_row = {"date": "overall"}
    for metric in METRICS:
        for key in MODEL_KEYS:
            raw_row[f"{metric}_{key}"] = raw_metrics[key][metric]

    # ── Row 2: z-scores ───────────────────────────────────────────────────────
    zscore_rows = {}
    for metric in METRICS:
        metric_vals = {key: raw_metrics[key][metric] for key in MODEL_KEYS}
        zscore_rows[metric] = winsorized_zscore_row(metric_vals)

    zscore_row = {"date": "zscore"}
    for metric in METRICS:
        for key in MODEL_KEYS:
            zscore_row[f"{metric}_{key}"] = zscore_rows[metric][key]

    # ── Row 3: Borda ranking ──────────────────────────────────────────────────
    borda_scores  = borda_rank(zscore_rows)
    # Rank from scores (1 = highest score = best)
    sorted_models = sorted(borda_scores, key=lambda m: borda_scores[m], reverse=True)
    borda_ranks   = {m: sorted_models.index(m) + 1 for m in MODEL_KEYS}

    borda_row = {"date": "borda_rank"}
    for metric in METRICS:
        for key in MODEL_KEYS:
            # Put the model's overall rank in the sharpe columns; NaN elsewhere
            borda_row[f"{metric}_{key}"] = borda_ranks[key] if metric == "sharpe" else np.nan

    # Borda score row (raw points, for reference)
    borda_score_row = {"date": "borda_score"}
    for metric in METRICS:
        for key in MODEL_KEYS:
            borda_score_row[f"{metric}_{key}"] = borda_scores[key] if metric == "sharpe" else np.nan

    result = pd.DataFrame(
        [raw_row, zscore_row, borda_score_row, borda_row],
        columns=all_cols,
    )

    out_path = os.path.join(OUT_DIR, f"model_performance_{out_name}.csv")
    result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved model_performance_{out_name}.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

for freq_label, out_name in FREQUENCIES.items():
    build_freq_csv(freq_label, out_name)
