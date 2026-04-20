"""
Crisis period model ranking analysis (csv_3.py)

Output per frequency (one CSV each):
  Section 1 - Raw risk/return metrics  : rows = models,  cols = crisis x metric
  Section 2 - Z-scores (winsorized +-3sd): same layout as section 1
  Section 3 - Borda scores             : rows = 4 methods, cols = models
  Section 4 - Final rankings           : rows = 4 methods, cols = models
"""
import os
import glob
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

BASE     = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "data", "latex")

CRISIS_PERIODS = [
    ("Dotcom Crash",       "dotcom",  "2000-03-23", "2002-10-09", "2007-05-31"),
    ("GFC (Full)",         "gfc",     "2007-10-09", "2009-03-09", "2013-03-28"),
    ("Monetary Policy",    "mon_pol", "2018-09-21", "2018-12-24", "2019-04-23"),
    ("COVID-19",           "covid19", "2020-02-19", "2020-03-23", "2020-08-12"),
    ("Russia/Ukraine",     "russia",  "2022-01-03", "2022-10-12", "2024-01-19"),
    ("Trade Policy Shock", "trade",   "2025-02-19", "2025-04-08", "2025-06-26"),
]

FREQUENCIES = {
    "Monthly":     "monthly",
    "Quarterly":   "quarterly",
    "Semi-Annual": "semi_annual",
    "Yearly":      "annual",
}

# (display_name, data_subdir, filename_template, is_lstm)
MODELS = [
    ("xgboost",       "xgboost",                "portfolio_xgb_{freq}.csv",        False),
    ("random_forest", "random_forest",           "portfolio_rf_{freq}.csv",         False),
    ("lstm",          "lstm",                    "portfolio_lstm_{freq}.csv",       False),
    ("equal_weight",  "equal_weight",            "portfolio_ew_{freq}.csv",         False),
    ("hrp",           "hrp",                     "portfolio_hrp_{freq}.csv",        False),
    ("market_cap",    "market_cap",              "portfolio_mktcap_{freq}.csv",     False),
    ("markowitz",     "markowitz",               "portfolio_markowitz_{freq}.csv",  False),
    ("markowitz_unc", "markowitz_unconstrained", "portfolio_markowitz_{freq}.csv",  False),
]
MODEL_KEYS = [m[0] for m in MODELS]

METRICS = ["sharpe", "sortino", "ulcer", "max_dd", "cum_ret", "calmar"]

# +1 = higher is better, -1 = lower is better (controls Borda ranking direction)
METRIC_DIRECTION = {
    "sharpe":  1,
    "sortino": 1,
    "ulcer":  -1,
    "max_dd":  1,   # stored as negative; less negative (higher) = better
    "cum_ret": 1,
    "calmar":  1,
}

TRADING_DAYS = 252


# ── Data loading ──────────────────────────────────────────────────────────────

def load_returns(data_subdir, file_tpl, freq, is_lstm):
    folder = os.path.join(DATA_DIR, data_subdir)
    tpl    = file_tpl.replace("{freq}", freq)
    if is_lstm:
        run_files = sorted(glob.glob(os.path.join(folder, tpl)))
        if not run_files:
            return None
        series = [
            pd.read_csv(f, parse_dates=["date"]).set_index("date")["log_return"]
            for f in run_files
        ]
        return pd.concat(series, axis=1).mean(axis=1)
    path = os.path.join(folder, tpl)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["log_return"]


def slice_crisis(r, d1, d3):
    if r is None:
        return None
    s = r[(r.index >= pd.Timestamp(d1)) & (r.index <= pd.Timestamp(d3))]
    return s if len(s) >= 5 else None


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
    ulcer  = float(np.sqrt((dd ** 2).mean()))
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan

    return {"sharpe": sharpe, "sortino": sortino, "ulcer": ulcer,
            "max_dd": max_dd, "cum_ret": cum_ret, "calmar": calmar}


# ── Crisis weights ────────────────────────────────────────────────────────────

def equal_weights():
    n = len(CRISIS_PERIODS)
    return np.ones(n) / n


def duration_weights():
    d = np.array(
        [(pd.Timestamp(d3) - pd.Timestamp(d1)).days for _, _, d1, _, d3 in CRISIS_PERIODS],
        dtype=float,
    )
    return d / d.sum()


def severity_weights(bm):
    sevs = []
    for _, _, d1, _, d3 in CRISIS_PERIODS:
        r = slice_crisis(bm, d1, d3)
        if r is None:
            sevs.append(0.0)
        else:
            cv = np.exp(r.cumsum())
            sevs.append(abs(float(((cv - cv.cummax()) / cv.cummax()).min())))
    s = np.array(sevs, dtype=float)
    return s / s.sum() if s.sum() > 0 else np.ones(len(s)) / len(s)


def vix_weights():
    FALLBACK = {"dotcom": 27.0, "gfc": 32.0, "mon_pol": 20.0,
                "covid19": 45.0, "russia": 24.0, "trade": 25.0}
    avgs = []
    if HAS_YF:
        try:
            raw = yf.download(
                "^VIX",
                start=CRISIS_PERIODS[0][2],
                end=CRISIS_PERIODS[-1][4],
                progress=False,
                auto_adjust=True,
            )["Close"].squeeze()
            for _, short, d1, _, d3 in CRISIS_PERIODS:
                mask = (raw.index >= pd.Timestamp(d1)) & (raw.index <= pd.Timestamp(d3))
                avgs.append(float(raw[mask].mean()) if mask.any() else FALLBACK[short])
        except Exception:
            avgs = [FALLBACK[s] for _, s, *_ in CRISIS_PERIODS]
    else:
        avgs = [FALLBACK[s] for _, s, *_ in CRISIS_PERIODS]
    v = np.array(avgs, dtype=float)
    return v / v.sum()


# ── Z-scores ──────────────────────────────────────────────────────────────────

def winsorized_zscore(df):
    result = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) < 2:
            continue
        mu, sigma = vals.mean(), vals.std()
        result[col] = ((df[col] - mu) / sigma).clip(-3, 3) if sigma > 0 else 0.0
    return result


# ── Borda count ───────────────────────────────────────────────────────────────

def borda_count(zdir_df, weights):
    """
    zdir_df : rows=models, cols=crisis_metric — already direction-adjusted (higher=better).
    weights : one weight per crisis, summing to 1.
    Returns a pd.Series of total Borda points per model.
    """
    n      = len(zdir_df)
    scores = pd.Series(0.0, index=zdir_df.index)
    for ci, (_, short, *_) in enumerate(CRISIS_PERIODS):
        w = weights[ci]
        for metric in METRICS:
            col = f"{short}_{metric}"
            if col not in zdir_df.columns:
                continue
            ranks   = zdir_df[col].rank(ascending=False, method="average", na_option="bottom")
            scores += (n - ranks) * w
    return scores


# ── CSV assembly helpers ──────────────────────────────────────────────────────

def make_blank(cols):
    return pd.DataFrame([[np.nan] * len(cols)], columns=cols)


def make_header(title, cols):
    row          = make_blank(cols)
    row[cols[0]] = title
    return row


def to_unified(df, label_col, value_cols, all_cols):
    out             = pd.DataFrame(np.nan, index=range(len(df)), columns=all_cols)
    out[all_cols[0]] = df[label_col].values
    for c in value_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").values
    return out


# ── Per-frequency build ───────────────────────────────────────────────────────

def build_freq_csv(freq_label, out_name, bm_returns, w_vix):
    crisis_shorts = [s for _, s, *_ in CRISIS_PERIODS]
    metric_cols   = [f"{s}_{m}" for s in crisis_shorts for m in METRICS]
    all_cols      = ["label"] + metric_cols + MODEL_KEYS

    # Load all model return series
    model_returns = {
        key: load_returns(subdir, tpl, freq_label, is_lstm)
        for key, subdir, tpl, is_lstm in MODELS
    }

    # ── Raw metrics ──────────────────────────────────────────────────────────
    raw_data = {}
    for key in MODEL_KEYS:
        r_full = model_returns[key]
        row = {}
        for _, short, d1, _, d3 in CRISIS_PERIODS:
            for m, v in calc_metrics(slice_crisis(r_full, d1, d3)).items():
                row[f"{short}_{m}"] = v
        raw_data[key] = row

    raw_df = pd.DataFrame(raw_data).T.reset_index().rename(columns={"index": "model"})

    # ── Z-scores ─────────────────────────────────────────────────────────────
    metrics_only = raw_df.set_index("model")[metric_cols]
    zscore_df    = winsorized_zscore(metrics_only)
    zscore_out   = zscore_df.reset_index()

    # Direction-adjusted copy: flip lower-is-better metrics so higher always = better
    zdir_df = zscore_df.copy()
    for metric in METRICS:
        if METRIC_DIRECTION[metric] == -1:
            for short in crisis_shorts:
                col = f"{short}_{metric}"
                if col in zdir_df.columns:
                    zdir_df[col] *= -1

    # ── Crisis weights ────────────────────────────────────────────────────────
    w_eq  = equal_weights()
    w_dur = duration_weights()
    w_sev = severity_weights(bm_returns)

    # ── Borda scores ──────────────────────────────────────────────────────────
    borda_results = {
        "borda_equal":    borda_count(zdir_df, w_eq),
        "borda_duration": borda_count(zdir_df, w_dur),
        "borda_severity": borda_count(zdir_df, w_sev),
        "borda_vix":      borda_count(zdir_df, w_vix),
    }
    borda_df = pd.DataFrame([
        {"method": name, **scores.to_dict()}
        for name, scores in borda_results.items()
    ])

    # ── Final rankings ────────────────────────────────────────────────────────
    rank_df = pd.DataFrame([
        {"method": name.replace("borda_", "rank_"),
         **scores.rank(ascending=False, method="min").astype(int).to_dict()}
        for name, scores in borda_results.items()
    ])

    # ── Assemble CSV ──────────────────────────────────────────────────────────
    blank = make_blank(all_cols)
    result = pd.concat([
        make_header("=== RAW METRICS ===",                all_cols),
        to_unified(raw_df,    "model",  metric_cols, all_cols),
        blank,
        make_header("=== Z-SCORES (winsorized +-3sd) ===", all_cols),
        to_unified(zscore_out, "model", metric_cols, all_cols),
        blank,
        make_header("=== BORDA SCORES ===",               all_cols),
        to_unified(borda_df,  "method", MODEL_KEYS,  all_cols),
        blank,
        make_header("=== FINAL RANKINGS (1=best) ===",    all_cols),
        to_unified(rank_df,   "method", MODEL_KEYS,  all_cols),
    ], ignore_index=True)

    out_path = os.path.join(OUT_DIR, f"crisis_ranking_{out_name}.csv")
    result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  Saved crisis_ranking_{out_name}.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading benchmark returns...")
bm_returns = (
    pd.read_csv(
        os.path.join(DATA_DIR, "benchmark", "portfolio.csv"), parse_dates=["date"]
    ).set_index("date")["log_returns_per_day"]
)

print("Fetching VIX weights (via yfinance)...")
w_vix = vix_weights()

for freq_label, out_name in FREQUENCIES.items():
    print(f"\nBuilding {freq_label}...")
    build_freq_csv(freq_label, out_name, bm_returns, w_vix)

print("\nDone.")
