import os
import numpy as np
import pandas as pd

BASE     = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "data", "latex")

CRISIS_PERIODS = [
    ("Early Credit Crunch",             "ecc",     "2007-10-09", "2008-09-15", "2010-04-23"),
    ("Acute GFC Crash",                 "agfc",    "2008-09-15", "2009-03-09", "2010-04-23"),
    ("European Debt + US Debt Ceiling", "eu_debt", "2010-04-23", "2011-10-03", "2012-03-23"),
]

FREQUENCIES = {
    "Monthly":     "monthly",
    "Quarterly":   "quarterly",
    "Semi-Annual": "semi_annual",
    "Yearly":      "annual",
}

MODELS = [
    ("xgboost",       "xgboost",                "portfolio_xgb_{freq}.csv",       False),
    ("random_forest", "random_forest",           "portfolio_rf_{freq}.csv",        False),
    ("lstm",          "lstm",                    "portfolio_lstm_{freq}.csv",      False),
    ("equal_weight",  "equal_weight",            "portfolio_ew_{freq}.csv",        False),
    ("hrp",           "hrp",                     "portfolio_hrp_{freq}.csv",       False),
    ("market_cap",    "market_cap",              "portfolio_mktcap_{freq}.csv",    False),
    ("markowitz",     "markowitz",               "portfolio_markowitz_{freq}.csv", False),
    ("markowitz_unc", "markowitz_unconstrained", "portfolio_markowitz_{freq}.csv", False),
]
MODEL_KEYS = [m[0] for m in MODELS]

METRICS = ["sharpe", "sortino", "ulcer", "max_dd", "cum_ret", "calmar"]

TRADING_DAYS = 252


def load_returns(data_subdir, file_tpl, freq):
    path = os.path.join(DATA_DIR, data_subdir, file_tpl.replace("{freq}", freq))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["log_return"]


def slice_crisis(r, d1, d3):
    if r is None:
        return None
    s = r[(r.index >= pd.Timestamp(d1)) & (r.index <= pd.Timestamp(d3))]
    return s if len(s) >= 5 else None


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


def build_freq_csv(freq_label, out_name):
    model_returns = {
        key: load_returns(subdir, tpl, freq_label)
        for key, subdir, tpl, _ in MODELS
    }

    rows = []
    for _, short, d1, _, d3 in CRISIS_PERIODS:
        for metric in METRICS:
            row = {"crisis": f"{short}_{metric}"}
            for key in MODEL_KEYS:
                r = slice_crisis(model_returns[key], d1, d3)
                row[key] = calc_metrics(r)[metric]
            rows.append(row)

    result = pd.DataFrame(rows, columns=["crisis"] + MODEL_KEYS)
    out_path = os.path.join(OUT_DIR, f"subcrisis_metrics_{out_name}.csv")
    result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved subcrisis_metrics_{out_name}.csv  ({len(result)} rows)")


os.makedirs(OUT_DIR, exist_ok=True)

for freq_label, out_name in FREQUENCIES.items():
    build_freq_csv(freq_label, out_name)
