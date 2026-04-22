import os
import glob
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "results")
METRICS_DIR = os.path.join(BASE, "metrics")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "data", "latex")

CRISIS_PERIODS = [
    ("Dotcom Crash",                    "dotcom",  "2000-03-23", "2002-10-09", "2007-05-31"),
    ("GFC (Full)",                      "gfc",     "2007-10-09", "2009-03-09", "2013-03-28"),
    ("Early Credit Crunch",             "ecc",     "2007-10-09", "2008-09-15", "2010-04-23"),
    ("Acute GFC Crash",                 "agfc",    "2008-09-15", "2009-03-09", "2010-04-23"),
    ("European Debt + US Debt Ceiling", "eu_debt", "2010-04-23", "2011-10-03", "2012-03-23"),
    ("Monetary Policy",                 "mon_pol", "2018-09-21", "2018-12-24", "2019-04-23"),
    ("COVID-19",                        "covid19", "2020-02-19", "2020-03-23", "2020-08-12"),
    ("Russia/Ukraine",                  "russia",  "2022-01-03", "2022-10-12", "2024-01-19"),
    ("Trade Policy Shock",              "trade",   "2025-02-19", "2025-04-08", "2025-06-26"),
]

# (freq label for filenames, output filename)
FREQUENCIES = {
    "Monthly":    "monthly",
    "Quarterly":  "quarterly",
    "Semi-Annual": "semi_annual",
    "Yearly":     "annual",
}

MODELS = {
    "xgboost":       {"abbr": "xgb",  "data_dir": "xgboost",       "metrics_dir": "xgboost"},
    "random_forest":  {"abbr": "rf",   "data_dir": "random_forest",  "metrics_dir": "random_forest"},
    "lstm":           {"abbr": "lstm", "data_dir": "lstm",           "metrics_dir": "lstm"},
}

TRADING_DAYS_PER_YEAR = 252


def sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def load_portfolio_returns(model_key: str, freq: str) -> pd.DataFrame | None:
    cfg = MODELS[model_key]
    path = os.path.join(DATA_DIR, cfg["data_dir"], f"portfolio_{cfg['abbr']}_{freq}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])[["date", "log_return"]]


def load_statistics(model_key: str, freq: str) -> pd.DataFrame | None:
    cfg = MODELS[model_key]
    if model_key == "lstm":
        path = os.path.join(DATA_DIR, cfg["data_dir"], f"portfolio_lstm_{freq}_statistics.csv")
    else:
        path = os.path.join(METRICS_DIR, cfg["metrics_dir"], f"statistics_{freq}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["rebalance_date"])
    return df[["rebalance_date", "Directional_Accuracy"]]


def compute_metrics(
    stats: pd.DataFrame | None,
    returns: pd.DataFrame | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> tuple[float, float]:
    """Return (da_mean, sharpe_mean) for the given window. None = full period."""
    if stats is not None:
        s = stats if start is None else stats[
            (stats["rebalance_date"] >= start) & (stats["rebalance_date"] <= end)
        ]
        da = s["Directional_Accuracy"].mean() if len(s) > 0 else np.nan
    else:
        da = np.nan

    if returns is not None:
        r = returns if start is None else returns[
            (returns["date"] >= start) & (returns["date"] <= end)
        ]
        sp = sharpe(r["log_return"]) if len(r) > 0 else np.nan
    else:
        sp = np.nan

    return da, sp


def build_freq_csv(freq_label: str, out_name: str) -> None:
    # Pre-load data for all models
    model_data = {}
    for model_key in MODELS:
        model_data[model_key] = {
            "stats": load_statistics(model_key, freq_label),
            "returns": load_portfolio_returns(model_key, freq_label),
        }

    rows = []

    # Overall row (full sample)
    row = {"crisis": "overall"}
    for model_key in MODELS:
        da, sp = compute_metrics(
            model_data[model_key]["stats"],
            model_data[model_key]["returns"],
            start=None, end=None,
        )
        row[f"da_mean_crisis_{model_key}"] = da
        row[f"sharpe_mean_crisis_{model_key}"] = sp
    rows.append(row)

    # One row per crisis (peak to recovery = date1 to date3)
    for full_name, _, date1, _, date3 in CRISIS_PERIODS:
        start = pd.Timestamp(date1)
        end = pd.Timestamp(date3)
        row = {"crisis": full_name}
        for model_key in MODELS:
            da, sp = compute_metrics(
                model_data[model_key]["stats"],
                model_data[model_key]["returns"],
                start=start, end=end,
            )
            row[f"da_mean_crisis_{model_key}"] = da
            row[f"sharpe_mean_crisis_{model_key}"] = sp
        rows.append(row)

    result = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f"ml_crisis_{out_name}.csv")
    result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved ml_crisis_{out_name}.csv  ({len(result)} rows)")


os.makedirs(OUT_DIR, exist_ok=True)

for freq_label, out_name in FREQUENCIES.items():
    build_freq_csv(freq_label, out_name)
