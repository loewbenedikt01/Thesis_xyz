import os
import numpy as np
import pandas as pd

BASE     = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "data", "latex")

FREQUENCIES = [
    ("Monthly",     "monthly"),
    ("Quarterly",   "quarterly"),
    ("Semi-Annual", "semi_annual"),
    ("Yearly",      "annual"),
]

# (display_name, source, data_subdir, file_template)
# source: "statistics" → results/metrics/{data_subdir}/statistics_{freq}.csv
#         "details"    → results/data/{data_subdir}/{file_template}
MODELS = [
    ("xgboost",       "statistics", "xgboost",                ""),
    ("random_forest", "statistics", "random_forest",           ""),
    ("lstm",          "details",    "lstm",                    "portfolio_lstm_{freq}_details.csv"),
    ("equal_weight",  "details",    "equal_weight",            "portfolio_ew_{freq}_details.csv"),
    ("hrp",           "details",    "hrp",                     "portfolio_hrp_{freq}_details.csv"),
    ("market_cap",    "details",    "market_cap",              "portfolio_mktcap_{freq}_details.csv"),
    ("markowitz",     "details",    "markowitz",               "portfolio_markowitz_{freq}_details.csv"),
    ("markowitz_unc", "details",    "markowitz_unconstrained", "portfolio_markowitz_{freq}_details.csv"),
]
MODEL_KEYS = [m[0] for m in MODELS]


def load_turnover(source, data_subdir, file_tpl, freq_label):
    """
    Returns a DataFrame with columns [invest_year, turnover],
    one row per rebalancing date (already deduplicated).
    Turnover is the portfolio-level turnover at that rebalancing.
    """
    if source == "statistics":
        path = os.path.join(BASE, "metrics", data_subdir, f"statistics_{freq_label}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, usecols=["rebalance_date", "invest_year", "turnover"])
        # Statistics files already have one row per rebalancing date
        return df[["invest_year", "turnover"]].copy()

    else:  # details
        path = os.path.join(DATA_DIR, data_subdir, file_tpl.replace("{freq}", freq_label))
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, usecols=["rebalance_date", "invest_year", "turnover"])
        # Details files have one row per ticker — deduplicate to one row per rebalancing date
        df = df.drop_duplicates(subset="rebalance_date")[["invest_year", "turnover"]]
        return df.copy()


def avg_annual_turnover(df):
    """Sum turnover within each year, then average across years."""
    if df is None or len(df) == 0:
        return np.nan
    annual = df.groupby("invest_year")["turnover"].sum()
    return float(annual.mean())


def build_csv():
    rows = []
    for freq_label, freq_out in FREQUENCIES:
        row = {"frequency": freq_out}
        for key, source, data_subdir, file_tpl in MODELS:
            df = load_turnover(source, data_subdir, file_tpl, freq_label)
            row[f"turn_{key}"] = avg_annual_turnover(df)
        rows.append(row)

    result = pd.DataFrame(rows, columns=["frequency"] + [f"turn_{k}" for k in MODEL_KEYS])
    out_path = os.path.join(OUT_DIR, "turnover_summary.csv")
    result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved turnover_summary.csv  ({len(result)} rows)")


os.makedirs(OUT_DIR, exist_ok=True)
build_csv()
