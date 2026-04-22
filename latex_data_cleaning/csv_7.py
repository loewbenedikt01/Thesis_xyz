import os
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


def load_cum_log_return(data_subdir, file_tpl, freq_label):
    path = os.path.join(DATA_DIR, data_subdir, file_tpl.replace("{freq}", freq_label))
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    df["cum_log_return"] = df["log_return"].cumsum()
    return df[["cum_log_return"]]


def build_freq_csv(freq_label, out_name):
    combined = None
    for key, data_subdir, file_tpl in MODELS:
        series = load_cum_log_return(data_subdir, file_tpl, freq_label)
        if series is None:
            print(f"  [{key}] file not found — skipping")
            continue
        series = series.rename(columns={"cum_log_return": f"cum_log_return_{key}"})
        combined = series if combined is None else combined.join(series, how="outer")

    if combined is None:
        print(f"No data for {freq_label}, skipping.")
        return

    combined = combined.sort_index().reset_index().rename(columns={"date": "date"})
    out_path = os.path.join(OUT_DIR, f"cum_log_returns_{out_name}.csv")
    combined.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved cum_log_returns_{out_name}.csv  ({len(combined)} rows, {len(combined.columns)} cols)")


os.makedirs(OUT_DIR, exist_ok=True)

for freq_label, out_name in FREQUENCIES.items():
    build_freq_csv(freq_label, out_name)
