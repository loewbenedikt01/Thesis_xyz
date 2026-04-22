import os
import pandas as pd

BASE        = os.path.join(os.path.dirname(__file__), "..", "results")
METRICS_DIR = os.path.join(BASE, "metrics")
OUT_DIR     = os.path.join(BASE, "data", "latex")

MODELS = ["xgboost", "random_forest", "lstm"]

FREQ_FILE_MAP = {
    "monthly":     "statistics_Monthly.csv",
    "quarterly":   "statistics_Quarterly.csv",
    "semi_annual": "statistics_Semi-Annual.csv",
    "annual":      "statistics_Yearly.csv",
}


def load_da(model: str, filename: str) -> pd.DataFrame | None:
    if model == "lstm":
        freq_label = filename.replace("statistics_", "").replace(".csv", "")
        path = os.path.join(BASE, "data", "lstm", f"portfolio_lstm_{freq_label}_statistics.csv")
    else:
        path = os.path.join(METRICS_DIR, model, filename)

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, parse_dates=["rebalance_date"])
    return (
        df[["rebalance_date", "Directional_Accuracy"]]
        .rename(columns={"rebalance_date": "date", "Directional_Accuracy": f"da_{model}"})
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_freq_csv(freq: str, filename: str) -> None:
    merged = None
    for model in MODELS:
        df = load_da(model, filename)
        if df is None:
            print(f"  [{model}] {filename} not found — skipping")
            continue
        merged = df if merged is None else merged.merge(df, on="date", how="outer")

    if merged is None:
        print(f"No data for {freq}, skipping.")
        return

    merged = merged.sort_values("date").reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, f"ml_da_daily_{freq}.csv")
    merged.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved ml_da_daily_{freq}.csv  ({len(merged)} rows)")


os.makedirs(OUT_DIR, exist_ok=True)

for freq, filename in FREQ_FILE_MAP.items():
    build_freq_csv(freq, filename)
