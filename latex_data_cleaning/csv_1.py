import os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "results")
METRICS_DIR = os.path.join(BASE, "metrics")
OUT_DIR = os.path.join(BASE, "data", "latex")

MODELS = ["xgboost", "random_forest", "lstm"]

FREQ_FILE_MAP = {
    "monthly": "statistics_Monthly.csv",
    "quarterly": "statistics_Quarterly.csv",
    "semi_annual": "statistics_Semi-Annual.csv",
    "annual": "statistics_Yearly.csv",
}

AGG = {
    "Directional_Accuracy": ["mean", "min", "max"],
    "Spearman": ["mean"],
    "MAE": ["mean", "std"],
    "R2_rank": ["mean", "std"],
}

COL_RENAME = {
    "Directional_Accuracy_mean": "da_mean",
    "Directional_Accuracy_min": "da_min",
    "Directional_Accuracy_max": "da_max",
    "Spearman_mean": "spearman_mean",
    "MAE_mean": "mae_mean",
    "MAE_std": "mae_sd",
    "R2_rank_mean": "r2_rank_mean",
    "R2_rank_std": "r2_rank_sd",
}


def load_and_aggregate(model: str, filename: str) -> pd.DataFrame | None:
    if model == "lstm":
        # lstm statistics live in data/lstm/ as portfolio_lstm_{freq}_statistics.csv
        freq_label = filename.replace("statistics_", "").replace(".csv", "")
        path = os.path.join(BASE, "data", "lstm", f"portfolio_lstm_{freq_label}_statistics.csv")
    else:
        path = os.path.join(METRICS_DIR, model, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    agg = df.groupby("invest_year").agg(AGG)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.rename(columns=COL_RENAME).reset_index().rename(columns={"invest_year": "date"})
    agg = agg.rename(columns={c: f"{c}_{model}" for c in agg.columns if c != "date"})
    return agg


def build_freq_csv(freq: str, filename: str) -> None:
    merged = None
    for model in MODELS:
        df = load_and_aggregate(model, filename)
        if df is None:
            print(f"  [{model}] {filename} not found — skipping")
            continue
        merged = df if merged is None else merged.merge(df, on="date", how="outer")

    if merged is None:
        print(f"No data for {freq}, skipping.")
        return

    merged = merged.sort_values("date").reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, f"{freq}.csv")
    merged.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved {freq}.csv  ({len(merged)} rows, {len(merged.columns)} cols)")


os.makedirs(OUT_DIR, exist_ok=True)

for freq, filename in FREQ_FILE_MAP.items():
    build_freq_csv(freq, filename)
