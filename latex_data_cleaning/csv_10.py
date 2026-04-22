import os
import pandas as pd

BASE        = os.path.join(os.path.dirname(__file__), "..", "results")
METRICS_DIR = os.path.join(BASE, "metrics")
OUT_DIR     = os.path.join(BASE, "data", "latex")

FREQ_FILE_MAP = {
    "monthly":     "statistics_Monthly.csv",
    "quarterly":   "statistics_Quarterly.csv",
    "semi_annual": "statistics_Semi-Annual.csv",
    "annual":      "statistics_Yearly.csv",
}

# (model_key, stat_path_fn, columns_to_extract)
MODELS = [
    (
        "xgboost",
        lambda freq_file: os.path.join(METRICS_DIR, "xgboost", freq_file),
        ["best_learning_rate", "best_max_depth", "best_n_estimators",
         "best_reg_lambda", "best_subsample", "best_colsample"],
    ),
    (
        "random_forest",
        lambda freq_file: os.path.join(METRICS_DIR, "random_forest", freq_file),
        ["best_depth", "best_n_estimators", "best_min_leaf"],
    ),
    (
        "lstm",
        lambda freq_file: os.path.join(
            BASE, "data", "lstm",
            f"portfolio_lstm_{freq_file.replace('statistics_', '').replace('.csv', '')}_statistics.csv"
        ),
        ["best_nodes", "best_batch", "best_lr"],
    ),
]


def load_hyperparams(path_fn, freq_file: str, model_key: str, columns: list) -> pd.DataFrame | None:
    path = path_fn(freq_file)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, parse_dates=["rebalance_date"])
    available = [c for c in columns if c in df.columns]
    if not available:
        return None

    result = df[["rebalance_date"] + available].copy()
    result = result.rename(columns={"rebalance_date": "date"})
    result = result.rename(columns={c: f"{c}_{model_key}" for c in available})
    return result.sort_values("date").reset_index(drop=True)


def build_freq_csv(freq: str, freq_file: str) -> None:
    merged = None
    for model_key, path_fn, columns in MODELS:
        df = load_hyperparams(path_fn, freq_file, model_key, columns)
        if df is None:
            print(f"  [{model_key}] {freq_file} not found — skipping")
            continue
        merged = df if merged is None else merged.merge(df, on="date", how="outer")

    if merged is None:
        print(f"No data for {freq}, skipping.")
        return

    merged = merged.sort_values("date").reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, f"ml_hyperparams_{freq}.csv")
    merged.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved ml_hyperparams_{freq}.csv  ({len(merged)} rows, {len(merged.columns)} cols)")


os.makedirs(OUT_DIR, exist_ok=True)

for freq, freq_file in FREQ_FILE_MAP.items():
    build_freq_csv(freq, freq_file)
