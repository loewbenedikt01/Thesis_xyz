import os
import glob
import pandas as pd

BASE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DATA_DIR = os.path.join(BASE, "data", "lstm")
OUT_DIR  = os.path.join(BASE, "metrics", "lstm")

# Columns that identify a rebalancing period — not averaged, taken from first run
STAT_ID_COLS = ["rebalance_date", "invest_year", "select_year", "n_stocks"]
DET_ID_COLS  = ["rebalance_date", "invest_year", "select_year", "ticker"]

FREQUENCIES = {
    "Monthly":     "Monthly",
    "Quarterly":   "Quarterly",
    "Semi-Annual": "Semi-Annual",
    "Yearly":      "Yearly",
}


def average_statistics(freq_label: str) -> None:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"portfolio_lstm_{freq_label}_run*_statistics.csv")))
    if not files:
        print(f"  [statistics/{freq_label}] No run files found — skipping.")
        return

    combined     = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    skip_cols    = set(STAT_ID_COLS) | {"run_number", "seed"}
    numeric_cols = [c for c in combined.columns if c not in skip_cols]

    id_part  = combined.groupby("rebalance_date")[STAT_ID_COLS[1:]].first().reset_index()
    avg_part = combined.groupby("rebalance_date")[numeric_cols].mean().reset_index()

    result = (
        id_part.merge(avg_part, on="rebalance_date")
        .sort_values("rebalance_date")
        .reset_index(drop=True)
    )
    out_path = os.path.join(DATA_DIR, f"portfolio_lstm_{freq_label}_statistics.csv")
    result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  [statistics/{freq_label}] Saved {os.path.basename(out_path)}  ({len(result)} rows, {len(files)} run(s))")


def average_details(freq_label: str) -> None:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"portfolio_lstm_{freq_label}_run*_details.csv")))
    if not files:
        print(f"  [details/{freq_label}] No run files found — skipping.")
        return

    combined     = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    skip_cols    = set(DET_ID_COLS) | {"run_number", "seed"}
    numeric_cols = [c for c in combined.columns if c not in skip_cols]

    id_part  = combined.groupby(["rebalance_date", "ticker"])[DET_ID_COLS[1:3]].first().reset_index()
    avg_part = combined.groupby(["rebalance_date", "ticker"])[numeric_cols].mean().reset_index()

    result = (
        id_part.merge(avg_part, on=["rebalance_date", "ticker"])
        .sort_values(["rebalance_date", "ticker"])
        .reset_index(drop=True)
    )
    out_path = os.path.join(DATA_DIR, f"portfolio_lstm_{freq_label}_details.csv")
    result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  [details/{freq_label}] Saved {os.path.basename(out_path)}  ({len(result)} rows, {len(files)} run(s))")


def average_portfolio_returns(freq_label: str) -> None:
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, f"portfolio_lstm_{freq_label}_run*.csv")))
    files = [f for f in all_files if "_statistics" not in f and "_details" not in f]
    if not files:
        print(f"  [returns/{freq_label}] No run files found — skipping.")
        return

    combined = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    result = (
        combined.groupby("date")[["log_return", "cumulative_value"]]
        .mean()
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    out_path = os.path.join(DATA_DIR, f"portfolio_lstm_{freq_label}.csv")
    result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  [returns/{freq_label}] Saved {os.path.basename(out_path)}  ({len(result)} rows, {len(files)} run(s))")


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

for freq_label, out_suffix in FREQUENCIES.items():
    print(f"\n[{freq_label}]")
    average_statistics(freq_label)
    average_details(freq_label)
    average_portfolio_returns(freq_label)

print("\nDone.")
