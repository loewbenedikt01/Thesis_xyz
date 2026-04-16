import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "results" / "data"
OUTPUT_DIR = BASE / "latex"
OUTPUT_DIR.mkdir(exist_ok=True)

FREQUENCIES = ["Monthly", "Quarterly", "Semi-Annual", "Yearly"]
ML_RUNS = 10  # number of random seeds per ML model


# ── Helpers ──────────────────────────────────────────────────────────────────

def rebalance_turnover(details_path: Path, turnover_col: str) -> pd.Series:
    """
    Load a details CSV and return a Series of turnover per rebalance date.

    Turnover is stored only on the first ticker row per rebalance date
    (it equals the sum of absolute weight changes across all stocks).
    Summing per date recovers the single non-zero value.
    """
    df = pd.read_csv(details_path, parse_dates=["rebalance_date"])
    per_rebalance = (
        df.groupby("rebalance_date")[turnover_col]
        .sum()
        .rename("turnover")
    )
    # Drop the very first rebalance (initial buy: always ~1.0, not a real rebalance)
    per_rebalance = per_rebalance.iloc[1:]
    return per_rebalance


def avg_annual_turnover(per_rebalance: pd.Series) -> float:
    """
    Sum turnovers within each calendar year, then average across years.
    Expressed as a fraction (e.g. 0.35 = 35% of portfolio traded per year).
    """
    annual = per_rebalance.groupby(per_rebalance.index.year).sum()
    return float(annual.mean())


def avg_annual_turnover_ml(paths: list[Path], turnover_col: str) -> float:
    """
    Average the per-run annual turnover across all ML runs.
    """
    run_avgs = []
    for p in paths:
        if not p.exists():
            continue
        per_rebalance = rebalance_turnover(p, turnover_col)
        run_avgs.append(avg_annual_turnover(per_rebalance))
    return float(np.mean(run_avgs)) if run_avgs else np.nan


# ── Per-strategy computation ─────────────────────────────────────────────────

def compute_all(freq: str) -> dict[str, float]:
    results: dict[str, float] = {}

    # Standard strategies (one details file per frequency)
    standard = [
        ("markowitz",               BASE / "markowitz"               / f"portfolio_{freq}_details.csv", "turnover_at_rebalance"),
        ("markowitz_unconstrained", BASE / "markowitz_unconstrained" / f"portfolio_{freq}_details.csv", "turnover_at_rebalance"),
        ("hrp",                     BASE / "hrp"                     / f"portfolio_{freq}_details.csv", "turnover_at_rebalance"),
        ("equal_weight",            BASE / "equal_weight"            / f"portfolio_{freq}_details.csv", "turnover_at_rebalance"),
        ("market_cap",              BASE / "market_cap"              / f"portfolio_{freq}_details.csv", "turnover_at_rebalance"),
    ]
    for name, path, col in standard:
        if not path.exists():
            print(f"  WARNING: {path.name} not found — skipping {name}")
            results[name] = np.nan
            continue
        per_reb = rebalance_turnover(path, col)
        results[name] = avg_annual_turnover(per_reb)
        print(f"  {name:<30} {results[name]:.4f}  ({len(per_reb)} rebalances)")

    # ML strategies (10 runs, average across them)
    ml_models = [
        ("xgboost", BASE / "xgboost",       "portfolio_xgb",  "turnover"),
        ("rf",      BASE / "random_forest",  "portfolio_rf",   "turnover"),
        ("lstm",    BASE / "lstm",           "portfolio_lstm", "turnover"),
    ]
    for name, folder, prefix, col in ml_models:
        paths = [folder / f"{prefix}_{freq}_{i}_details.csv" for i in range(1, ML_RUNS + 1)]
        val = avg_annual_turnover_ml(paths, col)
        results[name] = val
        found = sum(p.exists() for p in paths)
        print(f"  {name:<30} {val:.4f}  ({found}/{ML_RUNS} runs found)")

    # Benchmark: buy-and-hold, no rebalancing
    results["benchmark"] = 0.0
    print(f"  {'benchmark':<30} 0.0000  (buy-and-hold)")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    all_results: dict[str, dict[str, float]] = {}

    for freq in FREQUENCIES:
        print(f"\nFrequency: {freq}")
        all_results[freq] = compute_all(freq)

    # Build output table: rows = strategies, columns = frequencies
    df = pd.DataFrame(all_results)

    # Reorder rows for readability
    row_order = [
        "benchmark", "equal_weight", "market_cap",
        "markowitz", "markowitz_unconstrained", "hrp",
        "xgboost", "rf", "lstm",
    ]
    df = df.reindex([r for r in row_order if r in df.index])

    # Convert to percentage and round
    df_pct = (df * 100).round(2)
    df_pct.index.name = "strategy"

    out_path = OUTPUT_DIR / "turnover_avg_annual.csv"
    df_pct.to_csv(out_path, float_format="%.2f")

    print(f"\n\nAverage Annual Turnover (%) — one-way, excl. initial buy")
    print(df_pct.to_string())
    print(f"\n  Wrote {out_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()