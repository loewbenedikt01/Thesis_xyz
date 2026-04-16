import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "results" / "data"
OUTPUT_DIR = BASE / "latex"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Configurable frequency ──────────────────────────────────────────────────
FREQUENCY = "Monthly"  # options: "Monthly", "Quarterly", "Semi-Annual", "Yearly"

# ── Crisis periods: 6 main + 3 GFC sub-periods ─────────────────────────────
CRISIS_PERIODS = [
    ("dotcom",                        "2000-03-23", "2007-05-31"),
    ("gfc",                           "2007-10-09", "2013-03-28"),
    ("early_credit_crunch",           "2007-10-09", "2008-09-15"),
    ("acute_gfc_crash",               "2008-09-15", "2010-04-23"),
    ("european_debt_us_debt_ceiling", "2010-04-23", "2012-03-23"),
    ("monetary_policy",               "2018-09-21", "2019-04-23"),
    ("covid19",                       "2020-02-19", "2020-08-12"),
    ("inflation_rate_hike",           "2022-01-03", "2024-01-19"),
    ("trade_policy_shock",            "2025-02-19", "2025-06-26"),
]

FULL_PERIOD = ("full_period", "1998-01-01", "2025-12-31")


# ── Model configuration ──────────────────────────────────────────────────────
def get_model_configs(freq: str) -> list[tuple[str, str, str, str]]:
    return [
        ("benchmark",               "benchmark",               "portfolio.csv",                 "log_returns_per_day"),
        ("markowitz",               "markowitz",               f"portfolio_{freq}.csv",          "log_return"),
        ("markowitz_unconstrained", "markowitz_unconstrained", f"portfolio_{freq}.csv",          "log_return"),
        ("hrp",                     "hrp",                     f"portfolio_{freq}.csv",          "log_return"),
        ("equal_weight",            "equal_weight",            f"portfolio_{freq}.csv",          "log_return"),
        ("market_cap",              "market_cap",              f"portfolio_{freq}.csv",          "log_return"),
        ("xgboost",                 "xgboost",                 f"portfolio_xgb_{freq}_avg.csv",  "log_return"),
        ("rf",                      "random_forest",           f"portfolio_rf_{freq}_avg.csv",   "log_return"),
        ("lstm",                    "lstm",                    f"portfolio_lstm_{freq}_avg.csv", "log_return"),
    ]


def load_log_returns(folder: str, filename: str, log_return_col: str) -> pd.Series:
    """Load a model's log returns as a date-indexed Series."""
    path = BASE / folder / filename
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df[log_return_col].rename(folder)


def ulcer_index(log_returns: pd.Series) -> float:
    """
    Compute the Ulcer Index from a series of log returns.

    Formula:
        1. Convert log returns to a cumulative price level (starting at 1.0).
        2. Compute the running peak (maximum price seen up to each day).
        3. Drawdown (%) from peak: DD_t = (price_t - peak_t) / peak_t * 100
           This is always <= 0; at a new peak it equals 0.
        4. UI = sqrt( mean(DD_t²) )

    A higher UI indicates more sustained or severe drawdowns over the period.
    The running peak resets to the first observation in the window, so crisis-
    period UI reflects only intra-crisis drawdown severity.
    """
    price = np.exp(log_returns.cumsum())
    peak = price.cummax()
    drawdown_pct = (price - peak) / peak * 100  # <= 0
    return float(np.sqrt(np.mean(drawdown_pct ** 2)))


def compute_ulcer_indices(freq: str) -> None:
    model_configs = get_model_configs(freq)

    # Load all return series once
    all_series: dict[str, pd.Series] = {}
    for col_suffix, folder, filename, log_return_col in model_configs:
        try:
            all_series[col_suffix] = load_log_returns(folder, filename, log_return_col)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {col_suffix}")

    if not all_series:
        print("ERROR: no data loaded.")
        return

    # ── Crisis periods ──────────────────────────────────────────────────────
    crisis_rows = []
    for period_name, start_str, end_str in CRISIS_PERIODS:
        start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
        row: dict = {"period": period_name}
        for model_name, sr in all_series.items():
            window = sr.loc[start:end]
            if window.empty:
                print(f"  WARNING: no data for {model_name} in {period_name}")
                row[model_name] = np.nan
            else:
                row[model_name] = ulcer_index(window)
        crisis_rows.append(row)
        print(f"  Computed {period_name}")

    df_crisis = pd.DataFrame(crisis_rows).set_index("period")
    out_crisis = OUTPUT_DIR / f"ulcer_index_crisis_{freq.lower()}.csv"
    df_crisis.to_csv(out_crisis, float_format="%.6f")
    print(f"\n  Wrote {out_crisis.name}  ({len(df_crisis)} periods, {len(df_crisis.columns)} models)")

    # ── Full period (1998-2025) ─────────────────────────────────────────────
    period_name, start_str, end_str = FULL_PERIOD
    start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
    full_row: dict = {"period": period_name}
    for model_name, sr in all_series.items():
        window = sr.loc[start:end]
        if window.empty:
            full_row[model_name] = np.nan
        else:
            full_row[model_name] = ulcer_index(window)

    df_full = pd.DataFrame([full_row]).set_index("period")
    out_full = OUTPUT_DIR / f"ulcer_index_full_{freq.lower()}.csv"
    df_full.to_csv(out_full, float_format="%.6f")
    print(f"  Wrote {out_full.name}")


def main(frequency: str = FREQUENCY) -> None:
    print(f"Frequency: {frequency}\n")
    compute_ulcer_indices(frequency)
    print("\nDone.")


if __name__ == "__main__":
    main(FREQUENCY)