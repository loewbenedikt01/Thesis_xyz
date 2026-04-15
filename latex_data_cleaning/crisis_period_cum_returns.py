import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "results" / "data"
OUTPUT_DIR = BASE / "latex"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Configurable frequency ──────────────────────────────────────────────────
FREQUENCY =  "Quarterly"  # options: "Monthly", "Quarterly", "Semi-Annual", "Yearly"

# ── Crisis periods (name, start, end) ───────────────────────────────────────
CRISIS_PERIODS = [
    ("dotcom",                           "2000-03-23", "2007-05-31"),
    ("gfc",                              "2007-10-09", "2013-03-28"),
    ("early_credit_crunch",              "2007-10-09", "2008-09-15"),
    ("acute_gfc_crash",                  "2008-09-15", "2010-04-23"),
    ("european_debt_us_debt_ceiling",    "2010-04-23", "2012-03-23"),
    ("monetary_policy",                  "2018-09-21", "2019-04-23"),
    ("covid19",                          "2020-02-19", "2020-08-12"),
    ("inflation_rate_hike",              "2022-01-03", "2024-01-19"),
    ("trade_policy_shock",               "2025-02-19", "2025-06-26"),
]

# ── Model configuration ──────────────────────────────────────────────────────
# Each entry: (column_suffix, folder, filename, log_return_col)
# ML models (xgboost, rf, lstm) use the _avg file; others use their frequency file.
def get_model_configs(freq: str) -> list[tuple[str, str, str, str]]:
    return [
        ("benchmark",                "benchmark",              "portfolio.csv",                    "log_returns_per_day"),
        ("markowitz",                "markowitz",              f"portfolio_{freq}.csv",             "log_return"),
        ("markowitz_unconstrained",  "markowitz_unconstrained", f"portfolio_{freq}.csv",            "log_return"),
        ("hrp",                      "hrp",                    f"portfolio_{freq}.csv",             "log_return"),
        ("equal_weight",             "equal_weight",           f"portfolio_{freq}.csv",             "log_return"),
        ("market_cap",               "market_cap",             f"portfolio_{freq}.csv",             "log_return"),
        ("xgboost",                  "xgboost",                f"portfolio_xgb_{freq}_avg.csv",     "log_return"),
        ("rf",                       "random_forest",          f"portfolio_rf_{freq}_avg.csv",      "log_return"),
        ("lstm",                     "lstm",                   f"portfolio_lstm_{freq}_avg.csv",    "log_return"),
    ]


def load_log_returns(folder: str, filename: str, log_return_col: str) -> pd.Series:
    """Load a model's log returns as a date-indexed Series."""
    path = BASE / folder / filename
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df[log_return_col].rename(folder)


def build_crisis_csv(crisis_name: str, start_str: str, end_str: str, freq: str) -> None:
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str)

    model_configs = get_model_configs(freq)
    series_dict: dict[str, pd.Series] = {}

    for col_suffix, folder, filename, log_return_col in model_configs:
        try:
            lr = load_log_returns(folder, filename, log_return_col)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {col_suffix}")
            continue

        # Slice to crisis window (using nearest available dates)
        lr_crisis = lr.loc[start:end]
        if lr_crisis.empty:
            print(f"  WARNING: no data for {col_suffix} in [{start_str}, {end_str}]")
            continue

        # Cumulative log return starting at 0 on the first date.
        # shift(1) so the entry date shows 0 and each subsequent date
        # accumulates the returns earned from entry up to (but not including)
        # that day — i.e. the value on date[i] is the total return realised
        # between date[0] and date[i-1].
        cum = lr_crisis.cumsum().shift(1, fill_value=0.0)
        series_dict[f"cum_log_return_{col_suffix}"] = cum

    if not series_dict:
        print(f"  ERROR: no data loaded for {crisis_name}, skipping.")
        return

    # Align on dates (inner join — keep only dates present in ALL models)
    df_out = pd.DataFrame(series_dict)
    df_out.index.name = "date"
    df_out = df_out.dropna()  # drop any row where a model has no data

    # Format date column
    df_out.index = df_out.index.strftime("%Y-%m-%d")
    df_out = df_out.reset_index()

    out_path = OUTPUT_DIR / f"crisis_{crisis_name}_{freq.lower()}.csv"
    df_out.to_csv(out_path, index=False, float_format="%.10f")
    print(f"  Wrote {out_path.name}  ({len(df_out)} rows, {len(df_out.columns)-1} models)")


def main(frequency: str = FREQUENCY) -> None:
    print(f"Frequency: {frequency}\n")
    for crisis_name, start, end in CRISIS_PERIODS:
        print(f"Processing {crisis_name} [{start} -> {end}]")
        build_crisis_csv(crisis_name, start, end, frequency)
    print("\nDone.")


if __name__ == "__main__":
    main(FREQUENCY)
