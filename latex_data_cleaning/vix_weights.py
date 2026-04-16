import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "data" / "latex"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Crisis periods ───────────────────────────────────────────────────────────
# 6 main crises (used for z-score / sensitivity analysis weighting)
# + 3 GFC sub-periods (used for GFC narrative section)
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

# Earliest and latest dates across all periods
DOWNLOAD_START = "1998-01-01"
DOWNLOAD_END   = "2025-12-31"


def download_vix() -> pd.Series:
    """Download daily VIX close prices from Yahoo Finance."""
    print("Downloading VIX (^VIX) from Yahoo Finance...")
    raw = yf.download("^VIX", start=DOWNLOAD_START, end=DOWNLOAD_END,
                      auto_adjust=False, progress=False)
    vix = raw["Close"].squeeze()
    vix.index = pd.to_datetime(vix.index)
    vix.name = "vix"
    print(f"  Downloaded {len(vix)} trading days  "
          f"({vix.index[0].date()} -> {vix.index[-1].date()})")
    return vix


def compute_vix_stats(vix: pd.Series) -> pd.DataFrame:
    """
    For each crisis period compute:
      - avg_vix      : mean VIX over the window (used as sensitivity weight)
      - max_vix      : peak VIX (useful for GFC narrative)
      - min_vix      : trough VIX
      - std_vix      : volatility of VIX itself
      - trading_days : number of observations in the window
    """
    rows = []
    for period_name, start_str, end_str in CRISIS_PERIODS:
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str)
        window = vix.loc[start:end]

        if window.empty:
            print(f"  WARNING: no VIX data for {period_name} [{start_str} → {end_str}]")
            rows.append({
                "period":        period_name,
                "avg_vix":       np.nan,
                "max_vix":       np.nan,
                "min_vix":       np.nan,
                "std_vix":       np.nan,
                "trading_days":  0,
            })
        else:
            rows.append({
                "period":        period_name,
                "avg_vix":       round(window.mean(), 4),
                "max_vix":       round(window.max(), 4),
                "min_vix":       round(window.min(), 4),
                "std_vix":       round(window.std(ddof=1), 4),
                "trading_days":  len(window),
            })
            print(f"  {period_name:<35}  avg={window.mean():.2f}  "
                  f"max={window.max():.2f}  days={len(window)}")

    return pd.DataFrame(rows).set_index("period")


def main() -> None:
    vix = download_vix()

    df = compute_vix_stats(vix)

    out_path = OUTPUT_DIR / "vix_crisis_stats.csv"
    df.to_csv(out_path, float_format="%.4f")
    print(f"\n  Wrote {out_path.name}  ({len(df)} periods)")

    # Print a quick summary for sanity-checking
    print("\nSummary (avg VIX per period):")
    print(df[["avg_vix", "max_vix", "trading_days"]].to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()