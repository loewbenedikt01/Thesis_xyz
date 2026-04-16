"""
Jobson-Korkie-Memmel (JKM) Sharpe Ratio Significance Test
==========================================================
Tests H0: SR_i == SR_j for all pairs of strategies.

Reference:
    Memmel, C. (2003). Performance Hypothesis Testing with the Sharpe Ratio.
    Finance Letters, 1(1), 21-23.  (corrects the original Jobson-Korkie 1981 formula)

The z-statistic is:
    z = (SR_i - SR_j) / sqrt(V / T)

where (per-period Sharpe ratios, T = number of daily observations):
    V = 2 - 2*rho + (SR_i^2 + SR_j^2)/2
          - rho^2 * (SR_i^2 + SR_j^2)/2
          + rho * SR_i * SR_j

    rho = correlation(r_i, r_j)

Working with per-period (daily) returns keeps the formula exact.
Annualised Sharpe ratios are reported separately for context only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

BASE = Path(__file__).parent.parent / "results" / "data"
OUTPUT_DIR = BASE / "latex"
OUTPUT_DIR.mkdir(exist_ok=True)

FREQUENCY = "Monthly"   # primary frequency chosen for the thesis


# ── Model configuration (same as other scripts) ──────────────────────────────
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


def load_returns(freq: str) -> pd.DataFrame:
    """Load daily log returns for all models into a single aligned DataFrame."""
    configs = get_model_configs(freq)
    series: dict[str, pd.Series] = {}
    for name, folder, filename, col in configs:
        path = BASE / folder / filename
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").set_index("date")
            series[name] = df[col].rename(name)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {name}")

    # Inner join: only dates present in ALL models
    aligned = pd.DataFrame(series).dropna()
    print(f"  Loaded {len(aligned)} overlapping trading days across {len(aligned.columns)} models")
    return aligned


# ── JKM test ─────────────────────────────────────────────────────────────────

def jkm_test(r_i: np.ndarray, r_j: np.ndarray) -> tuple[float, float, float, float]:
    """
    Jobson-Korkie-Memmel test for equality of two Sharpe ratios.

    Parameters
    ----------
    r_i, r_j : daily log return arrays (same length)

    Returns
    -------
    sr_i      : annualised Sharpe ratio of i
    sr_j      : annualised Sharpe ratio of j
    z_stat    : JKM z-statistic (asymptotically N(0,1) under H0)
    p_value   : two-tailed p-value
    """
    T = len(r_i)
    mu_i, mu_j   = r_i.mean(), r_j.mean()
    sig_i, sig_j = r_i.std(ddof=1), r_j.std(ddof=1)

    # Per-period Sharpe ratios (used in the variance formula)
    sr_i_pp = mu_i / sig_i
    sr_j_pp = mu_j / sig_j

    # Correlation between the two return series
    rho = np.corrcoef(r_i, r_j)[0, 1]

    # Memmel (2003) asymptotic variance of sqrt(T) * (SR_i - SR_j)
    V = (2 - 2 * rho
         + (sr_i_pp**2 + sr_j_pp**2) / 2
         - (rho**2) * (sr_i_pp**2 + sr_j_pp**2) / 2
         + rho * sr_i_pp * sr_j_pp)

    # Guard against numerical zero
    if V <= 0:
        return np.nan, np.nan, np.nan, np.nan

    z = (sr_i_pp - sr_j_pp) / np.sqrt(V / T)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Annualised Sharpe ratios for reporting (multiply per-period SR by sqrt(252))
    sr_i_ann = sr_i_pp * np.sqrt(252)
    sr_j_ann = sr_j_pp * np.sqrt(252)

    return sr_i_ann, sr_j_ann, float(z), p


def significance_stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ── Build output tables ───────────────────────────────────────────────────────

def run_all_tests(returns: pd.DataFrame) -> None:
    models   = list(returns.columns)
    r        = returns.values  # shape (T, n_models)
    col_idx  = {m: i for i, m in enumerate(models)}

    # ── 1. Full pairwise z-stat and p-value matrices ─────────────────────────
    z_mat = pd.DataFrame(np.nan, index=models, columns=models)
    p_mat = pd.DataFrame(np.nan, index=models, columns=models)

    for m_i, m_j in combinations(models, 2):
        i, j = col_idx[m_i], col_idx[m_j]
        _, _, z, p = jkm_test(r[:, i], r[:, j])
        z_mat.loc[m_i, m_j] = round(z, 4)
        z_mat.loc[m_j, m_i] = round(-z, 4)   # antisymmetric
        p_mat.loc[m_i, m_j] = round(p, 4)
        p_mat.loc[m_j, m_i] = round(p, 4)    # symmetric

    # Diagonal = 0 for z-mat, 1.0 for p-mat
    for m in models:
        z_mat.loc[m, m] = 0.0
        p_mat.loc[m, m] = 1.0

    freq_tag = FREQUENCY.lower()
    z_mat.index.name = "strategy"
    p_mat.index.name = "strategy"

    z_mat.to_csv(OUTPUT_DIR / f"jkm_zstats_{freq_tag}.csv", float_format="%.4f")
    p_mat.to_csv(OUTPUT_DIR / f"jkm_pvalues_{freq_tag}.csv", float_format="%.4f")
    print(f"  Wrote jkm_zstats_{freq_tag}.csv")
    print(f"  Wrote jkm_pvalues_{freq_tag}.csv")

    # ── 2. Each strategy vs benchmark ────────────────────────────────────────
    bench_idx = col_idx["benchmark"]
    vs_bench_rows = []
    for m in models:
        if m == "benchmark":
            continue
        i = col_idx[m]
        sr_m, sr_b, z, p = jkm_test(r[:, i], r[:, bench_idx])
        vs_bench_rows.append({
            "strategy":          m,
            "sharpe_annualised": round(sr_m, 4),
            "benchmark_sharpe":  round(sr_b, 4),
            "z_stat":            round(z, 4),
            "p_value":           round(p, 4),
            "significance":      significance_stars(p),
        })

    df_vs_bench = pd.DataFrame(vs_bench_rows).set_index("strategy")
    df_vs_bench.to_csv(OUTPUT_DIR / f"jkm_vs_benchmark_{freq_tag}.csv", float_format="%.4f")
    print(f"  Wrote jkm_vs_benchmark_{freq_tag}.csv")

    # ── 3. Top model vs runner-up ─────────────────────────────────────────────
    # Rank by annualised Sharpe
    sharpe_map: dict[str, float] = {}
    for m in models:
        i = col_idx[m]
        sr = returns.iloc[:, i].mean() / returns.iloc[:, i].std(ddof=1) * np.sqrt(252)
        sharpe_map[m] = sr

    ranked = sorted(sharpe_map.items(), key=lambda x: x[1], reverse=True)
    print("\n  Annualised Sharpe ranking:")
    for rank, (m, sr) in enumerate(ranked, 1):
        print(f"    {rank}. {m:<30}  SR = {sr:.4f}")

    top, runner_up = ranked[0][0], ranked[1][0]
    sr_top, sr_ru, z, p = jkm_test(
        r[:, col_idx[top]], r[:, col_idx[runner_up]]
    )
    top_vs_ru = pd.DataFrame([{
        "comparison":            f"{top} vs {runner_up}",
        "top_sharpe":            round(sr_top, 4),
        "runner_up_sharpe":      round(sr_ru, 4),
        "z_stat":                round(z, 4),
        "p_value":               round(p, 4),
        "significance":          significance_stars(p),
    }]).set_index("comparison")

    top_vs_ru.to_csv(OUTPUT_DIR / f"jkm_top_vs_runnerup_{freq_tag}.csv", float_format="%.4f")
    print(f"\n  Top: {top}  vs  Runner-up: {runner_up}")
    print(f"  z = {z:.4f},  p = {p:.4f}  {significance_stars(p)}")
    print(f"  Wrote jkm_top_vs_runnerup_{freq_tag}.csv")

    # ── 4. Print vs-benchmark table to console ────────────────────────────────
    print(f"\n  Each strategy vs benchmark (annualised Sharpe, two-tailed JKM test):")
    print(f"  {'Strategy':<30} {'SR':>7} {'Bench SR':>9} {'z':>8} {'p':>8}  Sig")
    print(f"  {'-'*30} {'-'*7} {'-'*9} {'-'*8} {'-'*8}  ---")
    for row in vs_bench_rows:
        print(f"  {row['strategy']:<30} "
              f"{row['sharpe_annualised']:>7.4f} "
              f"{row['benchmark_sharpe']:>9.4f} "
              f"{row['z_stat']:>8.4f} "
              f"{row['p_value']:>8.4f}  "
              f"{row['significance']}")


def main(frequency: str = FREQUENCY) -> None:
    print(f"Frequency: {frequency}\n")
    returns = load_returns(frequency)
    print()
    run_all_tests(returns)
    print("\nDone.")
    print("\nSignificance: *** p<0.01  ** p<0.05  * p<0.10")


if __name__ == "__main__":
    main(FREQUENCY)
