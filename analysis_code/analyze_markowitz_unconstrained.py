"""
analyze_markowitz.py — Markowitz Portfolio Analysis
=====================================================
Imports all metric functions, crisis definitions, and plot generation
from metrics.py. This script only handles:
  - Paths and settings
  - Loading benchmark and portfolio data
  - Running compute_all_metrics() per frequency
  - Exporting per-frequency CSVs
  - Generating the combined HTML report

Works for both constrained and unconstrained variants — just change
MODEL_NAME and DATA_SUBDIR to match the folder you want to analyse.

Place metrics.py in the project root (Thesis_xyz/) so the import works.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT FROM METRICS.PY
# ─────────────────────────────────────────────────────────────────────────────
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from metrics import (
    CRISIS_PERIODS,
    CRISIS_METRIC_LABELS,
    compute_all_metrics,
    metrics_to_dataframe,
    generate_dynamic_benchmark_report,
    generate_crisis_comparison_charts,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Change MODEL_NAME / DATA_SUBDIR to switch between constrained / unconstrained
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "Markowitz Unconstrained"   # used in HTML titles
DATA_SUBDIR = "markowitz_unconstrained"   # subfolder under results/{data,metrics,plots}

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH          = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
output_dir_metrics = DATA_PATH / "results" / "metrics" / DATA_SUBDIR
output_dir_plots   = DATA_PATH / "results" / "plots"   / DATA_SUBDIR
output_dir_metrics.mkdir(parents=True, exist_ok=True)
output_dir_plots.mkdir(parents=True, exist_ok=True)
DATA_DIR           = DATA_PATH / "results" / "data" / DATA_SUBDIR

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
start_date     = "1998-01-01"
end_date       = "2025-12-31"
risk_free_rate = 0.0
threshold_val  = 0.05

FREQUENCIES = {
    'Yearly':      pd.DateOffset(years=1),
    'Semi-Annual': pd.DateOffset(months=6),
    'Quarterly':   pd.DateOffset(months=3),
    'Monthly':     pd.DateOffset(months=1),
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
benchmark_csv     = DATA_PATH / "results" / "data" / "benchmark" / "portfolio.csv"
df_bench          = pd.read_csv(benchmark_csv, index_col='date', parse_dates=True).sort_index()
df_bench          = df_bench.loc[start_date:end_date].dropna(subset=['log_returns_per_day'])
benchmark_log_ret = df_bench['log_returns_per_day']

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — one iteration per rebalancing frequency
# ─────────────────────────────────────────────────────────────────────────────
all_divs = []

for freq_name in FREQUENCIES:

    csv_path = DATA_DIR / f"portfolio_{freq_name}.csv"
    if not csv_path.exists():
        print(f"Skipping {freq_name}: file not found at {csv_path}")
        continue

    df_port = pd.read_csv(csv_path, index_col='date', parse_dates=True).sort_index()
    df_port = df_port.loc[start_date:end_date].dropna(subset=['log_return'])

    log_ret   = df_port['log_return']
    price_ser = df_port['cumulative_value']

    # ── Step 1: compute metrics ───────────────────────────────────────────────
    results = compute_all_metrics(
        portfolio_log_returns = log_ret,
        price_series          = price_ser,
        benchmark_log_returns = benchmark_log_ret,
        rfr                   = risk_free_rate,
        freq                  = 'D',
    )

    # ── Step 2: export CSV ────────────────────────────────────────────────────
    df_metrics   = metrics_to_dataframe(results)
    metrics_file = output_dir_metrics / f"metrics_{freq_name}.csv"
    df_metrics.to_csv(metrics_file, index=False)
    print(f"[{freq_name}] Metrics exported to: {metrics_file}")

    # ── Step 3: performance report (7-panel HTML) ─────────────────────────────
    section_title = (
        f'<h2 style="font-family:Inter,Arial,sans-serif;color:#1E293B;'
        f'margin:40px 0 8px 70px">{MODEL_NAME}  ·  {freq_name} Rebalancing</h2>'
    )
    div = generate_dynamic_benchmark_report(
        portfolio_df        = df_port,
        precomputed_results = results,
        title               = f"{MODEL_NAME} Portfolio  ·  {freq_name} Rebalancing",
        threshold           = threshold_val,
    )

    # ── Step 4: crisis comparison charts (portfolio vs benchmark) ─────────────
    crisis_header = (
        f'<h3 style="font-family:Inter,Arial,sans-serif;color:#1E293B;'
        f'margin:32px 0 4px 70px">Crisis Period Analysis  ·  {freq_name}</h3>'
    )
    try:
        crisis_div = generate_crisis_comparison_charts(
            portfolio_df = df_port,
            benchmark_df = df_bench,
            title_prefix = f"{freq_name}  ·  ",
        )
    except Exception as e:
        print(f"[{freq_name}] Crisis charts error: {e}")
        crisis_div = (
            f'<p style="font-family:Inter,Arial;color:#DC2626;margin:20px 70px">'
            f'Crisis charts could not be generated: {e}</p>'
        )

    all_divs.append(section_title + div + crisis_header + crisis_div)
    print(f"[{freq_name}] Charts built.")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE COMBINED HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────
combined_report_path = output_dir_plots / "report_all_frequencies.html"

combined_html = (
    f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{MODEL_NAME} Portfolio — All Frequencies</title>
</head>
<body style="margin:0;background:#ffffff">
"""
    + "\n<hr style='border:none;border-top:2px solid #E9ECEF;margin:0 70px'>\n".join(all_divs)
    + """
</body>
</html>"""
)

combined_report_path.write_text(combined_html, encoding='utf-8')
print(f"\nCombined report exported to: {combined_report_path}")