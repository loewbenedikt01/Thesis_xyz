"""
analyze_benchmark.py — S&P 500 Buy & Hold Analysis
====================================================
Imports all metric functions, crisis definitions, and plot generation
from metrics.py. This script only handles:
  - Paths and settings
  - Data loading and portfolio construction
  - Running compute_all_metrics()
  - Exporting CSV + printing console summary
  - Generating the HTML report

Place metrics.py in the project root (Thesis_xyz/) so the import works.
"""

import sys
import pyarrow
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
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH            = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
benchmark_price_file = DATA_PATH / "benchmark_price.parquet"
output_dir_data      = DATA_PATH / "results" / "data"    / "benchmark"
output_dir_metrics   = DATA_PATH / "results" / "metrics" / "benchmark"
output_dir_plots     = DATA_PATH / "results" / "plots"   / "benchmark"
output_dir_data.mkdir(parents=True, exist_ok=True)
output_dir_metrics.mkdir(parents=True, exist_ok=True)
output_dir_plots.mkdir(parents=True, exist_ok=True)

data_file    = output_dir_data    / "portfolio.csv"
metrics_file = output_dir_metrics / "metrics_buy_hold.csv"
report_path  = output_dir_plots   / "benchmark_interactive_report.html"

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
start_date     = "1998-01-01"
end_date       = "2025-12-31"
ticker         = "^GSPC"
risk_free_rate = 0.0
threshold_val  = 0.05      # drawdown episode threshold for HTML report

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & BUILD PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading data from {benchmark_price_file}...")

df_raw = pd.read_parquet(benchmark_price_file)
df_raw.index = pd.to_datetime(df_raw.index)
df_raw = df_raw.sort_index()

if ticker in df_raw.columns:
    price_series = df_raw[ticker]
elif 'Close' in df_raw.columns:
    price_series = df_raw['Close']
else:
    price_series = df_raw.iloc[:, 0]

price_series = price_series.loc[start_date:end_date]

portfolio = pd.DataFrame(index=price_series.index)
portfolio['ticker']              = ticker
portfolio['price']               = price_series
portfolio['returns_per_day']     = portfolio['price'].pct_change()
portfolio['log_returns_per_day'] = np.log(portfolio['price'] / portfolio['price'].shift(1))

portfolio.index      = portfolio.index.strftime('%Y-%m-%d')
portfolio.index.name = 'date'
portfolio            = portfolio.dropna()
portfolio            = portfolio[['ticker', 'price', 'returns_per_day', 'log_returns_per_day']]

portfolio.to_csv(data_file)
print(f"Portfolio data exported to: {data_file}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — COMPUTE ALL METRICS
# ─────────────────────────────────────────────────────────────────────────────
log_ret = portfolio['log_returns_per_day']
price   = portfolio['price']

results = compute_all_metrics(
    portfolio_log_returns = log_ret,
    price_series          = price,
    benchmark_log_returns = log_ret,   # S&P 500 IS the benchmark for this script
    rfr                   = risk_free_rate,
    freq                  = 'D',
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXPORT METRICS CSV
# ─────────────────────────────────────────────────────────────────────────────
df_metrics = metrics_to_dataframe(results)
df_metrics.to_csv(metrics_file, index=False)
print(f"Metrics exported to: {metrics_file}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
_SEP  = "=" * 72
_SEP2 = "-" * 72

def _fmtv(v, fmt):
    try:
        return fmt.format(v) if not (isinstance(v, float) and v != v) else 'N/A'
    except Exception:
        return str(v)

print(f"\n{_SEP}")
print("  OVERALL METRICS")
print(_SEP)
for _k, _lbl, _fmt in [
    ('cumulative_return',     'Cumulative Return',     '{:.2%}'),
    ('annualized_return',     'Annualized Return',     '{:.2%}'),
    ('annualized_volatility', 'Annualized Volatility', '{:.2%}'),
    ('maximum_drawdown',      'Maximum Drawdown',      '{:.2%}'),
    ('sharpe',                'Sharpe Ratio',          '{:.3f}'),
    ('sortino',               'Sortino Ratio',         '{:.3f}'),
    ('calmar',                'Calmar Ratio',          '{:.3f}'),
    ('omega',                 'Omega Ratio',           '{:.3f}'),
    ('ulcer_index',           'Ulcer Index',           '{:.4f}'),
    ('var_95',                'VaR 95%',               '{:.2%}'),
    ('cvar_95',               'CVaR 95%',              '{:.2%}'),
]:
    print(f"  {_lbl:<30} {_fmtv(results.get(_k, float('nan')), _fmt)}")

print(f"\n{_SEP}")
print("  CRISIS PERIOD METRICS")
print(_SEP)

_crisis_print_keys = [
    ('max_drawdown',            'Max Drawdown',       '{:.2%}'),
    ('days_to_trough',          'Days to Trough',     '{:.0f}'),
    ('days_trough_to_recovery', 'Days to Recovery',   '{:.0f}'),
    ('days_peak_to_breakeven',  'Peak to Breakeven',  '{:.0f}'),
    ('crisis_cum_return',       'Cum Return',         '{:.2%}'),
    ('crisis_ann_return',       'Ann Return',         '{:.2%}'),
    ('crisis_ann_volatility',   'Ann Volatility',     '{:.2%}'),
    ('crisis_sharpe',           'Sharpe',             '{:.3f}'),
    ('crisis_sortino',          'Sortino',            '{:.3f}'),
    ('crisis_calmar',           'Calmar',             '{:.3f}'),
    ('crisis_ulcer_index',      'Ulcer Index',        '{:.4f}'),
]

for _cname, _ckey, _cs, _ct, _ce in CRISIS_PERIODS:
    print(f"\n  {_cname}  [{_cs}  ->  trough {_ct}  ->  {_ce}]")
    print(f"  {_SEP2}")
    for _suffix, _lbl, _fmt in _crisis_print_keys:
        _v = results.get(f'{_ckey}_{_suffix}', float('nan'))
        print(f"    {_lbl:<28} {_fmtv(_v, _fmt)}")

print(f"\n{_SEP}\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GENERATE HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────
# Note: generate_dynamic_benchmark_report returns an HTML string.
# For the benchmark we write it to a standalone file with a full HTML wrapper.
html_body = generate_dynamic_benchmark_report(
    portfolio_df        = portfolio,
    precomputed_results = results,
    title               = "S&P 500  ·  Buy & Hold Performance Analysis",
    threshold           = threshold_val,
)

full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>S&P 500 — Buy & Hold Analysis</title>
</head>
<body style="margin:0;background:#ffffff">
{html_body}
</body>
</html>"""

report_path.write_text(full_html, encoding='utf-8')

print(f"Data file    : {data_file}")
print(f"Metrics file : {metrics_file}")
print(f"Report       : {report_path}")