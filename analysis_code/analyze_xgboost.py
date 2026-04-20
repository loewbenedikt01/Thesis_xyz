"""
analyze_xgboost.py — XGBoost Portfolio Analysis
=================================================
Imports all metric functions, crisis definitions, and plot generation
from metrics.py. This script retains XGB-specific logic:
  - load_and_average_runs()    : averages log returns across multiple seeds
  - load_and_average_stats()   : averages prediction quality stats across seeds
  - generate_xgb_statistics_report() : time-series + summary table for DA, Spearman etc.

Place metrics.py in the project root (Thesis_xyz/) so the import works.
"""

import sys
import pandas as pd
import plotly.graph_objects as go
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
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH          = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
output_dir_metrics = DATA_PATH / "results" / "metrics" / "xgboost"
output_dir_plots   = DATA_PATH / "results" / "plots"   / "xgboost"
output_dir_metrics.mkdir(parents=True, exist_ok=True)
output_dir_plots.mkdir(parents=True, exist_ok=True)
DATA_DIR           = DATA_PATH / "results" / "data" / "xgboost"

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
# XGB-SPECIFIC: MULTI-RUN LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def load_runs(freq_name: str):
    """
    Load portfolio_xgb_{freq}.csv (already averaged by the model).
    Returns (df, 1) or (None, 0) if missing.
    """
    p = DATA_DIR / f"portfolio_xgb_{freq_name}.csv"
    if not p.exists():
        return None, 0
    df = pd.read_csv(p, index_col='date', parse_dates=True).sort_index()
    df = df.loc[start_date:end_date].dropna(subset=['log_return'])
    return df[['log_return', 'cumulative_value']], 1


def load_stats(freq_name: str):
    """
    Load portfolio_xgb_{freq}_statistics.csv (already averaged by the model).
    Returns DataFrame or None if missing.
    """
    p = DATA_DIR / f"portfolio_xgb_{freq_name}_statistics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=['rebalance_date'])


# ─────────────────────────────────────────────────────────────────────────────
# XGB-SPECIFIC: STATISTICS REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_xgb_statistics_report(stats_df: pd.DataFrame, freq_name: str) -> str:
    """
    Builds an HTML section with:
      - Time-series chart of all prediction quality metrics over time
      - Summary table (mean, std, min, max per metric)
    Returns HTML string (no CDN tag — caller includes plotlyjs once).
    """
    if stats_df is None or stats_df.empty:
        return ('<p style="font-family:Inter,Arial;color:#64748B;margin:20px 70px">'
                'No statistics data available.</p>')

    BG      = "#FFFFFF"; PANEL = "#F8F9FA"; GRID  = "#E9ECEF"
    TEXT    = "#1E293B"; SUBTEXT = "#64748B"

    # metric_key: (display_label, higher_is_better, colour)
    METRIC_META = {
        'RMSE'                : ('RMSE',                 False, '#1D6FA4'),
        'MSE'                 : ('MSE',                  False, '#7C3AED'),
        'MAE'                 : ('MAE',                  False, '#D97706'),
        'R2_rank'             : ('R² (rank)',             True,  '#16A34A'),
        'R2_raw_vs_zero'      : ('R²oos vs zero',         True,  '#84CC16'),
        'Spearman'            : ('Spearman Correlation',  True,  '#DC2626'),
        'Directional_Accuracy': ('Directional Accuracy', True,  '#0891B2'),
        'Geometric_Score'     : ('Geometric Score',       True,  '#059669'),
    }

    available = [c for c in METRIC_META if c in stats_df.columns]
    if not available:
        return ('<p style="font-family:Inter,Arial;color:#64748B;margin:20px 70px">'
                'No statistics columns found.</p>')

    # ── Time-series chart ─────────────────────────────────────────────────────
    fig_ts = go.Figure()
    for col in available:
        label, _, color = METRIC_META[col]
        s = stats_df.set_index('rebalance_date')[col].dropna()
        fig_ts.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode='lines+markers', name=label,
            line=dict(color=color, width=1.8),
            marker=dict(size=4),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.4f}}<extra></extra>",
        ))

    fig_ts.update_layout(
        title=dict(
            text=f"XGBoost Prediction Quality Over Time  ·  {freq_name}",
            font=dict(size=15, color=TEXT, family='Inter, Arial'),
            x=0.5, xanchor='center',
        ),
        height=420,
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family='Inter, Arial', size=11, color=TEXT),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=10)),
        margin=dict(l=60, r=40, t=70, b=40),
        xaxis=dict(showgrid=False, tickfont=dict(color=SUBTEXT)),
        yaxis=dict(gridcolor=GRID, tickfont=dict(color=SUBTEXT)),
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    rows_m, rows_mean, rows_std, rows_min, rows_max = [], [], [], [], []
    for col in available:
        label, _, _ = METRIC_META[col]
        s = stats_df[col].dropna()
        rows_m.append(label)
        rows_mean.append(f"{s.mean():.4f}")
        rows_std.append(f"{s.std():.4f}")
        rows_min.append(f"{s.min():.4f}")
        rows_max.append(f"{s.max():.4f}")

    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Mean</b>", "<b>Std Dev</b>",
                    "<b>Min</b>", "<b>Max</b>"],
            fill_color='#1D6FA4',
            font=dict(color='white', size=12, family='Inter, Arial'),
            align='left', height=30, line_color='white',
        ),
        cells=dict(
            values=[rows_m, rows_mean, rows_std, rows_min, rows_max],
            fill_color=[['#F8F9FA'] * len(rows_m)] * 5,
            font=dict(color=TEXT, size=11, family='Inter, Arial'),
            align='left', height=26, line_color=GRID,
        ),
    ))
    fig_tbl.update_layout(
        height=60 + 30 * len(rows_m),
        margin=dict(l=60, r=40, t=10, b=10),
        paper_bgcolor=BG,
    )

    return (fig_ts.to_html(full_html=False, include_plotlyjs=False)
            + fig_tbl.to_html(full_html=False, include_plotlyjs=False))


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

    # ── Load portfolio ────────────────────────────────────────────────────────
    df_port, _ = load_runs(freq_name)
    if df_port is None:
        print(f"Skipping {freq_name}: no run files found.")
        continue
    print(f"[{freq_name}] Loaded.")

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

    # ── Step 2: export metrics CSV ────────────────────────────────────────────
    df_metrics   = metrics_to_dataframe(results)
    metrics_file = output_dir_metrics / f"metrics_{freq_name}.csv"
    df_metrics.to_csv(metrics_file, index=False)
    print(f"[{freq_name}] Metrics exported.")

    # ── Step 3: load, average, export statistics ──────────────────────────────
    stats_df = load_stats(freq_name)
    if stats_df is not None:
        stats_file = output_dir_metrics / f"statistics_{freq_name}.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"[{freq_name}] Statistics exported.")

    # ── Step 4: build HTML sections ───────────────────────────────────────────
    section_title = (
        f'<h2 style="font-family:Inter,Arial,sans-serif;color:#1E293B;'
        f'margin:40px 0 8px 70px">XGBoost  ·  {freq_name} Rebalancing</h2>'
    )

    div = generate_dynamic_benchmark_report(
        portfolio_df        = df_port,
        precomputed_results = results,
        title               = f"XGBoost  ·  {freq_name} Rebalancing",
        threshold           = threshold_val,
    )

    stats_header = (
        f'<h3 style="font-family:Inter,Arial,sans-serif;color:#1E293B;'
        f'margin:32px 0 4px 70px">Model Prediction Quality  ·  {freq_name}</h3>'
    )
    stats_div = generate_xgb_statistics_report(stats_df, freq_name)

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

    all_divs.append(
        section_title + div + stats_header + stats_div + crisis_header + crisis_div
    )
    print(f"[{freq_name}] All charts built.")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE COMBINED HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────
combined_report_path = output_dir_plots / "report_all_frequencies.html"

combined_html = (
    """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>XGBoost Portfolio — All Frequencies</title>
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