"""
Portfolio Scoring System
========================
Computes Crisis Scores (CS), Overall Score (OS), and Total Score for each
model × rebalancing-frequency combination, following the formulas:

    ScoreTotal = (mean(CS_c, c=1..6) × 0.80) + (OS × 0.20)
    CS_c       = Σ σ(m, c)   for m in 7 crisis metrics
    OS         = Σ σ(k)      for k in 7 overall metrics

Folder structure expected
─────────────────────────
results/
  metrics/
    benchmark/
      metrics_Monthly.csv
      metrics_Quarterly.csv
      metrics_Semi-Annual.csv
      metrics_Yearly.csv
    equal_weight/   (same files)
    hrp/            ...
    lstm/
    market_cap/
    markowitz/
    markowitz_uncon.../
    random_forest/
    xgboost/

CSV layout expected (metrics_<Freq>.csv)
─────────────────────────────────────────
Column 0  : period label  e.g. "Dotcom Crash", "GFC", …, "Overall"
Columns 1+: metric values with headers matching METRIC_COLS / OVERALL_COLS

The six crisis rows must carry one of these labels (case-insensitive match):
  'dotcom crash', 'gfc', 'monetary policy', 'covid-19',
  'russia/ukraine', 'trade policy shock'

The overall row must contain 'overall' somewhere in its label.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

RESULTS_DIR   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz\results")
METRICS_DIR   = RESULTS_DIR / "metrics"
BENCHMARK_DIR = METRICS_DIR / "benchmark"
DATA_DIR      = RESULTS_DIR / "data"        # raw portfolio CSVs (for Q1 computation)
OUTPUT_DIR    = RESULTS_DIR / "plots" / "all_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCIES = ["Monthly", "Quarterly", "Semi-Annual", "Yearly"]

MODELS = [
    "equal_weight",
    "hrp",
    "lstm",
    "market_cap",
    "markowitz",
    "markowitz_unconstrained",   # adjust to exact folder name
    "random_forest",
    "xgboost",
]

# ── Column names inside the metrics CSVs ──────────────────────────────────────
# Crisis-period metrics (relative scoring, except Sharpe & Sortino)
CRISIS_RELATIVE_COLS = [
    "Max Drawdown (%)",
    "Days: Trough to Breakeven",
    "Days: Peak to Breakeven (Total)",
    "Cumulative Return (Crisis)",
    "Annualized Volatility (Crisis)",
]
CRISIS_RATIO_COLS = [      # use absolute-difference scoring
    "Sharpe Ratio (Crisis)",
    "Sortino Ratio (Crisis)",
]
CRISIS_METRIC_COLS = CRISIS_RELATIVE_COLS + CRISIS_RATIO_COLS   # 7 metrics

# Overall metrics (same scoring logic as crisis relative / ratio)
OVERALL_RELATIVE_COLS = [
    "Annualized Return",
    "Annualized Volatility",
    "Value at Risk (95%)",
    "CVaR / Expected Shortfall",
    "1st Quartile Average Drawdown",
    "1st Quartile Average Duration to Trough",
    "1st Quartile Average Recovery Duration",
]
OVERALL_RATIO_COLS = []    # none for overall
OVERALL_METRIC_COLS = OVERALL_RELATIVE_COLS + OVERALL_RATIO_COLS  # 7 metrics

CRISIS_LABELS = [
    "Dotcom Crash",
    "GFC",
    "Monetary Policy",
    "COVID-19",
    "Russia/Ukraine",
    "Trade Policy Shock",
]

# Metrics stored as negative values where a SMALLER MAGNITUDE is better
# (e.g. drawdown -0.45 vs -0.30: -0.30 is better because |−0.30| < |−0.45|).
# abs() is applied to both sides before computing δ so the formula rewards
# the portfolio that has a smaller absolute loss.
ABS_MAGNITUDE_COLS = {
    "Max Drawdown (%)",
    "Value at Risk (95%)",
    "CVaR / Expected Shortfall",
    "1st Quartile Average Drawdown",
}

# Metrics where a HIGHER value is better.
# δ is negated before scoring so portfolio > benchmark → δ_eff < 0 → positive σ.
# NOTE: "Cumulative Return (Crisis)" belongs here — NOT in ABS_MAGNITUDE_COLS —
# because during many crises both sides are positive (e.g. Russia/Ukraine +16.77%
# vs +1.54%).  abs() would treat the larger positive return as "worse", which is
# the opposite of the correct direction.
HIGHER_IS_BETTER_COLS = {
    "Annualized Return",
    "Cumulative Return (Crisis)",
}

# ─── Drawdown / quartile helpers ─────────────────────────────────────────────

def _find_drawdown_episodes(cum_returns: pd.Series, threshold: float = 0.05) -> list:
    """Return list of drawdown episodes that breach -threshold.
    Each episode is a dict with dd_pct, days_to_trough, days_to_recovery."""
    dd = (cum_returns / cum_returns.cummax()) - 1
    episodes = []

    is_underwater = dd < 0
    if not is_underwater.any():
        return episodes

    group_ids = (is_underwater != is_underwater.shift()).cumsum()

    for _, grp in cum_returns[is_underwater].groupby(group_ids[is_underwater]):
        if dd.loc[grp.index].min() > -threshold:
            continue

        trough_date = dd.loc[grp.index].idxmin()
        trough_val  = cum_returns.loc[trough_date]

        group_start = grp.index[0]
        peak_date   = cum_returns.loc[:group_start].idxmax()
        peak_val    = cum_returns.loc[peak_date]

        max_dd         = (trough_val / peak_val) - 1
        days_to_trough = (pd.Timestamp(trough_date) - pd.Timestamp(peak_date)).days

        post = cum_returns.loc[trough_date:]
        hits = post[post >= peak_val]
        if not hits.empty:
            recovery_date    = hits.index[0]
            days_to_recovery = (pd.Timestamp(recovery_date) - pd.Timestamp(trough_date)).days
        else:
            days_to_recovery = (pd.Timestamp(cum_returns.index[-1]) - pd.Timestamp(trough_date)).days

        episodes.append({
            'dd_pct':           max_dd,
            'days_to_trough':   days_to_trough,
            'days_to_recovery': days_to_recovery,
        })

    return episodes


def compute_q1_metrics(model: str, frequency: str, threshold: float = 0.05) -> dict:
    """Load portfolio CSV(s), compute drawdown episodes per run, average Q1 stats.

    For multi-run models (xgboost, random_forest) every matching file is treated
    as one run; Q1 stats are computed per run then averaged across runs.
    For single-file models the single run is used directly.

    Returns a dict with the three Q1 column names, or NaN values on failure.
    """
    nan_result = {
        "1st Quartile Average Drawdown":              np.nan,
        "1st Quartile Average Duration to Trough":    np.nan,
        "1st Quartile Average Recovery Duration":     np.nan,
    }

    # Benchmark stores a single frequency-independent file
    if model == "benchmark":
        csv_files = list((DATA_DIR / "benchmark").glob("portfolio.csv"))
    else:
        data_dir = DATA_DIR / model
        if not data_dir.exists():
            return nan_result
        # Match e.g. portfolio_Yearly.csv  OR  portfolio_xgb_Yearly_1.csv
        csv_files = sorted(data_dir.glob(f"*{frequency}*.csv"))

    if not csv_files:
        return nan_result

    per_run_q1: list[dict] = []

    for csv_path in csv_files:
        try:
            df  = pd.read_csv(csv_path, index_col='date', parse_dates=True).sort_index()
            # benchmark uses 'price'; portfolio models use 'cumulative_value'
            if 'cumulative_value' in df.columns:
                cum = df['cumulative_value'].dropna()
            elif 'price' in df.columns:
                cum = df['price'].dropna()
            else:
                continue
            if len(cum) < 20:
                continue
            episodes = _find_drawdown_episodes(cum, threshold)
            if not episodes:
                continue

            sorted_eps = sorted(episodes, key=lambda e: e['dd_pct'])  # worst first
            q_size     = max(1, len(sorted_eps) // 4)
            q1_eps     = sorted_eps[:q_size]

            per_run_q1.append({
                "1st Quartile Average Drawdown":           np.mean([abs(e['dd_pct'])           for e in q1_eps]),
                "1st Quartile Average Duration to Trough": np.mean([e['days_to_trough']        for e in q1_eps]),
                "1st Quartile Average Recovery Duration":  np.mean([e['days_to_recovery']      for e in q1_eps]),
            })
        except Exception:
            continue

    if not per_run_q1:
        return nan_result

    return {
        "1st Quartile Average Drawdown":           np.mean([r["1st Quartile Average Drawdown"]           for r in per_run_q1]),
        "1st Quartile Average Duration to Trough": np.mean([r["1st Quartile Average Duration to Trough"] for r in per_run_q1]),
        "1st Quartile Average Recovery Duration":  np.mean([r["1st Quartile Average Recovery Duration"]  for r in per_run_q1]),
    }


# ─── Scoring functions ────────────────────────────────────────────────────────

def score_relative(delta: float) -> float:
    """σ(δ) for relative metrics (δ = (y - x) / |x|)."""
    if delta < -0.10:
        return +2.0
    elif delta < -0.05:
        return +1.0
    elif delta < +0.10:
        return  0.0
    elif delta < +0.15:
        return -1.0
    elif delta < +0.20:
        return -1.5
    else:
        return -2.0


def score_absolute(delta: float) -> float:
    """σ(Δ) for Sharpe and Sortino (Δ = y - x, absolute difference)."""
    if delta > +0.20:
        return +2.0
    elif delta > +0.05:
        return +1.0
    elif delta >= -0.20:
        return  0.0
    elif delta >= -0.30:
        return -1.0
    elif delta >= -0.40:
        return -1.5
    else:
        return -2.0


def compute_delta_relative(benchmark_val: float, portfolio_val: float) -> float:
    """δ = (y - x) / |x|;  returns NaN if benchmark is zero."""
    if benchmark_val == 0 or np.isnan(benchmark_val):
        return np.nan
    return (portfolio_val - benchmark_val) / abs(benchmark_val)


def compute_delta_absolute(benchmark_val: float, portfolio_val: float) -> float:
    """Δ = y - x."""
    return portfolio_val - benchmark_val


# ─── CSV loading helpers ──────────────────────────────────────────────────────

def _parse_value(v) -> float:
    """Convert a formatted value string ('8.50%', '0.562', 'N/A') to float."""
    if pd.isna(v) or str(v).strip() in ('N/A', ''):
        return np.nan
    s = str(v).strip()
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_metrics_csv(folder: Path, frequency: str) -> pd.DataFrame | None:
    """Load and pivot a metrics CSV into wide format.

    The CSV has three columns: Category, Metric, Value.
    e.g.  'Returns' | 'Annualized Return' | '8.50%'
          'Risk'    | 'Annualized Volatility' | '15.20%'
          'Dotcom Crash' | 'Max Drawdown (%)' | '-45.20%'

    The benchmark only produces metrics_buy_hold.csv (frequency-independent),
    so that file is used as a fallback when the frequency-specific file is absent.

    Output: one row per Category (crisis rows kept as-is), plus a synthetic
    'overall' row that merges all non-crisis categories so the existing
    get_row(df, 'overall') lookup continues to work.
    """
    path = folder / f"metrics_{frequency}.csv"
    if not path.exists():
        path = folder / "metrics_buy_hold.csv"   # benchmark fallback
    if not path.exists():
        return None

    raw = pd.read_csv(path)
    if raw.shape[1] < 3:
        return None

    raw.columns = ['category', 'metric', 'value'] + list(raw.columns[3:])
    raw['val_f'] = raw['value'].apply(_parse_value)

    # Pivot: index = Category, columns = Metric name, values = parsed float
    wide = (
        raw.pivot_table(index='category', columns='metric', values='val_f', aggfunc='first')
        .reset_index()
        .rename(columns={'category': 'period'})
    )
    wide.columns.name = None
    wide['period_norm'] = wide['period'].str.strip().str.lower()

    # Synthetic 'overall' row: take the first non-NaN value per metric across
    # all non-crisis categories (Returns, Risk, Ratios, Benchmark, …)
    crisis_norms = {lbl.lower() for lbl in CRISIS_LABELS}
    non_crisis   = wide[~wide['period_norm'].isin(crisis_norms)]
    if not non_crisis.empty:
        overall_vals: dict = {'period': 'overall', 'period_norm': 'overall'}
        for col in wide.columns:
            if col in ('period', 'period_norm'):
                continue
            vals = non_crisis[col].dropna()
            if not vals.empty:
                overall_vals[col] = vals.iloc[0]
        wide = pd.concat([wide, pd.DataFrame([overall_vals])], ignore_index=True)

    return wide


def get_row(df: pd.DataFrame, label_norm: str) -> pd.Series | None:
    """Return the first row whose normalised label contains label_norm (case-insensitive)."""
    query   = label_norm.strip().lower()
    matches = df[df["period_norm"].str.contains(query, regex=False)]
    if matches.empty:
        return None
    return matches.iloc[0]


# ─── Per-metric scoring ───────────────────────────────────────────────────────

def score_metric(col: str,
                 bmark_row: pd.Series,
                 port_row: pd.Series,
                 is_ratio: bool) -> float:
    """Compute σ for a single metric column, handling missing values.

    For metrics in ABS_MAGNITUDE_COLS (stored as negatives, e.g. drawdown),
    both values are converted to absolute magnitudes before computing δ so
    that a smaller loss correctly scores as better (δ < 0 → positive σ).
    """
    try:
        bval = float(bmark_row[col])
        pval = float(port_row[col])
    except (KeyError, ValueError, TypeError):
        return np.nan

    if np.isnan(bval) or np.isnan(pval):
        return np.nan

    if col in ABS_MAGNITUDE_COLS:
        bval = abs(bval)
        pval = abs(pval)

    if is_ratio:
        delta = compute_delta_absolute(bval, pval)
        return score_absolute(delta)
    else:
        delta = compute_delta_relative(bval, pval)
        if col in HIGHER_IS_BETTER_COLS:
            delta = -delta   # portfolio > benchmark → δ < 0 → positive σ
        return score_relative(delta)


# ─── Main scoring logic ───────────────────────────────────────────────────────

def score_model_frequency(model: str, frequency: str) -> dict:
    """
    Compute all scores for one model × frequency pair.
    Returns a dict with keys:
        model, frequency,
        CS_<crisis>, …  (6 crisis scores),
        OS,
        ScoreTotal,
        detail_<crisis>_<metric>  (optional per-metric breakdown)
    """
    result = {"model": model, "frequency": frequency}

    bmark_df  = load_metrics_csv(BENCHMARK_DIR, frequency)
    model_dir = METRICS_DIR / model
    port_df   = load_metrics_csv(model_dir, frequency)

    if bmark_df is None:
        result["error"] = f"Benchmark CSV missing for {frequency}"
        return result
    if port_df is None:
        result["error"] = f"Model CSV missing for {model}/{frequency}"
        return result

    # ── Crisis scores ──────────────────────────────────────────────────────
    crisis_scores = []
    for label in CRISIS_LABELS:
        brow = get_row(bmark_df, label)
        prow = get_row(port_df,  label)

        if brow is None:
            print(f"    [WARN] Benchmark row missing for crisis '{label}' [{frequency}]")
        if prow is None:
            print(f"    [WARN] {model} row missing for crisis '{label}' [{frequency}]")

        if brow is None or prow is None:
            result[f"CS_{label}"] = np.nan
            crisis_scores.append(np.nan)
            continue

        cs = 0.0
        for col in CRISIS_RELATIVE_COLS:
            s = score_metric(col, brow, prow, is_ratio=False)
            result[f"detail_{label}_{col}"] = s
            cs += s if not np.isnan(s) else 0.0

        for col in CRISIS_RATIO_COLS:
            s = score_metric(col, brow, prow, is_ratio=True)
            result[f"detail_{label}_{col}"] = s
            cs += s if not np.isnan(s) else 0.0

        result[f"CS_{label}"] = cs
        crisis_scores.append(cs)

    # ── Overall score ──────────────────────────────────────────────────────
    brow_os = get_row(bmark_df, "overall")
    prow_os = get_row(port_df,  "overall")

    # Compute Q1 drawdown metrics from raw portfolio data and inject into rows
    # (the metrics CSVs do not contain these; we calculate them here directly)
    bmark_q1 = compute_q1_metrics("benchmark", frequency)
    port_q1  = compute_q1_metrics(model, frequency)

    if brow_os is not None:
        brow_os = brow_os.copy()
        for k, v in bmark_q1.items():
            brow_os[k] = v
    if prow_os is not None:
        prow_os = prow_os.copy()
        for k, v in port_q1.items():
            prow_os[k] = v

    os_score = 0.0
    if brow_os is not None and prow_os is not None:
        for col in OVERALL_RELATIVE_COLS:
            s = score_metric(col, brow_os, prow_os, is_ratio=False)
            result[f"detail_overall_{col}"] = s
            os_score += s if not np.isnan(s) else 0.0
        for col in OVERALL_RATIO_COLS:
            s = score_metric(col, brow_os, prow_os, is_ratio=True)
            result[f"detail_overall_{col}"] = s
            os_score += s if not np.isnan(s) else 0.0
    else:
        os_score = np.nan

    result["OS"] = os_score

    # ── Total score ────────────────────────────────────────────────────────
    valid_cs = [v for v in crisis_scores if not np.isnan(v)]
    if valid_cs and not np.isnan(os_score):
        avg_cs = np.mean(valid_cs)
        result["ScoreTotal"] = round(avg_cs * 0.80 + os_score * 0.20, 4)
    else:
        result["ScoreTotal"] = np.nan

    return result


# ─── Run all combinations ─────────────────────────────────────────────────────

def run_scoring(models: list[str] = MODELS,
                frequencies: list[str] = FREQUENCIES) -> pd.DataFrame:
    """Evaluate every model × frequency combination.
    Returns a DataFrame with one row per (model, frequency).
    """
    rows = []
    for model in models:
        for freq in frequencies:
            print(f"  Scoring  {model:<30}  [{freq}]")
            rows.append(score_model_frequency(model, freq))

    df = pd.DataFrame(rows)

    # ── Structured column order ───────────────────────────────────────────
    # model | frequency | ScoreTotal | OS | detail_overall_* |
    # CS_<crisis> | detail_<crisis>_* | ...
    id_cols     = ["model", "frequency"]
    total_cols  = ["ScoreTotal", "OS"]
    os_detail   = [f"detail_overall_{c}" for c in OVERALL_METRIC_COLS]
    crisis_cols = []
    for lbl in CRISIS_LABELS:
        crisis_cols.append(f"CS_{lbl}")
        for c in CRISIS_METRIC_COLS:
            crisis_cols.append(f"detail_{lbl}_{c}")
    other_cols = [c for c in df.columns
                  if c not in id_cols + total_cols + os_detail + crisis_cols]

    ordered = [c for c in id_cols + total_cols + os_detail + crisis_cols + other_cols
               if c in df.columns]
    return df[ordered]


# ─── CSV export ───────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: str = "scoring_results.csv") -> None:
    """Save results to a clean CSV with numeric values rounded to 4 dp."""
    out = df.copy()
    float_cols = out.select_dtypes(include="number").columns
    out[float_cols] = out[float_cols].round(4)
    out.to_csv(path, index=False)
    print(f"✓ CSV saved  →  {path}")


# ─── HTML report ──────────────────────────────────────────────────────────────

def _score_color(val) -> str:
    """Return a CSS background colour based on score magnitude."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "#F8F9FA"
    if np.isnan(v):
        return "#F8F9FA"
    if v >= 1.5:   return "#166534"   # dark green
    if v >= 0.5:   return "#4ADE80"   # green
    if v >= -0.5:  return "#F8F9FA"   # neutral
    if v >= -1.5:  return "#FCA5A5"   # light red
    return "#DC2626"                  # dark red

def _score_text_color(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "#1E293B"
    if np.isnan(v):
        return "#1E293B"
    if v >= 1.5 or v <= -1.5:
        return "#FFFFFF"
    return "#1E293B"

def _fmt(val) -> str:
    try:
        v = float(val)
        if np.isnan(v):
            return "—"
        return f"{v:+.3f}"
    except (TypeError, ValueError):
        return str(val) if val else "—"


def generate_html(df: pd.DataFrame, path: str = "scoring_results.html") -> None:
    """Write a styled HTML report with one table per frequency."""

    BG      = "#FFFFFF"
    HDR     = "#1D6FA4"
    TEXT    = "#1E293B"
    SUBTEXT = "#64748B"
    GRID    = "#E9ECEF"

    # ── short display names for detail columns ────────────────────────────
    metric_short = {
        "Max Drawdown (%)":                        "Max DD",
        "Days: Trough to Breakeven":               "Trough→BEven",
        "Days: Peak to Breakeven (Total)":         "Peak→BEven",
        "Cumulative Return (Crisis)":              "Cum Ret",
        "Annualized Volatility (Crisis)":          "Ann Vol",
        "Sharpe Ratio (Crisis)":                   "Sharpe",
        "Sortino Ratio (Crisis)":                  "Sortino",
        "Annualized Return":                       "Ann Ret",
        "Annualized Volatility":                   "Ann Vol",
        "Value at Risk (95%)":                     "VaR 95%",
        "CVaR / Expected Shortfall":               "CVaR",
        "1st Quartile Average Drawdown":           "Q1 DD",
        "1st Quartile Average Duration to Trough": "Q1 Dur",
        "1st Quartile Average Recovery Duration":  "Q1 Rec",
    }

    def col_label(c: str) -> str:
        for k, v in metric_short.items():
            if k in c:
                return v
        return c.replace("detail_overall_", "").replace("detail_", "")

    sections = []

    for freq in FREQUENCIES:
        sub = df[df["frequency"] == freq].copy()
        if sub.empty:
            continue

        if "ScoreTotal" in sub.columns:
            sub = sub.sort_values("ScoreTotal", ascending=False)

        # ── build column groups ───────────────────────────────────────────
        # Group 1: identity + totals + OS detail
        grp_id    = ["model", "ScoreTotal", "OS"]
        grp_os    = [f"detail_overall_{c}" for c in OVERALL_METRIC_COLS if f"detail_overall_{c}" in sub.columns]
        # Group 2: per crisis
        grp_cs    = []
        for lbl in CRISIS_LABELS:
            cs_col = f"CS_{lbl}"
            det    = [f"detail_{lbl}_{c}" for c in CRISIS_METRIC_COLS if f"detail_{lbl}_{c}" in sub.columns]
            grp_cs.append((lbl, cs_col, det))

        all_cols = grp_id + grp_os + [c for _, cs, det in grp_cs for c in ([cs] + det)]
        display_cols = [c for c in all_cols if c in sub.columns]

        # ── header rows ───────────────────────────────────────────────────
        # Row 1: group spans
        spans = [
            ("Model",    1, HDR),
            ("ScoreTotal", 1, "#0F4C75"),
            ("Overall Score", 1 + len(grp_os), "#1A5276"),
        ]
        for lbl, cs_col, det in grp_cs:
            spans.append((lbl, 1 + len(det), "#1D3A5A"))

        hdr1 = "".join(
            f'<th colspan="{span}" style="background:{bg};color:#fff;'
            f'padding:8px 6px;border:1px solid {GRID};white-space:nowrap">{label}</th>'
            for label, span, bg in spans
        )

        # Row 2: individual column labels
        hdr2_cells = []
        for c in display_cols:
            label = col_label(c) if c not in ("model", "ScoreTotal") else c.replace("ScoreTotal", "Total")
            hdr2_cells.append(
                f'<th style="background:{HDR};color:#fff;padding:6px 5px;'
                f'border:1px solid {GRID};font-size:11px;white-space:nowrap">{label}</th>'
            )
        hdr2 = "".join(hdr2_cells)

        # ── data rows ─────────────────────────────────────────────────────
        rows_html = []
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            row_bg = "#FAFAFA" if rank % 2 == 0 else BG
            cells  = []
            for c in display_cols:
                val = row.get(c, np.nan)
                if c == "model":
                    cells.append(
                        f'<td style="background:{row_bg};padding:6px 8px;'
                        f'border:1px solid {GRID};font-weight:600;white-space:nowrap">{val}</td>'
                    )
                else:
                    bg  = _score_color(val)
                    fg  = _score_text_color(val)
                    cells.append(
                        f'<td style="background:{bg};color:{fg};padding:5px 6px;'
                        f'border:1px solid {GRID};text-align:right;font-size:12px">{_fmt(val)}</td>'
                    )
            rows_html.append(f'<tr>{"".join(cells)}</tr>')

        table = f"""
<h2 style="font-family:Inter,Arial;color:{TEXT};margin:36px 0 8px">
  {freq} Rebalancing — Leaderboard
</h2>
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-family:Inter,Arial,sans-serif;
              font-size:12px;width:100%;background:{BG}">
  <thead>
    <tr>{hdr1}</tr>
    <tr>{hdr2}</tr>
  </thead>
  <tbody>
    {"".join(rows_html)}
  </tbody>
</table>
</div>
<p style="font-family:Inter,Arial;font-size:11px;color:{SUBTEXT};margin:4px 0 0">
  Score range: CS ∈ [−14, +14] &nbsp;·&nbsp; OS ∈ [−14, +14] &nbsp;·&nbsp;
  ScoreTotal ∈ [−14, +14] &nbsp;·&nbsp;
  Detail columns show individual σ values (−2 … +2).
</p>
"""
        sections.append(table)

    # ── colour legend ─────────────────────────────────────────────────────
    legend_items = [
        ("#166534", "#fff", "≥ +1.5  strong outperform"),
        ("#4ADE80", "#1E293B", "+0.5 … +1.5  outperform"),
        ("#F8F9FA", "#1E293B", "−0.5 … +0.5  neutral"),
        ("#FCA5A5", "#1E293B", "−1.5 … −0.5  underperform"),
        ("#DC2626", "#fff", "≤ −1.5  strong underperform"),
    ]
    legend_html = "".join(
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'margin:2px;border-radius:3px;font-size:11px;display:inline-block">{lbl}</span>'
        for bg, fg, lbl in legend_items
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Portfolio Scoring Results</title>
  <style>
    body {{ margin: 20px 40px; background: {BG}; }}
    h1   {{ font-family: Inter, Arial; color: {TEXT}; }}
    p    {{ font-family: Inter, Arial; color: {SUBTEXT}; font-size: 13px; }}
  </style>
</head>
<body>
<h1>Portfolio Scoring Results</h1>
<p>
  ScoreTotal = (mean CS × 0.80) + (OS × 0.20) &nbsp;|&nbsp;
  CS = Σ σ(m, c) over 7 crisis metrics &nbsp;|&nbsp;
  OS = Σ σ(k) over 7 overall metrics
</p>
<div style="margin:8px 0 24px">{legend_html}</div>
{"<hr style='border:none;border-top:1px solid #E9ECEF;margin:32px 0'>".join(sections)}
</body>
</html>"""

    Path(path).write_text(html, encoding="utf-8")
    print(f"✓ HTML saved →  {path}")


# ─── Actual-values export ────────────────────────────────────────────────────

# Format specs for each metric column (for display in the actual-values table)
_FMT = {
    "Max Drawdown (%)":                         "{:.2%}",
    "Days: Trough to Breakeven":                "{:.0f}d",
    "Days: Peak to Breakeven (Total)":          "{:.0f}d",
    "Cumulative Return (Crisis)":               "{:.2%}",
    "Annualized Volatility (Crisis)":           "{:.2%}",
    "Sharpe Ratio (Crisis)":                    "{:.3f}",
    "Sortino Ratio (Crisis)":                   "{:.3f}",
    "Annualized Return":                        "{:.2%}",
    "Annualized Volatility":                    "{:.2%}",
    "Value at Risk (95%)":                      "{:.2%}",
    "CVaR / Expected Shortfall":                "{:.2%}",
    "1st Quartile Average Drawdown":            "{:.2%}",
    "1st Quartile Average Duration to Trough":  "{:.0f}d",
    "1st Quartile Average Recovery Duration":   "{:.0f}d",
}

def _fmt_actual(col: str, val) -> str:
    try:
        v = float(val)
        if np.isnan(v):
            return "—"
        fmt = _FMT.get(col, "{:.4f}")
        return fmt.format(v)
    except (TypeError, ValueError):
        return "—"


def collect_actual_values(models: list[str] = MODELS,
                          frequencies: list[str] = FREQUENCIES) -> pd.DataFrame:
    """Build a DataFrame of raw metric values (not σ scores) for every
    model × frequency combination.  Columns mirror the scoring table:
      model | frequency |
      <crisis>_<metric>  (one per crisis × metric)  |
      overall_<metric>   (one per overall metric)
    """
    rows = []
    for model in models:
        for freq in frequencies:
            row: dict = {"model": model, "frequency": freq}

            port_df  = load_metrics_csv(METRICS_DIR / model, freq)
            bmark_df = load_metrics_csv(BENCHMARK_DIR, freq)

            # ── Crisis metrics ────────────────────────────────────────────
            for label in CRISIS_LABELS:
                prow  = get_row(port_df,  label) if port_df  is not None else None
                brow  = get_row(bmark_df, label) if bmark_df is not None else None
                short = label.replace("/", "-").replace(" ", "_").lower()

                for col in CRISIS_METRIC_COLS:
                    key = f"{short}_{col}"
                    try:
                        row[key] = float(prow[col]) if prow is not None else np.nan
                    except (KeyError, TypeError, ValueError):
                        row[key] = np.nan
                    # benchmark column
                    bkey = f"bm_{short}_{col}"
                    try:
                        row[bkey] = float(brow[col]) if brow is not None else np.nan
                    except (KeyError, TypeError, ValueError):
                        row[bkey] = np.nan

            # ── Overall metrics ───────────────────────────────────────────
            prow_os  = get_row(port_df,  "overall") if port_df  is not None else None
            brow_os  = get_row(bmark_df, "overall") if bmark_df is not None else None

            # inject Q1 values
            port_q1  = compute_q1_metrics(model,       freq)
            bmark_q1 = compute_q1_metrics("benchmark", freq)
            if prow_os is not None:
                prow_os = prow_os.copy()
                for k, v in port_q1.items():
                    prow_os[k] = v
            if brow_os is not None:
                brow_os = brow_os.copy()
                for k, v in bmark_q1.items():
                    brow_os[k] = v

            for col in OVERALL_METRIC_COLS:
                key = f"overall_{col}"
                try:
                    row[key] = float(prow_os[col]) if prow_os is not None else np.nan
                except (KeyError, TypeError, ValueError):
                    row[key] = np.nan
                bkey = f"bm_overall_{col}"
                try:
                    row[bkey] = float(brow_os[col]) if brow_os is not None else np.nan
                except (KeyError, TypeError, ValueError):
                    row[bkey] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def save_actual_csv(df: pd.DataFrame, path: str) -> None:
    """Save the actual-values DataFrame to CSV (raw floats, rounded to 6 dp)."""
    out = df.copy()
    float_cols = out.select_dtypes(include="number").columns
    out[float_cols] = out[float_cols].round(6)
    out.to_csv(path, index=False)
    print(f"✓ Actual-values CSV saved  →  {path}")


def generate_actual_html(df: pd.DataFrame, path: str) -> None:
    """Write a styled HTML table of raw metric values, grouped by frequency.
    Each section shows portfolio values with the benchmark value in parentheses.
    """
    BG      = "#FFFFFF"
    HDR     = "#0F4C75"
    TEXT    = "#1E293B"
    SUBTEXT = "#64748B"
    GRID    = "#E9ECEF"

    metric_short = {
        "Max Drawdown (%)":                         "Max DD",
        "Days: Trough to Breakeven":                "Trough→BEven",
        "Days: Peak to Breakeven (Total)":          "Peak→BEven",
        "Cumulative Return (Crisis)":               "Cum Ret",
        "Annualized Volatility (Crisis)":           "Ann Vol",
        "Sharpe Ratio (Crisis)":                    "Sharpe",
        "Sortino Ratio (Crisis)":                   "Sortino",
        "Annualized Return":                        "Ann Ret",
        "Annualized Volatility":                    "Ann Vol",
        "Value at Risk (95%)":                      "VaR 95%",
        "CVaR / Expected Shortfall":                "CVaR",
        "1st Quartile Average Drawdown":            "Q1 DD",
        "1st Quartile Average Duration to Trough":  "Q1 Dur",
        "1st Quartile Average Recovery Duration":   "Q1 Rec",
    }

    sections = []

    for freq in FREQUENCIES:
        sub = df[df["frequency"] == freq].copy()
        if sub.empty:
            continue

        # ── Build display columns ─────────────────────────────────────────
        # crisis groups: (label, short, [col, ...])
        crisis_groups = []
        for label in CRISIS_LABELS:
            short = label.replace("/", "-").replace(" ", "_").lower()
            cols  = [f"{short}_{c}" for c in CRISIS_METRIC_COLS if f"{short}_{c}" in sub.columns]
            crisis_groups.append((label, short, cols))

        overall_cols = [f"overall_{c}" for c in OVERALL_METRIC_COLS if f"overall_{c}" in sub.columns]

        # ── Header row 1: group spans ─────────────────────────────────────
        spans = [
            ("Model", 1, HDR),
            ("Overall Metrics", len(overall_cols), "#1A5276"),
        ]
        for label, _, cols in crisis_groups:
            spans.append((label, len(cols), "#1D3A5A"))

        hdr1 = "".join(
            f'<th colspan="{sp}" style="background:{bg};color:#fff;padding:8px 6px;'
            f'border:1px solid {GRID};white-space:nowrap">{lbl}</th>'
            for lbl, sp, bg in spans if sp > 0
        )

        # ── Header row 2: individual metric labels ────────────────────────
        display_cols = ["model"] + overall_cols + \
                       [c for _, _, cols in crisis_groups for c in cols]

        def _col_label(c):
            for k, v in metric_short.items():
                if k in c:
                    return v
            return c
        hdr2 = "".join(
            f'<th style="background:{HDR};color:#fff;padding:6px 5px;border:1px solid {GRID};'
            f'font-size:11px;white-space:nowrap">{"Model" if c == "model" else _col_label(c)}</th>'
            for c in display_cols
        )

        # ── Data rows ─────────────────────────────────────────────────────
        rows_html = []
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            row_bg = "#FAFAFA" if rank % 2 == 0 else BG
            cells  = []
            for c in display_cols:
                val = row.get(c, np.nan)
                if c == "model":
                    cells.append(
                        f'<td style="background:{row_bg};padding:6px 8px;border:1px solid {GRID};'
                        f'font-weight:600;white-space:nowrap">{val}</td>'
                    )
                else:
                    # find underlying metric name
                    metric_col = next((m for m in CRISIS_METRIC_COLS + OVERALL_METRIC_COLS if m in c), c)
                    # benchmark value
                    bm_key = "bm_" + c
                    bm_val = row.get(bm_key, np.nan)
                    port_str = _fmt_actual(metric_col, val)
                    bm_str   = _fmt_actual(metric_col, bm_val)
                    display  = port_str if bm_str == "—" else f"{port_str}<br><small style='color:{SUBTEXT}'>({bm_str})</small>"
                    cells.append(
                        f'<td style="background:{row_bg};padding:5px 6px;border:1px solid {GRID};'
                        f'text-align:right;font-size:12px">{display}</td>'
                    )
            rows_html.append(f'<tr>{"".join(cells)}</tr>')

        table = f"""
<h2 style="font-family:Inter,Arial;color:{TEXT};margin:36px 0 8px">
  {freq} — Actual Metric Values
</h2>
<p style="font-family:Inter,Arial;font-size:12px;color:{SUBTEXT};margin:0 0 6px">
  Portfolio value shown; benchmark value in parentheses.
</p>
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-family:Inter,Arial,sans-serif;font-size:12px;width:100%">
  <thead>
    <tr>{hdr1}</tr>
    <tr>{hdr2}</tr>
  </thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>
</div>"""
        sections.append(table)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Portfolio — Actual Metric Values</title>
  <style>body {{ margin: 20px 40px; background: {BG}; }}</style>
</head>
<body>
<h1 style="font-family:Inter,Arial;color:{TEXT}">Portfolio — Actual Metric Values</h1>
<p style="font-family:Inter,Arial;color:{SUBTEXT};font-size:13px">
  Raw values from the metrics CSVs. Portfolio value / (Benchmark in parentheses).
</p>
{"<hr style='border:none;border-top:1px solid #E9ECEF;margin:32px 0'>".join(sections)}
</body>
</html>"""

    Path(path).write_text(html, encoding="utf-8")
    print(f"✓ Actual-values HTML saved →  {path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running portfolio scoring system...\n")
    results_df = run_scoring()

    csv_path  = OUTPUT_DIR / "scoring_results.csv"
    html_path = OUTPUT_DIR / "scoring_results.html"
    save_csv(results_df, str(csv_path))
    generate_html(results_df, str(html_path))

    print("\nCollecting actual metric values...")
    actual_df   = collect_actual_values()
    actual_csv  = OUTPUT_DIR / "actual_values.csv"
    actual_html = OUTPUT_DIR / "actual_values.html"
    save_actual_csv(actual_df, str(actual_csv))
    generate_actual_html(actual_df, str(actual_html))

    # Console leaderboard
    summary_cols = ["model", "frequency", "ScoreTotal", "OS"] + \
                   [f"CS_{l}" for l in CRISIS_LABELS]
    available = [c for c in summary_cols if c in results_df.columns]
    if "ScoreTotal" in results_df.columns:
        summary = results_df[available].sort_values("ScoreTotal", ascending=False)
    else:
        summary = results_df[available]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:+.3f}".format)
    print("\n" + "=" * 80)
    print("LEADERBOARD  (ScoreTotal range: [−14, +14])")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)