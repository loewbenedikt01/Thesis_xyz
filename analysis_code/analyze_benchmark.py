


import pyarrow
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats



# Path to directory
DATA_PATH = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
benchmark_price_file = DATA_PATH / "benchmark_price.parquet"
output_dir_data = DATA_PATH / "results" / "data" / "benchmark"
output_dir_metrics = DATA_PATH / "results" / "metrics" / "benchmark"
output_dir_plots = DATA_PATH / "results" / "plots" / "benchmark"
output_dir_data.mkdir(parents=True, exist_ok=True) 
output_dir_metrics.mkdir(parents=True, exist_ok=True)
output_dir_plots.mkdir(parents=True, exist_ok=True)

data_file = output_dir_data / "portfolio.csv"
metrics_file = output_dir_metrics / "metrics_buy_hold.csv"
# plots_file = output_dir_plots / 

# Date Range
start_date = "1998-01-01"
end_date = "2025-12-31"
ticker = "^GSPC"

print(f"Loading data from {benchmark_price_file}...")


df = pd.read_parquet(benchmark_price_file)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

if ticker in df.columns:
    price_series = df[ticker]
elif 'Close' in df.columns:
    price_series = df['Close']
else:
    price_series = df.iloc[:, 0]

price_series = price_series.loc[start_date:end_date]

portfolio = pd.DataFrame(index=price_series.index)
portfolio['ticker'] = ticker
portfolio['price'] = price_series

portfolio['returns_per_day'] = portfolio['price'].pct_change()
portfolio['log_returns_per_day'] = np.log(portfolio['price'] / portfolio['price'].shift(1))

portfolio.index = portfolio.index.strftime('%Y-%m-%d')
portfolio.index.name = 'date'
portfolio = portfolio.dropna()
portfolio = portfolio[['ticker', 'price', 'returns_per_day', 'log_returns_per_day']]

# Export portfolio data
portfolio.to_csv(data_file)
print(f"Portfolio data exported to: {data_file}")

# --------
# Create the Metrics
# --------

# Helpers for Frequency

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

def _annualized_factor(freq: str) -> int:
    """Return the annualization factor based on return frequency"""
    if freq == 'D':
        return TRADING_DAYS_PER_YEAR
    elif freq == 'M':
        return MONTHS_PER_YEAR
    else:
        raise ValueError(f"freq must be 'D' or 'M', got '{freq}'")

# Return Metrics

def cumulative_return(log_returns_per_day: pd.Series) -> float:
    """Total cumulative return over the full period"""
    total_log_return = log_returns_per_day.sum()
    return np.exp(total_log_return) - 1

def annualized_return(log_returns_per_day: pd.Series, freq: str = 'D') -> float:
    """Annualized return, accounting for compounding"""
    total_log_return = log_returns_per_day.sum()
    n = len(log_returns_per_day)
    ann_f = _annualized_factor(freq)
    annual_log_return = (total_log_return / n) * ann_f
    return np.exp(annual_log_return) - 1

# Risk Metrics

def annualized_volatility(log_returns_per_day: pd.Series, freq: str = 'D') -> float:
    """Annualized standard deviation of returns"""
    std_dev = log_returns_per_day.std(ddof=1)
    ann_f = _annualized_factor(freq)
    ann_volatility = std_dev * np.sqrt(ann_f)
    return ann_volatility

def maximum_drawdown(price_series: pd.Series) -> float:
    """Maximum Drawdown"""
    rolling_max = price_series.cummax()
    drawdowns = (price_series / rolling_max) - 1
    return drawdowns.min()

def drawdown_series(price_series: pd.Series) -> pd.Series:
    """Return the full drawdown time series"""
    rolling_max = price_series.cummax()
    drawdown_series = (price_series / rolling_max) - 1
    return drawdown_series

def max_drawdown_duration(price_series: pd.Series) -> int:
    """Duration of Drawdown"""
    rolling_max = price_series.cummax()
    is_at_peak = price_series >= rolling_max
    underwater_groups = is_at_peak.cumsum()
    durations = price_series.groupby(underwater_groups).cumcount()
    return durations.max()

def recovery_duration(price_series: pd.Series) -> int:
    """Duration of Recovery"""
    rolling_max = price_series.cummax()
    drawdown = (price_series / rolling_max) - 1
    mdd_date = drawdown.idxmin()
    peak_at_mdd = rolling_max.loc[mdd_date]
    recovery_series = price_series.loc[mdd_date:]
    recovered_hits = recovery_series[recovery_series >= peak_at_mdd]
    if recovered_hits.empty:
        return len(recovery_series)
    recovery_date = recovered_hits.index[0]
    return len(price_series.loc[mdd_date:recovery_date]) - 1

def value_at_risk(log_returns_per_day: pd.Series, confidence_level: float = 0.95) -> float:
    """Historical Value at Risk (VaR 95%)"""
    return log_returns_per_day.quantile(1 - confidence_level)

def conditional_value_at_risk(log_returns_per_day: pd.Series, confidence_level: float = 0.95) -> float:
    """Historical Conditional Value at Risk (CVaR 95%)"""
    var_threshold = value_at_risk(log_returns_per_day, confidence_level)
    return log_returns_per_day[log_returns_per_day <= var_threshold].mean()

# Risk Adjusted Metrics
# Risk Free Rate, adjust for different results
risk_free_rate = 0.0

def sharpe_ratio(log_returns_per_day: pd.Series, risk_free_rate, freq: str = 'D') -> float:
    """Annualized Sharpe Ratio"""
    ann_ret = annualized_return(log_returns_per_day, freq)
    ann_vol = annualized_volatility(log_returns_per_day, freq)
    return (ann_ret - risk_free_rate) / ann_vol

def sortino_ratio(log_returns_per_day: pd.Series, risk_free_rate, freq: str = 'D') -> float:
    """Annualized Sortino Ratio"""
    ann_ret = annualized_return(log_returns_per_day, freq)
    ann_f = _annualized_factor(freq)
    downside_returns = log_returns_per_day[log_returns_per_day < 0]
    downside_vol = np.sqrt((downside_returns**2).mean()) * np.sqrt(ann_f)
    return (ann_ret - risk_free_rate) / downside_vol

def calmar_ratio(log_returns_per_day: pd.Series, price_series: pd.Series, freq: str = 'D') -> float:
    """Calmar Ratio (Return/Max Drawdown)"""
    ann_ret = annualized_return(log_returns_per_day, freq)
    mdd = abs(maximum_drawdown(price_series))
    return ann_ret / mdd if mdd != 0 else np.nan

def omega_ratio(log_returns_per_day: pd.Series, threshold: float = 0.0) -> float:
    """Omega Ratio (Probability-weighted gains vs losses)"""
    gains = log_returns_per_day[log_returns_per_day > threshold].sum()
    losses = abs(log_returns_per_day[log_returns_per_day < threshold].sum())
    return gains / losses if losses != 0 else np.inf

def _benchmark_breakeven(price_series: pd.Series) -> float:
    """Helper: compute time to breakeven after max drawdown using Price."""
    rolling_max = price_series.cummax()
    dd_series = (price_series / rolling_max) - 1
    trough_idx = dd_series.idxmin()
    peak_value = price_series.loc[:trough_idx].max()
    post_trough = price_series.loc[trough_idx:]
    recovered = post_trough[post_trough >= peak_value]
    if recovered.empty:
        return np.nan
    return len(price_series.loc[trough_idx:recovered.index[0]])

def crisis_metrics(log_returns_per_day: pd.Series, price_series: pd.Series, risk_free_rate, freq: str = 'D') -> dict:
    """Compute crisis-specific metrics"""
    rolling_max = price_series.cummax()
    dd_series = (price_series / rolling_max) - 1
    trough_idx = dd_series.idxmin()
    peak_idx = price_series.loc[:trough_idx].idxmax()
    phase1_log_ret = log_returns_per_day.loc[peak_idx:trough_idx]
    phase2_log_ret = log_returns_per_day.loc[trough_idx:]
    phase1_price = price_series.loc[peak_idx:trough_idx]
    phase2_price = price_series.loc[trough_idx:]
    time_to_breakeven = _benchmark_breakeven(price_series)
    return_speed = phase2_log_ret.mean() if len(phase2_log_ret) > 0 else np.nan
    return {
        'phase1_max_drawdown'        : maximum_drawdown(phase1_price),
        'phase1_duration_to_trough'  : len(phase1_price),
        'phase1_sharpe'              : sharpe_ratio(phase1_log_ret, risk_free_rate, freq) if len(phase1_log_ret) > 1 else np.nan,
        'phase1_sortino'             : sortino_ratio(phase1_log_ret, risk_free_rate, freq) if len(phase1_log_ret) > 1 else np.nan,
        'phase2_time_to_breakeven'   : time_to_breakeven,
        'phase2_return_speed'        : return_speed,
        'phase2_calmar'              : calmar_ratio(phase2_log_ret, phase2_price, freq) if len(phase2_log_ret) > 1 else np.nan,
        'phase2_volatility'          : annualized_volatility(phase2_log_ret, freq) if len(phase2_log_ret) > 1 else np.nan,
    }

# Calculate the Metrics
def compute_all_metrics(portfolio_log_returns: pd.Series, price_series: pd.Series, benchmark_log_returns: pd.Series, risk_free_rate, freq: str = 'D') -> dict:
    """Compute all portfolio evaluation metrics"""
    results = {}
    log_ret, bench_ret = portfolio_log_returns.align(benchmark_log_returns, join='inner')
    results['cumulative_return']      = cumulative_return(log_ret)
    results['annualized_return']      = annualized_return(log_ret, freq)
    results['benchmark_cum_return']   = cumulative_return(bench_ret)
    results['benchmark_ann_return']   = annualized_return(bench_ret, freq)
    results['annualized_volatility']  = annualized_volatility(log_ret, freq)
    results['maximum_drawdown']       = maximum_drawdown(price_series)
    results['dd_duration_to_trough']  = max_drawdown_duration(price_series)
    results['recovery_duration']      = recovery_duration(price_series)
    results['var_95']                 = value_at_risk(log_ret, 0.95)
    results['cvar_95']                = conditional_value_at_risk(log_ret, 0.95)
    results['sharpe']                 = sharpe_ratio(log_ret, risk_free_rate, freq)
    results['sortino']                = sortino_ratio(log_ret, risk_free_rate, freq)
    results['calmar']                 = calmar_ratio(log_ret, price_series, freq)
    results['omega']                  = omega_ratio(log_ret)
    crisis_results = crisis_metrics(log_ret, price_series, risk_free_rate, freq)
    results.update(crisis_results)
    return results

# Create DataFrame
def metrics_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dict to a clean formatted DataFrame for display/export."""
    labels = {
        'cumulative_return'          : ('Returns',   'Cumulative Return',           '{:.2%}'),
        'annualized_return'          : ('Returns',   'Annualized Return',           '{:.2%}'),
        'benchmark_cum_return'       : ('Benchmark', 'Benchmark Cumulative Return', '{:.2%}'),
        'benchmark_ann_return'       : ('Benchmark', 'Benchmark Annualized Return', '{:.2%}'),
        'annualized_volatility'      : ('Risk',      'Annualized Volatility',       '{:.2%}'),
        'maximum_drawdown'           : ('Risk',      'Maximum Drawdown',            '{:.2%}'),
        'dd_duration_to_trough'      : ('Risk',      'DD Duration (periods)',       '{:.0f}'),
        'recovery_duration'          : ('Risk',      'Recovery Duration (periods)', '{:.0f}'),
        'var_95'                     : ('Risk',      'Value at Risk (95%)',         '{:.2%}'),
        'cvar_95'                    : ('Risk',      'CVaR / Expected Shortfall',   '{:.2%}'),
        'sharpe'                     : ('Ratios',    'Sharpe Ratio',                '{:.3f}'),
        'sortino'                    : ('Ratios',    'Sortino Ratio',               '{:.3f}'),
        'calmar'                     : ('Ratios',    'Calmar Ratio',                '{:.3f}'),
        'omega'                      : ('Ratios',    'Omega Ratio',                 '{:.3f}'),
        # Adding Crisis Labels
        'phase1_max_drawdown'        : ('Crisis',    'Phase 1 Max Drawdown',        '{:.2%}'),
        'phase1_duration_to_trough'  : ('Crisis',    'Phase 1 Duration',            '{:.0f}'),
        'phase2_time_to_breakeven'   : ('Crisis',    'Phase 2 Recovery Time',       '{:.0f}'),
        'phase2_return_speed'        : ('Crisis',    'Phase 2 Avg Daily Return',    '{:.4%}')
    }
    
    rows = []
    for key, value in results.items():
        if key in labels:
            category, name, fmt = labels[key]
            try:
                # Handle NaNs and formatting
                formatted = fmt.format(value) if not (isinstance(value, float) and np.isnan(value)) else 'N/A'
            except:
                formatted = str(value)
            rows.append({'Category': category, 'Metric': name, 'Value': formatted})
            
    return pd.DataFrame(rows)


# --------
# Create the Plots
# --------

def _find_drawdown_episodes(df, threshold):
    """
    Find drawdown episodes where dd breaches -threshold.
    Groups by full underwater periods (dd < 0) so a single drawdown that
    temporarily recovers above the threshold is still counted as ONE episode.
    Returns list of dicts: peak_date, trough_date, recovery_date,
    dd_pct, days_to_trough, days_to_recovery, recovered.
    """
    cum = df['cum_return']
    dd  = df['drawdown']
    episodes = []

    is_underwater = dd < 0
    if not is_underwater.any():
        return episodes

    group_ids = (is_underwater != is_underwater.shift()).cumsum()

    for _, grp in df[is_underwater].groupby(group_ids[is_underwater]):
        # Skip episodes that never breach the threshold
        if grp['drawdown'].min() > -threshold:
            continue
        trough_date = grp['drawdown'].idxmin()
        dd_pct      = dd.loc[trough_date]

        group_start = grp.index[0]
        peak_date   = cum.loc[:group_start].idxmax()
        peak_val    = cum.loc[peak_date]

        post        = cum.loc[trough_date:]
        hits        = post[post >= peak_val]
        if not hits.empty:
            recovery_date = hits.index[0]
            recovered     = True
        else:
            recovery_date = cum.index[-1]
            recovered     = False

        episodes.append({
            'peak_date'        : peak_date,
            'trough_date'      : trough_date,
            'recovery_date'    : recovery_date,
            'dd_pct'           : dd_pct,
            'days_to_trough'   : (pd.Timestamp(trough_date)   - pd.Timestamp(peak_date)).days,
            'days_to_recovery' : (pd.Timestamp(recovery_date) - pd.Timestamp(trough_date)).days,
            'recovered'        : recovered,
        })

    return episodes


def _build_drawdown_table(episodes):
    """
    Build overall stats + quartile breakdown table data from drawdown episodes.
    Returns (header_values, cell_values) ready for go.Table.
    """
    if not episodes:
        return [], []

    dds  = [abs(ep['dd_pct'])           for ep in episodes]
    durs = [ep['days_to_trough']        for ep in episodes]
    recs = [ep['days_to_recovery']      for ep in episodes]

    n = len(episodes)

    # Sort by severity (worst first) and split into quartiles
    sorted_eps = sorted(episodes, key=lambda e: e['dd_pct'])   # most negative first
    q_size     = max(1, n // 4)
    quartiles  = [sorted_eps[i * q_size:(i + 1) * q_size] for i in range(4)]
    # put any remainder into Q4
    if len(sorted_eps) > 4 * q_size:
        quartiles[3] += sorted_eps[4 * q_size:]

    def stats(eps_list):
        if not eps_list:
            return "—", "—", "—", 0
        d  = [abs(e['dd_pct'])      for e in eps_list]
        du = [e['days_to_trough']   for e in eps_list]
        re = [e['days_to_recovery'] for e in eps_list]
        return (
            f"{np.mean(d):.1%}",
            f"{np.mean(du):.0f}d",
            f"{np.mean(re):.0f}d",
            len(eps_list),
        )

    rows = []
    # Overall
    rows.append(("All episodes", f"{n}", f"{np.mean(dds):.1%}", f"{np.mean(durs):.0f}d", f"{np.mean(recs):.0f}d"))
    # Quartiles: Q1 = most severe
    labels = ["Q1 — most severe (top 25%)", "Q2", "Q3", "Q4 — mildest (bottom 25%)"]
    for label, q_eps in zip(labels, quartiles):
        avg_dd, avg_dur, avg_rec, cnt = stats(q_eps)
        rows.append((label, str(cnt), avg_dd, avg_dur, avg_rec))

    headers = ["Quartile", "Count", "Avg DD %", "Avg Duration to Trough", "Avg Recovery Duration"]
    cells   = list(zip(*rows))   # transpose
    return headers, cells


def generate_dynamic_benchmark_report(portfolio_df, output_path, threshold=0.05):
    """Creates a white-themed HTML report: cumulative return + drawdown table, underwater, return distribution."""
    df = portfolio_df.copy()
    df.index = pd.to_datetime(df.index)
    df['cum_return'] = (1 + df['returns_per_day']).cumprod()
    df['drawdown']   = (df['cum_return'] / df['cum_return'].cummax()) - 1

    episodes = _find_drawdown_episodes(df, threshold)
    worst    = min(episodes, key=lambda e: e['dd_pct']) if episodes else None

    # ── Colours ───────────────────────────────────────────────────────────────
    BG        = "#FFFFFF"
    PANEL     = "#F8F9FA"
    GRID      = "#E9ECEF"
    LINE_BLUE = "#1D6FA4"
    RED       = "#DC2626"
    GREEN     = "#16A34A"
    TEXT      = "#1E293B"
    SUBTEXT   = "#64748B"
    TBL_HDR   = "#1D6FA4"
    TBL_ROW_B = "#FFFFFF"
    Q1_COLOR  = "#FEE2E2"   # lightest red
    Q2_COLOR  = "#FEF9C3"
    Q3_COLOR  = "#DCFCE7"
    Q4_COLOR  = "#DBEAFE"   # lightest blue

    # ── Layout: 4 rows (cum return, table, underwater, distribution) ───────────
    # ── Prepare annual + monthly return data ──────────────────────────────────
    df_annual  = df['returns_per_day'].resample('YE').apply(lambda r: (1 + r).prod() - 1)
    df_annual.index = df_annual.index.year

    monthly    = df['returns_per_day'].resample('ME').apply(lambda r: (1 + r).prod() - 1)
    monthly_df = monthly.to_frame('ret')
    monthly_df['year']  = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    heat_pivot = monthly_df.pivot(index='year', columns='month', values='ret')
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    heat_pivot.columns = [month_names[m - 1] for m in heat_pivot.columns]
    heat_z     = heat_pivot.values
    heat_years = [str(y) for y in heat_pivot.index]

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        row_heights=[0.28, 0.12, 0.14, 0.16, 0.14, 0.32],
        subplot_titles=[
            f"Cumulative Return  —  drawdown episodes > {threshold:.0%} highlighted",
            f"Drawdown Episode Summary (threshold {threshold:.0%})",
            "Underwater Plot (Drawdown %)",
            "Daily Return Distribution",
            "Annual Returns",
            "Monthly Returns Heatmap",
        ],
        specs=[
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
    )

    # ── Chart 1: Cumulative Return ─────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['cum_return'],
            mode='lines',
            line=dict(color=LINE_BLUE, width=2),
            name="Cum. Return",
            hovertemplate="%{x|%b %d, %Y}<br>Cum. Return: %{y:.1%}<extra></extra>",
        ),
        row=1, col=1,
    )

    for ep in episodes:
        is_worst    = worst is not None and ep['trough_date'] == worst['trough_date']
        red_alpha   = 0.25 if is_worst else 0.12
        green_alpha = 0.20 if is_worst else 0.10

        fig.add_vrect(
            x0=ep['peak_date'], x1=ep['trough_date'],
            fillcolor=RED, opacity=red_alpha,
            layer="below", line_width=0,
            row=1, col=1,
        )
        fig.add_vrect(
            x0=ep['trough_date'], x1=ep['recovery_date'],
            fillcolor=GREEN, opacity=green_alpha,
            layer="below", line_width=0,
            row=1, col=1,
        )

        rec_label = f"↑ {ep['days_to_recovery']}d" if ep['recovered'] else "↑ n/a"
        prefix    = "<b>★ Worst</b><br>" if is_worst else ""
        ay_offset = -70 if is_worst else -48
        ann_color  = "#991B1B" if is_worst else RED

        fig.add_annotation(
            x=ep['trough_date'],
            y=df.loc[ep['trough_date'], 'cum_return'],
            xref="x", yref="y",
            text=(
                f"{prefix}"
                f"<b>{ep['dd_pct']:.1%}</b><br>"
                f"↓ {ep['days_to_trough']}d  {rec_label}"
            ),
            showarrow=True,
            arrowhead=2,
            arrowcolor=ann_color,
            arrowwidth=1.2,
            arrowsize=0.9,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=ann_color,
            borderwidth=1,
            font=dict(size=9, color=TEXT),
            ax=0, ay=ay_offset,
            row=1, col=1,
        )

    fig.update_yaxes(
        tickformat=".0%", title_text="Cumulative Return",
        gridcolor=GRID, zerolinecolor=GRID,
        title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT),
        row=1, col=1,
    )
    fig.update_xaxes(
        dtick="M12", tickformat="%Y", tickangle=0,
        showgrid=False, tickfont=dict(color=SUBTEXT),
        row=1, col=1,
    )

    # ── Table: Drawdown Summary ────────────────────────────────────────────────
    headers, cells = _build_drawdown_table(episodes)
    if headers:
        row_colors = [TBL_ROW_B, Q1_COLOR, Q2_COLOR, Q3_COLOR, Q4_COLOR]
        fill_colors = [[row_colors[r] for r in range(len(cells[0]))] for _ in cells]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in headers],
                    fill_color=TBL_HDR,
                    font=dict(color="white", size=12, family="Inter, Arial"),
                    align="left",
                    height=30,
                    line_color="white",
                ),
                cells=dict(
                    values=cells,
                    fill_color=fill_colors,
                    font=dict(color=TEXT, size=11, family="Inter, Arial"),
                    align="left",
                    height=26,
                    line_color=GRID,
                ),
            ),
            row=2, col=1,
        )

    # ── Chart 3: Underwater ────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['drawdown'],
            fill='tozeroy',
            mode='lines',
            line=dict(color=RED, width=1),
            fillcolor='rgba(220,38,38,0.20)',
            name="Drawdown",
            hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2%}<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="x2 domain",
        y0=-threshold, y1=-threshold, yref="y2",
        line=dict(dash="dash", color=SUBTEXT, width=1),
    )
    fig.add_annotation(
        x=1, xref="x2 domain",
        y=-threshold, yref="y2",
        text=f"{threshold:.0%} threshold",
        showarrow=False, xanchor="right", yanchor="top",
        font=dict(color=SUBTEXT, size=10),
    )
    fig.update_yaxes(
        tickformat=".0%", title_text="Drawdown",
        gridcolor=GRID, zerolinecolor=GRID,
        title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT),
        row=3, col=1,
    )
    fig.update_xaxes(
        dtick="M12", tickformat="%Y", tickangle=0,
        showgrid=False, tickfont=dict(color=SUBTEXT),
        row=3, col=1,
    )

    # ── Chart 4: Return Distribution ──────────────────────────────────────────
    ret_vals = df['returns_per_day'].dropna()
    neg_vals = ret_vals[ret_vals <  0]
    pos_vals = ret_vals[ret_vals >= 0]
    bin_size = 0.001

    fig.add_trace(
        go.Histogram(
            x=neg_vals, name="Negative",
            xbins=dict(size=bin_size),
            marker_color='rgba(220,38,38,0.70)',
            hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=pos_vals, name="Positive",
            xbins=dict(size=bin_size),
            marker_color='rgba(22,163,74,0.70)',
            hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
        ),
        row=4, col=1,
    )
    fig.add_shape(
        type="line",
        x0=0, x1=0, xref="x3",
        y0=0, y1=1, yref="y3 domain",
        line=dict(dash="dot", color=TEXT, width=1),
    )
    mean_ret = float(ret_vals.mean())
    fig.add_shape(
        type="line",
        x0=mean_ret, x1=mean_ret, xref="x3",
        y0=0, y1=1, yref="y3 domain",
        line=dict(dash="dash", color="#D97706", width=1.2),
    )
    fig.add_annotation(
        x=mean_ret, xref="x3",
        y=1, yref="y3 domain",
        text=f"mean {mean_ret:.3%}",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(color="#D97706", size=10),
    )
    fig.update_xaxes(
        tickformat=".1%", title_text="Daily Return",
        gridcolor=GRID, tickfont=dict(color=SUBTEXT),
        title_font=dict(color=SUBTEXT),
        row=4, col=1,
    )
    fig.update_yaxes(
        title_text="Count", gridcolor=GRID,
        title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT),
        row=4, col=1,
    )

    # ── Chart 5: Annual Returns Bar ───────────────────────────────────────────
    bar_colors = [GREEN if v >= 0 else RED for v in df_annual.values]
    fig.add_trace(
        go.Bar(
            x=[str(y) for y in df_annual.index],
            y=df_annual.values,
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in df_annual.values],
            textposition='outside',
            textfont=dict(size=9, color=TEXT),
            hovertemplate="Year: %{x}<br>Return: %{y:.2%}<extra></extra>",
            name="Annual Return",
        ),
        row=5, col=1,
    )
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="x4 domain",
        y0=0, y1=0, yref="y4",
        line=dict(color=SUBTEXT, width=1),
    )
    fig.update_yaxes(
        tickformat=".0%", title_text="Return",
        gridcolor=GRID, zerolinecolor=GRID,
        title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT),
        row=5, col=1,
    )
    fig.update_xaxes(
        tickfont=dict(color=SUBTEXT), showgrid=False,
        tickangle=-45, row=5, col=1,
    )

    # ── Chart 6: Monthly Heatmap ──────────────────────────────────────────────
    abs_max = float(np.nanmax(np.abs(heat_z)))
    fig.add_trace(
        go.Heatmap(
            z=heat_z,
            x=month_names,
            y=heat_years,
            colorscale=[
                [0.0,  "#DC2626"],
                [0.5,  "#FFFFFF"],
                [1.0,  "#16A34A"],
            ],
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in heat_z],
            texttemplate="%{text}",
            textfont=dict(size=9, color=TEXT),
            hovertemplate="Month: %{x}<br>Year: %{y}<br>Return: %{z:.2%}<extra></extra>",
            showscale=True,
            colorbar=dict(
                tickformat=".0%",
                thickness=12,
                len=0.16,
                y=0.05,
                title=dict(text="Return", font=dict(size=10, color=SUBTEXT)),
                tickfont=dict(size=9, color=SUBTEXT),
            ),
        ),
        row=6, col=1,
    )
    fig.update_yaxes(
        title_text="Year", autorange="reversed",
        title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT, size=10),
        row=6, col=1,
    )
    fig.update_xaxes(
        tickfont=dict(color=SUBTEXT), showgrid=False,
        side="bottom", row=6, col=1,
    )

    # ── Subplot title style ────────────────────────────────────────────────────
    for ann in fig['layout']['annotations']:
        if not ann.showarrow:
            ann['font'] = dict(color=TEXT, size=13, family="Inter, Arial")

    # ── Global layout ──────────────────────────────────────────────────────────
    fig.update_layout(
        height=2500,
        barmode='overlay',
        title=dict(
            text="S&P 500  ·  Buy & Hold Performance Analysis",
            font=dict(size=20, color=TEXT, family="Inter, Arial"),
            x=0.5, xanchor='center', y=0.99,
        ),
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, Arial, sans-serif", size=12, color=TEXT),
        showlegend=False,
        margin=dict(l=70, r=50, t=80, b=50),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_color=TEXT,
            bordercolor=GRID,
        ),
    )
    fig.update_xaxes(linecolor=GRID, mirror=False)
    fig.update_yaxes(linecolor=GRID, mirror=False)

    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"HTML report saved to {output_path}")


# ------
# Export to CSV
# ------

log_ret = portfolio['log_returns_per_day']
price = portfolio['price']

results = compute_all_metrics(
    portfolio_log_returns=log_ret,
    price_series=price,
    benchmark_log_returns=log_ret,
    risk_free_rate=risk_free_rate,
    freq='D',
)

df_metrics = metrics_to_dataframe(results)
df_metrics.to_csv(metrics_file, index=False)

# Change 0.05 to 0.10 for 10%, etc.
threshold_val = 0.05            # Drawdown highlighting
report_path = output_dir_plots / "benchmark_interactive_report.html"
generate_dynamic_benchmark_report(portfolio, report_path, threshold=threshold_val)

print(f"Metrics exported to: {metrics_file}")



print(f"Successfully exported to: {data_file}")