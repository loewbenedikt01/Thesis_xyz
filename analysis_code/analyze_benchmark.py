"""
Benchmark Analysis — v2
========================
Changes vs original:
  1.  3 new crisis periods added (Early Credit Crunch, Acute GFC Crash,
      European Debt + US Debt Ceiling Crisis)
  2.  Peak logic fixed: max drawdown measured relative to the PRICE AT
      window_start, not the all-time high before trough
  3.  calmar_crisis metric added per crisis
  4.  ulcer_index metric added per crisis
  5.  CRISIS_METRIC_LABELS updated to include the two new metrics
  6.  named_crisis_metrics() rewritten to use window_start price as anchor
"""

import pyarrow
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


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

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
start_date = "1998-01-01"
end_date   = "2025-12-31"
ticker     = "^GSPC"

risk_free_rate       = 0.0
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR       = 12

# ─────────────────────────────────────────────────────────────────────────────
# CRISIS PERIODS
# Format: (label, short_key, window_start, trough_date, window_end)
#
# Peak logic (FIX 2):
#   max_drawdown is measured as (trough_price / window_start_price) - 1
#   i.e. the drawdown FROM the start of the defined crisis window,
#   NOT from an all-time high before the trough.
#
# Three new crises added (FIX 1):
#   - Early Credit Crunch    : 2007-10-09 → 2008-09-15 → 2010-04-23
#   - Acute GFC Crash        : 2008-09-15 → 2009-03-09 → 2010-04-23
#   - European Debt + US DC  : 2010-04-23 → 2011-10-03 → 2012-03-23
# ─────────────────────────────────────────────────────────────────────────────
CRISIS_PERIODS = [
    ('Dotcom Crash',                         'dotcom',   '2000-03-23', '2002-10-09', '2007-05-31'),
    ('GFC (Full)',                            'gfc',      '2007-10-09', '2009-03-09', '2013-03-28'),
    ('Early Credit Crunch',                  'ecc',      '2007-10-09', '2008-09-15', '2010-04-23'),
    ('Acute GFC Crash',                      'agfc',     '2008-09-15', '2009-03-09', '2010-04-23'),
    ('European Debt + US Debt Ceiling',      'eu_debt',  '2010-04-23', '2011-10-03', '2012-03-23'),
    ('Monetary Policy',                      'mon_pol',  '2018-09-21', '2018-12-24', '2019-04-23'),
    ('COVID-19',                             'covid19',  '2020-02-19', '2020-03-23', '2020-08-12'),
    ('Russia/Ukraine',                       'russia',   '2022-01-03', '2022-10-12', '2024-01-19'),
    ('Trade Policy Shock',                   'trade',    '2025-02-19', '2025-04-08', '2025-06-26'),
]

# Metrics reported per crisis — order determines CSV column order
CRISIS_METRIC_LABELS = [
    ('max_drawdown',            'Max Drawdown (%)',                 '{:.2%}'),
    ('days_to_trough',          'Days: Peak to Trough',             '{:.0f}'),
    ('days_trough_to_recovery', 'Days: Trough to Breakeven',        '{:.0f}'),
    ('days_peak_to_breakeven',  'Days: Peak to Breakeven (Total)',  '{:.0f}'),
    ('crisis_cum_return',       'Cumulative Return (Crisis)',        '{:.2%}'),
    ('crisis_ann_return',       'Annualized Return (Crisis)',        '{:.2%}'),
    ('crisis_ann_volatility',   'Annualized Volatility (Crisis)',    '{:.2%}'),
    ('crisis_sharpe',           'Sharpe Ratio (Crisis)',             '{:.3f}'),
    ('crisis_sortino',          'Sortino Ratio (Crisis)',            '{:.3f}'),
    ('crisis_calmar',           'Calmar Ratio (Crisis)',             '{:.3f}'),   # FIX 3
    ('crisis_ulcer_index',      'Ulcer Index (Crisis)',              '{:.4f}'),   # FIX 4
]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
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
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _annualized_factor(freq: str) -> int:
    if freq == 'D':
        return TRADING_DAYS_PER_YEAR
    elif freq == 'M':
        return MONTHS_PER_YEAR
    raise ValueError(f"freq must be 'D' or 'M', got '{freq}'")


# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def cumulative_return(log_returns: pd.Series) -> float:
    return np.exp(log_returns.sum()) - 1

def annualized_return(log_returns: pd.Series, freq: str = 'D') -> float:
    n = len(log_returns)
    ann_f = _annualized_factor(freq)
    return np.exp((log_returns.sum() / n) * ann_f) - 1

def annualized_volatility(log_returns: pd.Series, freq: str = 'D') -> float:
    return log_returns.std(ddof=1) * np.sqrt(_annualized_factor(freq))

def maximum_drawdown(price_series: pd.Series) -> float:
    rolling_max = price_series.cummax()
    return ((price_series / rolling_max) - 1).min()

def drawdown_series(price_series: pd.Series) -> pd.Series:
    rolling_max = price_series.cummax()
    return (price_series / rolling_max) - 1

def max_drawdown_duration(price_series: pd.Series) -> int:
    rolling_max  = price_series.cummax()
    is_at_peak   = price_series >= rolling_max
    underwater_groups = is_at_peak.cumsum()
    durations    = price_series.groupby(underwater_groups).cumcount()
    return durations.max()

def recovery_duration(price_series: pd.Series) -> int:
    rolling_max  = price_series.cummax()
    drawdown     = (price_series / rolling_max) - 1
    mdd_date     = drawdown.idxmin()
    peak_at_mdd  = rolling_max.loc[mdd_date]
    recovery_ser = price_series.loc[mdd_date:]
    hits         = recovery_ser[recovery_ser >= peak_at_mdd]
    if hits.empty:
        return len(recovery_ser)
    return len(price_series.loc[mdd_date:hits.index[0]]) - 1

def value_at_risk(log_returns: pd.Series, confidence_level: float = 0.95) -> float:
    return log_returns.quantile(1 - confidence_level)

def conditional_value_at_risk(log_returns: pd.Series, confidence_level: float = 0.95) -> float:
    var = value_at_risk(log_returns, confidence_level)
    return log_returns[log_returns <= var].mean()

def sharpe_ratio(log_returns: pd.Series, rfr: float, freq: str = 'D') -> float:
    return (annualized_return(log_returns, freq) - rfr) / annualized_volatility(log_returns, freq)

def sortino_ratio(log_returns: pd.Series, rfr: float, freq: str = 'D') -> float:
    ann_ret       = annualized_return(log_returns, freq)
    ann_f         = _annualized_factor(freq)
    down_ret      = log_returns[log_returns < 0]
    downside_vol  = np.sqrt((down_ret ** 2).mean()) * np.sqrt(ann_f)
    return (ann_ret - rfr) / downside_vol if downside_vol != 0 else np.nan

def calmar_ratio(log_returns: pd.Series, price_series: pd.Series, freq: str = 'D') -> float:
    ann_ret = annualized_return(log_returns, freq)
    mdd     = abs(maximum_drawdown(price_series))
    return ann_ret / mdd if mdd != 0 else np.nan

def omega_ratio(log_returns: pd.Series, threshold: float = 0.0) -> float:
    gains  = log_returns[log_returns >  threshold].sum()
    losses = abs(log_returns[log_returns < threshold].sum())
    return gains / losses if losses != 0 else np.inf

def ulcer_index(price_series: pd.Series) -> float:
    """
    Ulcer Index = sqrt(mean(drawdown²))
    Measures both depth and duration of drawdowns.
    All drawdowns are measured from the running peak (standard definition).
    """
    rolling_max = price_series.cummax()
    dd          = (price_series / rolling_max) - 1      # always <= 0
    return float(np.sqrt((dd ** 2).mean()))


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 + 3 + 4: CRISIS METRICS — peak anchored to window_start price
# ─────────────────────────────────────────────────────────────────────────────
def named_crisis_metrics(log_returns: pd.Series, price_series: pd.Series,
                          freq: str = 'D') -> dict:
    """
    For each crisis in CRISIS_PERIODS:

    Peak logic (FIX 2):
        The 'peak' is defined as the price on window_start (the first available
        trading day >= the defined start date). Max drawdown is computed as
        (trough_price / window_start_price) - 1, i.e. the drop from the moment
        the crisis begins — NOT from the all-time high before the trough.

    Additional metrics (FIX 3 + 4):
        crisis_calmar    : annualized return / |max drawdown within window|
        crisis_ulcer_index: ulcer index computed on the crisis price window
    """
    results = {}

    for _, crisis_key, window_start, defined_trough, window_end in CRISIS_PERIODS:

        # ── Align dates to actual trading days ───────────────────────────────
        idx = price_series.index

        start_hits = idx[idx >= window_start]
        if start_hits.empty:
            continue
        actual_start = start_hits[0]

        trough_hits = idx[idx >= defined_trough]
        if trough_hits.empty:
            continue
        actual_trough = trough_hits[0]

        end_hits = idx[idx >= window_end]
        actual_end = end_hits[0] if not end_hits.empty else idx[-1]

        # ── Crisis window price series ────────────────────────────────────────
        crisis_prices = price_series.loc[actual_start:actual_end]
        crisis_ret    = log_returns.loc[actual_start:actual_end]

        if crisis_prices.empty or len(crisis_ret) < 2:
            continue

        # ── FIX 2: anchor drawdown to window_start price ──────────────────────
        # Max drawdown = lowest price in window relative to the opening price
        anchor_price  = crisis_prices.iloc[0]          # price on the crisis start date
        trough_price  = crisis_prices.min()            # lowest price in window
        trough_date   = crisis_prices.idxmin()         # date of that trough

        max_dd        = (trough_price / anchor_price) - 1   # always <= 0

        # Days: start → trough
        days_to_trough = (pd.Timestamp(trough_date) - pd.Timestamp(actual_start)).days

        # Days: trough → recovery back to anchor_price (search full series, not just window)
        post_trough    = price_series.loc[trough_date:]
        recovered_hits = post_trough[post_trough >= anchor_price]
        if not recovered_hits.empty:
            recovery_date           = recovered_hits.index[0]
            days_trough_to_recovery = (pd.Timestamp(recovery_date) - pd.Timestamp(trough_date)).days
            days_peak_to_breakeven  = (pd.Timestamp(recovery_date) - pd.Timestamp(actual_start)).days
        else:
            days_trough_to_recovery = np.nan
            days_peak_to_breakeven  = np.nan

        # ── Standard crisis window metrics ────────────────────────────────────
        ann_ret = annualized_return(crisis_ret, freq)   if len(crisis_ret) > 1 else np.nan
        ann_vol = annualized_volatility(crisis_ret, freq) if len(crisis_ret) > 1 else np.nan

        def _crisis_sortino(lr):
            _ann_ret = annualized_return(lr, freq)
            _ds_vol  = np.sqrt((np.minimum(lr, 0) ** 2).mean()) * np.sqrt(_annualized_factor(freq))
            return _ann_ret / _ds_vol if _ds_vol != 0 else np.nan

        # FIX 3: calmar within crisis window
        crisis_calmar = ann_ret / abs(max_dd) if (not np.isnan(ann_ret) and max_dd != 0) else np.nan

        # FIX 4: ulcer index within crisis window
        # Drawdowns computed relative to the running peak WITHIN the window
        crisis_ulcer = ulcer_index(crisis_prices)

        k = crisis_key
        results[f'{k}_max_drawdown']            = max_dd
        results[f'{k}_days_to_trough']          = days_to_trough
        results[f'{k}_days_trough_to_recovery'] = days_trough_to_recovery
        results[f'{k}_days_peak_to_breakeven']  = days_peak_to_breakeven
        results[f'{k}_crisis_cum_return']       = cumulative_return(crisis_ret)
        results[f'{k}_crisis_ann_return']       = ann_ret
        results[f'{k}_crisis_ann_volatility']   = ann_vol
        results[f'{k}_crisis_sharpe']           = sharpe_ratio(crisis_ret, 0.0, freq) if len(crisis_ret) > 1 else np.nan
        results[f'{k}_crisis_sortino']          = _crisis_sortino(crisis_ret)          if len(crisis_ret) > 1 else np.nan
        results[f'{k}_crisis_calmar']           = crisis_calmar                         # FIX 3
        results[f'{k}_crisis_ulcer_index']      = crisis_ulcer                          # FIX 4

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MASTER METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(portfolio_log_returns: pd.Series,
                        price_series: pd.Series,
                        benchmark_log_returns: pd.Series,
                        rfr: float,
                        freq: str = 'D') -> dict:
    results = {}
    log_ret, bench_ret = portfolio_log_returns.align(benchmark_log_returns, join='inner')

    results['cumulative_return']     = cumulative_return(log_ret)
    results['annualized_return']     = annualized_return(log_ret, freq)
    results['benchmark_cum_return']  = cumulative_return(bench_ret)
    results['benchmark_ann_return']  = annualized_return(bench_ret, freq)
    results['annualized_volatility'] = annualized_volatility(log_ret, freq)
    results['maximum_drawdown']      = maximum_drawdown(price_series)
    results['dd_duration_to_trough'] = max_drawdown_duration(price_series)
    results['recovery_duration']     = recovery_duration(price_series)
    results['var_95']                = value_at_risk(log_ret, 0.95)
    results['cvar_95']               = conditional_value_at_risk(log_ret, 0.95)
    results['sharpe']                = sharpe_ratio(log_ret, rfr, freq)
    results['sortino']               = sortino_ratio(log_ret, rfr, freq)
    results['calmar']                = calmar_ratio(log_ret, price_series, freq)
    results['omega']                 = omega_ratio(log_ret)
    results['ulcer_index']           = ulcer_index(price_series)

    results.update(named_crisis_metrics(log_ret, price_series, freq))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# METRICS → DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
def metrics_to_dataframe(results: dict) -> pd.DataFrame:
    labels = {
        'cumulative_return'     : ('Returns',   'Cumulative Return',           '{:.2%}'),
        'annualized_return'     : ('Returns',   'Annualized Return',           '{:.2%}'),
        'benchmark_cum_return'  : ('Benchmark', 'Benchmark Cumulative Return', '{:.2%}'),
        'benchmark_ann_return'  : ('Benchmark', 'Benchmark Annualized Return', '{:.2%}'),
        'annualized_volatility' : ('Risk',      'Annualized Volatility',       '{:.2%}'),
        'maximum_drawdown'      : ('Risk',      'Maximum Drawdown',            '{:.2%}'),
        'var_95'                : ('Risk',      'Value at Risk (95%)',         '{:.2%}'),
        'cvar_95'               : ('Risk',      'CVaR / Expected Shortfall',   '{:.2%}'),
        'ulcer_index'           : ('Risk',      'Ulcer Index (Full Period)',    '{:.4f}'),
        'sharpe'                : ('Ratios',    'Sharpe Ratio',                '{:.3f}'),
        'sortino'               : ('Ratios',    'Sortino Ratio',               '{:.3f}'),
        'calmar'                : ('Ratios',    'Calmar Ratio',                '{:.3f}'),
        'omega'                 : ('Ratios',    'Omega Ratio',                 '{:.3f}'),
    }

    crisis_labels = {}
    for crisis_name, crisis_key, _, _, _ in CRISIS_PERIODS:
        for metric_suffix, metric_label, fmt in CRISIS_METRIC_LABELS:
            crisis_labels[f'{crisis_key}_{metric_suffix}'] = (crisis_name, metric_label, fmt)

    def _fmt(value, fmt):
        try:
            return fmt.format(value) if not (isinstance(value, float) and np.isnan(value)) else 'N/A'
        except Exception:
            return str(value)

    rows = []
    for key, value in results.items():
        if key in labels:
            cat, name, fmt = labels[key]
            rows.append({'Category': cat, 'Metric': name, 'Value': _fmt(value, fmt)})
        elif key in crisis_labels:
            cat, name, fmt = crisis_labels[key]
            rows.append({'Category': cat, 'Metric': name, 'Value': _fmt(value, fmt)})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def _find_drawdown_episodes(df, threshold):
    cum = df['cum_return']
    dd  = df['drawdown']
    episodes = []
    is_underwater = dd < 0
    if not is_underwater.any():
        return episodes
    group_ids = (is_underwater != is_underwater.shift()).cumsum()
    for _, grp in df[is_underwater].groupby(group_ids[is_underwater]):
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
            'peak_date'       : peak_date,
            'trough_date'     : trough_date,
            'recovery_date'   : recovery_date,
            'dd_pct'          : dd_pct,
            'days_to_trough'  : (pd.Timestamp(trough_date)   - pd.Timestamp(peak_date)).days,
            'days_to_recovery': (pd.Timestamp(recovery_date) - pd.Timestamp(trough_date)).days,
            'recovered'       : recovered,
        })
    return episodes


def _build_drawdown_table(episodes):
    if not episodes:
        return [], []
    dds  = [abs(ep['dd_pct'])      for ep in episodes]
    durs = [ep['days_to_trough']   for ep in episodes]
    recs = [ep['days_to_recovery'] for ep in episodes]
    n    = len(episodes)
    sorted_eps = sorted(episodes, key=lambda e: e['dd_pct'])
    q_size     = max(1, n // 4)
    quartiles  = [sorted_eps[i * q_size:(i + 1) * q_size] for i in range(4)]
    if len(sorted_eps) > 4 * q_size:
        quartiles[3] += sorted_eps[4 * q_size:]

    def stats(eps_list):
        if not eps_list:
            return "—", "—", "—", 0
        d  = [abs(e['dd_pct'])      for e in eps_list]
        du = [e['days_to_trough']   for e in eps_list]
        re = [e['days_to_recovery'] for e in eps_list]
        return f"{np.mean(d):.1%}", f"{np.mean(du):.0f}d", f"{np.mean(re):.0f}d", len(eps_list)

    rows = [("All episodes", f"{n}", f"{np.mean(dds):.1%}", f"{np.mean(durs):.0f}d", f"{np.mean(recs):.0f}d")]
    labels_q = ["Q1 — most severe (top 25%)", "Q2", "Q3", "Q4 — mildest (bottom 25%)"]
    for lbl, q_eps in zip(labels_q, quartiles):
        avg_dd, avg_dur, avg_rec, cnt = stats(q_eps)
        rows.append((lbl, str(cnt), avg_dd, avg_dur, avg_rec))

    headers = ["Quartile", "Count", "Avg DD %", "Avg Duration to Trough", "Avg Recovery Duration"]
    cells   = list(zip(*rows))
    return headers, cells


def generate_dynamic_benchmark_report(portfolio_df, output_path, precomputed_results: dict, threshold=0.05):
    df = portfolio_df.copy()
    df.index = pd.to_datetime(df.index)
    df['cum_return'] = (1 + df['returns_per_day']).cumprod()
    df['drawdown']   = (df['cum_return'] / df['cum_return'].cummax()) - 1

    episodes = _find_drawdown_episodes(df, threshold)
    worst    = min(episodes, key=lambda e: e['dd_pct']) if episodes else None

    _named_eps = []
    for _cname, _, _c_start, _c_trough, _c_end in CRISIS_PERIODS:
        _idx = df.index
        _pd  = _idx[_idx >= pd.Timestamp(_c_start)][0]   if any(_idx >= pd.Timestamp(_c_start))  else None
        _td  = _idx[_idx >= pd.Timestamp(_c_trough)][0]  if any(_idx >= pd.Timestamp(_c_trough)) else None
        _rd  = _idx[_idx >= pd.Timestamp(_c_end)][0]     if any(_idx >= pd.Timestamp(_c_end))    else _idx[-1]
        if _pd is None or _td is None:
            continue
        _pv = df['cum_return'].loc[_pd]
        _tv = df['cum_return'].loc[_td]
        _named_eps.append({'name': _cname, 'peak_date': _pd, 'trough_date': _td,
                           'recovery_date': _rd, 'dd_pct': (_tv / _pv) - 1})

    BG        = "#FFFFFF"; PANEL     = "#F8F9FA"; GRID      = "#E9ECEF"
    LINE_BLUE = "#1D6FA4"; RED       = "#DC2626"; GREEN     = "#16A34A"
    TEXT      = "#1E293B"; SUBTEXT   = "#64748B"; TBL_HDR   = "#1D6FA4"
    TBL_ROW_B = "#FFFFFF"
    Q1_COLOR  = "#FEE2E2"; Q2_COLOR  = "#FEF9C3"
    Q3_COLOR  = "#DCFCE7"; Q4_COLOR  = "#DBEAFE"

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

    # ── Use precomputed crisis metrics (already computed in RUN section) ────────
    _crisis_results = precomputed_results

    # Columns: one per crisis; rows: each metric
    _crisis_col_names  = [cp[0] for cp in CRISIS_PERIODS]   # full names
    _crisis_keys       = [cp[1] for cp in CRISIS_PERIODS]

    # Row labels = metric names; one column per crisis + one Overall column
    _tbl_metrics = [
        ('max_drawdown',            'Max Drawdown',         '{:.2%}'),
        ('days_to_trough',          'Days to Trough',       '{:.0f}'),
        ('days_trough_to_recovery', 'Days to Recovery',     '{:.0f}'),
        ('days_peak_to_breakeven',  'Days Peak→Breakeven',  '{:.0f}'),
        ('crisis_cum_return',       'Cum Return',           '{:.2%}'),
        ('crisis_ann_return',       'Ann Return',           '{:.2%}'),
        ('crisis_ann_volatility',   'Ann Volatility',       '{:.2%}'),
        ('crisis_sharpe',           'Sharpe',               '{:.3f}'),
        ('crisis_sortino',          'Sortino',              '{:.3f}'),
        ('crisis_calmar',           'Calmar',               '{:.3f}'),
        ('crisis_ulcer_index',      'Ulcer Index',          '{:.4f}'),
    ]

    def _fv(v, fmt):
        try:
            return fmt.format(v) if not (isinstance(v, float) and np.isnan(v)) else 'N/A'
        except Exception:
            return 'N/A'

    # header row: Metric | Crisis1 | Crisis2 | ...
    _tbl_header = ['Metric'] + _crisis_col_names
    # one list per column (plotly table format)
    _metric_col   = [m[1] for m in _tbl_metrics]  # row labels
    _crisis_cols  = []
    for ck in _crisis_keys:
        col_vals = []
        for suffix, _, fmt in _tbl_metrics:
            key = f'{ck}_{suffix}'
            col_vals.append(_fv(_crisis_results.get(key, np.nan), fmt))
        _crisis_cols.append(col_vals)

    _tbl_cells_values = [_metric_col] + _crisis_cols

    # Alternating row fill colours for readability
    _n_rows = len(_tbl_metrics)
    _row_fills = ['#F8F9FA' if i % 2 == 0 else '#FFFFFF' for i in range(_n_rows)]
    _tbl_fill  = [_row_fills] * len(_tbl_cells_values)  # same pattern every column

    fig = make_subplots(
        rows=7, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        row_heights=[0.25, 0.08, 0.12, 0.12, 0.14, 0.12, 0.30],
        subplot_titles=[
            f"Cumulative Return  —  drawdown episodes > {threshold:.0%} highlighted",
            f"Drawdown Episode Summary (threshold {threshold:.0%})",
            "Named Crisis Periods — Key Metrics",
            "Underwater Plot (Drawdown %)",
            "Daily Return Distribution",
            "Annual Returns",
            "Monthly Returns Heatmap",
        ],
        specs=[
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
    )

    # Chart 1: Cumulative Return
    fig.add_trace(go.Scatter(x=df.index, y=df['cum_return'], mode='lines',
                             line=dict(color=LINE_BLUE, width=2), name="Cum. Return",
                             hovertemplate="%{x|%b %d, %Y}<br>Cum. Return: %{y:.1%}<extra></extra>"),
                  row=1, col=1)
    for ep in episodes:
        is_worst = worst is not None and ep['trough_date'] == worst['trough_date']
        fig.add_vrect(x0=ep['peak_date'], x1=ep['trough_date'],
                      fillcolor=RED, opacity=0.25 if is_worst else 0.12,
                      layer="below", line_width=0, row=1, col=1)
        fig.add_vrect(x0=ep['trough_date'], x1=ep['recovery_date'],
                      fillcolor=GREEN, opacity=0.20 if is_worst else 0.10,
                      layer="below", line_width=0, row=1, col=1)
        rec_label = f"↑ {ep['days_to_recovery']}d" if ep['recovered'] else "↑ n/a"
        prefix    = "<b>★ Worst</b><br>" if is_worst else ""
        fig.add_annotation(x=ep['trough_date'], y=df.loc[ep['trough_date'], 'cum_return'],
                           xref="x", yref="y",
                           text=f"{prefix}<b>{ep['dd_pct']:.1%}</b><br>↓ {ep['days_to_trough']}d  {rec_label}",
                           showarrow=True, arrowhead=2,
                           arrowcolor="#991B1B" if is_worst else RED,
                           arrowwidth=1.2, arrowsize=0.9,
                           bgcolor="rgba(255,255,255,0.92)",
                           bordercolor="#991B1B" if is_worst else RED, borderwidth=1,
                           font=dict(size=9, color=TEXT), ax=0, ay=-70 if is_worst else -48,
                           row=1, col=1)
    _y_top = float(df['cum_return'].max())
    for _ep in _named_eps:
        fig.add_vrect(x0=_ep['peak_date'], x1=_ep['trough_date'],
                      fillcolor='rgba(220,38,38,0.40)', opacity=1,
                      layer='below', line_width=1.5, line_color='black', row=1, col=1)
        fig.add_vrect(x0=_ep['trough_date'], x1=_ep['recovery_date'],
                      fillcolor='rgba(22,163,74,0.28)', opacity=1,
                      layer='below', line_width=1.5, line_color='black', row=1, col=1)
        fig.add_annotation(x=_ep['peak_date'], y=_y_top, xref='x', yref='y',
                           text=f"<b>{_ep['name']}</b>", showarrow=False,
                           xanchor='left', yanchor='top',
                           font=dict(size=9, color='black', family='Inter, Arial'),
                           bgcolor='rgba(255,255,255,0.85)',
                           bordercolor='black', borderwidth=1)
    fig.update_yaxes(tickformat=".0%", title_text="Cumulative Return",
                     gridcolor=GRID, zerolinecolor=GRID,
                     title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT), row=1, col=1)
    fig.update_xaxes(dtick="M12", tickformat="%Y", tickangle=0,
                     showgrid=False, tickfont=dict(color=SUBTEXT), row=1, col=1)

    # ── Table row 2: Drawdown Episode Summary ────────────────────────────────
    headers, cells = _build_drawdown_table(episodes)
    if headers:
        row_colors  = [TBL_ROW_B, Q1_COLOR, Q2_COLOR, Q3_COLOR, Q4_COLOR]
        fill_colors = [[row_colors[r] for r in range(len(cells[0]))] for _ in cells]
        fig.add_trace(go.Table(
            header=dict(values=[f"<b>{h}</b>" for h in headers],
                        fill_color=TBL_HDR, font=dict(color="white", size=12),
                        align="left", height=30, line_color="white"),
            cells=dict(values=cells, fill_color=fill_colors,
                       font=dict(color=TEXT, size=11), align="left",
                       height=26, line_color=GRID)), row=2, col=1)

    # ── Table row 3: Named Crisis Metrics ────────────────────────────────────
    fig.add_trace(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in _tbl_header],
            fill_color=TBL_HDR,
            font=dict(color="white", size=11, family="Inter, Arial"),
            align="left", height=28, line_color="white",
        ),
        cells=dict(
            values=_tbl_cells_values,
            fill_color=_tbl_fill,
            font=dict(color=TEXT, size=10, family="Inter, Arial"),
            align="left", height=24, line_color=GRID,
        ),
    ), row=3, col=1)

    # ── Chart 4: Underwater ───────────────────────────────────────────────────
    # NOTE: two tables in rows 2 and 3 means the first xy after them is x2/y2
    fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], fill='tozeroy', mode='lines',
                             line=dict(color=RED, width=1), fillcolor='rgba(220,38,38,0.20)',
                             name="Drawdown",
                             hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2%}<extra></extra>"),
                  row=4, col=1)
    fig.add_shape(type="line", x0=0, x1=1, xref="x2 domain",
                  y0=-threshold, y1=-threshold, yref="y2",
                  line=dict(dash="dash", color=SUBTEXT, width=1))
    fig.add_annotation(x=1, xref="x2 domain", y=-threshold, yref="y2",
                       text=f"{threshold:.0%} threshold",
                       showarrow=False, xanchor="right", yanchor="top",
                       font=dict(color=SUBTEXT, size=10))
    for _ep in _named_eps:
        fig.add_shape(type='rect', x0=_ep['peak_date'], x1=_ep['trough_date'], y0=0, y1=1,
                      xref='x2', yref='y2 domain',
                      fillcolor='rgba(220,38,38,0.30)', opacity=1,
                      layer='below', line_width=1.5, line_color='black')
        fig.add_shape(type='rect', x0=_ep['trough_date'], x1=_ep['recovery_date'], y0=0, y1=1,
                      xref='x2', yref='y2 domain',
                      fillcolor='rgba(22,163,74,0.20)', opacity=1,
                      layer='below', line_width=1.5, line_color='black')
    fig.update_yaxes(tickformat=".0%", title_text="Drawdown", gridcolor=GRID, zerolinecolor=GRID,
                     title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT), row=4, col=1)
    fig.update_xaxes(dtick="M12", tickformat="%Y", tickangle=0,
                     showgrid=False, tickfont=dict(color=SUBTEXT), row=4, col=1)

    # ── Chart 5: Return Distribution ─────────────────────────────────────────
    ret_vals = df['returns_per_day'].dropna()
    bin_size = 0.001
    fig.add_trace(go.Histogram(x=ret_vals[ret_vals < 0], name="Negative",
                               xbins=dict(size=bin_size), marker_color='rgba(220,38,38,0.70)',
                               hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>"),
                  row=5, col=1)
    fig.add_trace(go.Histogram(x=ret_vals[ret_vals >= 0], name="Positive",
                               xbins=dict(size=bin_size), marker_color='rgba(22,163,74,0.70)',
                               hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>"),
                  row=5, col=1)
    fig.add_shape(type="line", x0=0, x1=0, xref="x3", y0=0, y1=1, yref="y3 domain",
                  line=dict(dash="dot", color=TEXT, width=1))
    mean_ret = float(ret_vals.mean())
    fig.add_shape(type="line", x0=mean_ret, x1=mean_ret, xref="x3", y0=0, y1=1,
                  yref="y3 domain", line=dict(dash="dash", color="#D97706", width=1.2))
    fig.add_annotation(x=mean_ret, xref="x3", y=1, yref="y3 domain",
                       text=f"mean {mean_ret:.3%}", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color="#D97706", size=10))
    fig.update_xaxes(tickformat=".1%", title_text="Daily Return", gridcolor=GRID,
                     tickfont=dict(color=SUBTEXT), title_font=dict(color=SUBTEXT), row=5, col=1)
    fig.update_yaxes(title_text="Count", gridcolor=GRID,
                     title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT), row=5, col=1)

    # ── Chart 6: Annual Returns ───────────────────────────────────────────────
    bar_colors = [GREEN if v >= 0 else RED for v in df_annual.values]
    fig.add_trace(go.Bar(x=[str(y) for y in df_annual.index], y=df_annual.values,
                         marker_color=bar_colors,
                         text=[f"{v:.1%}" for v in df_annual.values],
                         textposition='outside', textfont=dict(size=9, color=TEXT),
                         hovertemplate="Year: %{x}<br>Return: %{y:.2%}<extra></extra>",
                         name="Annual Return"), row=6, col=1)
    fig.add_shape(type="line", x0=0, x1=1, xref="x4 domain", y0=0, y1=0, yref="y4",
                  line=dict(color=SUBTEXT, width=1))
    fig.update_yaxes(tickformat=".0%", title_text="Return", gridcolor=GRID, zerolinecolor=GRID,
                     title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT), row=6, col=1)
    fig.update_xaxes(tickfont=dict(color=SUBTEXT), showgrid=False, tickangle=-45, row=6, col=1)

    # ── Chart 7: Monthly Heatmap ──────────────────────────────────────────────
    abs_max = float(np.nanmax(np.abs(heat_z)))
    fig.add_trace(go.Heatmap(
        z=heat_z, x=month_names, y=heat_years,
        colorscale=[[0.0, "#DC2626"], [0.5, "#FFFFFF"], [1.0, "#16A34A"]],
        zmid=0, zmin=-abs_max, zmax=abs_max,
        text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in heat_z],
        texttemplate="%{text}", textfont=dict(size=9, color=TEXT),
        hovertemplate="Month: %{x}<br>Year: %{y}<br>Return: %{z:.2%}<extra></extra>",
        showscale=True,
        colorbar=dict(tickformat=".0%", thickness=12, len=0.16, y=0.05,
                      title=dict(text="Return", font=dict(size=10, color=SUBTEXT)),
                      tickfont=dict(size=9, color=SUBTEXT))), row=7, col=1)
    fig.update_yaxes(title_text="Year", autorange="reversed",
                     title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT, size=10),
                     row=7, col=1)
    fig.update_xaxes(tickfont=dict(color=SUBTEXT), showgrid=False, side="bottom", row=7, col=1)

    for ann in fig['layout']['annotations']:
        if not ann.showarrow:
            ann['font'] = dict(color=TEXT, size=13, family="Inter, Arial")

    fig.update_layout(
        height=2500, barmode='overlay',
        title=dict(text="S&P 500  ·  Buy & Hold Performance Analysis",
                   font=dict(size=20, color=TEXT, family="Inter, Arial"),
                   x=0.5, xanchor='center', y=0.99),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="Inter, Arial, sans-serif", size=12, color=TEXT),
        showlegend=False,
        margin=dict(l=70, r=50, t=80, b=50),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color=TEXT, bordercolor=GRID),
    )
    fig.update_xaxes(linecolor=GRID, mirror=False)
    fig.update_yaxes(linecolor=GRID, mirror=False)
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"HTML report saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
log_ret = portfolio['log_returns_per_day']
price   = portfolio['price']

# Step 1: compute all metrics
results = compute_all_metrics(
    portfolio_log_returns = log_ret,
    price_series          = price,
    benchmark_log_returns = log_ret,   # S&P 500 IS the benchmark for this script
    rfr                   = risk_free_rate,
    freq                  = 'D',
)

# Step 2: export CSV
df_metrics = metrics_to_dataframe(results)
df_metrics.to_csv(metrics_file, index=False)
print(f"Metrics exported to: {metrics_file}")

# Step 3: print console summary (before HTML so output is visible even if HTML crashes)
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

# Step 4: generate HTML report (uses results for the crisis metrics table)
threshold_val = 0.05
report_path   = output_dir_plots / "benchmark_interactive_report.html"
generate_dynamic_benchmark_report(portfolio, report_path, precomputed_results=results, threshold=threshold_val)

print(f"Data file: {data_file}")
print(f"Metrics file: {metrics_file}")
print(f"Report: {report_path}")