"""
metrics.py — Single Source of Truth for All Portfolio Metrics
==============================================================
Import this module in every analysis script instead of duplicating formulas.

Usage in an analysis script:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from metrics import (
        compute_all_metrics,
        metrics_to_dataframe,
        generate_dynamic_benchmark_report,
        generate_crisis_comparison_charts,
        CRISIS_PERIODS,
        CRISIS_METRIC_LABELS,
    )

═══════════════════════════════════════════════════════════════════════════════
WHAT IS CALCULATED
═══════════════════════════════════════════════════════════════════════════════

A. CONSTANTS
   ─────────────────────────────────────────────────────────────────────────────
   TRADING_DAYS_PER_YEAR   252 trading days used for annualisation
   MONTHS_PER_YEAR         12
   CRISIS_PERIODS          9 named crisis windows with start / trough / end dates
   CRISIS_METRIC_LABELS    11 per-crisis metrics with display names and formats

B. RETURN METRICS
   ─────────────────────────────────────────────────────────────────────────────
   cumulative_return       Total return over the full period
                           Formula: exp(sum(log_returns)) - 1

   annualized_return       Geometric annualised return
                           Formula: exp((sum(log_returns) / n) * ann_factor) - 1

C. RISK METRICS
   ─────────────────────────────────────────────────────────────────────────────
   annualized_volatility   Annualised standard deviation of log returns
                           Formula: std(log_returns, ddof=1) * sqrt(ann_factor)

   maximum_drawdown        Worst peak-to-trough decline in price
                           Formula: min((price / cummax(price)) - 1)

   drawdown_series         Full time series of drawdown at each date

   max_drawdown_duration   Longest consecutive period spent below a prior peak
                           (in trading days)

   recovery_duration       Days from the global MDD trough back to the
                           pre-trough peak level

   value_at_risk           Historical VaR at the given confidence level
                           (default 95%). Returns the threshold daily loss
                           exceeded only 5% of the time.
                           Formula: quantile(log_returns, 1 - confidence)

   conditional_value_at_risk  Expected Shortfall (CVaR / ES) at 95%.
                           Average loss on the worst 5% of days.
                           Formula: mean(log_returns <= VaR threshold)

   ulcer_index             sqrt(mean(drawdown²)) — penalises both depth and
                           duration of being underwater. Higher = worse.
                           Formula: sqrt(mean(((price / cummax) - 1)²))

D. RISK-ADJUSTED RATIOS
   ─────────────────────────────────────────────────────────────────────────────
   sharpe_ratio            (annualized_return - rfr) / annualized_volatility
                           Uses full distribution (upside + downside).

   sortino_ratio           (annualized_return - rfr) / downside_deviation
                           Only penalises negative returns. Downside deviation
                           = sqrt(mean(min(r, 0)²)) * sqrt(ann_factor)

   calmar_ratio            annualized_return / |maximum_drawdown|
                           Reward per unit of worst observed drawdown.

   omega_ratio             sum(gains above threshold) / sum(losses below threshold)
                           Probability-weighted gain/loss ratio. > 1 = net positive.

E. BENCHMARK-RELATIVE METRICS
   ─────────────────────────────────────────────────────────────────────────────
   beta                    Sensitivity of portfolio returns to benchmark returns.
                           Formula: cov(p, b) / var(b). > 1 = more volatile.

   alpha                   Jensen's alpha — excess return vs CAPM expectation.
                           Formula: ann_port - (rfr + beta * (ann_bench - rfr))

   information_ratio       Active return / tracking error (annualised).
                           Active return = portfolio - benchmark per period.
                           Higher = more consistent alpha per unit of active risk.

   modified_information_ratio  Active annualised return / |MDD of active returns|
                           Uses drawdown instead of tracking error as the
                           denominator — penalises sustained underperformance.

   treynor_ratio           (annualized_return - rfr) / beta
                           Return per unit of systematic (market) risk only.

F. CRISIS-PERIOD METRICS  (computed for each of 9 named crises)
   ─────────────────────────────────────────────────────────────────────────────
   Peak logic: anchor = price on window_start (NOT the all-time high).
   Drawdown = (trough / anchor) - 1. Recovery = first date >= anchor.

   max_drawdown            Drop from window_start price to the intra-window trough
   days_to_trough          Calendar days from window_start to trough date
   days_trough_to_recovery Calendar days from trough back to anchor price level
   days_peak_to_breakeven  Total calendar days from window_start to recovery
   crisis_cum_return       Cumulative return over the full crisis window
   crisis_ann_return       Annualised return over the full crisis window
   crisis_ann_volatility   Annualised volatility over the full crisis window
   crisis_sharpe           Sharpe ratio within the crisis window (rfr = 0)
   crisis_sortino          Sortino ratio within the crisis window
   crisis_calmar           crisis_ann_return / |max_drawdown within window|
   crisis_ulcer_index      Ulcer index computed on the crisis price window

G. PLOT / REPORT FUNCTIONS
   ─────────────────────────────────────────────────────────────────────────────
   generate_dynamic_benchmark_report()
       7-panel HTML report:
         1. Cumulative return with algorithmic + named crisis shading
         2. Drawdown episode summary table (quartile breakdown)
         3. Named crisis metrics table (all 9 crises × 11 metrics)
         4. Underwater (drawdown) chart
         5. Daily return distribution histogram
         6. Annual returns bar chart
         7. Monthly returns heatmap

   generate_crisis_comparison_charts()
       Per-crisis HTML charts: portfolio vs benchmark normalised to 1.0 at
       crisis start, plus a side-by-side metrics comparison table.

   _find_drawdown_episodes()   Internal: detect all drawdown episodes above threshold
   _build_drawdown_table()     Internal: build quartile summary table data

H. MASTER COMPUTATION FUNCTIONS
   ─────────────────────────────────────────────────────────────────────────────
   compute_all_metrics()   Runs all metrics A–F and returns a flat dict.
                           Accepts portfolio log returns, price series,
                           benchmark log returns, risk-free rate, frequency.

   metrics_to_dataframe()  Converts the results dict to a formatted
                           Category / Metric / Value DataFrame for CSV export.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# A. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR       = 12

# Crisis periods — (label, short_key, window_start, trough_date, window_end)
# Peak/drawdown logic: anchor = price on window_start, NOT all-time high.
CRISIS_PERIODS = [
    ('Dotcom Crash',                    'dotcom',   '2000-03-23', '2002-10-09', '2007-05-31'),
    ('GFC (Full)',                       'gfc',      '2007-10-09', '2009-03-09', '2013-03-28'),
    ('Early Credit Crunch',             'ecc',      '2007-10-09', '2008-09-15', '2010-04-23'),
    ('Acute GFC Crash',                 'agfc',     '2008-09-15', '2009-03-09', '2010-04-23'),
    ('European Debt + US Debt Ceiling', 'eu_debt',  '2010-04-23', '2011-10-03', '2012-03-23'),
    ('Monetary Policy',                 'mon_pol',  '2018-09-21', '2018-12-24', '2019-04-23'),
    ('COVID-19',                        'covid19',  '2020-02-19', '2020-03-23', '2020-08-12'),
    ('Russia/Ukraine',                  'russia',   '2022-01-03', '2022-10-12', '2024-01-19'),
    ('Trade Policy Shock',              'trade',    '2025-02-19', '2025-04-08', '2025-06-26'),
]

# Per-crisis metric definitions: (key_suffix, display_label, format_string)
CRISIS_METRIC_LABELS = [
    ('max_drawdown',            'Max Drawdown (%)',                '{:.2%}'),
    ('days_to_trough',          'Days: Peak to Trough',            '{:.0f}'),
    ('days_trough_to_recovery', 'Days: Trough to Breakeven',       '{:.0f}'),
    ('days_peak_to_breakeven',  'Days: Peak to Breakeven (Total)', '{:.0f}'),
    ('crisis_cum_return',       'Cumulative Return (Crisis)',       '{:.2%}'),
    ('crisis_ann_return',       'Annualized Return (Crisis)',       '{:.2%}'),
    ('crisis_ann_volatility',   'Annualized Volatility (Crisis)',   '{:.2%}'),
    ('crisis_sharpe',           'Sharpe Ratio (Crisis)',            '{:.3f}'),
    ('crisis_sortino',          'Sortino Ratio (Crisis)',           '{:.3f}'),
    ('crisis_calmar',           'Calmar Ratio (Crisis)',            '{:.3f}'),
    ('crisis_ulcer_index',      'Ulcer Index (Crisis)',             '{:.4f}'),
]


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _annualized_factor(freq: str) -> int:
    if freq == 'D':
        return TRADING_DAYS_PER_YEAR
    elif freq == 'M':
        return MONTHS_PER_YEAR
    raise ValueError(f"freq must be 'D' or 'M', got '{freq}'")


# ─────────────────────────────────────────────────────────────────────────────
# B. RETURN METRICS
# ─────────────────────────────────────────────────────────────────────────────
def cumulative_return(log_returns: pd.Series) -> float:
    """Total cumulative return: exp(sum(log_returns)) - 1"""
    return float(np.exp(log_returns.sum()) - 1)


def annualized_return(log_returns: pd.Series, freq: str = 'D') -> float:
    """Geometric annualised return accounting for compounding."""
    n     = len(log_returns)
    ann_f = _annualized_factor(freq)
    return float(np.exp((log_returns.sum() / n) * ann_f) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# C. RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
def annualized_volatility(log_returns: pd.Series, freq: str = 'D') -> float:
    """Annualised standard deviation of log returns (ddof=1)."""
    return float(log_returns.std(ddof=1) * np.sqrt(_annualized_factor(freq)))


def maximum_drawdown(price_series: pd.Series) -> float:
    """Worst peak-to-trough decline in the price series. Returns a negative number."""
    rolling_max = price_series.cummax()
    return float(((price_series / rolling_max) - 1).min())


def drawdown_series(price_series: pd.Series) -> pd.Series:
    """Full time series of drawdown at each date (always <= 0)."""
    rolling_max = price_series.cummax()
    return (price_series / rolling_max) - 1


def max_drawdown_duration(price_series: pd.Series) -> int:
    """
    Longest consecutive run of trading days spent below a prior peak.
    Measures how long an investor stays underwater at its worst stretch.
    """
    rolling_max       = price_series.cummax()
    is_at_peak        = price_series >= rolling_max
    underwater_groups = is_at_peak.cumsum()
    durations         = price_series.groupby(underwater_groups).cumcount()
    return int(durations.max())


def recovery_duration(price_series: pd.Series) -> int:
    """
    Trading days from the global MDD trough back to the pre-trough peak level.
    If the series never recovers, returns remaining length from trough.
    """
    rolling_max  = price_series.cummax()
    drawdown     = (price_series / rolling_max) - 1
    mdd_date     = drawdown.idxmin()
    peak_at_mdd  = rolling_max.loc[mdd_date]
    recovery_ser = price_series.loc[mdd_date:]
    hits         = recovery_ser[recovery_ser >= peak_at_mdd]
    if hits.empty:
        return len(recovery_ser)
    return int(len(price_series.loc[mdd_date:hits.index[0]]) - 1)


def value_at_risk(log_returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.
    Returns the daily loss threshold exceeded only (1 - confidence_level)% of days.
    """
    return float(log_returns.quantile(1 - confidence_level))


def conditional_value_at_risk(log_returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR / ES) at the given confidence level.
    Average return on the worst (1 - confidence_level)% of days.
    """
    var = value_at_risk(log_returns, confidence_level)
    return float(log_returns[log_returns <= var].mean())


def ulcer_index(price_series: pd.Series) -> float:
    """
    Ulcer Index = sqrt(mean(drawdown²)).
    Measures both depth and duration of being underwater.
    All drawdowns computed from the running peak (standard definition).
    Higher values = more painful drawdown profile.
    """
    rolling_max = price_series.cummax()
    dd          = (price_series / rolling_max) - 1
    return float(np.sqrt((dd ** 2).mean()))


# ─────────────────────────────────────────────────────────────────────────────
# D. RISK-ADJUSTED RATIOS
# ─────────────────────────────────────────────────────────────────────────────
def sharpe_ratio(log_returns: pd.Series, rfr: float = 0.0,
                 freq: str = 'D') -> float:
    """
    Annualised Sharpe Ratio.
    (annualized_return - rfr) / annualized_volatility.
    Uses full return distribution (not downside-only).
    """
    ann_ret = annualized_return(log_returns, freq)
    ann_vol = annualized_volatility(log_returns, freq)
    return float((ann_ret - rfr) / ann_vol) if ann_vol != 0 else np.nan


def sortino_ratio(log_returns: pd.Series, rfr: float = 0.0,
                  freq: str = 'D') -> float:
    """
    Annualised Sortino Ratio.
    (annualized_return - rfr) / downside_deviation.
    Downside deviation = sqrt(mean(min(r, 0)²)) * sqrt(ann_factor).
    Penalises only negative returns, not upside volatility.
    """
    ann_ret      = annualized_return(log_returns, freq)
    ann_f        = _annualized_factor(freq)
    downside_vol = np.sqrt((np.minimum(log_returns, 0) ** 2).mean()) * np.sqrt(ann_f)
    return float((ann_ret - rfr) / downside_vol) if downside_vol != 0 else np.nan


def calmar_ratio(log_returns: pd.Series, price_series: pd.Series,
                 freq: str = 'D') -> float:
    """
    Calmar Ratio = annualized_return / |maximum_drawdown|.
    Reward per unit of the worst observed drawdown.
    """
    ann_ret = annualized_return(log_returns, freq)
    mdd     = abs(maximum_drawdown(price_series))
    return float(ann_ret / mdd) if mdd != 0 else np.nan


def omega_ratio(log_returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega Ratio = sum(gains above threshold) / sum(losses below threshold).
    Probability-weighted ratio of gains to losses. Value > 1 = net positive.
    """
    gains  = log_returns[log_returns >  threshold].sum()
    losses = abs(log_returns[log_returns < threshold].sum())
    return float(gains / losses) if losses != 0 else np.inf


# ─────────────────────────────────────────────────────────────────────────────
# E. BENCHMARK-RELATIVE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Beta — sensitivity of portfolio returns to benchmark returns.
    Formula: cov(p, b) / var(b).
    Beta > 1: amplifies market moves. Beta < 1: dampens them.
    """
    aligned_p, aligned_b = portfolio_returns.align(benchmark_returns, join='inner')
    cov = np.cov(aligned_p.values, aligned_b.values)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan


def alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
          rfr: float = 0.0, freq: str = 'D') -> float:
    """
    Jensen's Alpha — annualised excess return vs CAPM expectation.
    Formula: ann_port - (rfr + beta * (ann_bench - rfr)).
    Positive alpha = portfolio outperforms its risk-adjusted benchmark expectation.
    """
    b      = beta(portfolio_returns, benchmark_returns)
    ann_p  = annualized_return(portfolio_returns, freq)
    ann_bm = annualized_return(benchmark_returns, freq)
    return float(ann_p - (rfr + b * (ann_bm - rfr)))


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                      freq: str = 'D') -> float:
    """
    Information Ratio = annualised active return / tracking error.
    Active return = portfolio return - benchmark return per period.
    Tracking error = annualised std of active returns.
    Higher IR = more consistent alpha delivery per unit of active risk.
    """
    ann_f                = _annualized_factor(freq)
    aligned_p, aligned_b = portfolio_returns.align(benchmark_returns, join='inner')
    active_returns       = aligned_p - aligned_b
    if active_returns.std() == 0:
        return np.nan
    return float((active_returns.mean() / active_returns.std()) * np.sqrt(ann_f))


def modified_information_ratio(portfolio_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                freq: str = 'D') -> float:
    """
    Modified IR = annualised active return / |MDD of active returns|.
    Replaces tracking error with maximum drawdown of active returns.
    Penalises sustained underperformance more severely than random noise.
    """
    aligned_p, aligned_b = portfolio_returns.align(benchmark_returns, join='inner')
    active_returns       = aligned_p - aligned_b
    ann_active           = annualized_return(active_returns, freq)
    max_dd               = maximum_drawdown(active_returns)
    return float(ann_active / abs(max_dd)) if max_dd != 0 else np.nan


def treynor_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                  rfr: float = 0.0, freq: str = 'D') -> float:
    """
    Treynor Ratio = (annualized_return - rfr) / beta.
    Return per unit of systematic (market) risk.
    Useful for comparing diversified portfolios where idiosyncratic risk is low.
    """
    b       = beta(portfolio_returns, benchmark_returns)
    ann_ret = annualized_return(portfolio_returns, freq)
    return float((ann_ret - rfr) / b) if b != 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# F. CRISIS-PERIOD METRICS
# ─────────────────────────────────────────────────────────────────────────────
def named_crisis_metrics(log_returns: pd.Series, price_series: pd.Series,
                          freq: str = 'D') -> dict:
    """
    Compute 11 metrics for each crisis in CRISIS_PERIODS.

    Peak logic:
        anchor_price = price on window_start (first available trading day).
        max_drawdown = (trough_price / anchor_price) - 1.
        Recovery = first date >= anchor_price after the trough
                   (searched over the FULL series, not just the window,
                   so long recoveries are captured correctly).
    """
    results = {}

    log_returns  = log_returns.copy()
    price_series = price_series.copy()
    log_returns.index  = pd.to_datetime(log_returns.index)
    price_series.index = pd.to_datetime(price_series.index)

    for _, crisis_key, window_start, defined_trough, window_end in CRISIS_PERIODS:
        idx = price_series.index

        start_hits = idx[idx >= pd.Timestamp(window_start)]
        if start_hits.empty:
            continue
        actual_start = start_hits[0]

        trough_hits = idx[idx >= pd.Timestamp(defined_trough)]
        if trough_hits.empty:
            continue

        end_hits   = idx[idx >= pd.Timestamp(window_end)]
        actual_end = end_hits[0] if not end_hits.empty else idx[-1]

        crisis_prices = price_series.loc[actual_start:actual_end]
        crisis_ret    = log_returns.loc[actual_start:actual_end]

        if crisis_prices.empty or len(crisis_ret) < 2:
            continue

        # ── Anchor-based drawdown ─────────────────────────────────────────────
        anchor_price = crisis_prices.iloc[0]
        trough_price = crisis_prices.min()
        trough_date  = crisis_prices.idxmin()
        max_dd       = (trough_price / anchor_price) - 1

        days_to_trough = (pd.Timestamp(trough_date) - pd.Timestamp(actual_start)).days

        post_trough    = price_series.loc[trough_date:]
        recovered_hits = post_trough[post_trough >= anchor_price]
        if not recovered_hits.empty:
            recovery_date           = recovered_hits.index[0]
            days_trough_to_recovery = (pd.Timestamp(recovery_date) - pd.Timestamp(trough_date)).days
            days_peak_to_breakeven  = (pd.Timestamp(recovery_date) - pd.Timestamp(actual_start)).days
        else:
            days_trough_to_recovery = np.nan
            days_peak_to_breakeven  = np.nan

        # ── Per-crisis statistics ─────────────────────────────────────────────
        ann_ret = annualized_return(crisis_ret, freq)    if len(crisis_ret) > 1 else np.nan
        ann_vol = annualized_volatility(crisis_ret, freq) if len(crisis_ret) > 1 else np.nan

        crisis_calmar = (ann_ret / abs(max_dd)
                         if (not np.isnan(ann_ret) and max_dd != 0)
                         else np.nan)
        crisis_ulcer  = ulcer_index(crisis_prices)

        k = crisis_key
        results[f'{k}_max_drawdown']            = max_dd
        results[f'{k}_days_to_trough']          = days_to_trough
        results[f'{k}_days_trough_to_recovery'] = days_trough_to_recovery
        results[f'{k}_days_peak_to_breakeven']  = days_peak_to_breakeven
        results[f'{k}_crisis_cum_return']       = cumulative_return(crisis_ret)
        results[f'{k}_crisis_ann_return']       = ann_ret
        results[f'{k}_crisis_ann_volatility']   = ann_vol
        results[f'{k}_crisis_sharpe']           = (sharpe_ratio(crisis_ret, 0.0, freq)
                                                    if len(crisis_ret) > 1 else np.nan)
        results[f'{k}_crisis_sortino']          = (sortino_ratio(crisis_ret, 0.0, freq)
                                                    if len(crisis_ret) > 1 else np.nan)
        results[f'{k}_crisis_calmar']           = crisis_calmar
        results[f'{k}_crisis_ulcer_index']      = crisis_ulcer

    return results


# ─────────────────────────────────────────────────────────────────────────────
# H. MASTER COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(portfolio_log_returns: pd.Series,
                        price_series: pd.Series,
                        benchmark_log_returns: pd.Series,
                        rfr: float = 0.0,
                        freq: str = 'D') -> dict:
    """
    Run all metrics and return a flat results dict.

    Parameters
    ----------
    portfolio_log_returns   log returns of the portfolio
    price_series            portfolio cumulative value series (for drawdown metrics)
    benchmark_log_returns   log returns of the benchmark (e.g. S&P 500)
    rfr                     risk-free rate (annualised, default 0.0)
    freq                    'D' for daily, 'M' for monthly
    """
    results = {}
    log_ret, bench_ret = portfolio_log_returns.align(benchmark_log_returns, join='inner')
    price_aligned      = price_series.reindex(log_ret.index).ffill()

    # ── Returns ───────────────────────────────────────────────────────────────
    results['cumulative_return']     = cumulative_return(log_ret)
    results['annualized_return']     = annualized_return(log_ret, freq)
    results['benchmark_cum_return']  = cumulative_return(bench_ret)
    results['benchmark_ann_return']  = annualized_return(bench_ret, freq)

    # ── Risk ──────────────────────────────────────────────────────────────────
    results['annualized_volatility'] = annualized_volatility(log_ret, freq)
    results['maximum_drawdown']      = maximum_drawdown(price_aligned)
    results['dd_duration_to_trough'] = max_drawdown_duration(price_aligned)
    results['recovery_duration']     = recovery_duration(price_aligned)
    results['var_95']                = value_at_risk(log_ret, 0.95)
    results['cvar_95']               = conditional_value_at_risk(log_ret, 0.95)
    results['ulcer_index']           = ulcer_index(price_aligned)

    # ── Risk-adjusted ratios ──────────────────────────────────────────────────
    results['sharpe']                = sharpe_ratio(log_ret, rfr, freq)
    results['sortino']               = sortino_ratio(log_ret, rfr, freq)
    results['calmar']                = calmar_ratio(log_ret, price_aligned, freq)
    results['omega']                 = omega_ratio(log_ret)

    # ── Benchmark-relative ────────────────────────────────────────────────────
    results['beta']                  = beta(log_ret, bench_ret)
    results['alpha']                 = alpha(log_ret, bench_ret, rfr, freq)
    results['information_ratio']     = information_ratio(log_ret, bench_ret, freq)
    results['modified_ir']           = modified_information_ratio(log_ret, bench_ret, freq)
    results['treynor']               = treynor_ratio(log_ret, bench_ret, rfr, freq)

    # ── Crisis metrics ────────────────────────────────────────────────────────
    results.update(named_crisis_metrics(log_ret, price_aligned, freq))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# H. METRICS → DATAFRAME  (for CSV export)
# ─────────────────────────────────────────────────────────────────────────────
def metrics_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert the results dict to a Category / Metric / Value DataFrame."""
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
        'beta'                  : ('Benchmark', 'Beta',                        '{:.3f}'),
        'alpha'                 : ('Benchmark', 'Alpha (Annualized)',          '{:.2%}'),
        'information_ratio'     : ('Benchmark', 'Information Ratio',           '{:.3f}'),
        'modified_ir'           : ('Benchmark', 'Modified Information Ratio',  '{:.3f}'),
        'treynor'               : ('Benchmark', 'Treynor Ratio',               '{:.3f}'),
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
# G. PLOT HELPERS — INTERNAL
# ─────────────────────────────────────────────────────────────────────────────
def _find_drawdown_episodes(df: pd.DataFrame, threshold: float) -> list:
    """
    Find all drawdown episodes where the drawdown breaches -threshold.
    Groups by continuous underwater periods so a single drawdown that
    temporarily recovers above the threshold still counts as ONE episode.
    """
    cum           = df['cum_return']
    dd            = df['drawdown']
    episodes      = []
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


def _build_drawdown_table(episodes: list) -> tuple:
    """Build quartile breakdown table data. Returns (headers, cells) for go.Table."""
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

    def _stats(eps_list):
        if not eps_list:
            return "—", "—", "—", 0
        d  = [abs(e['dd_pct'])      for e in eps_list]
        du = [e['days_to_trough']   for e in eps_list]
        re = [e['days_to_recovery'] for e in eps_list]
        return f"{np.mean(d):.1%}", f"{np.mean(du):.0f}d", f"{np.mean(re):.0f}d", len(eps_list)

    rows = [("All episodes", f"{n}", f"{np.mean(dds):.1%}",
             f"{np.mean(durs):.0f}d", f"{np.mean(recs):.0f}d")]
    qlbls = ["Q1 — most severe (top 25%)", "Q2", "Q3", "Q4 — mildest (bottom 25%)"]
    for lbl, q_eps in zip(qlbls, quartiles):
        avg_dd, avg_dur, avg_rec, cnt = _stats(q_eps)
        rows.append((lbl, str(cnt), avg_dd, avg_dur, avg_rec))

    headers = ["Quartile", "Count", "Avg DD %", "Avg Duration to Trough", "Avg Recovery Duration"]
    return headers, list(zip(*rows))


# ─────────────────────────────────────────────────────────────────────────────
# G. PLOT FUNCTIONS — PUBLIC
# ─────────────────────────────────────────────────────────────────────────────
def generate_dynamic_benchmark_report(portfolio_df: pd.DataFrame,
                                       precomputed_results: dict,
                                       title: str = "Portfolio Performance Analysis",
                                       threshold: float = 0.05) -> str:
    """
    Generate a 7-panel interactive HTML report as a string (use include_plotlyjs='cdn').

    Panels:
      1. Cumulative return — algorithmic drawdown episodes + named crisis bands
      2. Drawdown episode summary table (quartile breakdown)
      3. Named crisis metrics table (9 crises × 11 metrics)
      4. Underwater plot (drawdown %)
      5. Daily return distribution histogram (negative/positive coloured)
      6. Annual returns bar chart
      7. Monthly returns heatmap

    Parameters
    ----------
    portfolio_df         Must contain 'returns_per_day' or 'log_return' column,
                         and optionally 'cumulative_value'.
    precomputed_results  Output of compute_all_metrics() — used for the crisis table.
    title                Chart title string.
    threshold            Drawdown threshold for episode detection (default 5%).
    """
    df = portfolio_df.copy()
    df.index = pd.to_datetime(df.index)

    if 'returns_per_day' not in df.columns:
        df['returns_per_day'] = np.exp(df['log_return']) - 1
    if 'cum_return' not in df.columns:
        df['cum_return'] = (df['cumulative_value'] if 'cumulative_value' in df.columns
                            else (1 + df['returns_per_day']).cumprod())
    df['drawdown'] = (df['cum_return'] / df['cum_return'].cummax()) - 1

    episodes = _find_drawdown_episodes(df, threshold)
    worst    = min(episodes, key=lambda e: e['dd_pct']) if episodes else None

    _named_eps = []
    for _cname, _, _c_start, _c_trough, _c_end in CRISIS_PERIODS:
        _idx = df.index
        _pd  = _idx[_idx >= pd.Timestamp(_c_start)][0]   if any(_idx >= pd.Timestamp(_c_start))  else None
        _td  = _idx[_idx >= pd.Timestamp(_c_trough)][0]  if any(_idx >= pd.Timestamp(_c_trough)) else None
        _rd  = (_idx[_idx >= pd.Timestamp(_c_end)][0]
                if any(_idx >= pd.Timestamp(_c_end)) else _idx[-1])
        if _pd is None or _td is None:
            continue
        _pv = df['cum_return'].loc[_pd]
        _tv = df['cum_return'].loc[_td]
        _named_eps.append({'name': _cname, 'peak_date': _pd, 'trough_date': _td,
                           'recovery_date': _rd, 'dd_pct': (_tv / _pv) - 1})

    # ── Colours ───────────────────────────────────────────────────────────────
    BG        = "#FFFFFF"; PANEL     = "#F8F9FA"; GRID      = "#E9ECEF"
    LINE_BLUE = "#1D6FA4"; RED       = "#DC2626"; GREEN     = "#16A34A"
    TEXT      = "#1E293B"; SUBTEXT   = "#64748B"; TBL_HDR   = "#1D6FA4"
    TBL_ROW_B = "#FFFFFF"
    Q1_COLOR  = "#FEE2E2"; Q2_COLOR  = "#FEF9C3"
    Q3_COLOR  = "#DCFCE7"; Q4_COLOR  = "#DBEAFE"

    # ── Annual + monthly data ─────────────────────────────────────────────────
    df_annual  = df['returns_per_day'].resample('YE').apply(lambda r: (1 + r).prod() - 1)
    df_annual.index = df_annual.index.year
    monthly         = df['returns_per_day'].resample('ME').apply(lambda r: (1 + r).prod() - 1)
    monthly_df      = monthly.to_frame('ret')
    monthly_df['year']  = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    heat_pivot      = monthly_df.pivot(index='year', columns='month', values='ret')
    month_names     = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    heat_pivot.columns = [month_names[m - 1] for m in heat_pivot.columns]
    heat_z          = heat_pivot.values
    heat_years      = [str(y) for y in heat_pivot.index]

    # ── Named crisis metrics table data ──────────────────────────────────────
    _crisis_col_names = [cp[0] for cp in CRISIS_PERIODS]
    _crisis_keys      = [cp[1] for cp in CRISIS_PERIODS]
    _tbl_m = [
        ('max_drawdown',            'Max Drawdown',        '{:.2%}'),
        ('days_to_trough',          'Days to Trough',      '{:.0f}'),
        ('days_trough_to_recovery', 'Days to Recovery',    '{:.0f}'),
        ('days_peak_to_breakeven',  'Days Peak→Breakeven', '{:.0f}'),
        ('crisis_cum_return',       'Cum Return',          '{:.2%}'),
        ('crisis_ann_return',       'Ann Return',          '{:.2%}'),
        ('crisis_ann_volatility',   'Ann Volatility',      '{:.2%}'),
        ('crisis_sharpe',           'Sharpe',              '{:.3f}'),
        ('crisis_sortino',          'Sortino',             '{:.3f}'),
        ('crisis_calmar',           'Calmar',              '{:.3f}'),
        ('crisis_ulcer_index',      'Ulcer Index',         '{:.4f}'),
    ]

    def _fv(v, fmt):
        try:
            return fmt.format(v) if not (isinstance(v, float) and np.isnan(v)) else 'N/A'
        except Exception:
            return 'N/A'

    _tbl_header  = ['Metric'] + _crisis_col_names
    _metric_col  = [m[1] for m in _tbl_m]
    _crisis_cols = [[_fv(precomputed_results.get(f'{ck}_{s}', np.nan), f)
                     for s, _, f in _tbl_m] for ck in _crisis_keys]
    _tbl_values  = [_metric_col] + _crisis_cols
    _row_fills   = ['#F8F9FA' if i % 2 == 0 else '#FFFFFF' for i in range(len(_tbl_m))]
    _tbl_fill    = [_row_fills] * len(_tbl_values)

    # ── 7-row subplot ─────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=False, vertical_spacing=0.04,
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
        specs=[[{"type": "xy"}], [{"type": "table"}], [{"type": "table"}],
               [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
    )

    # Row 1: Cumulative Return
    fig.add_trace(go.Scatter(x=df.index, y=df['cum_return'], mode='lines',
                             line=dict(color=LINE_BLUE, width=2), name="Cum. Return",
                             hovertemplate="%{x|%b %d, %Y}<br>Cum. Return: %{y:.1%}<extra></extra>"),
                  row=1, col=1)
    for ep in episodes:
        iw = worst is not None and ep['trough_date'] == worst['trough_date']
        fig.add_vrect(x0=ep['peak_date'], x1=ep['trough_date'],
                      fillcolor=RED, opacity=0.25 if iw else 0.12,
                      layer="below", line_width=0, row=1, col=1)
        fig.add_vrect(x0=ep['trough_date'], x1=ep['recovery_date'],
                      fillcolor=GREEN, opacity=0.20 if iw else 0.10,
                      layer="below", line_width=0, row=1, col=1)
        rec_label = f"↑ {ep['days_to_recovery']}d" if ep['recovered'] else "↑ n/a"
        fig.add_annotation(x=ep['trough_date'], y=df.loc[ep['trough_date'], 'cum_return'],
                           xref="x", yref="y",
                           text=f"{'<b>★ Worst</b><br>' if iw else ''}"
                                f"<b>{ep['dd_pct']:.1%}</b><br>"
                                f"↓ {ep['days_to_trough']}d  {rec_label}",
                           showarrow=True, arrowhead=2,
                           arrowcolor="#991B1B" if iw else RED,
                           arrowwidth=1.2, arrowsize=0.9,
                           bgcolor="rgba(255,255,255,0.92)",
                           bordercolor="#991B1B" if iw else RED, borderwidth=1,
                           font=dict(size=9, color=TEXT), ax=0, ay=-70 if iw else -48,
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
                           bgcolor='rgba(255,255,255,0.85)', bordercolor='black', borderwidth=1)
    fig.update_yaxes(tickformat=".0%", title_text="Cumulative Return",
                     gridcolor=GRID, zerolinecolor=GRID,
                     title_font=dict(color=SUBTEXT), tickfont=dict(color=SUBTEXT), row=1, col=1)
    fig.update_xaxes(dtick="M12", tickformat="%Y", tickangle=0,
                     showgrid=False, tickfont=dict(color=SUBTEXT), row=1, col=1)

    # Row 2: Drawdown Episode Table
    headers, cells = _build_drawdown_table(episodes)
    if headers:
        rc = [TBL_ROW_B, Q1_COLOR, Q2_COLOR, Q3_COLOR, Q4_COLOR]
        fc = [[rc[r] for r in range(len(cells[0]))] for _ in cells]
        fig.add_trace(go.Table(
            header=dict(values=[f"<b>{h}</b>" for h in headers],
                        fill_color=TBL_HDR, font=dict(color="white", size=12, family="Inter, Arial"),
                        align="left", height=30, line_color="white"),
            cells=dict(values=cells, fill_color=fc,
                       font=dict(color=TEXT, size=11, family="Inter, Arial"),
                       align="left", height=26, line_color=GRID)), row=2, col=1)

    # Row 3: Named Crisis Metrics Table
    fig.add_trace(go.Table(
        header=dict(values=[f"<b>{h}</b>" for h in _tbl_header],
                    fill_color=TBL_HDR, font=dict(color="white", size=11, family="Inter, Arial"),
                    align="left", height=28, line_color="white"),
        cells=dict(values=_tbl_values, fill_color=_tbl_fill,
                   font=dict(color=TEXT, size=10, family="Inter, Arial"),
                   align="left", height=24, line_color=GRID)), row=3, col=1)

    # Row 4: Underwater (two tables → first xy = x2/y2)
    fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], fill='tozeroy', mode='lines',
                             line=dict(color=RED, width=1), fillcolor='rgba(220,38,38,0.20)',
                             name="Drawdown",
                             hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2%}<extra></extra>"),
                  row=4, col=1)
    fig.add_shape(type="line", x0=0, x1=1, xref="x2 domain",
                  y0=-threshold, y1=-threshold, yref="y2",
                  line=dict(dash="dash", color=SUBTEXT, width=1))
    fig.add_annotation(x=1, xref="x2 domain", y=-threshold, yref="y2",
                       text=f"{threshold:.0%} threshold", showarrow=False,
                       xanchor="right", yanchor="top", font=dict(color=SUBTEXT, size=10))
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

    # Row 5: Return Distribution
    ret_vals = df['returns_per_day'].dropna()
    fig.add_trace(go.Histogram(x=ret_vals[ret_vals < 0], name="Negative",
                               xbins=dict(size=0.001), marker_color='rgba(220,38,38,0.70)',
                               hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>"),
                  row=5, col=1)
    fig.add_trace(go.Histogram(x=ret_vals[ret_vals >= 0], name="Positive",
                               xbins=dict(size=0.001), marker_color='rgba(22,163,74,0.70)',
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

    # Row 6: Annual Returns
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

    # Row 7: Monthly Heatmap
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
        height=2800, barmode='overlay',
        title=dict(text=title, font=dict(size=20, color=TEXT, family="Inter, Arial"),
                   x=0.5, xanchor='center', y=0.99),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="Inter, Arial, sans-serif", size=12, color=TEXT),
        showlegend=False, margin=dict(l=70, r=50, t=80, b=50),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color=TEXT, bordercolor=GRID),
    )
    fig.update_xaxes(linecolor=GRID, mirror=False)
    fig.update_yaxes(linecolor=GRID, mirror=False)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def generate_crisis_comparison_charts(portfolio_df: pd.DataFrame,
                                       benchmark_df: pd.DataFrame,
                                       title_prefix: str = "") -> str:
    """
    For each crisis in CRISIS_PERIODS, generate a chart (portfolio vs benchmark,
    normalised to 1.0 at crisis start) and a side-by-side metrics table.
    Returns a combined HTML string (no CDN tag — caller must include plotlyjs once).

    Metrics compared per crisis:
        Cumulative Return, Annualised Return, Annualised Volatility,
        Max Drawdown, Sharpe, Sortino, Calmar, Ulcer Index, VaR 95%, CVaR 95%, Omega.
    Color coding: green = portfolio better, red = portfolio worse.
    """
    BG      = "#FFFFFF"; PANEL   = "#F8F9FA"; GRID    = "#E9ECEF"
    TEXT    = "#1E293B"; SUBTEXT = "#64748B"
    PORT_C  = "#1D6FA4"; BENCH_C = "#F97316"

    bm_log   = benchmark_df['log_returns_per_day'].sort_index()
    port_log = portfolio_df['log_return'].sort_index()
    divs     = []

    def _m(lr):
        price = np.exp(lr.cumsum())
        vol   = annualized_volatility(lr)
        sr    = annualized_return(lr) / vol if vol != 0 and len(lr) > 1 else np.nan
        dv    = np.sqrt((np.minimum(lr, 0)**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        so    = annualized_return(lr) / dv if dv != 0 else np.nan
        mdd   = abs(maximum_drawdown(price))
        ca    = annualized_return(lr) / mdd if mdd != 0 else np.nan
        return {
            'Cumulative Return'     : f"{cumulative_return(lr):.2%}",
            'Annualized Return'     : f"{annualized_return(lr):.2%}",
            'Annualized Volatility' : f"{vol:.2%}",
            'Max Drawdown'          : f"{maximum_drawdown(price):.2%}",
            'Sharpe Ratio'          : f"{sr:.3f}" if not np.isnan(sr) else 'N/A',
            'Sortino Ratio'         : f"{so:.3f}" if not np.isnan(so) else 'N/A',
            'Calmar Ratio'          : f"{ca:.3f}" if not np.isnan(ca) else 'N/A',
            'Ulcer Index'           : f"{ulcer_index(price):.4f}",
            'VaR 95%'               : f"{value_at_risk(lr, 0.95):.2%}",
            'CVaR 95%'              : f"{conditional_value_at_risk(lr, 0.95):.2%}",
            'Omega Ratio'           : f"{omega_ratio(lr):.3f}",
        }

    def _color(name, pv_str, bv_str):
        try:
            pv = float(pv_str.replace('%', ''))
            bv = float(bv_str.replace('%', ''))
        except Exception:
            return TEXT, TEXT
        lower_better = name in ('Annualized Volatility', 'Ulcer Index')
        if pv > bv:
            return ('#166534' if not lower_better else '#DC2626', TEXT)
        elif pv < bv:
            return ('#DC2626' if not lower_better else '#166534', TEXT)
        return TEXT, TEXT

    for cname, _, c_start, _, c_end in CRISIS_PERIODS:
        p_lr = port_log.loc[c_start:c_end].dropna()
        b_lr = bm_log.loc[c_start:c_end].dropna()
        if p_lr.empty or b_lr.empty:
            divs.append(f'<p style="font-family:Inter,Arial;color:{SUBTEXT};margin:20px 70px">'
                        f'{cname}: insufficient data.</p>')
            continue

        p_lr, b_lr = p_lr.align(b_lr, join='inner')
        p_cum = np.exp(p_lr.cumsum())
        b_cum = np.exp(b_lr.cumsum())
        p_m   = _m(p_lr)
        b_m   = _m(b_lr)
        mnames = list(p_m.keys())
        pc    = [_color(mn, p_m[mn], b_m[mn])[0] for mn in mnames]
        bc    = [_color(mn, p_m[mn], b_m[mn])[1] for mn in mnames]

        fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45],
                            vertical_spacing=0.06,
                            specs=[[{"type": "xy"}], [{"type": "table"}]],
                            subplot_titles=[
                                f"{cname}  ·  Cumulative Return  ({c_start} → {c_end})",
                                "Metrics Comparison"])

        fig.add_trace(go.Scatter(x=p_cum.index, y=p_cum.values, mode='lines',
                                 name='Portfolio', line=dict(color=PORT_C, width=2.5),
                                 hovertemplate="%{x|%b %d, %Y}<br>Portfolio: %{y:.3f}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=b_cum.index, y=b_cum.values, mode='lines',
                                 name='Benchmark (S&P 500)',
                                 line=dict(color=BENCH_C, width=2.5, dash='dash'),
                                 hovertemplate="%{x|%b %d, %Y}<br>Benchmark: %{y:.3f}<extra></extra>"),
                      row=1, col=1)
        fig.add_shape(type='line', x0=0, x1=1, xref='x domain', y0=1.0, y1=1.0, yref='y',
                      line=dict(color=SUBTEXT, width=1, dash='dot'))
        fig.update_yaxes(title_text="Normalized Value (1.0 = start)", gridcolor=GRID,
                         tickformat=".2f", title_font=dict(color=SUBTEXT),
                         tickfont=dict(color=SUBTEXT), row=1, col=1)
        fig.update_xaxes(showgrid=False, tickfont=dict(color=SUBTEXT),
                         dtick="M1", tickformat="%b %Y", tickangle=-45, row=1, col=1)

        fig.add_trace(go.Table(
            header=dict(values=["<b>Metric</b>", "<b>Portfolio</b>", "<b>Benchmark</b>"],
                        fill_color='#1D6FA4',
                        font=dict(color='white', size=12, family='Inter, Arial'),
                        align='left', height=30, line_color='white'),
            cells=dict(values=[mnames, [p_m[mn] for mn in mnames], [b_m[mn] for mn in mnames]],
                       fill_color=[['#F8F9FA'] * len(mnames),
                                   ['rgba(29,111,164,0.08)'] * len(mnames),
                                   ['rgba(249,115,22,0.08)'] * len(mnames)],
                       font=dict(color=[[TEXT]*len(mnames), pc, bc],
                                 size=11, family='Inter, Arial'),
                       align='left', height=26, line_color=GRID)), row=2, col=1)

        for ann in fig['layout']['annotations']:
            ann['font'] = dict(color=TEXT, size=13, family='Inter, Arial')

        fig.update_layout(
            height=800,
            title=dict(text=f"{title_prefix}{cname}  ·  Crisis Period Analysis",
                       font=dict(size=16, color=TEXT, family='Inter, Arial'),
                       x=0.5, xanchor='center'),
            paper_bgcolor=BG, plot_bgcolor=PANEL,
            font=dict(family='Inter, Arial, sans-serif', size=12, color=TEXT),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1, font=dict(color=TEXT, size=11)),
            margin=dict(l=70, r=50, t=80, b=40))
        fig.update_xaxes(linecolor=GRID, mirror=False)
        fig.update_yaxes(linecolor=GRID, mirror=False)
        divs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return "\n<hr style='border:none;border-top:1px solid #E9ECEF;margin:20px 70px'>\n".join(divs)