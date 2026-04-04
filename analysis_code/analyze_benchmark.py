


import pyarrow
import pandas as pd
import numpy as np
import plotly
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
print(f"Metrics exported to: {metrics_file}")



print(f"Successfully exported to: {data_file}")