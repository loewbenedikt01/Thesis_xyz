"""
Markowitz Mean-Variance Portfolio
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from pypfopt import EfficientFrontier, risk_models, expected_returns

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import universe

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
FREQUENCIES = {
    'Yearly':      pd.DateOffset(years=1),
    'Quarterly':   pd.DateOffset(months=3),
    'Monthly':     pd.DateOffset(months=1),
}

LOOKBACK_MONTHS  = 60
MIN_COMPLETENESS = 0.50
WEIGHT_MAX       = 0.10
WEIGHT_MIN       = 0.01
TC_BPS = 50

start_invest = pd.Timestamp("1998-01-01")
end_invest   = pd.Timestamp("2025-12-31")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
output_name = "portfolio_markowitz_"
DATA_SUFFIX = "_tc"
DATA_PATH   = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
prices_file = DATA_PATH / "universe_prices.parquet"
output_dir  = DATA_PATH / "results" / "data" / f"markowitz{DATA_SUFFIX}"
output_dir.mkdir(parents=True, exist_ok=True)

all_prices = pd.read_parquet(prices_file)
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for label, offset in FREQUENCIES.items():
    print(f"\n=== Markowitz [{label}] | lookback={LOOKBACK_MONTHS}mo | TC={TC_BPS}bps ===")

    current_date          = start_invest
    portfolio_value       = 1.0
    last_end_weights      = pd.Series(dtype=float)
    portfolio_performance = []
    rebalance_details     = []

    while current_date < end_invest:
        next_rebalance = current_date + offset

        # ── Universe selection ────────────────────────────────────────────────
        invest_year = current_date.year
        select_year = invest_year - 1

        if select_year not in universe.tickers:
            current_date = next_rebalance
            continue

        year_tickers = [t[0] for t in universe.tickers[select_year]]

        trading_days_ahead = all_prices.index[all_prices.index >= current_date]
        if trading_days_ahead.empty:
            current_date = next_rebalance
            continue
        actual_trade_date = trading_days_ahead[0]

        # ── Lookback prices ───────────────────────────────────────────────────
        lb_start  = actual_trade_date - pd.DateOffset(months=LOOKBACK_MONTHS)
        lb_end    = actual_trade_date - pd.Timedelta(days=1)
        available = [t for t in year_tickers if t in all_prices.columns]
        lb_data   = all_prices.loc[lb_start:lb_end, available]

        if lb_data.empty:
            current_date = next_rebalance
            continue

        coverage      = lb_data.notnull().sum() / len(lb_data)
        valid_tickers = coverage[coverage >= MIN_COMPLETENESS].index.tolist()

        if len(valid_tickers) < 2:
            current_date = next_rebalance
            continue

        lb_prices_final = lb_data[valid_tickers].ffill(limit=5).dropna(how='all')
        valid_tickers   = lb_prices_final.columns.tolist()

        if len(valid_tickers) < 2:
            current_date = next_rebalance
            continue

        # ── Mean-Variance Optimisation ────────────────────────────────────────
        target_weights = None
        opt_method     = 'none'

        try:
            mu = expected_returns.mean_historical_return(lb_prices_final)
            S  = risk_models.sample_cov(lb_prices_final)

            # Primary: maximum Sharpe ratio
            try:
                ef      = EfficientFrontier(mu, S, weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
                weights = ef.max_sharpe()
                opt_method = 'max_sharpe'
            except Exception:
                ef      = EfficientFrontier(mu, S, weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
                weights = ef.min_volatility()
                opt_method = 'min_volatility'
                print(f"  [{label}] {current_date.date()}: "
                      f"max_sharpe failed — using min_volatility")

            target_weights = pd.Series(ef.clean_weights())
            target_weights = target_weights[target_weights > 0]

        except Exception as e:
            print(f"  [{label}] {current_date.date()}: optimisation failed ({e}) "
                  f"— skipping period")
            current_date = next_rebalance
            continue

        if target_weights is None or target_weights.empty:
            current_date = next_rebalance
            continue

        # ── Transaction cost at rebalance ──────────────────────────────
        turnover = target_weights.sub(last_end_weights, fill_value=0).abs().sum()
        prev_value = portfolio_value
        if TC_BPS > 0:
            portfolio_value *= (1 - turnover * TC_BPS / 10_000)

        # ── Rebalance logging ─────────────────────────────────────────────────
        for ticker, w in target_weights.items():
            rebalance_details.append({
                'rebalance_date'  : actual_trade_date.strftime('%Y-%m-%d'),
                'invest_year'     : invest_year,
                'select_year'     : select_year,
                'ticker'          : ticker,
                'assigned_weight' : w,
                'opt_method'      : opt_method,
                'turnover'        : round(turnover, 6) if ticker == target_weights.index[0] else 0,
                'tc_drag_bps'     : round(turnover * TC_BPS, 4) if ticker == target_weights.index[0] else 0,
            })

        # ── Daily portfolio drift ─────────────────────────────────────────────
        active_weights = target_weights.copy()

        period_prices = all_prices.loc[
            actual_trade_date - pd.Timedelta(days=5)
            : next_rebalance   - pd.Timedelta(days=1),
            active_weights.index,
        ]

        if period_prices.empty:
            current_date = next_rebalance
            continue

        daily_rets = period_prices.pct_change().dropna(how='all')
        daily_rets = daily_rets[daily_rets.index >= actual_trade_date]

        for day_ts, day_ret in daily_rets.iterrows():
            day_ret         = day_ret.fillna(0)
            day_pct         = (active_weights * day_ret).sum()
            portfolio_value *= (1 + day_pct)
            active_weights   = active_weights * (1 + day_ret) / (1 + day_pct)

            portfolio_performance.append({
                'date'            : day_ts.strftime('%Y-%m-%d'),
                'log_return'      : np.log(portfolio_value / prev_value),
                'cumulative_value': portfolio_value,
            })
            prev_value = portfolio_value

        last_end_weights = active_weights
        current_date     = next_rebalance

    # ── Export ────────────────────────────────────────────────────────────────
    pd.DataFrame(portfolio_performance).to_csv(
        output_dir / f"{output_name}{DATA_SUFFIX}_{label}.csv", index=False
    )
    pd.DataFrame(rebalance_details).to_csv(
        output_dir / f"{output_name}{DATA_SUFFIX}_{label}_details.csv", index=False
    )
    print(f"  [{label}] Done — saved to {output_dir}")

print(f"\nAll frequencies complete. Files exported to {output_dir}")