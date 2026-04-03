


import pyarrow
import pandas as pd
import numpy as np
import plotly
import matplotlib as plt
from pathlib import Path

# Path to directory
DATA_PATH = Path(r"C:\Users\benel\OneDrive\Desktop\Python\Thesis_xyz")
benchmark_price_file = DATA_PATH / "benchmark_price.parquet"
output_dir = DATA_PATH / "results" / "data" / "benchmark"
output_dir.mkdir(parents=True, exist_ok=True) 

data_file = output_dir / "portfolio.csv"

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

# Export to CSV
portfolio.to_csv(data_file)

print(f"Successfully exported to: {data_file}")


# --------
# Create the Metrics
# --------






# --------
# Create the Plots
# --------

