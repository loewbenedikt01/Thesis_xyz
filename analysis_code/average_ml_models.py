import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "results" / "data"

models = {
    "xgboost": "xgb",
    "lstm": "lstm",
    "random_forest": "rf",
}
frequencies = ["Monthly", "Quarterly", "Semi-Annual", "Yearly"]
runs = range(1, 11)

for model_dir, prefix in models.items():
    folder = BASE / model_dir
    for freq in frequencies:
        dfs = []
        for run in runs:
            path = folder / f"portfolio_{prefix}_{freq}_{run}.csv"
            df = pd.read_csv(path, parse_dates=["date"])
            dfs.append(df.set_index("date")["log_return"])

        avg_log_return = pd.concat(dfs, axis=1).mean(axis=1)

        result = pd.DataFrame({
            "date": avg_log_return.index,
            "log_return": avg_log_return.values,
            "cumulative_return": np.exp(avg_log_return.cumsum().values),
        })

        out_path = folder / f"portfolio_{prefix}_{freq}_avg.csv"
        result.to_csv(out_path, index=False)
        print(f"Wrote {out_path.name}")
