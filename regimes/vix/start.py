import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
df = pd.read_parquet(project_root / 'vix_price.parquet')

print(df.tail())