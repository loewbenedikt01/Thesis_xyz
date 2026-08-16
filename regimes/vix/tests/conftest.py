import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regimes.vix.thresholds import calibrate_thresholds  # noqa: E402

MIN_SEG = 20

# Reduced-scale calibration used across the test suite for tractable runtime
# (a few seconds instead of the ~10 minute production calibration). ARL0 is
# also reduced to 2000 so that paths of a few thousand days aren't mostly
# right-censored before their first false alarm — the production cache
# (regimes/vix/cache/thresholds_arl10000.npz, ARL0=10000) is what the actual
# backtest uses; this fixture only exercises the same code path.
TEST_ARL0 = 2000
TEST_N_PATHS = 3000
TEST_T_MAX = 500


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_arl0() -> int:
    return TEST_ARL0


@pytest.fixture(scope="session")
def test_thresholds() -> np.ndarray:
    return calibrate_thresholds(
        n_paths=TEST_N_PATHS,
        t_max=TEST_T_MAX,
        min_seg=MIN_SEG,
        arl0=TEST_ARL0,
        seed=123,
        verbose=False,
    )


@pytest.fixture(scope="session")
def vix_log_returns(project_root) -> pd.Series:
    df = pd.read_parquet(project_root / "vix_price.parquet")
    s = df.iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    lr = np.log(s / s.shift(1)).dropna()
    lr.name = "vix_log_return"
    return lr


@pytest.fixture(scope="session")
def spx_log_returns(project_root) -> pd.Series:
    df = pd.read_parquet(project_root / "benchmark_price.parquet")
    s = df.iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    lr = np.log(s / s.shift(1)).dropna()
    lr.name = "spx_log_return"
    return lr
