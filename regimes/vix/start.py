"""
regimes/vix/start.py — end-to-end demo of the nowcast-and-persistence
change-point regime detector (Nystrup, Hansen, Madsen & Lindström, 2016).

Pipeline:
    1. Load VIX and S&P 500 daily prices, compute log-returns.
    2. Load (or, on first run, calibrate — see regimes/vix/thresholds.py)
       the sequential detection thresholds.
    3. Run the change-point detector on VIX log-returns — the paper finds
       VIX-based change points more informative for dynamic asset
       allocation than S&P 500-based ones.
    4. Estimate regime volatility and map it to an equity exposure using
       S&P 500 returns — a DIFFERENT series from the one change points are
       detected in. Two series, two roles; they are not conflated.
    5. Print a summary of detected regimes and save the daily state to CSV.

Run from anywhere — paths resolve relative to the project root:
    python regimes/vix/start.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regimes.vix import detector, state
from regimes.vix import thresholds as th
from metrics import CRISIS_PERIODS


def load_log_returns(parquet_path: Path) -> pd.Series:
    df = pd.read_parquet(parquet_path)
    s = df.iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return np.log(s / s.shift(1)).dropna()


def label_crisis(date: pd.Timestamp) -> str:
    for name, _key, c_start, _trough, c_end in CRISIS_PERIODS:
        if pd.Timestamp(c_start) <= date <= pd.Timestamp(c_end):
            return name
    return ""


def main() -> None:
    print("=== VIX / S&P 500 change-point regime detector ===\n")

    vix_lr = load_log_returns(PROJECT_ROOT / "vix_price.parquet")
    spx_lr = load_log_returns(PROJECT_ROOT / "benchmark_price.parquet")
    print(f"VIX log-returns : {vix_lr.index.min().date()} -> {vix_lr.index.max().date()}  (n={len(vix_lr)})")
    print(f"SPX log-returns : {spx_lr.index.min().date()} -> {spx_lr.index.max().date()}  (n={len(spx_lr)})")

    # Change points are detected on VIX; volatility is estimated on SPX.
    # Restrict to the overlapping window since state.compute_state() needs
    # SPX returns defined over the entire monitored window.
    overlap_start = max(vix_lr.index.min(), spx_lr.index.min())
    vix_lr = vix_lr.loc[overlap_start:]
    print(f"\nUsing overlapping window from {overlap_start.date()}  (n={len(vix_lr)} VIX obs)\n")

    print("Loading (or calibrating — first run takes several minutes) thresholds...")
    t0 = time.time()
    h = th.load_or_generate_thresholds()
    print(f"  thresholds ready in {time.time() - t0:.1f}s\n")

    print("Running sequential detector on VIX log-returns...")
    det = detector.detect(vix_lr, h, min_seg=20, startup=21)
    n_detections = int(det["detected"].sum())
    print(f"  {n_detections} change points detected over {len(det)} monitored days\n")

    spx_aligned = spx_lr.reindex(det.index).ffill()
    st = state.compute_state(spx_aligned, det, mode="frozen", exposure_rule="switching")

    mean_exposure = float(st["exposure"].mean())
    print(f"Realised mean exposure (frozen, switching @ 20% vol): {mean_exposure:.1%}\n")

    hits = det[det["detected"]].copy()
    hits["crisis"] = hits["tau_hat"].apply(label_crisis)

    print(f"{'Detected':<12} {'tau_hat':<12} {'delay(d)':>8}  crisis")
    print("-" * 62)
    for date, row in hits.iterrows():
        print(
            f"{str(date.date()):<12} {str(row['tau_hat'].date()):<12} "
            f"{row['detection_delay']:>8.0f}  {row['crisis']}"
        )

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    st.to_csv(out_dir / "vix_regime_state.csv")
    det.to_csv(out_dir / "vix_detections.csv")
    print(f"\nSaved daily state to    {out_dir / 'vix_regime_state.csv'}")
    print(f"Saved detection log to  {out_dir / 'vix_detections.csv'}")


if __name__ == "__main__":
    main()
