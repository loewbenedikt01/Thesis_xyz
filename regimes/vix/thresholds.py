"""
thresholds.py — sequential threshold calibration for the Mood-based detector.

The detector fires when D_max,t > h_t. The sequence h_2, h_3, ... is
calibrated so that the *conditional* false-alarm rate is held constant at
alpha = 1 / ARL0:

    P( D_max,t > h_t | D_max,i <= h_i for all i < t ) = alpha

h_t therefore depends on the whole preceding threshold sequence and has no
closed form — it is calibrated once via Monte Carlo simulation and cached.

Distribution-free shortcut: the Mood statistic depends only on ranks, so
under H0 its null distribution is independent of the data-generating
distribution for any continuous F. Thresholds are simulated from
np.random.standard_normal and remain valid for fat-tailed financial returns —
this is the property that makes the whole approach robust to the
kurtosis-11-ish nature of daily return series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mood import mood_max_batch

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Defaults matching the paper / spec. T_MAX=1500 covers the longest inter-
# change-point distances the paper reports (up to ~6 years of trading days);
# thresholds are held constant beyond T_MAX (documented conservatism).
DEFAULT_N_PATHS = 20000
DEFAULT_T_MAX = 1500
DEFAULT_MIN_SEG = 20
DEFAULT_ARL0 = 10000
DEFAULT_SEED = 0


def calibrate_thresholds(
    n_paths: int = DEFAULT_N_PATHS,
    t_max: int = DEFAULT_T_MAX,
    min_seg: int = DEFAULT_MIN_SEG,
    arl0: int = DEFAULT_ARL0,
    seed: int = DEFAULT_SEED,
    min_alive: int = 500,
    verbose: bool = True,
) -> np.ndarray:
    """
    Simulate `n_paths` i.i.d. standard-normal paths of length `t_max` and
    calibrate a sequential threshold h[t] such that, conditional on having
    survived (not fired) up to t, P(D_max,t > h[t]) = 1 / arl0.

    Paths that exceed a threshold are excluded from all subsequent quantile
    estimates (they are "dead") — this enforces the conditioning in the
    definition above. All paths are simulated up front (memory: n_paths *
    t_max * 8 bytes, e.g. 20000*1500*8 ~= 240MB).

    Returns
    -------
    h : np.ndarray, shape (t_max + 1,). h[t] is the threshold to apply when
        the current monitored segment holds exactly t observations.
        h[0 .. 2*min_seg - 1] = np.inf (no valid Mood split exists yet).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_paths, t_max))
    alive = np.ones(n_paths, dtype=bool)
    alpha = 1.0 / arl0

    h = np.full(t_max + 1, np.inf)
    min_t = 2 * min_seg
    last_h = np.inf

    for t in range(min_t, t_max + 1):
        n_alive = int(alive.sum())
        if n_alive < min_alive:
            if verbose:
                print(f"  [thresholds] stopping at t={t}: only {n_alive} paths alive")
            h[t:] = last_h
            break

        D, _ = mood_max_batch(X[:, :t], min_seg=min_seg)
        D_alive = D[alive]
        h_t = float(np.quantile(D_alive, 1.0 - alpha))
        h[t] = h_t
        last_h = h_t
        alive &= D <= h_t

        if verbose and (t % 100 == 0 or t == t_max):
            print(f"  [thresholds] t={t:5d}  h_t={h_t:.4f}  alive={int(alive.sum())}")

    return h


def _cache_path(cache_dir: Path, arl0: int) -> Path:
    return cache_dir / f"thresholds_arl{arl0}.npz"


def load_or_generate_thresholds(
    n_paths: int = DEFAULT_N_PATHS,
    t_max: int = DEFAULT_T_MAX,
    min_seg: int = DEFAULT_MIN_SEG,
    arl0: int = DEFAULT_ARL0,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = None,
    force: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load a cached threshold array if one exists with matching calibration
    parameters, otherwise simulate it (expensive — see `calibrate_thresholds`)
    and cache it to `cache_dir / thresholds_arl{arl0}.npz`.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, arl0)

    if cache_file.exists() and not force:
        data = np.load(cache_file)
        meta_matches = (
            int(data["n_paths"]) == n_paths
            and int(data["t_max"]) == t_max
            and int(data["min_seg"]) == min_seg
            and int(data["arl0"]) == arl0
            and int(data["seed"]) == seed
        )
        if meta_matches:
            if verbose:
                print(f"  [thresholds] loaded cache: {cache_file}")
            return data["h"]
        if verbose:
            print(f"  [thresholds] cache at {cache_file} has different params — regenerating")

    if verbose:
        print(
            f"  [thresholds] calibrating: n_paths={n_paths} t_max={t_max} "
            f"min_seg={min_seg} arl0={arl0} seed={seed} (this can take several minutes)"
        )
    h = calibrate_thresholds(
        n_paths=n_paths, t_max=t_max, min_seg=min_seg, arl0=arl0, seed=seed, verbose=verbose
    )
    np.savez(
        cache_file, h=h, n_paths=n_paths, t_max=t_max, min_seg=min_seg, arl0=arl0, seed=seed
    )
    if verbose:
        print(f"  [thresholds] cached to {cache_file}")
    return h
