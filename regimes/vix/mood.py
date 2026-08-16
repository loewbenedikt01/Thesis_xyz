"""
mood.py — Mood test statistic for detecting scale (variance) changes.

At time t we hold x_1 .. x_t (daily log-returns, time order). For a candidate
split k, sample A = {x_1 .. x_{k-1}} (size n_A = k-1) and B = {x_k .. x_t}
(size n_B = t-k+1), with n = t.

    M'_{k,t} = sum over x_i in A of ( r(x_i) - (n+1)/2 )^2
    mu       = n_A * (n^2 - 1) / 12
    var      = n_A * n_B * (n + 1) * (n^2 - 4) / 180
    M_{k,t}  = | (M'_{k,t} - mu) / sqrt(var) |

r(x_i) is the rank of x_i in the pooled sample of all t observations. Ranks
are recomputed at every t, because adding an observation changes them.

The statistic depends only on ranks, so under H0 its distribution is
independent of the data-generating distribution for any continuous F — this
is what makes threshold calibration in `thresholds.py` valid even though
financial returns are fat-tailed.

`tau_hat` convention (fixed and depended on downstream): `tau_hat` is the
0-indexed position of the *first observation of the post-change segment* —
i.e. x[tau_hat] is the first observation of B. Equivalently, tau_hat == n_A
at the maximising split.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _mood_M_from_ranks(ranks: np.ndarray, n: int, min_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared core: given ranks (1..n, ties averaged) in time order, compute the
    standardised |Mood statistic| M_j for every valid split n_A = j in
    [min_seg, n - min_seg], plus the corresponding j values.
    """
    c = (ranks - (n + 1) / 2.0) ** 2
    prefix = np.cumsum(c)                       # prefix[j-1] = M'_{k,t} for n_A = j

    j = np.arange(min_seg, n - min_seg + 1)      # candidate n_A values
    Mp = prefix[j - 1]
    n_A = j.astype(float)
    n_B = n - n_A
    mu = n_A * (n ** 2 - 1) / 12.0
    var = n_A * n_B * (n + 1) * (n ** 2 - 4) / 180.0
    M = np.abs((Mp - mu) / np.sqrt(var))
    return M, j


def mood_max(x: np.ndarray, min_seg: int = 20) -> tuple[float, int]:
    """
    Mood test statistic maximised over all valid split points of x.

    Parameters
    ----------
    x       : 1-D array of observations in time order (x[0] earliest).
    min_seg : minimum size of both sub-samples A and B.

    Returns
    -------
    D_max   : the maximum standardised |Mood statistic| over all valid splits.
              0.0 if no valid split exists (len(x) < 2 * min_seg).
    tau_hat : 0-indexed position of the first observation of segment B.
              -1 if no valid split exists.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_seg:
        return 0.0, -1

    ranks = rankdata(x, method="average")
    M, j = _mood_M_from_ranks(ranks, n, min_seg)

    best = int(np.argmax(M))
    return float(M[best]), int(j[best])


def mood_max_batch(X: np.ndarray, min_seg: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised mood_max over many independent paths at once — used only by
    `thresholds.py` for Monte Carlo calibration.

    Parameters
    ----------
    X : shape (n_paths, t). Each row a path, time order along axis 1.

    Ranks are computed via argsort (no tie averaging). Valid for continuous
    simulated innovations (e.g. standard normal); do NOT reuse this function
    on real return data where ties are possible — use `mood_max` there.

    Returns
    -------
    D_max   : shape (n_paths,)
    tau_hat : shape (n_paths,) — -1 filled if no valid split exists at this t.
    """
    X = np.asarray(X, dtype=float)
    n_paths, n = X.shape
    if n < 2 * min_seg:
        return np.zeros(n_paths), np.full(n_paths, -1)

    order = np.argsort(X, axis=1)
    ranks = np.empty_like(order, dtype=float)
    fill = np.broadcast_to(np.arange(1, n + 1, dtype=float), (n_paths, n))
    np.put_along_axis(ranks, order, fill, axis=1)

    c = (ranks - (n + 1) / 2.0) ** 2
    prefix = np.cumsum(c, axis=1)

    j = np.arange(min_seg, n - min_seg + 1)
    Mp = prefix[:, j - 1]
    n_A = j[None, :].astype(float)
    n_B = n - n_A
    mu = n_A * (n ** 2 - 1) / 12.0
    var = n_A * n_B * (n + 1) * (n ** 2 - 4) / 180.0
    M = np.abs((Mp - mu) / np.sqrt(var))

    best = np.argmax(M, axis=1)
    D_max = M[np.arange(n_paths), best]
    tau_hat = j[best]
    return D_max, tau_hat
