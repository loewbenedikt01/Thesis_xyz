"""
regimes.vix — Nowcast-and-persistence change-point regime detector.

Implements Nystrup, Hansen, Madsen & Lindström (2016), "Detecting change points
in VIX and S&P 500: A new approach to dynamic asset allocation", adapted to a
monthly-rebalanced portfolio pipeline.

Core principle: nothing is forecast. At each rebalance date T we (a) detect
whether a change point has occurred using only data up to T, (b) estimate
volatility in the current regime from observations since the last change
point, (c) map that volatility to an equity exposure, and (d) hold it until
the next rebalance. The forecast content comes entirely from regime
persistence, not from a transition model.

Modules:
    mood        — Mood test statistic for scale (variance) changes
    thresholds  — Monte Carlo calibration of sequential detection thresholds
    detector    — sequential change-point detection with restart
    state       — EWMA volatility estimate and exposure mapping
    integrate   — applying the regime signal to model portfolio weights
"""

from . import mood, thresholds, detector, state, integrate

__all__ = ["mood", "thresholds", "detector", "state", "integrate"]
