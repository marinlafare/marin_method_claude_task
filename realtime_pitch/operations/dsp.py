from __future__ import annotations

import math

import numpy as np


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def dbfs_from_rms(r: float) -> float:
    return float(20.0 * math.log10(max(r, 1e-12)))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def parabolic_interpolation(y_minus: float, y0: float, y_plus: float) -> float:
    """Returns sub-sample offset of peak given 3 points (parabolic fit)."""
    denom = y_minus - 2.0 * y0 + y_plus
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y_minus - y_plus) / denom
