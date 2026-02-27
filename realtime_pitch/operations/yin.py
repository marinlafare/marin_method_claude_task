from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from operations.dsp import clamp, hann, parabolic_interpolation


_HANN_CACHE: dict[int, np.ndarray] = {}
_FFT_N_CACHE: dict[int, int] = {}


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _get_hann(n: int) -> np.ndarray:
    w = _HANN_CACHE.get(n)
    if w is None:
        w = hann(n).astype(np.float32)
        _HANN_CACHE[n] = w
    return w


def _get_fft_n(n: int) -> int:
    k = _FFT_N_CACHE.get(n)
    if k is None:
        k = _next_pow2(2 * n)
        _FFT_N_CACHE[n] = k
    return k


def _difference(x: np.ndarray, max_tau: int) -> np.ndarray:
    """YIN difference function."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    d = np.zeros(max_tau + 1, dtype=np.float32)
    if max_tau < 1 or n < 2:
        return d

    w = _get_hann(n)
    xw = x * w

    x2 = xw * xw
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(x2, dtype=np.float64, out=prefix[1:])

    fft_n = _get_fft_n(n)
    X = np.fft.rfft(xw, n=fft_n)
    ac = np.fft.irfft(X * np.conj(X), n=fft_n).astype(np.float32)

    taus = np.arange(1, max_tau + 1, dtype=np.int32)
    energy = prefix[n] - prefix[taus]
    energy_shift = prefix[n - taus]
    d[1:] = energy + energy_shift - 2.0 * ac[1 : max_tau + 1]
    d[1:] = np.maximum(d[1:], 0.0)

    return d


def _cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    cmnd = np.zeros_like(d, dtype=np.float32)
    cmnd[0] = 1.0
    if len(d) <= 1:
        return cmnd

    running = np.cumsum(d[1:], dtype=np.float64)
    taus = np.arange(1, len(d), dtype=np.float64)
    cmnd[1:] = (d[1:] * taus / np.maximum(running, 1e-12)).astype(np.float32)
    return cmnd


def _absolute_threshold(cmnd: np.ndarray, threshold: float) -> Optional[int]:
    """Find the first minimum below threshold."""
    for tau in range(2, len(cmnd) - 1):
        if cmnd[tau] < threshold and cmnd[tau] < cmnd[tau - 1]:
            while tau + 1 < len(cmnd) and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            return tau
    return None


def yin_pitch(
    x: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 1000.0,
    threshold: float = 0.07,
) -> Tuple[Optional[float], float]:
    """Simple YIN pitch estimate.

    Returns: (f0_hz | None, confidence in [0,1]).
    """

    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < 64:
        return None, 0.0

    w = hann(n)
    xw = x * w

    max_tau = int(sr / max(fmin, 1e-6))
    min_tau = int(sr / max(fmax, 1e-6))

    max_tau = int(clamp(max_tau, 2, n // 2))
    min_tau = int(clamp(min_tau, 2, max_tau - 1))

    d = _difference(xw, max_tau)
    cmnd = _cumulative_mean_normalized_difference(d)

    win = cmnd[min_tau:max_tau]
    tau = _absolute_threshold(win, threshold)

    if tau is None:
        tau = int(np.argmin(win))

    tau = tau + min_tau

    if (tau * 2) < max_tau and (tau * 2) < (len(cmnd) - 1):
        if float(cmnd[tau * 2]) + 0.01 < float(cmnd[tau]):
            tau = tau * 2

    if 1 <= tau < len(cmnd) - 1:
        off = parabolic_interpolation(cmnd[tau - 1], cmnd[tau], cmnd[tau + 1])
    else:
        off = 0.0

    tau_f = float(tau) + float(off)
    f0 = float(sr) / max(tau_f, 1e-6)

    conf = 1.0 - float(clamp(cmnd[tau], 0.0, 1.0))

    if not (fmin <= f0 <= fmax):
        return None, float(conf)

    return float(f0), float(conf)
