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


def _mpm_difference(x: np.ndarray, max_tau: int) -> np.ndarray:
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


def _mpm_cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    cmnd = np.zeros_like(d, dtype=np.float32)
    cmnd[0] = 1.0
    if len(d) <= 1:
        return cmnd

    running = np.cumsum(d[1:], dtype=np.float64)
    taus = np.arange(1, len(d), dtype=np.float64)
    cmnd[1:] = (d[1:] * taus / np.maximum(running, 1e-12)).astype(np.float32)
    return cmnd


def _pick_tau(win: np.ndarray, thresh: float) -> int:
    """Prefer a good local minimum; otherwise fall back to global minimum."""
    for i in range(2, len(win) - 1):
        if win[i] < thresh and win[i] < win[i - 1]:
            j = i
            while j + 1 < len(win) and win[j + 1] < win[j]:
                j += 1
            return int(j)

    return int(np.argmin(win))


def _interp_mag(spec: np.ndarray, freqs: np.ndarray, hz: float) -> float:
    if hz <= float(freqs[0]) or hz >= float(freqs[-1]):
        return 0.0
    return float(np.interp(hz, freqs, spec))


def _harmonic_profile(
    spec: np.ndarray,
    freqs: np.ndarray,
    f0_hz: float,
    nyq_hz: float,
    harmonics: int = 8,
) -> tuple[float, float]:
    total = 0.0
    fundamental = 0.0
    for harmonic_idx in range(1, int(max(1, harmonics)) + 1):
        target_hz = f0_hz * float(harmonic_idx)
        if target_hz > nyq_hz:
            break
        weight = 1.0 / np.sqrt(float(harmonic_idx))
        amp = _interp_mag(spec, freqs, target_hz)
        total += float(weight) * amp
        if harmonic_idx == 1:
            fundamental = amp
    return float(total), float(fundamental)


def _fundamental_quality(total: float, fundamental: float) -> float:
    harmonic = max(float(total) - float(fundamental), 1e-9)
    return float(fundamental) / harmonic


def _resolve_subharmonic_from_frame(xw: np.ndarray, sr: int, f0_hz: float, fmin: float, fmax: float) -> float:
    doubled = float(f0_hz) * 2.0
    if doubled > float(fmax):
        return float(f0_hz)

    n = int(xw.size)
    if n < 128:
        return float(f0_hz)
    n_fft = _next_pow2(max(n, 4096))
    spec = np.abs(np.fft.rfft(xw, n=n_fft)).astype(np.float32)
    spec = np.power(spec + 1e-9, 0.6).astype(np.float32, copy=False)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr)).astype(np.float32)
    nyq = float(freqs[-1])
    if doubled > nyq:
        return float(f0_hz)

    low_total, low_fund = _harmonic_profile(spec, freqs, float(f0_hz), nyq)
    hi_total, hi_fund = _harmonic_profile(spec, freqs, doubled, nyq)
    low_q = _fundamental_quality(low_total, low_fund)
    hi_q = _fundamental_quality(hi_total, hi_fund)
    total_ratio = hi_total / max(low_total, 1e-9)
    fund_ratio = hi_fund / max(low_fund, 1e-9)

    # More permissive promotion for period-doubling errors: if the doubled
    # candidate keeps comparable total energy and clearly improves the
    # fundamental/quality profile, choose it.
    promote = (
        (fund_ratio >= 0.95 and total_ratio >= 0.72)
        or (fund_ratio >= 1.08 and total_ratio >= 0.60)
        or (hi_q >= 1.18 * low_q and total_ratio >= 0.58)
    )
    return float(doubled if promote else f0_hz)


def mpm_pitch(
    x: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 1000.0,
    thresh: float = 0.25,
) -> Tuple[Optional[float], float]:
    """Compact MPM-like pitch estimator.

    Returns: (f0_hz | None, confidence in [0,1]).
    """

    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < 64:
        return None, 0.0

    xw = x * hann(n)

    max_tau = int(sr / max(fmin, 1e-6))
    min_tau = int(sr / max(fmax, 1e-6))

    max_tau = int(clamp(max_tau, 2, n // 2))
    min_tau = int(clamp(min_tau, 2, max_tau - 1))

    d = _mpm_difference(xw, max_tau)
    cmnd = _mpm_cumulative_mean_normalized_difference(d)

    win = cmnd[min_tau:max_tau]
    if win.size < 8:
        return None, 0.0

    tau = _pick_tau(win, float(thresh)) + min_tau

    # Period-doubling guard: if tau/2 has a similarly good CMND valley,
    # prefer the shorter period (higher octave).
    half_tau = tau // 2
    if half_tau >= min_tau and half_tau < len(cmnd):
        half_ok = float(cmnd[half_tau]) <= (float(cmnd[tau]) * 1.20)
        if half_ok:
            tau = half_tau
            quarter_tau = tau // 2
            if quarter_tau >= min_tau and quarter_tau < len(cmnd):
                quarter_ok = float(cmnd[quarter_tau]) <= (float(cmnd[tau]) * 1.10)
                if quarter_ok:
                    tau = quarter_tau

    if 1 <= tau < len(cmnd) - 1:
        off = parabolic_interpolation(cmnd[tau - 1], cmnd[tau], cmnd[tau + 1])
    else:
        off = 0.0

    tau_f = float(tau) + float(off)
    f0 = float(sr) / max(tau_f, 1e-6)

    c = 1.0 - float(clamp(float(cmnd[tau]), 0.0, 1.0))

    f0 = _resolve_subharmonic_from_frame(xw, sr, f0, fmin, fmax)

    if not (fmin <= f0 <= fmax):
        return None, float(c)

    return float(f0), float(c)
