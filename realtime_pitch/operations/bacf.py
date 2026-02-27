from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from operations.dsp import clamp, hann, parabolic_interpolation


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def bacf_pitch(
    x: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 1000.0,
) -> Tuple[Optional[float], float]:
    """Bitstream Autocorrelation (BACF): binary stream + normalized ACF."""

    x = np.asarray(x, dtype=np.float32)
    n = int(x.size)
    if n < 128:
        return None, 0.0

    min_tau = int(max(2, sr / max(fmax, 1e-6)))
    max_tau = int(min(n - 2, max(min_tau + 3, sr / max(fmin, 1e-6))))
    if max_tau <= min_tau + 2:
        return None, 0.0

    xw = x * hann(n)
    med = float(np.median(xw))
    bits_u8 = (xw >= med).astype(np.uint8)
    bits = (2.0 * bits_u8.astype(np.float32)) - 1.0

    n_fft = _next_pow2(2 * n)
    spec = np.fft.rfft(bits, n=n_fft)
    ac = np.fft.irfft(spec * np.conj(spec), n=n_fft).astype(np.float32)[:n]
    denom = np.arange(n, 0, -1, dtype=np.float32)
    ac = ac / np.maximum(denom, 1.0)

    ac0 = float(max(ac[0], 1e-6))
    seg = ac[min_tau : max_tau + 1]
    if seg.size < 5:
        return None, 0.0

    seg_sm = seg.copy()
    seg_sm[1:-1] = 0.2 * seg[:-2] + 0.6 * seg[1:-1] + 0.2 * seg[2:]

    peaks = np.where((seg_sm[1:-1] > seg_sm[:-2]) & (seg_sm[1:-1] >= seg_sm[2:]))[0] + 1
    if peaks.size > 0:
        gate = max(0.05 * ac0, float(np.mean(seg_sm)) + 0.015 * ac0)
        valid = peaks[seg_sm[peaks] >= gate]
        best_local = int(valid[np.argmax(seg_sm[valid])]) if valid.size > 0 else int(np.argmax(seg_sm))
    else:
        best_local = int(np.argmax(seg_sm))

    tau = int(min_tau + best_local)

    # Subharmonic guard.
    half_tau = tau // 2
    if half_tau >= min_tau and ac[half_tau] >= 0.92 * ac[tau]:
        tau = half_tau

    if 1 <= tau < (n - 1):
        off = float(parabolic_interpolation(ac[tau - 1], ac[tau], ac[tau + 1]))
    else:
        off = 0.0
    tau_f = float(max(1e-6, tau + off))
    f0 = float(sr / tau_f)

    best = float(ac[tau])
    second = float(np.partition(seg_sm, -2)[-2]) if seg_sm.size >= 2 else 0.0
    peak_strength = float(clamp(best / ac0, 0.0, 1.0))
    contrast = float(clamp((best - second) / max(abs(best), 1e-6), 0.0, 1.0))
    conf = float(clamp(0.65 * peak_strength + 0.35 * contrast, 0.0, 0.999))

    if not (float(fmin) <= f0 <= float(fmax)):
        return None, conf
    return f0, conf
