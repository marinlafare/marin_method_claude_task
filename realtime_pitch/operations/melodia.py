from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from operations.dsp import clamp, hann, parabolic_interpolation


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def melodia_pitch(
    x: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 1000.0,
    harmonics: int = 8,
    n_fft_min: int = 4096,
    debug: bool = False,
) -> Tuple[Optional[float], float, Optional[Dict[str, float]]]:
    x = np.asarray(x, dtype=np.float32)
    n = int(x.size)
    if n < 128:
        return None, 0.0, None

    n_fft = _next_pow2(max(n, int(n_fft_min)))
    xw = x * hann(n)

    spec = np.abs(np.fft.rfft(xw, n=n_fft)).astype(np.float32)
    if spec.size < 8:
        return None, 0.0, None

    # Dynamic-range compression helps stabilize harmonic summation.
    spec = np.power(spec + 1e-9, 0.6).astype(np.float32, copy=False)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr)).astype(np.float32)
    nyq = float(freqs[-1])

    cand_mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    cand_freqs = freqs[cand_mask]
    if cand_freqs.size < 4:
        return None, 0.0, None

    scores = np.zeros(cand_freqs.size, dtype=np.float32)
    fundamentals = np.interp(cand_freqs, freqs, spec).astype(np.float32)
    scores += 1.25 * fundamentals

    for harmonic_idx in range(2, int(max(1, harmonics)) + 1):
        weight = 1.0 / math.sqrt(float(harmonic_idx))
        target = cand_freqs * float(harmonic_idx)
        valid = target <= nyq
        if not np.any(valid):
            break
        scores[valid] += float(weight) * np.interp(target[valid], freqs, spec).astype(np.float32)

    # Penalize candidates whose immediate neighborhood is equally strong.
    side_hi = np.interp(np.minimum(cand_freqs * 1.03, nyq), freqs, spec).astype(np.float32)
    side_lo = np.interp(np.maximum(cand_freqs * 0.97, float(freqs[1])), freqs, spec).astype(np.float32)
    scores -= 0.12 * (side_hi + side_lo)

    # Mild subharmonic penalty to reduce octave-down bias.
    sub = np.interp(np.maximum(cand_freqs * 0.5, float(freqs[1])), freqs, spec).astype(np.float32)
    scores -= 0.08 * sub

    best_i = int(np.argmax(scores))
    best_score = float(scores[best_i])
    if best_score <= 0.0:
        return None, 0.0, None

    if scores.size >= 2:
        second = float(np.partition(scores, -2)[-2])
    else:
        second = 0.0

    score_mean = float(np.mean(np.maximum(scores, 0.0)))
    contrast = (best_score - second) / max(best_score, 1e-9)
    prominence = best_score / max(best_score + score_mean, 1e-9)
    conf = float(clamp(0.65 * contrast + 0.35 * prominence, 0.0, 0.999))

    cand_f0 = float(cand_freqs[best_i])
    debug_payload: Optional[Dict[str, float]] = None
    if debug:
        debug_payload = {
            "cand_f0": float(cand_f0),
            "octave_promoted": 0.0,
        }

    bin_hz = float(sr) / float(n_fft)
    bin_pos = cand_f0 / max(bin_hz, 1e-9)
    b = int(round(bin_pos))
    if 1 <= b < (spec.size - 1):
        y0 = float(math.log(spec[b] + 1e-12))
        ym = float(math.log(spec[b - 1] + 1e-12))
        yp = float(math.log(spec[b + 1] + 1e-12))
        off = float(parabolic_interpolation(ym, y0, yp))
        f0 = (float(b) + off) * bin_hz
    else:
        f0 = cand_f0

    if not (float(fmin) <= f0 <= float(fmax)):
        return None, conf, debug_payload

    if debug and debug_payload is None:
        debug_payload = {"cand_f0": float(cand_f0), "octave_promoted": 0.0}
    return float(f0), conf, debug_payload
