from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from operations.dsp import clamp, hann, parabolic_interpolation


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _log_candidates(fmin: float, fmax: float, step_cents: float = 20.0) -> np.ndarray:
    if fmax <= fmin:
        return np.zeros(0, dtype=np.float32)
    ratio = 2.0 ** (float(step_cents) / 1200.0)
    vals: list[float] = []
    f = float(fmin)
    while f <= float(fmax):
        vals.append(f)
        f *= ratio
    return np.asarray(vals, dtype=np.float32)


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
        weight = 1.0 / math.sqrt(float(harmonic_idx))
        amp = _interp_mag(spec, freqs, target_hz)
        total += float(weight) * amp
        if harmonic_idx == 1:
            fundamental = amp
    return float(total), float(fundamental)


def _fundamental_quality(total: float, fundamental: float) -> float:
    harmonic = max(float(total) - float(fundamental), 1e-9)
    return float(fundamental) / harmonic


def _resolve_upward_octave(
    spec: np.ndarray,
    freqs: np.ndarray,
    base_f0_hz: float,
    nyq_hz: float,
    fmax_hz: float,
    harmonics: int = 8,
) -> float:
    rows: list[tuple[float, float, float, float, float]] = []
    for mult in (1.0, 2.0, 4.0):
        hz = float(base_f0_hz) * mult
        if hz > float(nyq_hz) or hz > float(fmax_hz):
            continue
        total, fundamental = _harmonic_profile(spec, freqs, hz, nyq_hz, harmonics=harmonics)
        quality = _fundamental_quality(total, fundamental)
        rows.append((mult, hz, total, fundamental, quality))

    if len(rows) < 2:
        return float(base_f0_hz)

    totals = np.asarray([r[2] for r in rows], dtype=np.float32)
    funds = np.asarray([r[3] for r in rows], dtype=np.float32)
    quals = np.asarray([r[4] for r in rows], dtype=np.float32)
    totals_n = totals / max(float(np.max(totals)), 1e-9)
    funds_n = funds / max(float(np.max(funds)), 1e-9)
    quals_n = quals / max(float(np.max(quals)), 1e-9)
    scores = (0.30 * totals_n) + (0.58 * funds_n) + (0.12 * quals_n)

    base_score = float(scores[0])
    base_total = float(totals[0])
    base_fund = float(funds[0])
    base_qual = float(quals[0])
    best_i = int(np.argmax(scores))
    best = rows[best_i]
    best_score = float(scores[best_i])
    best_total = float(best[2])
    best_fund = float(best[3])
    best_qual = float(best[4])

    promote = best_i > 0 and (
        (best_score >= base_score + 0.075 and (best_fund >= base_fund * 1.07 or best_qual >= base_qual * 1.12))
        or (best_total >= base_total * 1.28 and best_fund >= base_fund * 1.20)
    )
    if promote:
        return float(best[1])
    return float(base_f0_hz)


def swipe_pitch(
    x: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 1000.0,
    harmonics: int = 10,
) -> Tuple[Optional[float], float]:
    x = np.asarray(x, dtype=np.float32)
    n = int(x.size)
    if n < 128:
        return None, 0.0

    n_fft = _next_pow2(max(n, 4096))
    xw = x * hann(n)

    spec = np.abs(np.fft.rfft(xw, n=n_fft)).astype(np.float32)
    if spec.size < 8:
        return None, 0.0
    spec = np.sqrt(spec + 1e-9).astype(np.float32, copy=False)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr)).astype(np.float32)
    nyq = float(freqs[-1])

    cands = _log_candidates(float(fmin), float(fmax), step_cents=20.0)
    if cands.size < 4:
        return None, 0.0

    scores = np.full(cands.size, -1e12, dtype=np.float32)
    max_harmonics = int(max(1, harmonics))

    for idx, f0 in enumerate(cands):
        max_h = int(min(max_harmonics, math.floor(nyq / max(float(f0), 1e-6))))
        if max_h <= 0:
            continue
        harm = np.arange(1, max_h + 1, dtype=np.float32)
        weights = 1.0 / np.sqrt(harm)

        positive = harm * float(f0)
        pos_vals = np.interp(positive, freqs, spec).astype(np.float32)

        negative = (harm + 0.5) * float(f0)
        neg_valid = negative <= nyq
        neg_vals = np.interp(negative[neg_valid], freqs, spec).astype(np.float32)
        neg_weights = weights[: neg_vals.size]

        p_score = float(np.sum(weights * pos_vals))
        n_score = float(np.sum(neg_weights * neg_vals)) if neg_vals.size else 0.0
        scores[idx] = p_score - 0.35 * n_score

    best_i = int(np.argmax(scores))
    best_score = float(scores[best_i])
    if not np.isfinite(best_score) or best_score <= 0.0:
        return None, 0.0

    if scores.size >= 2:
        second = float(np.partition(scores, -2)[-2])
    else:
        second = 0.0

    score_mean = float(np.mean(np.maximum(scores, 0.0)))
    contrast = (best_score - second) / max(best_score, 1e-9)
    prominence = best_score / max(best_score + score_mean, 1e-9)
    conf = float(clamp(0.7 * contrast + 0.3 * prominence, 0.0, 0.999))

    if 1 <= best_i < (cands.size - 1):
        off = float(parabolic_interpolation(scores[best_i - 1], scores[best_i], scores[best_i + 1]))
        ratio = 2.0 ** (20.0 / 1200.0)
        f0 = float(cands[best_i]) * (ratio ** off)
    else:
        f0 = float(cands[best_i])

    f0 = _resolve_upward_octave(
        spec=spec,
        freqs=freqs,
        base_f0_hz=float(f0),
        nyq_hz=float(nyq),
        fmax_hz=float(fmax),
        harmonics=8,
    )

    if not (float(fmin) <= f0 <= float(fmax)):
        return None, conf

    return float(f0), conf
