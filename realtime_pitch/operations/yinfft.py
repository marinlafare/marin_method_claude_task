from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import aubio  # type: ignore
except Exception:  # pragma: no cover - runtime dependency gate
    aubio = None  # type: ignore


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _normalize_aubio_conf(raw_conf: float) -> float:
    """Generic aubio confidence normalization for methods with [0,1]-like confidence."""
    if not np.isfinite(raw_conf):
        return 0.0
    return float(max(0.0, min(0.999, raw_conf)))


def _normalize_yinfft_conf(raw_conf: float) -> float:
    """Map aubio confidence to [0, 1) without artificial floors.

    Different aubio builds can emit confidence in different ranges, including
    negative values on weak/unvoiced frames. We map smoothly so low-quality
    frames stay low-confidence instead of being boosted.
    """
    if not np.isfinite(raw_conf):
        return 0.0
    r = float(max(-2.0, min(2.0, raw_conf)))
    # yinfft on some builds outputs mostly negative confidence. Center around
    # -0.55 so voiced frames do not collapse to near-zero confidence.
    conf = 1.0 / (1.0 + np.exp(-2.8 * (r + 0.55)))
    return float(max(0.0, min(0.999, conf)))


class YinFftPitcher:
    def __init__(
        self,
        sr: int,
        hop_size: int,
        fmin: float = 50.0,
        fmax: float = 1000.0,
    ) -> None:
        if aubio is None:
            raise RuntimeError("aubio module is not installed")
        self._sr = int(sr)
        self._hop_size = int(max(64, hop_size))
        self._buf_size = int(max(1024, _next_pow2(self._hop_size * 4)))
        self._fmin = float(fmin)
        self._fmax = float(fmax)
        self._build()

    def _build(self) -> None:
        if aubio is None:
            raise RuntimeError("aubio module is not installed")
        self._pitch = aubio.pitch("yinfft", self._buf_size, self._hop_size, self._sr)
        self._pitch.set_unit("Hz")
        self._pitch.set_silence(-70.0)
        self._pitch.set_tolerance(0.85)

    def reconfigure(self, sr: int, hop_size: int, fmin: float, fmax: float) -> None:
        sr_i = int(sr)
        hop_i = int(max(64, hop_size))
        fmin_f = float(fmin)
        fmax_f = float(fmax)
        changed = (
            sr_i != self._sr
            or hop_i != self._hop_size
            or abs(fmin_f - self._fmin) > 1e-6
            or abs(fmax_f - self._fmax) > 1e-6
        )
        if not changed:
            return
        self._sr = sr_i
        self._hop_size = hop_i
        self._buf_size = int(max(1024, _next_pow2(self._hop_size * 4)))
        self._fmin = fmin_f
        self._fmax = fmax_f
        self._build()

    def process(self, x: np.ndarray) -> Tuple[Optional[float], float]:
        if aubio is None:
            return None, 0.0
        if x.size != self._hop_size:
            if x.size > self._hop_size:
                frame = x[: self._hop_size]
            else:
                frame = np.pad(x, (0, self._hop_size - x.size))
        else:
            frame = x
        frame = np.asarray(frame, dtype=np.float32, order="C")
        f0_hz = float(self._pitch(frame)[0])
        raw_conf = float(self._pitch.get_confidence())
        confidence = _normalize_yinfft_conf(raw_conf)
        if not np.isfinite(f0_hz) or f0_hz <= 0.0:
            return None, confidence
        if f0_hz < self._fmin or f0_hz > self._fmax:
            return None, confidence
        return float(f0_hz), float(confidence)
