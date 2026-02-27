from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_PYWORLD_IMPORT_ERROR: Optional[str] = None
try:  # pragma: no cover - runtime dependency gate
    import pyworld as pw  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency gate
    pw = None  # type: ignore
    _PYWORLD_IMPORT_ERROR = str(exc)


def _clamp01(v: float) -> float:
    return float(max(0.0, min(0.999, v)))


class PyworldPitcher:
    """Streaming WORLD (DIO + StoneMask) pitch wrapper."""

    def __init__(
        self,
        sr: int,
        hop_size: int,
        fmin: float = 50.0,
        fmax: float = 1000.0,
        buffer_s: float = 0.45,
    ) -> None:
        if pw is None:
            msg = "pyworld dependency is not installed"
            if _PYWORLD_IMPORT_ERROR:
                msg = f"{msg}: {_PYWORLD_IMPORT_ERROR}"
            raise RuntimeError(msg)
        self._sr = int(sr)
        self._hop_size = int(max(64, hop_size))
        self._fmin = float(fmin)
        self._fmax = float(fmax)
        self._buffer_s = float(max(0.18, buffer_s))
        self._buf = np.zeros(0, dtype=np.float32)
        self._sync_derived()

    def _sync_derived(self) -> None:
        hop_ms = 1000.0 * (self._hop_size / float(max(self._sr, 1)))
        self._frame_period_ms = float(max(5.0, min(20.0, hop_ms)))
        self._min_samples = max(int(round(self._sr * 0.10)), self._hop_size * 3)
        self._max_samples = max(int(round(self._sr * self._buffer_s)), self._hop_size * 8)

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
        if sr_i != self._sr or hop_i != self._hop_size:
            self._buf = np.zeros(0, dtype=np.float32)
        self._sr = sr_i
        self._hop_size = hop_i
        self._fmin = fmin_f
        self._fmax = fmax_f
        self._sync_derived()

    def process(self, x: np.ndarray) -> Tuple[Optional[float], float]:
        if pw is None:
            return None, 0.0

        if x.size != self._hop_size:
            if x.size > self._hop_size:
                frame = x[: self._hop_size]
            else:
                frame = np.pad(x, (0, self._hop_size - x.size))
        else:
            frame = x
        frame = np.asarray(frame, dtype=np.float32, order="C")

        self._buf = np.concatenate([self._buf, frame], dtype=np.float32)
        if self._buf.size > self._max_samples:
            self._buf = self._buf[-self._max_samples :]

        if self._buf.size < self._min_samples:
            return None, 0.0

        signal = self._buf.astype(np.float64, copy=False)
        raw_f0, t = pw.dio(
            signal,
            self._sr,
            f0_floor=float(self._fmin),
            f0_ceil=float(self._fmax),
            frame_period=float(self._frame_period_ms),
        )
        if raw_f0.size == 0:
            return None, 0.0

        refined = pw.stonemask(signal, raw_f0, t, self._sr)
        voiced = np.flatnonzero(np.isfinite(refined) & (refined > 0.0))
        if voiced.size == 0:
            return None, 0.0

        idx = int(voiced[-1])
        f0 = float(refined[idx])
        if not np.isfinite(f0) or f0 <= 0.0:
            return None, 0.0
        if f0 < self._fmin or f0 > self._fmax:
            return None, 0.0

        tail_idx = voiced[max(0, voiced.size - 6) :]
        tail = refined[tail_idx].astype(np.float64, copy=False)
        conf = 0.78
        if tail.size >= 2:
            med = float(np.median(tail))
            cents = np.abs(1200.0 * np.log2(np.maximum(tail, 1e-9) / max(med, 1e-9)))
            jitter = float(np.mean(cents))
            conf = _clamp01(0.95 - (jitter / 220.0))

        frame_rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float64, copy=False)))))
        conf *= _clamp01(max(0.20, min(1.0, frame_rms / 0.03)))
        return float(f0), float(conf)
