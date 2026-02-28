# pitch_guide_builder/algorithms/pyin.py
from __future__ import annotations

import numpy as np

from operations.models import AlgoTrack


def time_grid(n_frames: int, hop_s: float) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float32) * float(hop_s)


def yin_energy_track(y: np.ndarray, sr: int, hop_s: float, fmin: float, fmax: float) -> AlgoTrack:
    """librosa.yin fallback (always returns an f0), with energy-based confidence.

    This is a safety net for cases where pYIN/crepe produce almost no voiced frames.
    """

    try:
        import librosa
    except Exception as e:
        raise RuntimeError(f"librosa is required for YIN fallback: {e}")

    hop_length = max(1, int(round(hop_s * sr)))

    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    f0 = np.asarray(f0, dtype=np.float32)

    # Frame RMS as a crude confidence (0..1)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True).squeeze(0)
    rms = np.asarray(rms, dtype=np.float32)
    if rms.shape[0] != f0.shape[0]:
        n = min(rms.shape[0], f0.shape[0])
        rms = rms[:n]
        f0 = f0[:n]

    # Normalize RMS to 0..1 and gate very low-energy frames
    rmax = float(np.max(rms)) if rms.size else 0.0
    conf = (rms / (rmax + 1e-9)).astype(np.float32)

    # Consider the bottom 15% energy as unvoiced
    thr = float(np.quantile(conf, 0.15)) if conf.size else 0.0
    keep = np.isfinite(f0) & (f0 > 0) & (conf > thr)
    f0[~keep] = np.nan
    conf[~keep] = 0.0

    t = time_grid(len(f0), hop_s)
    return AlgoTrack(name="yin_energy", t=t, f0_hz=f0, conf=conf)