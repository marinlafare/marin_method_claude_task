# pitch_guide_builder/algorithms/pyin.py
from __future__ import annotations

import numpy as np

from operations.models import AlgoTrack


def time_grid(n_frames: int, hop_s: float) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float32) * float(hop_s)


def pyin_track(
    y: np.ndarray,
    sr: int,
    hop_s: float,
    fmin: float,
    fmax: float,
    prob_threshold: float = 0.10,
) -> AlgoTrack:
    """librosa.pyin track.

    Note: librosa.pyin can be overly conservative on some material.
    We gate voiced frames by voiced_prob >= prob_threshold.
    """

    try:
        import librosa
    except Exception as e:
        raise RuntimeError(f"librosa is required for pYIN: {e}")

    hop_length = max(1, int(round(hop_s * sr)))
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )

    f0 = np.asarray(f0, dtype=np.float32)
    conf = np.asarray(voiced_prob, dtype=np.float32)

    # Gate voiced frames by probability. (voiced_flag can be too strict.)
    keep = np.isfinite(f0) & (f0 > 0) & np.isfinite(conf) & (conf >= float(prob_threshold))
    f0[~keep] = np.nan

    t = time_grid(len(f0), hop_s)
    return AlgoTrack(name=f"pyin_p{prob_threshold:.2f}", t=t, f0_hz=f0, conf=conf)