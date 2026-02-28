# pitch_guide_builder/melody.py
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from operations.algorithms.pitch_fusion import (
    fuse_tracks,
    post_filter_energy_and_short_runs,
    summarize_track,
    trim_wrong_harmonic_onsets,
    viterbi_smooth_hz,
)
from operations.algorithms.pyin import pyin_track
from operations.algorithms.torchcrepe import torchcrepe_track
from operations.algorithms.yin import yin_energy_track
from operations.models import AlgoTrack


def _dbg_enabled() -> bool:
    return os.getenv("PGB_DEBUG_GPU", "0") == "1"


def load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 and resample to target_sr.

    Prefers soundfile+resampy if available; falls back to librosa.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        import soundfile as sf

        y, sr = sf.read(str(p), always_2d=False)
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32, copy=False)
    except Exception:
        try:
            import librosa

            y, sr = librosa.load(str(p), sr=None, mono=True)
            y = y.astype(np.float32, copy=False)
        except Exception as e:
            raise RuntimeError(f"Could not load audio (install soundfile or librosa): {e}")

    if sr != target_sr:
        try:
            import resampy

            y = resampy.resample(y, sr, target_sr).astype(np.float32, copy=False)
            sr = target_sr
        except Exception:
            try:
                import librosa

                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32, copy=False)
                sr = target_sr
            except Exception as e:
                raise RuntimeError(f"Resampling failed (install resampy or librosa): {e}")

    return y, sr


def get_melody(
    vocals_path: str,
    hop_ms: int = 10,
    fmin_hz: float = 50.0,
    fmax_hz: float = 1000.0,
) -> Tuple[List[float], List[Optional[float]]]:
    """Main API: return (timestamps, hz) from a vocals wav."""

    hop_s = float(hop_ms) / 1000.0

    y, sr = load_audio_mono(vocals_path, target_sr=16000)

    if _dbg_enabled():
        yy = np.asarray(y, dtype=np.float32)
        mx = float(np.max(np.abs(yy))) if yy.size else 0.0
        rms = float(np.sqrt(np.mean(yy * yy))) if yy.size else 0.0
        dur = float(yy.size) / float(sr) if sr else 0.0
        print(f"[pitch_guide_builder] audio sr={sr} dur={dur:.2f}s max_abs={mx:.6f} rms={rms:.6f}")

    tracks: List[AlgoTrack] = []

    pyin_tr = pyin_track(y, sr, hop_s, fmin_hz, fmax_hz, prob_threshold=0.10)
    tracks.append(pyin_tr)
    if _dbg_enabled():
        print(f"[pitch_guide_builder] track: {summarize_track(pyin_tr)}")

    f0p = np.asarray(pyin_tr.f0_hz, dtype=np.float32)
    voiced_p = int(np.count_nonzero(np.isfinite(f0p) & (f0p > 0)))
    if voiced_p < max(10, int(0.002 * f0p.shape[0])):
        pyin_soft = pyin_track(y, sr, hop_s, fmin_hz, fmax_hz, prob_threshold=0.05)
        tracks.append(pyin_soft)
        if _dbg_enabled():
            print(f"[pitch_guide_builder] track: {summarize_track(pyin_soft)}")

        yin_tr = yin_energy_track(y, sr, hop_s, fmin_hz, fmax_hz)
        tracks.append(yin_tr)
        if _dbg_enabled():
            print(f"[pitch_guide_builder] track: {summarize_track(yin_tr)}")

    crepe_tr = torchcrepe_track(y, sr, hop_s, fmin_hz, fmax_hz, conf_threshold=0.35)
    if crepe_tr is not None:
        tracks.append(crepe_tr)
        if _dbg_enabled():
            print(f"[pitch_guide_builder] track: {summarize_track(crepe_tr)}")

        f0c = np.asarray(crepe_tr.f0_hz, dtype=np.float32)
        voiced_c = int(np.count_nonzero(np.isfinite(f0c) & (f0c > 0)))
        if voiced_c < max(10, int(0.002 * f0c.shape[0])):
            crepe_soft = torchcrepe_track(y, sr, hop_s, fmin_hz, fmax_hz, conf_threshold=0.15)
            if crepe_soft is not None:
                tracks.append(crepe_soft)
                if _dbg_enabled():
                    print(f"[pitch_guide_builder] track: {summarize_track(crepe_soft)}")
    elif _dbg_enabled():
        print("[pitch_guide_builder] track: torchcrepe: unavailable")

    t, fused = fuse_tracks(tracks, hop_s=hop_s)

    smoothed = viterbi_smooth_hz(t, fused, fmin=fmin_hz, fmax=fmax_hz)

    smoothed = post_filter_energy_and_short_runs(
        y=y,
        sr=sr,
        hop_s=hop_s,
        f0_hz=np.asarray(smoothed, dtype=np.float32),
        energy_gate_db=-40.0,
        min_voiced_run_s=0.15,
    )

    smoothed = trim_wrong_harmonic_onsets(
        np.asarray(smoothed, dtype=np.float32),
        hop_s=hop_s,
        onset_dev_cents=250.0,
        stable_window_s=0.30,
        onset_hold_frames=5,
    )

    timestamps = [float(x) for x in t.tolist()]
    hz: List[Optional[float]] = []
    for v in smoothed.tolist():
        if v is None:
            hz.append(None)
        else:
            try:
                fv = float(v)
            except Exception:
                hz.append(None)
                continue
            hz.append(None if (not math.isfinite(fv) or fv <= 0) else fv)

    return timestamps, hz
