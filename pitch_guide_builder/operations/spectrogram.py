# pitch_guide_builder/spectrogram.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from database.storage import DATA_DIR, safe_name
from operations.melody import load_audio_mono


def _spectrogram_librosa(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    n_fft: int,
    kind: Literal["stft_mag", "mel"],
    n_mels: int,
    fmin_hz: float,
    fmax_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (spec, freqs)."""

    try:
        import librosa
    except Exception as e:
        raise RuntimeError(f"librosa required for spectrogram fallback: {e}")

    if kind == "stft_mag":
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
        spec = np.abs(stft).astype(np.float32, copy=False)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32, copy=False)
        return spec, freqs

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=float(fmin_hz),
        fmax=float(fmax_hz) if fmax_hz > 0 else None,
        power=2.0,
    )
    spec = np.asarray(S, dtype=np.float32)
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=float(fmin_hz), fmax=float(fmax_hz)).astype(
        np.float32, copy=False
    )
    return spec, freqs


def _spectrogram_audioflux(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    n_fft: int,
    kind: Literal["stft_mag", "mel"],
    n_mels: int,
    fmin_hz: float,
    fmax_hz: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        import audioflux as af  # type: ignore
    except Exception:
        return None

    try:
        if kind == "stft_mag":
            if hasattr(af, "BFT"):
                bft = af.BFT(num=hop_length, radix2_exp=int(np.log2(n_fft)), samplate=sr)
                Z = bft.bft(y)
                spec = np.abs(np.asarray(Z)).astype(np.float32)
                if hasattr(af, "fft_frequencies"):
                    freqs = np.asarray(af.fft_frequencies(sr, n_fft), dtype=np.float32)
                else:
                    freqs = np.linspace(0, sr / 2, spec.shape[0], dtype=np.float32)
                return spec, freqs

            if hasattr(af, "STFT"):
                stft = af.STFT(num=hop_length, radix2_exp=int(np.log2(n_fft)), samplate=sr)
                Z = stft.stft(y)
                spec = np.abs(np.asarray(Z)).astype(np.float32)
                freqs = np.linspace(0, sr / 2, spec.shape[0], dtype=np.float32)
                return spec, freqs

            return None

        if hasattr(af, "MelSpectrogram"):
            mel = af.MelSpectrogram(
                num=hop_length,
                radix2_exp=int(np.log2(n_fft)),
                samplate=sr,
                mel_num=n_mels,
                low_fre=fmin_hz,
                high_fre=fmax_hz,
            )
            S = mel.spectrogram(y)
            spec = np.asarray(S, dtype=np.float32)
            freqs = np.linspace(float(fmin_hz), float(fmax_hz), spec.shape[0], dtype=np.float32)
            return spec, freqs

        return None
    except Exception:
        return None


def compute_spectrogram_file(
    vocals_path: str,
    hop_ms: int = 10,
    n_fft: int = 2048,
    kind: Literal["stft_mag", "mel"] = "stft_mag",
    n_mels: int = 128,
    fmin_hz: float = 0.0,
    fmax_hz: float = 8000.0,
) -> Tuple[str, int, int, int, str, int, int]:
    """Compute a spectrogram and save as a compressed .npz under data/."""

    hop_s = float(hop_ms) / 1000.0
    y, sr = load_audio_mono(vocals_path, target_sr=16000)

    hop_length = max(1, int(round(hop_s * sr)))

    af_out = _spectrogram_audioflux(
        y,
        sr,
        hop_length=hop_length,
        n_fft=n_fft,
        kind=kind,
        n_mels=n_mels,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )

    if af_out is not None:
        spec, freqs = af_out
    else:
        spec, freqs = _spectrogram_librosa(
            y,
            sr,
            hop_length=hop_length,
            n_fft=n_fft,
            kind=kind,
            n_mels=n_mels,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )

    spec = np.asarray(spec, dtype=np.float32)
    if spec.ndim != 2:
        raise RuntimeError(f"Spectrogram must be 2D; got shape {spec.shape}")

    n_bins, n_frames = int(spec.shape[0]), int(spec.shape[1])
    times_s = (np.arange(n_frames, dtype=np.float32) * float(hop_length) / float(sr)).astype(np.float32)

    in_name = safe_name(Path(vocals_path).parent.name + "_" + Path(vocals_path).stem)
    key = f"{vocals_path}|{sr}|{hop_ms}|{n_fft}|{kind}|{n_mels}|{fmin_hz}|{fmax_hz}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    out_dir = DATA_DIR / "pitch_guides" / "spectrograms"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_name}_{kind}_{digest}.npz"

    np.savez_compressed(
        str(out_path),
        spec=spec,
        freqs_hz=np.asarray(freqs, dtype=np.float32),
        times_s=times_s,
        sr=np.int32(sr),
        hop_ms=np.int32(hop_ms),
        n_fft=np.int32(n_fft),
        kind=np.array([kind]),
        n_mels=np.int32(n_mels),
        fmin_hz=np.float32(fmin_hz),
        fmax_hz=np.float32(fmax_hz),
    )

    return str(out_path), int(sr), int(hop_ms), int(n_fft), str(kind), int(n_frames), int(n_bins)
