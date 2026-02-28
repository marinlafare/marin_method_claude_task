# pitch_guide_builder/models.py
from __future__ import annotations

"""Internal/core models.

Responsibility split:
- models.py: domain structures used by operations + algorithms (dataclasses / numpy)
- schemas.py: HTTP request/response shapes (Pydantic)

Rule of thumb:
- If it crosses the FastAPI boundary (validation/serialization): schemas.py
- If it represents internal service state / DSP outputs / filesystem-derived info: models.py
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal

import numpy as np


# ------------------------------
# Algorithm-facing structures


@dataclass(frozen=True)
class AlgoTrack:
    """A pitch track produced by an algorithm, on its native grid."""

    name: str
    t: np.ndarray  # seconds
    f0_hz: np.ndarray  # float Hz, np.nan for unvoiced
    conf: np.ndarray  # 0..1 confidence


# ------------------------------
# Service/domain structures (operations-facing)


@dataclass(frozen=True)
class UploadResult:
    upload_id: str
    wav_path: str
    converted_to_wav: bool


@dataclass(frozen=True)
class ExtractResult:
    """Internal representation for extracted melody."""

    timestamps_s: np.ndarray  # shape (T,)
    f0_hz: np.ndarray  # shape (T,), np.nan for unvoiced


@dataclass(frozen=True)
class MelodyIndexItemCore:
    """One available precomputed melody asset."""

    id: str
    label: str
    json_path: str


@dataclass(frozen=True)
class MelodyJsonCore:
    """Loaded precomputed melody JSON."""

    id: str
    json_path: str
    data: Dict[str, Any]


MelodyAssetKind = Literal["vocals", "accompaniment"]


@dataclass(frozen=True)
class MelodyAssetCore:
    """A non-JSON asset associated with a melody (e.g., accompaniment audio)."""

    kind: MelodyAssetKind
    filename: str
    path: str


SpectrogramKind = Literal["stft_mag", "mel"]


@dataclass(frozen=True)
class SpectrogramMeta:
    sr: int
    hop_ms: int
    n_fft: int
    kind: SpectrogramKind
    n_frames: int
    n_bins: int
    n_mels: int
    fmin_hz: float
    fmax_hz: float


@dataclass(frozen=True)
class SpectrogramResult:
    spec_path: str
    meta: SpectrogramMeta


__all__ = [
    "AlgoTrack",
    "UploadResult",
    "ExtractResult",
    "MelodyIndexItemCore",
    "MelodyJsonCore",
    "MelodyAssetKind",
    "MelodyAssetCore",
    "SpectrogramKind",
    "SpectrogramMeta",
    "SpectrogramResult",
]