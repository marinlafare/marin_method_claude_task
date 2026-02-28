# pitch_guide_builder/schemas.py
from __future__ import annotations

"""HTTP request/response schemas (Pydantic).

Routers should stay thin and import these types.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    ok: bool = True
    upload_id: str
    wav_path: str
    converted_to_wav: bool


class ExtractByIdRequest(BaseModel):
    upload_id: str = Field(..., description="ID returned by POST /pitch-guide/upload")
    hop_ms: int = Field(10, ge=5, le=100, description="Frame hop size in milliseconds")
    fmin_hz: float = Field(50.0, gt=0, description="Minimum expected F0 in Hz")
    fmax_hz: float = Field(1000.0, gt=0, description="Maximum expected F0 in Hz")


class ExtractByPathRequest(BaseModel):
    vocals_path: str = Field(
        ...,
        description="Dev-only. Path to vocals wav inside the container (e.g. data/.../vocals.wav)",
    )
    hop_ms: int = Field(10, ge=5, le=100, description="Frame hop size in milliseconds")
    fmin_hz: float = Field(50.0, gt=0, description="Minimum expected F0 in Hz")
    fmax_hz: float = Field(1000.0, gt=0, description="Maximum expected F0 in Hz")


class ExtractResponse(BaseModel):
    ok: bool
    timestamps: List[float]
    hz: List[Optional[float]]


class SpectrogramByIdRequest(BaseModel):
    upload_id: str = Field(..., description="ID returned by POST /pitch-guide/upload")
    hop_ms: int = Field(10, ge=5, le=100, description="Frame hop size in milliseconds")
    n_fft: int = Field(2048, ge=256, le=8192, description="FFT window size")
    kind: Literal["stft_mag", "mel"] = Field("stft_mag", description="Spectrogram type")
    n_mels: int = Field(128, ge=16, le=512, description="Number of mel bins (kind=mel)")
    fmin_hz: float = Field(0.0, ge=0, description="Min freq (Hz) for mel or plotting")
    fmax_hz: float = Field(8000.0, gt=0, description="Max freq (Hz) for mel or plotting")


class SpectrogramByPathRequest(BaseModel):
    vocals_path: str = Field(
        ...,
        description="Dev-only. Path to vocals wav inside the container (e.g. data/.../vocals.wav)",
    )
    hop_ms: int = Field(10, ge=5, le=100, description="Frame hop size in milliseconds")
    n_fft: int = Field(2048, ge=256, le=8192, description="FFT window size")
    kind: Literal["stft_mag", "mel"] = Field("stft_mag", description="Spectrogram type")
    n_mels: int = Field(128, ge=16, le=512, description="Number of mel bins (kind=mel)")
    fmin_hz: float = Field(0.0, ge=0, description="Min freq (Hz) for mel or plotting")
    fmax_hz: float = Field(8000.0, gt=0, description="Max freq (Hz) for mel or plotting")


class SpectrogramResponse(BaseModel):
    ok: bool
    spec_path: str
    sr: int
    hop_ms: int
    n_fft: int
    kind: str
    n_frames: int
    n_bins: int


class MelodyIndexItem(BaseModel):
    id: str
    label: str
    json_path: str


class MelodyIndexResponse(BaseModel):
    ok: bool = True
    items: List[MelodyIndexItem]


class MelodyJsonResponse(BaseModel):
    ok: bool = True
    id: str
    json_path: str
    data: Dict[str, Any]


class MelodyAssetItem(BaseModel):
    kind: Literal["vocals", "accompaniment"]
    filename: str
    url: str


class MelodyAssetsResponse(BaseModel):
    ok: bool = True
    id: str
    assets: List[MelodyAssetItem]


__all__ = [
    "UploadResponse",
    "ExtractByIdRequest",
    "ExtractByPathRequest",
    "ExtractResponse",
    "SpectrogramByIdRequest",
    "SpectrogramByPathRequest",
    "SpectrogramResponse",
    "MelodyIndexItem",
    "MelodyIndexResponse",
    "MelodyJsonResponse",
    "MelodyAssetItem",
    "MelodyAssetsResponse",
]