# pitch_guide_builder/operations/library.py
"""Service operations.

Routers should stay thin: they validate request shapes and map exceptions to HTTP.
All actual work (filesystem scanning, JSON loading, DSP calls) lives here.

IMPORTANT: PitchCEP (AudioFlux) has been fully removed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from database.storage import AUDIO_DATA_DIR
from operations.models import MelodyAssetCore, MelodyAssetKind, MelodyIndexItemCore, MelodyJsonCore

# Keep the historical public API stable.
from operations.melody import get_melody
from operations.spectrogram import compute_spectrogram_file

# ------------------------------
# Precomputed melody library (./data/audio_data/<song>/melody/*.json)

_MELODY_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,180}$")


def _validate_melody_id(melody_id: str) -> str:
    if not _MELODY_ID_RE.fullmatch(melody_id or ""):
        raise ValueError("Invalid melody_id")
    return melody_id


def _song_dir_for_id(melody_id: str) -> Path:
    melody_id = _validate_melody_id(melody_id)
    song_dir = AUDIO_DATA_DIR / melody_id
    if not song_dir.exists() or not song_dir.is_dir():
        raise FileNotFoundError(f"Melody not found: {melody_id}")
    return song_dir


def _find_melody_json_for_id(melody_id: str) -> Path:
    song_dir = _song_dir_for_id(melody_id)

    base = song_dir / "melody"
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Melody not found: {melody_id}")

    preferred = base / f"{melody_id}.json"
    if preferred.exists() and preferred.is_file():
        return preferred

    candidates = sorted(p for p in base.glob("*.json") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No melody json found for: {melody_id}")

    return candidates[0]


def list_melody_index() -> List[MelodyIndexItemCore]:
    """Return an index of available melody JSONs."""

    items: List[MelodyIndexItemCore] = []

    base = AUDIO_DATA_DIR
    if not base.exists() or not base.is_dir():
        return items

    for song_dir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        melody_dir = song_dir / "melody"
        if not melody_dir.exists() or not melody_dir.is_dir():
            continue

        melody_id = song_dir.name
        if not _MELODY_ID_RE.fullmatch(melody_id):
            continue

        preferred = melody_dir / f"{melody_id}.json"
        if preferred.exists() and preferred.is_file():
            json_path = preferred
        else:
            candidates = sorted(p for p in melody_dir.glob("*.json") if p.is_file())
            if not candidates:
                continue
            json_path = candidates[0]

        items.append(
            MelodyIndexItemCore(
                id=melody_id,
                label=melody_id,
                json_path=str(json_path.as_posix()),
            )
        )

    return items


def load_melody_json(melody_id: str) -> MelodyJsonCore:
    """Load a specific precomputed melody JSON."""

    p = _find_melody_json_for_id(melody_id)

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read melody json: {e}")

    if not isinstance(data, dict):
        raise ValueError("Melody json must be an object")

    return MelodyJsonCore(id=melody_id, json_path=str(p.as_posix()), data=data)


# ------------------------------
# Melody audio assets (./data/audio_data/<song>/{vocals,accompaniment}.wav)


_KIND_TO_FILENAME = {
    "vocals": "vocals.wav",
    "accompaniment": "accompaniment.wav",
}


def list_melody_assets(melody_id: str) -> List[MelodyAssetCore]:
    """List available audio assets for a melody.

    Convention:
      data/audio_data/<song>/vocals.wav
      data/audio_data/<song>/accompaniment.wav
    """

    song_dir = _song_dir_for_id(melody_id)

    out: List[MelodyAssetCore] = []
    for kind, filename in _KIND_TO_FILENAME.items():
        p = song_dir / filename
        if p.exists() and p.is_file():
            out.append(MelodyAssetCore(kind=kind, filename=filename, path=str(p.as_posix())))

    return out


def load_melody_asset_path(melody_id: str, kind: MelodyAssetKind) -> Path:
    """Return the validated path to a melody asset file."""

    song_dir = _song_dir_for_id(melody_id)

    k = str(kind)
    filename = _KIND_TO_FILENAME.get(k)
    if not filename:
        raise ValueError("Invalid asset kind")

    p = song_dir / filename
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Asset not found: {melody_id}/{k}")

    return p


__all__ = [
    "get_melody",
    "compute_spectrogram_file",
    "list_melody_index",
    "load_melody_json",
    "list_melody_assets",
    "load_melody_asset_path",
]
