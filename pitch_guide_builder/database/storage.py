# pitch_guide_builder/database/storage.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from fastapi import HTTPException

# Stored under the shared /app/data volume (mounted at /app/data in compose)
DATA_DIR = Path(os.getenv("PGB_DATA_DIR", "/app/data"))
UPLOADS_DIR = DATA_DIR / "pitch_guides" / "uploads"
TMP_DIR = UPLOADS_DIR / "_tmp"

# Canonical location for precomputed assets checked into / mounted under /app/data
AUDIO_DATA_DIR = DATA_DIR / "audio_data"

# Dev-only: allow server-side filesystem path inputs (container paths).
ALLOW_PATH_INPUT = os.getenv("ALLOW_PATH_INPUT", "0") == "1"

# Strict allowlist for path-based extraction inputs.
_DEFAULT_ALLOWED_ROOTS = "/app/data/audio_data"
ALLOWED_PATH_ROOTS = [
    Path(x).resolve(strict=False)
    for x in os.getenv("PGB_ALLOWED_PATH_ROOTS", _DEFAULT_ALLOWED_ROOTS).split(":")
    if x.strip()
]

# If 1, keep uploaded wavs and tmp dir for debugging/caching.
PGB_KEEP_UPLOADS = os.getenv("PGB_KEEP_UPLOADS", "0") == "1"

# Strict allowlist: prevents path traversal and weird characters in upload_id.
_UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,180}$")


def safe_name(name: str) -> str:
    out: List[str] = []
    for ch in (name or ""):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s[:120] if s else "audio"


def validate_upload_id(upload_id: str) -> str:
    if not _UPLOAD_ID_RE.fullmatch(upload_id or ""):
        raise HTTPException(status_code=400, detail="Invalid upload_id")
    return upload_id


def upload_wav_path(upload_id: str) -> Path:
    upload_id = validate_upload_id(upload_id)
    return UPLOADS_DIR / f"{upload_id}.wav"


def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_upload_artifacts(upload_id: str) -> None:
    """Best-effort cleanup of upload staging artifacts for an upload_id."""

    if PGB_KEEP_UPLOADS:
        return

    # Delete wav created by /upload
    try:
        wav = upload_wav_path(upload_id)
        if wav.exists():
            wav.unlink()
    except Exception:
        pass

    # Remove TMP_DIR if it is empty
    try:
        if TMP_DIR.exists() and TMP_DIR.is_dir():
            if not any(TMP_DIR.iterdir()):
                TMP_DIR.rmdir()
    except Exception:
        pass

    # Remove UPLOADS_DIR if empty (optional cleanliness)
    try:
        if UPLOADS_DIR.exists() and UPLOADS_DIR.is_dir():
            if not any(UPLOADS_DIR.iterdir()):
                UPLOADS_DIR.rmdir()
    except Exception:
        pass


def validate_allowed_audio_path(vocals_path: str) -> Path:
    """Validate that a path-based extraction input is under allowed roots."""

    try:
        resolved = Path(vocals_path).resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Input file not found: {vocals_path}")
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid input path: {vocals_path}")

    if not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"Input path is not a file: {vocals_path}")

    if resolved.name != "vocals.wav":
        raise HTTPException(status_code=403, detail="Only vocals.wav files are allowed for path extraction")

    for root in ALLOWED_PATH_ROOTS:
        if resolved == root or root in resolved.parents:
            return resolved

    raise HTTPException(status_code=403, detail=f"Path not allowed: {resolved}")
