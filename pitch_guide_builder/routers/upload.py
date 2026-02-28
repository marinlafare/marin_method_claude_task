from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from database.storage import PGB_KEEP_UPLOADS, TMP_DIR, ensure_dirs, safe_name, upload_wav_path
from operations.ffmpeg import ffmpeg_to_wav
from operations.schemas import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    label: Optional[str] = Form(default=None),
) -> UploadResponse:
    ensure_dirs()

    base = safe_name(label or Path(file.filename or "upload").stem or "upload")
    upload_id = f"{base}_{uuid.uuid4().hex[:10]}"
    wav_path = upload_wav_path(upload_id)
    tmp_path = TMP_DIR / f"{upload_id}_{safe_name(file.filename or 'upload')}"

    try:
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        ext = (Path(file.filename or "").suffix or "").lower()
        converted = True
        if ext == ".wav":
            shutil.move(str(tmp_path), str(wav_path))
            converted = False
        else:
            ffmpeg_to_wav(tmp_path, wav_path)

        return UploadResponse(
            ok=True,
            upload_id=upload_id,
            wav_path=str(wav_path),
            converted_to_wav=converted,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg conversion failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

        if not PGB_KEEP_UPLOADS:
            try:
                if TMP_DIR.exists() and not any(TMP_DIR.iterdir()):
                    TMP_DIR.rmdir()
            except Exception:
                pass
