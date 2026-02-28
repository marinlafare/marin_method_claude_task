from fastapi import APIRouter, HTTPException

from database.storage import ALLOW_PATH_INPUT, cleanup_upload_artifacts, upload_wav_path, validate_allowed_audio_path
from operations.library import get_melody
from operations.schemas import ExtractByIdRequest, ExtractByPathRequest, ExtractResponse

router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractByIdRequest) -> ExtractResponse:
    wav_path = upload_wav_path(req.upload_id)
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown upload_id or missing file: {req.upload_id}")

    ok = False
    try:
        ts, hz = get_melody(
            str(wav_path),
            hop_ms=int(req.hop_ms),
            fmin_hz=float(req.fmin_hz),
            fmax_hz=float(req.fmax_hz),
        )
        ok = True
        return ExtractResponse(
            ok=True,
            timestamps=[float(x) for x in ts],
            hz=[None if x is None else float(x) for x in hz],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if ok:
            cleanup_upload_artifacts(req.upload_id)


@router.post("/extract-from-path", response_model=ExtractResponse)
def extract_from_path(req: ExtractByPathRequest) -> ExtractResponse:
    if not ALLOW_PATH_INPUT:
        raise HTTPException(status_code=403, detail="Path-based input disabled (set ALLOW_PATH_INPUT=1 for dev)")

    try:
        safe_path = validate_allowed_audio_path(req.vocals_path)
        ts, hz = get_melody(
            str(safe_path),
            hop_ms=int(req.hop_ms),
            fmin_hz=float(req.fmin_hz),
            fmax_hz=float(req.fmax_hz),
        )
        return ExtractResponse(
            ok=True,
            timestamps=[float(x) for x in ts],
            hz=[None if x is None else float(x) for x in hz],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
