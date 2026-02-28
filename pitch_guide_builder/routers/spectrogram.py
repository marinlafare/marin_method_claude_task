from fastapi import APIRouter, HTTPException

from database.storage import ALLOW_PATH_INPUT, upload_wav_path
from operations.library import compute_spectrogram_file
from operations.schemas import SpectrogramByIdRequest, SpectrogramByPathRequest, SpectrogramResponse

router = APIRouter()


@router.post("/spectrogram", response_model=SpectrogramResponse)
def spectrogram(req: SpectrogramByIdRequest) -> SpectrogramResponse:
    wav_path = upload_wav_path(req.upload_id)
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown upload_id or missing file: {req.upload_id}")

    try:
        out = compute_spectrogram_file(
            str(wav_path),
            hop_ms=int(req.hop_ms),
            n_fft=int(req.n_fft),
            kind=req.kind,
            n_mels=int(req.n_mels),
            fmin_hz=float(req.fmin_hz),
            fmax_hz=float(req.fmax_hz),
        )
        return SpectrogramResponse(
            ok=True,
            spec_path=str(out[0]),
            sr=int(out[1]),
            hop_ms=int(out[2]),
            n_fft=int(out[3]),
            kind=str(out[4]),
            n_frames=int(out[5]),
            n_bins=int(out[6]),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/spectrogram-from-path", response_model=SpectrogramResponse)
def spectrogram_from_path(req: SpectrogramByPathRequest) -> SpectrogramResponse:
    if not ALLOW_PATH_INPUT:
        raise HTTPException(status_code=403, detail="Path-based input disabled (set ALLOW_PATH_INPUT=1 for dev)")

    try:
        out = compute_spectrogram_file(
            req.vocals_path,
            hop_ms=int(req.hop_ms),
            n_fft=int(req.n_fft),
            kind=req.kind,
            n_mels=int(req.n_mels),
            fmin_hz=float(req.fmin_hz),
            fmax_hz=float(req.fmax_hz),
        )
        return SpectrogramResponse(
            ok=True,
            spec_path=str(out[0]),
            sr=int(out[1]),
            hop_ms=int(out[2]),
            n_fft=int(out[3]),
            kind=str(out[4]),
            n_frames=int(out[5]),
            n_bins=int(out[6]),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
