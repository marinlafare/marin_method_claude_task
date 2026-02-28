from fastapi import APIRouter

from .extract import router as extract_router
from .health import router as health_router
from .melodies import router as melodies_router
from .spectrogram import router as spectrogram_router
from .upload import router as upload_router

router = APIRouter()
router.include_router(health_router)
router.include_router(melodies_router)
router.include_router(upload_router)
router.include_router(extract_router)
router.include_router(spectrogram_router)
