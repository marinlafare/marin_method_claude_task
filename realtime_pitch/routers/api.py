from fastapi import APIRouter

from .health import router as health_router
from .stream import router as stream_router

router = APIRouter(prefix="/realtime", tags=["realtime_pitch"])
router.include_router(health_router)
router.include_router(stream_router)
