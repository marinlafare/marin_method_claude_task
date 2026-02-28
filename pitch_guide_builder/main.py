# pitch_guide_builder/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.api import router

app = FastAPI(title="pitch_guide_builder", version="0.1.0")

# Allow the dev UI (Vite) to call this service directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The router is mounted under /pitch-guide.
app.include_router(router, prefix="/pitch-guide")
