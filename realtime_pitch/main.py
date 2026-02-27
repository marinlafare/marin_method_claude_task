# realtime_pitch/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.api import router as realtime_router

app = FastAPI(title="realtime_pitch")

# Dev CORS (safe defaults for local UI). If you only use the Vite proxy, CORS isn't needed,
# but keeping it enabled avoids browser issues when hitting the API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(realtime_router)
