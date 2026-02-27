from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from operations.models import ENGINE_REV, LAST_REVISION_MX, MPM_PIPELINE_REV, RealtimePitchState

router = APIRouter()
logger = logging.getLogger("realtime_pitch")


async def _safe_send_json(ws: WebSocket, payload: Any) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        return


def _int16le_to_float32(data: bytes) -> np.ndarray:
    x_i16 = np.frombuffer(data, dtype="<i2")
    return (x_i16.astype(np.float32) / 32768.0).copy()


@router.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """WebSocket protocol.

    - First TEXT message: JSON config object
    - Subsequent BYTES messages: int16le mono PCM frames
    - Subsequent TEXT messages: treated as config updates
    """

    await ws.accept()
    state = RealtimePitchState()
    have_config = False
    t0 = time.time()

    try:
        while True:
            msg = await ws.receive()

            if msg.get("text") is not None:
                txt = msg["text"]
                try:
                    cfg = json.loads(txt)
                except Exception:
                    await _safe_send_json(ws, {"ok": False, "error": "bad_json"})
                    continue

                ok, err = state.apply_config(cfg)
                if not ok:
                    await _safe_send_json(ws, {"ok": False, "error": err})
                    continue

                logger.info("config_applied %s", json.dumps(cfg, sort_keys=True))
                have_config = True
                ack = {"ok": True, "type": "config_ack"}
                ack.update(state.config_ack_payload())
                await _safe_send_json(ws, ack)
                continue

            if msg.get("bytes") is not None:
                if not have_config:
                    await _safe_send_json(ws, {"ok": False, "error": "missing_config"})
                    continue

                raw = msg["bytes"]
                if not raw:
                    continue

                x = _int16le_to_float32(raw)
                r = await run_in_threadpool(state.process_chunk, x)

                if r.ok:
                    payload = {
                        "ok": True,
                        "type": "pitch",
                        "t": time.time() - t0,
                        "f0_hz": r.f0_hz,
                        "confidence": r.confidence,
                        "note": r.note,
                        "engine_rev": ENGINE_REV,
                        "mpm_pipeline_rev": MPM_PIPELINE_REV,
                        "last_revision": LAST_REVISION_MX,
                    }
                    if r.debug is not None:
                        payload["debug"] = r.debug
                    await _safe_send_json(ws, payload)
                else:
                    await _safe_send_json(
                        ws,
                        {
                            "ok": False,
                            "error": r.error or "no_pitch",
                            "engine_rev": ENGINE_REV,
                            "mpm_pipeline_rev": MPM_PIPELINE_REV,
                            "last_revision": LAST_REVISION_MX,
                        },
                    )

                continue

            await _safe_send_json(ws, {"ok": False, "error": "unsupported_message_type"})

    except WebSocketDisconnect:
        return
    except Exception as e:
        await _safe_send_json(ws, {"ok": False, "error": str(e)})
        return
