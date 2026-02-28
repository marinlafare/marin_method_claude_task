# pitch_guide_builder/algorithms/torchcrepe.py
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from operations.models import AlgoTrack


def time_grid(n_frames: int, hop_s: float) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float32) * float(hop_s)


def _dbg_enabled() -> bool:
    # Reuse the same flag you already set in docker-compose.yml
    return os.getenv("PGB_DEBUG_GPU", "0") == "1"


def torchcrepe_track(
    y: np.ndarray,
    sr: int,
    hop_s: float,
    fmin: float,
    fmax: float,
    conf_threshold: float = 0.35,
) -> Optional[AlgoTrack]:
    """torchcrepe-based pitch track (optional, GPU-accelerated when available).

    - Requires torch + torchcrepe installed.
    - Uses CUDA if available; otherwise runs on CPU.
    - Returns f0_hz and periodicity as confidence.

    Notes:
    - torchcrepe expects mono float audio in range [-1, 1].
    - It works best at 16kHz; we already resample to 16kHz in the pipeline.
    """

    try:
        import torch
        import torchcrepe  # type: ignore
    except Exception:
        return None

    try:
        hop_length = max(1, int(round(hop_s * sr)))

        # Ensure float32 and reasonable range
        yy = np.asarray(y, dtype=np.float32)
        mx = float(np.max(np.abs(yy))) if yy.size else 0.0
        if mx > 1.5:  # likely int16 scaled or hot signal
            yy = yy / (mx + 1e-9)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if _dbg_enabled():
            if device == "cuda":
                try:
                    idx = torch.cuda.current_device()
                    name = torch.cuda.get_device_name(idx)
                    print(f"[pitch_guide_builder] torchcrepe using CUDA device {idx}: {name}")
                except Exception:
                    print("[pitch_guide_builder] torchcrepe using CUDA")
            else:
                print("[pitch_guide_builder] torchcrepe using CPU")

        audio = torch.from_numpy(yy).to(device=device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            f0, pd = torchcrepe.predict(
                audio,
                sr,
                hop_length,
                fmin,
                fmax,
                model="full",
                batch_size=1024,
                device=device,
                return_periodicity=True,
            )

        f0_np = f0.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        pd_np = pd.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        f0_np[(~np.isfinite(f0_np)) | (f0_np <= 0) | (pd_np < float(conf_threshold))] = np.nan
        pd_np[~np.isfinite(f0_np)] = 0.0

        t = time_grid(len(f0_np), hop_s)
        return AlgoTrack(name=f"torchcrepe_{device}", t=t, f0_hz=f0_np, conf=pd_np)

    except Exception:
        return None