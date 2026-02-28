# pitch_guide_builder/ffmpeg.py
from __future__ import annotations

import subprocess
from pathlib import Path


def ffmpeg_to_wav(src: Path, dst: Path) -> None:
    """Convert to canonical 16k mono wav for downstream consistency."""

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    subprocess.run(cmd, check=True)