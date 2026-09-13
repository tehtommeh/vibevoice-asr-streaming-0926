"""Audio decoding helpers.

Everything the model sees is mono float32 at the checkpoint's sample rate
(24 kHz). ffmpeg does the decoding so any container the demo accepts -- mp3,
wav, m4a, flac, ogg, mp4, webm, mov -- lands in the same shape.
"""

from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np

MAX_SECONDS_DEFAULT = 3600


class AudioDecodeError(RuntimeError):
    pass


def decode_bytes(data: bytes, target_sr: int = 24000) -> np.ndarray:
    """Decode arbitrary encoded audio/video bytes to mono float32 at target_sr."""
    if not data:
        raise AudioDecodeError("empty upload")
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - image always ships ffmpeg
        raise AudioDecodeError("ffmpeg is not installed in the backend image") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise AudioDecodeError(detail[-1] if detail else "ffmpeg could not decode this file")
    audio = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        raise AudioDecodeError("no decodable audio stream in this file")
    return np.ascontiguousarray(audio)


def decode_file(path: str, target_sr: int = 24000) -> np.ndarray:
    with open(path, "rb") as fh:
        return decode_bytes(fh.read(), target_sr=target_sr)


def probe_duration(data: bytes) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        "pipe:0",
    ]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, check=True)
        return float(proc.stdout.decode().strip())
    except Exception:
        return None


def trim(audio: np.ndarray, sample_rate: int, max_seconds: float) -> tuple[np.ndarray, bool]:
    limit = int(max_seconds * sample_rate)
    if len(audio) <= limit:
        return audio, False
    return audio[:limit], True
