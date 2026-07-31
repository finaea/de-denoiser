"""Local STT client — same /recognize protocol ai-handler's local_stt uses:
POST raw 16 kHz mono s16 WAV bytes, JSON {text, language, confidence} back."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import aiohttp
import soundfile as sf

from . import config


async def transcribe(wav_path: Path) -> dict:
    data, rate = sf.read(wav_path, dtype="int16", always_2d=True)
    mono = data.mean(axis=1).astype("int16")
    if rate != config.WHISPER_SAMPLE_RATE:
        raise ValueError(f"expected {config.WHISPER_SAMPLE_RATE} Hz wav, got {rate}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(mono.tobytes())

    async with aiohttp.ClientSession() as session:
        async with session.post(
            config.WHISPER_URL,
            params={"boost_volume": str(config.WHISPER_BOOST_VOLUME).lower()},
            data=buf.getvalue(),
            headers={"Content-Type": "audio/wav"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()
    if "error" in body:
        raise RuntimeError(f"stt error: {body}")
    return {
        "text": (body.get("text") or "").strip(),
        "language": body.get("language"),
        "confidence": body.get("confidence"),
    }
