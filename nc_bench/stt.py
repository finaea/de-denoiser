"""STT client for a /recognize endpoint (WHISPER_URL):
POST raw 16 kHz mono s16 WAV bytes, JSON {text, language, confidence} back."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import aiohttp
import soundfile as sf

from . import config


def _load_mono(wav_path: Path):
    data, rate = sf.read(wav_path, dtype="int16", always_2d=True)
    if rate != config.WHISPER_SAMPLE_RATE:
        raise ValueError(f"expected {config.WHISPER_SAMPLE_RATE} Hz wav, got {rate}")
    return data.mean(axis=1).astype("int16"), rate


async def _recognize(mono, rate: int) -> dict:
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


async def transcribe(wav_path: Path) -> dict:
    """Whole file in one request — how the bench has always measured."""
    mono, rate = _load_mono(wav_path)
    return await _recognize(mono, rate)


def merge_turns(segments, pad_s: float, duration_s: float) -> list[list[float]]:
    """VAD spans -> the cut list actually sent, one entry per STT request.

    Start extended by pad_s (production's speech buffer includes prefix padding
    the stored spans don't), clamped to the file, sub-20 ms slivers dropped, and
    overlaps merged — sending the same audio twice would transcribe the same
    words twice and score as insertions. Pure so scripts/check_stt_turns.py can
    pin the edge cases without an STT server.
    """
    cuts: list[list[float]] = []
    for s, e in segments or []:
        s = max(0.0, float(s) - pad_s)
        e = min(duration_s, float(e))
        if e - s < 0.02:
            continue
        if cuts and s <= cuts[-1][1]:
            cuts[-1][1] = max(cuts[-1][1], e)
        else:
            cuts.append([s, e])
    return cuts


async def transcribe_turns(
    wav_path: Path, segments, pad_s: float = 0.5, cut_wav: Path | None = None
) -> dict:
    """Cut the file at its VAD spans and transcribe one request per turn — how
    production actually feeds this STT (livekit's VAD segments the call and each
    turn is one /recognize POST; the model never sees two minutes at once).

    Each turn's start is extended by `pad_s`: production's speech buffer includes
    prefix_padding_duration of audio from BEFORE the detected onset, and the
    stored spans don't. Turns that overlap after extension are merged — duplicated
    audio would transcribe the same words twice and score as insertions.

    Returns the turn transcripts joined in time order, with `mode`/`turns` so the
    result is never mistaken for a whole-file transcript. Zero spans returns empty
    text deliberately: if the VAD sent nothing, production hears nothing, and the
    WER should say so rather than quietly falling back to the whole file.
    """
    mono, rate = _load_mono(wav_path)
    cuts = merge_turns(segments, pad_s, len(mono) / rate)
    texts, langs, confs, pieces = [], [], [], []
    for s, e in cuts:
        piece = mono[int(s * rate): int(e * rate)]
        pieces.append(piece)
        r = await _recognize(piece, rate)
        if r["text"]:
            texts.append(r["text"])
            langs.append(r.get("language"))
            if r.get("confidence") is not None:
                confs.append((e - s, float(r["confidence"])))
    if cut_wav is not None:
        # exactly the audio the STT heard, concatenated — so "why did this WER
        # change" is answerable by ear, not by re-deriving the cuts
        import numpy as np

        sf.write(cut_wav, np.concatenate(pieces) if pieces
                 else mono[:0], rate, subtype="PCM_16")
    total = sum(d for d, _ in confs)
    return {
        "text": " ".join(texts),
        "language": next((x for x in langs if x), None),
        # weighted by turn length: a 10 s turn should count for more than a blip
        "confidence": round(sum(d * c for d, c in confs) / total, 4) if total else None,
        "mode": "segmented",
        "turns": len(cuts),
    }
