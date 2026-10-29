"""VAD bench: many VAD models, one shared segmenter, stored probabilities.

The registry is vad_candidates.json, one entry per model family. An entry
that lists two rates (the Silero ONNX releases: 8 and 16 kHz) expands into one
VAD per rate, `<id>-8k` and `<id>-16k`, because the same weights at two rates
behave like two different models: on PR-42, Silero v4.0 caught 7/7 missed
sounds at 8 kHz and 4/7 at 16 kHz. Treating the rate as a separate VAD also
gives each its own stored probabilities and its own row in every table.

A VAD can be unavailable (missing package, file or licence key) and says why,
exactly as an NC candidate does; the UI greys it out.

    from nc_bench import vads
    vads.registry()                          # expanded entries, with availability
    vads.infer("silero-v6.2-8k", pcm, 16000)  # -> probs, hop_s, t0_s, cpu_ms_per_s
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from .. import config

REGISTRY_FILE = config.PROJECT_ROOT / "vad_candidates.json"
MODELS_DIR = config.MODELS_DIR

# the Python package each loader needs; checked with find_spec so a missing
# package greys the VAD out instead of raising at import
_NEEDS = {
    "lk_inference": ["livekit.local_inference"],
    "silero_onnx": ["onnxruntime"],
    "firered": ["onnxruntime", "kaldi_native_fbank", "kaldiio"],
    "ten": ["ten_vad"],
    "earshot": ["earshot"],
    "vadonnx": ["vadonnx", "kaldi_native_fbank"],
    "fsmn8k": ["vadonnx", "kaldi_native_fbank", "onnxruntime"],
    "webrtc": ["webrtcvad"],
    "aic_vad": ["aic_sdk"],
    "lkaic_flag": [],
    "legacy_spans": [],
}


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _availability(e: dict) -> tuple[bool, str | None]:
    missing = [m for m in _NEEDS.get(e["loader"], []) if not _has(m)]
    if e["loader"] not in _NEEDS:
        return False, f"unknown loader {e['loader']!r}"
    if missing:
        return False, f"python package missing: {', '.join(missing)} (uv sync)"
    absent = [rel for rel in (e.get("files") or {}) if not (MODELS_DIR / rel).is_file()]
    if absent:
        return False, f"model file missing: models/{absent[0]} (run scripts/fetch_vads.py)"
    if e["loader"] == "aic_vad" and not config.AIC_LICENSE_KEY:
        return False, "AIC_LICENSE_KEY not set in .env"
    return True, None


def _expand(e: dict) -> list[dict]:
    rates = e.get("rates") or []
    if len(rates) <= 1:
        out = dict(e, family=e["id"], rate=rates[0] if rates else None)
        return [out]
    return [dict(e, id=f"{e['id']}-{r // 1000}k", family=e["id"], rate=r,
                 label=f"{e['label']} @ {r // 1000} kHz") for r in rates]


def registry() -> list[dict]:
    """Every VAD, expanded per rate, with `available` / `unavailable_reason`.

    Re-read on every call (cheap: one small JSON plus a few stat calls), so an
    edit to vad_candidates.json or a fetch_vads.py run shows up without a
    restart.
    """
    out = []
    for raw in json.loads(REGISTRY_FILE.read_text()):
        ok, why = _availability(raw)
        for e in _expand(raw):
            e["available"], e["unavailable_reason"] = ok, why
            e.setdefault("binary", False)
            out.append(e)
    return out


def by_id() -> dict[str, dict]:
    return {e["id"]: e for e in registry()}


# ------------------------------------------------------------------ inference

_models: dict[str, object] = {}


def _path(rel: str) -> Path:
    return MODELS_DIR / rel


def _build(e: dict):
    from . import loaders as L

    kind = e["loader"]
    if kind == "lk_inference":
        return L.LkInference()
    if kind == "silero_onnx":
        return L.SileroOnnx(_path(e["onnx"]), e["rate"], e.get("state_api", "v5"))
    if kind == "firered":
        return L.FireRed(_path(e["onnx"]), _path(e["cmvn"]))
    if kind == "ten":
        return L.Ten()
    if kind == "earshot":
        return L.Earshot()
    if kind == "vadonnx":
        return L.VadOnnx(e["vadonnx_name"], _path(e["onnx"]))
    if kind == "fsmn8k":
        return L.Fsmn8k(_path(e["onnx"]), _path(e["mvn"]))
    if kind == "webrtc":
        return L.WebRtc(e.get("mode", 3), e["rate"])
    if kind == "aic_vad":
        return L.AicVad(_path(e["model_file"]))
    raise ValueError(f"{e['id']}: loader {kind!r} does no inference")


def model(e: dict):
    """One instance per VAD per process: loading is ~50-300 ms each, and a
    bench run crosses every VAD with every source."""
    if e["id"] not in _models:
        m = _build(e)
        # one throwaway call, so the first real source is not billed for ORT's
        # lazy graph initialisation (it tripled FireRed-stream's first number).
        # Safe because every loader starts from fresh state on each run().
        m.run(np.zeros(8000, dtype=np.int16), 16000)
        _models[e["id"]] = m
    return _models[e["id"]]


def infer(e: dict | str, pcm: np.ndarray, rate: int) -> dict:
    """Per-window probabilities for mono int16 `pcm` at `rate`.

    CPU is process time across the whole call (resampling and features
    included, as the bake-off measured it), per second of audio. The model
    load is excluded: it is paid once per process, not per call.
    """
    if isinstance(e, str):
        e = by_id()[e]
    m = model(e)
    c0 = time.process_time()
    probs, hop, t0 = m.run(pcm, rate)
    cpu = time.process_time() - c0
    dur = len(pcm) / rate if rate else 0.0
    return {
        "probs": np.asarray(probs, dtype=np.float32),
        "hop_s": float(hop),
        "t0_s": float(t0),
        "cpu_ms_per_s": round(cpu * 1000 / dur, 2) if dur else None,
    }
