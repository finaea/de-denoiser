"""RNNoise (Valin 2018) — the classic baseline row, via ffmpeg's `arnndn`.

ffmpeg already ships the RNNoise filter and is already a dependency for
upload decoding, so the whole integration is one subprocess. It needs a weights
file (`models/rnnoise/<id>.rnnn`, from GregorR/rnnoise-models) and runs at
48 kHz — the pipeline resamples in and out.

ponytail: whole-file, not streaming. `arnndn` streams internally but driving it
through pipes chunk-by-chunk buys nothing for a baseline that needs only one score
from — so this processor buffers the input and shells out once in flush(), and
declares `whole_file` so the pipeline reports no per-block latency for it rather
than a misleading ~0 ms. Upgrade path: pipe s16le through a long-lived ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import config
from .base import Processor

_RATE = 48_000
# RNNoise works on 10 ms frames at 48 kHz
_ALGO_DELAY_MS = 10.0


def _model_path(model: str | None) -> Path:
    return config.MODELS_DIR / "rnnoise" / f"{model or config.RNNOISE_MODEL}.rnnn"


def rnnoise_available(model: str | None = None) -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not on PATH"
    path = _model_path(model)
    if not path.exists():
        return False, f"rnnoise weights missing: {path} — run scripts/fetch_models.py"
    return True, ""


class RNNoiseProcessor(Processor):
    rate = _RATE
    whole_file = True
    algo_delay_ms = _ALGO_DELAY_MS

    def __init__(self, model: str | None = None):
        self.name = f"rnnoise-{model or config.RNNOISE_MODEL}"
        self._model = _model_path(model)
        self._buf: list[np.ndarray] = []

    def process_block(self, x: np.ndarray) -> np.ndarray:
        self._buf.append(np.asarray(x, dtype=np.float32))
        return np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        if not self._buf:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._buf)
        self._buf = []
        tmp = Path(tempfile.mkdtemp(prefix="nc-rnnoise-"))
        src, dst = tmp / "in.wav", tmp / "out.wav"
        sf.write(src, audio, _RATE)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-af", f"arnndn=m={self._model}",
             "-ar", str(_RATE), "-ac", "1", str(dst)],
            capture_output=True,
        )
        if proc.returncode != 0 or not dst.exists():
            raise RuntimeError(f"ffmpeg arnndn failed: {proc.stderr.decode()[-300:]}")
        out, _ = sf.read(dst, dtype="float32")
        shutil.rmtree(tmp, ignore_errors=True)
        return out
