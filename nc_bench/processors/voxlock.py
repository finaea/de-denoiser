"""VoxLock (arbazsiddiqui/voxlock, Apache-2.0): caller-voice isolation.

Like Hush it removes competing voices, but by target-speaker extraction: a 27M
causal band-split RNN conditioned on ~3 s of ONE voice, so anything that isn't
that voice goes, not just noise. With no sample in hand, `live()` passes the
audio through untouched until it has heard 3 s of speech (10 ms frames 10 dB
over the 2 s noise floor), enrols on that and isolates from then on. The first
seconds of every call are therefore raw, and whatever it hears first is who it
keeps.

Run the way its own Pipecat filter ships it (`voxlock/pipecat.py`): int8, live
enrolment, 80 ms chunks. The output is sample-aligned with the input (the
stream drops its own STFT padding and warm-up), so there is no shift to
correct. ONNX is pinned to one thread like every processor here
(base.ort_options), so cpu_per_audio_s reads the whole cost; the filter's own
default is 2 threads.
"""

from __future__ import annotations

import numpy as np

from .base import Processor

_HOLD_MS = 24.0   # NFFT - HOP = 384 samples the stream keeps back for overlap-add


def voxlock_available(_spec: dict | None = None) -> tuple[bool, str]:
    try:
        import voxlock  # noqa: F401
    except ImportError:
        return False, "voxlock not installed in this venv (uv sync)"
    return True, ""


class VoxLockProcessor(Processor):
    rate = 16_000

    def __init__(self, precision: str = "int8", chunk_ms: int = 80, sample_s: float = 3.0):
        import voxlock

        self.name = f"voxlock:{precision}"
        # downloads the pinned ONNX revision from Hugging Face on first use
        self._stream = voxlock.VoxLock(precision=precision, threads=1).live(sec=sample_s)
        self._chunk = int(self.rate * chunk_ms / 1000)
        self._pending = np.zeros(0, np.float32)
        self.algo_delay_ms = chunk_ms + _HOLD_MS

    def process_block(self, x: np.ndarray) -> np.ndarray:
        self._pending = np.concatenate([self._pending, x])
        if len(self._pending) < self._chunk:
            return np.zeros(0, np.float32)
        chunk, self._pending = self._pending, np.zeros(0, np.float32)
        return self._stream.push(chunk)

    def flush(self) -> np.ndarray:
        head = self._stream.push(self._pending) if len(self._pending) else np.zeros(0, np.float32)
        return np.concatenate([head, self._stream.flush()])
