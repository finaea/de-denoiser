"""Offline/streaming DTLN noise suppressor (Westhausen & Meyer 2020, MIT).

Uses the same two ONNX models as livekit-plugins-dtln (copied into
models/dtln/), with the canonical real-time loop from breizhn/DTLN:
512-sample window, 128-sample shift, LSTM state carried across blocks,
overlap-add output. 16 kHz fixed.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Processor, ort_options

_BLOCK_LEN = 512
_BLOCK_SHIFT = 128


def dtln_available() -> tuple[bool, str]:
    m1 = config.DTLN_MODEL_DIR / "model_1.onnx"
    m2 = config.DTLN_MODEL_DIR / "model_2.onnx"
    if not (m1.exists() and m2.exists()):
        return False, f"DTLN models missing in {config.DTLN_MODEL_DIR}"
    return True, ""


class DTLNProcessor(Processor):
    name = "dtln"
    rate = 16_000
    # 512-sample window minus the 128-sample shift of overlap-add lookahead
    algo_delay_ms = (_BLOCK_LEN - _BLOCK_SHIFT) / 16_000 * 1000

    def __init__(self):
        import onnxruntime as ort

        # a fresh options object per session: ORT reads it at construction, and
        # sharing one across two sessions ties their thread pools together
        self._s1 = ort.InferenceSession(
            str(config.DTLN_MODEL_DIR / "model_1.onnx"), ort_options()
        )
        self._s2 = ort.InferenceSession(
            str(config.DTLN_MODEL_DIR / "model_2.onnx"), ort_options()
        )
        self._m1_in = [i.name for i in self._s1.get_inputs()]
        self._m2_in = [i.name for i in self._s2.get_inputs()]
        self._state1 = np.zeros(self._s1.get_inputs()[1].shape, dtype=np.float32)
        self._state2 = np.zeros(self._s2.get_inputs()[1].shape, dtype=np.float32)
        self._in_buf = np.zeros(_BLOCK_LEN, dtype=np.float32)
        self._out_buf = np.zeros(_BLOCK_LEN, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def _process_shift(self, shift: np.ndarray) -> np.ndarray:
        """One 128-sample hop through both stages; returns 128 output samples."""
        self._in_buf = np.roll(self._in_buf, -_BLOCK_SHIFT)
        self._in_buf[-_BLOCK_SHIFT:] = shift

        spec = np.fft.rfft(self._in_buf)
        mag = np.abs(spec).astype(np.float32).reshape(1, 1, -1)
        out1, self._state1 = self._s1.run(
            None, {self._m1_in[0]: mag, self._m1_in[1]: self._state1}
        )
        estimated = spec * out1.reshape(-1)
        block = np.fft.irfft(estimated).astype(np.float32).reshape(1, 1, -1)
        out2, self._state2 = self._s2.run(
            None, {self._m2_in[0]: block, self._m2_in[1]: self._state2}
        )

        self._out_buf = np.roll(self._out_buf, -_BLOCK_SHIFT)
        self._out_buf[-_BLOCK_SHIFT:] = 0.0
        self._out_buf += out2.reshape(-1)
        return self._out_buf[:_BLOCK_SHIFT].copy()

    def process_block(self, x: np.ndarray) -> np.ndarray:
        self._pending = np.concatenate(
            [self._pending, x.astype(np.float32, copy=False)]
        )
        outs = []
        while len(self._pending) >= _BLOCK_SHIFT:
            shift = self._pending[:_BLOCK_SHIFT]
            self._pending = self._pending[_BLOCK_SHIFT:]
            outs.append(self._process_shift(shift))
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs)

    def flush(self) -> np.ndarray:
        tail = len(self._pending)
        if tail == 0:
            return np.zeros(0, dtype=np.float32)
        pad = np.zeros(_BLOCK_SHIFT - tail, dtype=np.float32)
        out = self.process_block(pad)
        return out[:tail]
