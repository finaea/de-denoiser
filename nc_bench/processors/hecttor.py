"""Offline/streaming Hecttor ASR enhancer (same SDK + key ai-handler uses).

Feeds fixed get_chunk_size_samples() float32 chunks through
ASRSpeechEnhancer.process_chunk, exactly like the ai-handler integration —
minus the rtc plumbing, since here the audio is already mono float32 at the
enhancer's rate.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Processor


def hecttor_available() -> tuple[bool, str]:
    if not config.HECTTOR_API_KEY:
        return False, "HECTTOR_API_KEY not set in .env"
    try:
        import hecttor_sdk  # noqa: F401
    except ImportError:
        return False, "hecttor_sdk wheel not installed in this venv"
    return True, ""


class HecttorProcessor(Processor):
    name = "hecttor"

    def __init__(
        self,
        model: str | None = None,
        enhancer_weight: float | None = None,
        sample_rate: int | None = None,
        chunk_ms: int | None = None,
    ):
        from hecttor_sdk import (
            ASRSpeechEnhancer,
            ASRSpeechEnhancerConfig,
            ModelConfig,
        )

        self.rate = int(sample_rate or config.HECTTOR_SAMPLE_RATE)
        model = model or config.HECTTOR_MODEL
        weight = (
            config.HECTTOR_ENHANCER_WEIGHT
            if enhancer_weight is None
            else float(enhancer_weight)
        )
        self.name = f"hecttor:{model}@{weight:g}"

        cfg = ASRSpeechEnhancerConfig(
            api_key=config.HECTTOR_API_KEY,
            model_config=ModelConfig(model_name=model, enhancer_weight=weight),
            chunk_size_ms=int(chunk_ms or config.HECTTOR_CHUNK_MS),
            sample_rate=self.rate,
        )
        self._enh = ASRSpeechEnhancer()
        ok, err = self._enh.initialize(cfg)
        if not ok:
            raise RuntimeError(f"hecttor initialize failed: {err}")
        self._chunk = int(self._enh.get_chunk_size_samples())
        self.algo_delay_ms = self._chunk / self.rate * 1000
        self._buf = np.zeros(0, dtype=np.float32)

    def _drain(self) -> np.ndarray:
        outs = []
        while len(self._buf) >= self._chunk:
            chunk = np.ascontiguousarray(self._buf[: self._chunk])
            self._buf = self._buf[self._chunk :]
            result = self._enh.process_chunk(chunk)
            if isinstance(result, (tuple, list)):
                result = result[0]
            enhanced = np.asarray(result, dtype=np.float32).reshape(-1)
            if len(enhanced) < self._chunk:
                enhanced = np.pad(enhanced, (0, self._chunk - len(enhanced)))
            outs.append(enhanced[: self._chunk])
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)])
        return self._drain()

    def flush(self) -> np.ndarray:
        if len(self._buf) == 0:
            return np.zeros(0, dtype=np.float32)
        tail = len(self._buf)
        pad = self._chunk - (tail % self._chunk)
        if pad != self._chunk:
            self._buf = np.pad(self._buf, (0, pad))
        out = self._drain()
        return out[:tail]

    def close(self) -> None:
        try:
            self._enh.reset_caches()
        except Exception:
            pass
