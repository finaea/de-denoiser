"""ai-coustics enhancement via their own on-device SDK (aic-sdk on PyPI).

Not the LiveKit-Cloud-metered path — the Python plugin only exposes Krisp.
This uses ai-coustics' self-service license (30-day trial at ai-coustics.com);
set AIC_LICENSE_KEY and AIC_MODEL_ID in .env. The model file is downloaded
once into AIC_MODEL_DIR on first use, then loaded from disk.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Processor


def aic_available() -> tuple[bool, str]:
    if not config.AIC_LICENSE_KEY:
        return False, "AIC_LICENSE_KEY not set in .env (self-service trial at ai-coustics.com)"
    try:
        import aic_sdk  # noqa: F401
    except ImportError:
        return False, "aic-sdk not installed in this venv"
    if not config.AIC_MODEL_ID:
        return False, "AIC_MODEL_ID not set in .env (e.g. a QUAIL/Voice-Focus model id)"
    return True, ""


class AICProcessor(Processor):
    name = "aic"

    def __init__(self, model_id: str | None = None, enhancement_level: float | None = None):
        import aic_sdk

        model_id = model_id or config.AIC_MODEL_ID
        self.name = f"aic:{model_id}"
        config.AIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        existing = list(config.AIC_MODEL_DIR.glob(f"*{model_id}*"))
        if existing:
            model = aic_sdk.Model.from_file(str(existing[0]))
        else:
            path = aic_sdk.Model.download(model_id, str(config.AIC_MODEL_DIR))
            model = aic_sdk.Model.from_file(str(path))

        self.rate = int(model.get_optimal_sample_rate())
        cfg = aic_sdk.ProcessorConfig.optimal(model, sample_rate=self.rate, num_channels=1)
        self._proc = aic_sdk.Processor(model, config.AIC_LICENSE_KEY)
        self._proc.initialize(cfg)
        self._frames = int(cfg.num_frames)
        self.algo_delay_ms = self._frames / self.rate * 1000
        if enhancement_level is not None:
            ctx = self._proc.get_processor_context()
            ctx.set_parameter(aic_sdk.ProcessorParameter.EnhancementLevel, float(enhancement_level))
        self._buf = np.zeros(0, dtype=np.float32)

    def _drain(self) -> np.ndarray:
        outs = []
        while len(self._buf) >= self._frames:
            block = np.ascontiguousarray(self._buf[: self._frames]).reshape(1, -1)
            self._buf = self._buf[self._frames :]
            enhanced = np.asarray(self._proc.process(block), dtype=np.float32).reshape(-1)
            outs.append(enhanced)
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)])
        return self._drain()

    def flush(self) -> np.ndarray:
        tail = len(self._buf)
        if tail == 0:
            return np.zeros(0, dtype=np.float32)
        pad = self._frames - (tail % self._frames)
        if pad != self._frames:
            self._buf = np.pad(self._buf, (0, pad))
        return self._drain()[:tail]
