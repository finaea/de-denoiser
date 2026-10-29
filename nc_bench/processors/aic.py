"""ai-coustics enhancement via their own on-device SDK (`aic-sdk` 3.2 on PyPI).

Not the LiveKit-hosted path: that is the official `livekit-plugins-ai-coustics`
plugin (`lkaic-*` candidates, cloud rail, see lk_cloud.py), which embeds four
models on an older core. This one runs any catalog model offline, including the
8 kHz and Rook Multi Speaker models the plugin does not ship, with ai-coustics'
own licence key (AIC_LICENSE_KEY in .env).

Model files are fetched once into AIC_MODEL_DIR (scripts/fetch_models.py does all
of them up front) and loaded from disk after that. The SDK saves them under an
underscored name plus a build suffix — `quail-vf-2.2-l-16khz` lands as
`quail_vf_2_2_l_16khz_horgwub0_v14.aicmodel` — so the lookup below converts the id
rather than globbing for it as-is.

Each model runs at its own native rate and optimal block size
(`ProcessorConfig.optimal`), so the 8 kHz models see 8 kHz input: the pipeline
resamples per stage.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import config
from .base import Processor


def aic_available(spec: dict | None = None) -> tuple[bool, str]:
    if not config.AIC_LICENSE_KEY:
        return False, "AIC_LICENSE_KEY not set in .env (self-service trial at ai-coustics.com)"
    try:
        import aic_sdk  # noqa: F401
    except ImportError:
        return False, "aic-sdk not installed in this venv"
    if not ((spec or {}).get("model") or config.AIC_MODEL_ID):
        return False, "no ai-coustics model id in the spec (or AIC_MODEL_ID in .env)"
    return True, ""


def model_file(model_id: str) -> Path | None:
    """The cached model file for an id, or None if it has not been fetched.

    The name is the underscored id plus `_<build>_v<n>`, matched whole: an id
    must not pick up a longer one's file (`quail-ms-l` vs `quail_ms_l_16khz_…`,
    `quail-ms-l-16khz` vs `quail_ms_l_8khz_…`). If an older build is still on
    disk next to a newer one, the newest file wins.
    """
    stem = model_id.replace("-", "_").replace(".", "_")
    pattern = re.compile(re.escape(stem) + r"_[a-z0-9]+_v\d+\.aicmodel")
    found = sorted((p for p in config.AIC_MODEL_DIR.glob(f"{stem}_*.aicmodel")
                    if pattern.fullmatch(p.name)),
                   key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def fetch_model(model_id: str) -> Path:
    """Local path of the model, downloading it from ai-coustics' CDN if missing."""
    import aic_sdk

    config.AIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return model_file(model_id) or Path(aic_sdk.Model.download(model_id, str(config.AIC_MODEL_DIR)))


class AICProcessor(Processor):
    name = "aic"

    def __init__(self, model_id: str | None = None, enhancement_level: float | None = None):
        import aic_sdk

        model_id = model_id or config.AIC_MODEL_ID
        path = fetch_model(model_id)
        model = aic_sdk.Model.from_file(str(path))
        # the glob is by name; the id inside the file is what it actually is. It
        # carries the build too (`quail-ms-l-16khz-dtv5nvgu-v20`), which goes into
        # the chain name so a stored run says exactly which weights produced it
        build_id = model.get_id()
        if build_id != model_id and not build_id.startswith(model_id + "-"):
            raise RuntimeError(f"{path.name} holds model '{build_id}', not '{model_id}'")

        self.rate = int(model.get_optimal_sample_rate())
        cfg = aic_sdk.ProcessorConfig.optimal(model)
        self._proc = aic_sdk.Processor(model, config.AIC_LICENSE_KEY, cfg)
        self._frames = int(cfg.block_size)
        ctx = self._proc.get_context()
        if enhancement_level is not None:
            ctx.set_parameter(aic_sdk.ProcessorParameter.EnhancementLevel, float(enhancement_level))
        self.enhancement_level = float(ctx.get_parameter(aic_sdk.ProcessorParameter.EnhancementLevel))
        self.name = f"aic:{build_id}" + (
            f"@{self.enhancement_level:g}" if enhancement_level is not None else ""
        )
        # The SDK's own figure for how far process() output trails its input, at
        # this config. This wrapper adds no shift of its own (whole blocks in,
        # whole blocks out), so it is also the output shift selfcheck measures.
        # A live stream additionally waits one block to fill, hence the sum.
        self.output_delay_ms = int(ctx.get_audio_delay()) / self.rate * 1000
        self.algo_delay_ms = self._frames / self.rate * 1000 + self.output_delay_ms
        self._buf = np.zeros(0, dtype=np.float32)

    def _drain(self) -> np.ndarray:
        outs = []
        while len(self._buf) >= self._frames:
            block = np.ascontiguousarray(self._buf[: self._frames])
            self._buf = self._buf[self._frames :]
            outs.append(np.asarray(self._proc.process(block), dtype=np.float32).reshape(-1))
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

    def close(self) -> None:
        # ends the SDK's telemetry session now rather than at garbage collection
        try:
            self._proc.terminate_session()
        except Exception:
            pass
