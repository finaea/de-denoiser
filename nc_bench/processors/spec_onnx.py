"""Per-frame spectral ONNX denoisers: DPDFNet, GTCRN, UL-UNAS.

All three ship streaming ONNX exports with the *same* shape of interface — one
STFT frame in (complex as a trailing `[..., 2]` axis) plus opaque cache/state
tensors, one enhanced frame + updated caches out — so one wrapper serves them
all. Framing (n_fft / hop / window / rate) is read from the model's ONNX
metadata when it carries it (DPDFNet and the sherpa-exported GTCRN do) and from
`_MODELS` otherwise.

The STFT loop mirrors sherpa-onnx's `online-speech-denoiser-stft-impl.h`:
roll the analysis buffer by one hop, window, rfft, model, irfft, window,
overlap-add, emit the oldest hop — skipping the first frame, so every emitted
hop has full overlap and the output is sample-aligned with the input. The
overlap-add is divided by the summed window² (torch.istft's normalisation),
which matters for plain-hann models (UL-UNAS) and is a no-op for the
Princen-Bradley windows (vorbis, hann-sqrt).
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Processor

# id -> file under models/ + any framing the ONNX metadata doesn't carry.
# extra_delay_ms = lookahead *inside* the graph, on top of the STFT framing: the
# frame it hands back is that much older than the frame fed in. Measured by
# scripts/selfcheck.py's lag probe (2026-07-31) — DPDFNet's deep-filter stage
# returns audio 30 ms old at every size, so its true live latency is 40 ms.
_MODELS: dict[str, dict] = {
    # DPDFNet (Ceva, Apache-2.0) — metadata-complete, incl. the native 8 kHz pair
    "dpdfnet2-8k": {"path": "dpdfnet/dpdfnet2_8khz.onnx", "extra_delay_ms": 30.0},
    "dpdfnet8-8k": {"path": "dpdfnet/dpdfnet8_8khz.onnx", "extra_delay_ms": 30.0},
    "dpdfnet-baseline": {"path": "dpdfnet/baseline.onnx", "extra_delay_ms": 30.0},
    "dpdfnet2": {"path": "dpdfnet/dpdfnet2.onnx", "extra_delay_ms": 30.0},
    "dpdfnet8": {"path": "dpdfnet/dpdfnet8.onnx", "extra_delay_ms": 30.0},
    # GTCRN (23.7 k params) — sherpa-onnx's export carries metadata
    "gtcrn": {"path": "gtcrn/gtcrn_simple.onnx"},
    # UL-UNAS — bare export; framing from the repo's stream/ reference
    "ulunas": {
        "path": "ulunas/ulunas_stream_simple.onnx",
        "n_fft": 512,
        "hop": 256,
        "window": "hann",
        "rate": 16_000,
    },
}


def _window(kind: str, n: int) -> np.ndarray:
    """Analysis = synthesis window, matching each model's training STFT."""
    if kind == "vorbis":
        s = np.sin(np.pi * (np.arange(n) + 0.5) / n)
        return np.sin(0.5 * np.pi * s * s).astype(np.float32)
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)
    if kind == "hann_sqrt":
        return np.sqrt(hann).astype(np.float32)
    return hann


def spec_onnx_available(model: str | None) -> tuple[bool, str]:
    model = model or ""
    if model not in _MODELS:
        return False, f"unknown spec-onnx model '{model}' (see nc_bench/processors/spec_onnx.py)"
    path = config.MODELS_DIR / _MODELS[model]["path"]
    if not path.exists():
        return False, f"model file missing: {path} — run scripts/fetch_models.py"
    return True, ""


class SpecOnnxProcessor(Processor):
    def __init__(self, model: str):
        import onnxruntime as ort

        cfg = _MODELS[model]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(config.MODELS_DIR / cfg["path"]), opts, providers=["CPUExecutionProvider"]
        )
        md = self._sess.get_modelmeta().custom_metadata_map

        self.name = model
        self.rate = int(cfg.get("rate") or md["sample_rate"])
        self._n = int(cfg.get("n_fft") or md["n_fft"])
        self._hop = int(cfg.get("hop") or md["hop_length"])
        self._win = _window(cfg.get("window") or md.get("window_type", "hann"), self._n)
        # streaming delay: (n_fft - hop) of framing, plus whatever the graph
        # itself holds back
        self.algo_delay_ms = (self._n - self._hop) / self.rate * 1000 + cfg.get(
            "extra_delay_ms", 0.0
        )

        ins = self._sess.get_inputs()
        self._spec_name, self._spec_shape = ins[0].name, list(ins[0].shape)
        self._cache_names = [i.name for i in ins[1:]]
        self._caches = [np.zeros(i.shape, dtype=np.float32) for i in ins[1:]]

        self._analysis = np.zeros(self._n, dtype=np.float32)
        self._ola = np.zeros(self._n, dtype=np.float32)
        self._wsq = np.zeros(self._n, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._started = False
        self._consumed = 0  # real input samples in
        self._emitted = 0  # output samples out (sample-aligned with the input)

    def _process_hop(self, hop: np.ndarray) -> np.ndarray:
        self._analysis = np.roll(self._analysis, -self._hop)
        self._analysis[-self._hop :] = hop

        spec = np.fft.rfft(self._analysis * self._win)
        frame = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)
        feeds = {self._spec_name: frame.reshape(self._spec_shape)}
        feeds.update(dict(zip(self._cache_names, self._caches)))
        outs = self._sess.run(None, feeds)
        enh = np.asarray(outs[0], dtype=np.float32).reshape(-1, 2)
        self._caches = [np.asarray(o, dtype=np.float32) for o in outs[1:]]

        y = np.fft.irfft(enh[:, 0] + 1j * enh[:, 1], n=self._n).astype(np.float32)
        for buf in (self._ola, self._wsq):
            buf[:-self._hop] = buf[self._hop :]
            buf[-self._hop :] = 0.0
        self._ola += y * self._win
        self._wsq += self._win * self._win

        if not self._started:  # first frame has no full overlap yet
            self._started = True
            return np.zeros(0, dtype=np.float32)
        out = (self._ola[: self._hop] / np.maximum(self._wsq[: self._hop], 1e-8)).copy()
        self._emitted += len(out)
        return out

    def _feed(self, x: np.ndarray) -> np.ndarray:
        self._pending = np.concatenate([self._pending, x])
        outs = []
        while len(self._pending) >= self._hop:
            hop, self._pending = self._pending[: self._hop], self._pending[self._hop :]
            outs.append(self._process_hop(hop))
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        self._consumed += len(x)
        return self._feed(x)

    def flush(self) -> np.ndarray:
        """Pad to a hop, then drain the overlap-add tail with zero hops."""
        outs = []
        if self._pending.size:
            outs.append(self._feed(np.zeros(self._hop - self._pending.size, dtype=np.float32)))
        while self._started and self._emitted < self._consumed:
            outs.append(self._process_hop(np.zeros(self._hop, dtype=np.float32)))
        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        overshoot = self._emitted - self._consumed
        return out[: len(out) - overshoot] if overshoot > 0 else out
