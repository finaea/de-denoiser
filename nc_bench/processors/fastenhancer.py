"""FastEnhancer (aask1357, arXiv:2509.21867) — wav2wav streaming ONNX.

The published wav2wav exports do STFT/iSTFT inside the graph: feed one hop of
new samples, get one hop back, with the model carrying its own input/output
caches. Everything we need is in the graph: `wav_in` is [1, hop] and
`cache_in_0` is [1, n_fft - hop], so hop size (which differs per size variant —
t/s 256, m 160, l 100) never has to be hard-coded.

Per upstream's docs the returned stream *lags* the input by n_fft - hop samples,
so the first that many output samples are dropped to keep the output
sample-aligned with the input (scoring compares gap windows by time).
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Processor, ort_options

# id -> file under models/ (the DNS-trained 16 kHz wav2wav release)
_MODELS = {
    "fastenhancer-t": "fastenhancer/fastenhancer_t_dns.onnx",
    "fastenhancer-s": "fastenhancer/fastenhancer_s_dns.onnx",
    "fastenhancer-l": "fastenhancer/fastenhancer_l_dns.onnx",
}


def fastenhancer_available(model: str | None) -> tuple[bool, str]:
    model = model or ""
    if model not in _MODELS:
        return False, f"unknown fastenhancer model '{model}'"
    path = config.MODELS_DIR / _MODELS[model]
    if not path.exists():
        return False, f"model file missing: {path} — run scripts/fetch_models.py"
    return True, ""


class FastEnhancerProcessor(Processor):
    rate = 16_000

    def __init__(self, model: str):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(
            str(config.MODELS_DIR / _MODELS[model]), ort_options(),
            providers=["CPUExecutionProvider"],
        )
        ins = self._sess.get_inputs()
        self.name = model
        self._hop = int(ins[0].shape[-1])
        self._wav_name = ins[0].name
        self._cache_names = [i.name for i in ins[1:]]
        self._caches = [np.zeros(i.shape, dtype=np.float32) for i in ins[1:]]
        # cache_in_0 holds the n_fft - hop samples of lookahead the model buffers
        self._delay = int(ins[1].shape[-1])
        self.algo_delay_ms = self._delay / self.rate * 1000
        self._skip = self._delay  # output lags input by exactly this many samples
        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed = 0  # real input samples in
        self._emitted = 0  # aligned output samples out

    def _run_hop(self, hop: np.ndarray) -> np.ndarray:
        feeds = {self._wav_name: hop.reshape(1, self._hop)}
        feeds.update(dict(zip(self._cache_names, self._caches)))
        outs = self._sess.run(None, feeds)
        self._caches = [np.asarray(o, dtype=np.float32) for o in outs[1:]]
        y = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        if self._skip:
            drop = min(self._skip, len(y))
            self._skip -= drop
            y = y[drop:]
        self._emitted += len(y)
        return y

    def _feed(self, x: np.ndarray) -> np.ndarray:
        self._pending = np.concatenate([self._pending, x])
        outs = []
        while len(self._pending) >= self._hop:
            hop, self._pending = self._pending[: self._hop], self._pending[self._hop :]
            outs.append(self._run_hop(hop))
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        self._consumed += len(x)
        return self._feed(x)

    def flush(self) -> np.ndarray:
        """Pad to a hop, then push zero hops until the delayed tail is out."""
        outs = []
        if self._pending.size:
            outs.append(self._feed(np.zeros(self._hop - self._pending.size, dtype=np.float32)))
        while self._emitted < self._consumed:
            outs.append(self._run_hop(np.zeros(self._hop, dtype=np.float32)))
        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        overshoot = self._emitted - self._consumed
        return out[: len(out) - overshoot] if overshoot > 0 else out
