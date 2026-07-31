"""Hush / Weya NC (pulp-vision/Hush, Apache-2.0) via its prebuilt native library.

The only *open* candidate here that targets the same job as Krisp BVC and Hecttor
`coda-vi`: suppressing a **competing human voice**, not just stationary noise.
It's DeepFilterNet3 retrained with a speaker-discriminative objective (60% of
training samples carry a background speaker), 16 kHz native — telephony's rate,
no wideband resampling — and fully causal.

Their published ONNX bundle is the three raw DFN graphs (encoder + ERB decoder +
DF decoder), which take features and return gains, not audio. All the DSP that
turns audio into features and gains back into audio lives in the shipped Rust
library, so we drive that through ctypes instead of reimplementing it:

    model_load_from_path(bundle) -> session_create(model, input_sr, atten_lim_db)
    -> get_frame_length() -> process_frame(in, out) per frame -> reset/free

`atten_lim_db` caps how much the model may attenuate. Their scale runs the
opposite way to what the name suggests: **100 = unlimited** (their default) and
**0 = no attenuation at all**, i.e. passthrough. Lower it to preserve ambience,
or sweep it if full strength turns out to hurt ASR.
"""

from __future__ import annotations

import ctypes
import platform

import numpy as np

from .. import config
from .base import Processor

_LIB_NAMES = {"Darwin": "libweya_nc.dylib", "Linux": "libweya_nc.so", "Windows": "weya_nc.dll"}
_BUNDLE = "advanced_dfnet16k_model_best_onnx.tar.gz"


def _paths() -> tuple:
    d = config.MODELS_DIR / "hush"
    return d / _LIB_NAMES.get(platform.system(), "libweya_nc.so"), d / _BUNDLE


def hush_available(_spec: dict | None = None) -> tuple[bool, str]:
    lib, bundle = _paths()
    for p in (lib, bundle):
        if not p.exists():
            return False, f"Hush file missing: {p} — run scripts/fetch_models.py"
    return True, ""


class HushProcessor(Processor):
    rate = 16_000
    # README: ~20 ms algorithmic latency, zero lookahead — confirmed by the lag
    # probe in scripts/selfcheck.py
    algo_delay_ms = 20.0

    def __init__(self, atten_lim_db: float | None = None):
        lib_path, bundle = _paths()
        self.name = "hush"
        lib = ctypes.CDLL(str(lib_path))
        lib.weya_nc_model_load_from_path.argtypes = [ctypes.c_char_p]
        lib.weya_nc_model_load_from_path.restype = ctypes.c_void_p
        lib.weya_nc_session_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float]
        lib.weya_nc_session_create.restype = ctypes.c_void_p
        lib.weya_nc_get_frame_length.argtypes = [ctypes.c_void_p]
        lib.weya_nc_get_frame_length.restype = ctypes.c_size_t
        lib.weya_nc_get_sample_rate.argtypes = [ctypes.c_void_p]
        lib.weya_nc_get_sample_rate.restype = ctypes.c_size_t
        lib.weya_nc_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.weya_nc_process_frame.restype = ctypes.c_float
        lib.weya_nc_session_free.argtypes = [ctypes.c_void_p]
        lib.weya_nc_model_free.argtypes = [ctypes.c_void_p]
        self._lib = lib

        self._model = lib.weya_nc_model_load_from_path(str(bundle).encode())
        if not self._model:
            raise RuntimeError(f"Hush: could not load model bundle {bundle}")
        self._session = lib.weya_nc_session_create(
            self._model,
            self.rate,
            float(config.HUSH_ATTEN_LIM_DB if atten_lim_db is None else atten_lim_db),
        )
        if not self._session:
            lib.weya_nc_model_free(self._model)
            raise RuntimeError("Hush: could not create session")
        self.rate = int(lib.weya_nc_get_sample_rate(self._session))
        self._frame = int(lib.weya_nc_get_frame_length(self._session))
        self._out = (ctypes.c_float * self._frame)()
        self._pending = np.zeros(0, dtype=np.float32)

    def _run_frame(self, frame: np.ndarray) -> np.ndarray:
        buf = (ctypes.c_float * self._frame)(*frame)
        self._lib.weya_nc_process_frame(self._session, buf, self._out)
        return np.frombuffer(self._out, dtype=np.float32).copy()

    def process_block(self, x: np.ndarray) -> np.ndarray:
        self._pending = np.concatenate([self._pending, x.astype(np.float32, copy=False)])
        outs = []
        while len(self._pending) >= self._frame:
            frame, self._pending = self._pending[: self._frame], self._pending[self._frame :]
            outs.append(self._run_frame(frame))
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        tail = len(self._pending)
        if not tail:
            return np.zeros(0, dtype=np.float32)
        out = self.process_block(np.zeros(self._frame - tail, dtype=np.float32))
        return out[:tail]

    def close(self) -> None:
        if getattr(self, "_session", None):
            self._lib.weya_nc_session_free(self._session)
            self._session = None
        if getattr(self, "_model", None):
            self._lib.weya_nc_model_free(self._model)
            self._model = None
