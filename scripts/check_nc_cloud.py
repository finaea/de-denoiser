"""Do the T1 NC candidates and the cloud replay rail do what they claim, with every
outside dependency faked?

No network, no ai-coustics licence, no LiveKit, nothing under data/runs/ or
models/. Covers

- nc_bench/processors/aic.py against a fake `aic_sdk` (model lookup, build-id
  check, rate/delay bookkeeping, block buffering, flush length, close);
- pipeline.py's `output_delay_ms` roll-up;
- nc_bench/lk_cloud.py against fake `livekit.plugins.*` modules;
- nc_bench/cloud_replay.py: the pure alignment/gate helpers, `_finish_batch` on
  synthetic 24 kHz streams, `replay()` with `_run_batch` stubbed, `_timed()`;
- the cloud-replay wiring in nc_bench/server.py (`_load_candidates`,
  `_check_runnable`, `_process_run` with everything heavy stubbed);
- scripts/fetch_models.py `fetch_aic`;
- candidates.json sanity.

Real credentials in .env are neutralised before anything imports nc_bench.config,
and DATA_DIR / AIC_MODEL_DIR point at a temp dir, so this cannot touch the real
runs or models whatever the machine has configured.

    .venv/bin/python scripts/check_nc_cloud.py

A check listed in KNOWN_BUGS is a test that currently fails because of a
production bug; it is reported but does not fail the script.
"""

import asyncio
import contextlib
import enum
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SANDBOX = Path(tempfile.mkdtemp(prefix="nc-check-nc-cloud-")).resolve()
for _k, _v in {
    "LIVEKIT_URL": "wss://bogus.invalid",
    "LIVEKIT_API_KEY": "bogus-key",
    "LIVEKIT_API_SECRET": "bogus-secret",
    "AIC_LICENSE_KEY": "",
    "AIC_MODEL_ID": "",
    "AIC_MODEL_DIR": str(_SANDBOX / "aic-models"),
    "DATA_DIR": str(_SANDBOX / "data"),
}.items():
    os.environ[_k] = _v  # load_dotenv does not override what is already set

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nc_bench import cloud_replay as cr  # noqa: E402
from nc_bench import config, lk_cloud, pipeline  # noqa: E402
from nc_bench import processors as procs  # noqa: E402
from nc_bench.processors import aic as aicmod  # noqa: E402
from nc_bench.processors.base import Passthrough, Processor  # noqa: E402

KNOWN_BUGS: set[str] = set()

_MISSING = object()


class Env:
    """Monkeypatching that undoes itself."""

    def __init__(self):
        self.undo = []

    def attr(self, obj, name, value):
        self.undo.append(("attr", obj, name, getattr(obj, name, _MISSING)))
        setattr(obj, name, value)

    def module(self, name, mod):
        self.undo.append(("mod", name, sys.modules.get(name, _MISSING)))
        sys.modules[name] = mod

    def plugin(self, name, mod):
        """livekit.plugins.<name>: `from livekit.plugins import x` reads the package
        attribute first, so a real plugin imported earlier would shadow a fake."""
        import livekit.plugins as lp

        self.module(f"livekit.plugins.{name}", mod)
        self.attr(lp, name, mod)

    def env(self, key, value):
        self.undo.append(("env", key, os.environ.get(key, _MISSING)))
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def close(self):
        for kind, a, b, *rest in reversed(self.undo):
            if kind == "attr":
                setattr(a, b, rest[0]) if rest[0] is not _MISSING else delattr(a, b)
            elif kind == "mod":
                if b is _MISSING:
                    sys.modules.pop(a, None)
                else:
                    sys.modules[a] = b
            else:
                if b is _MISSING:
                    os.environ.pop(a, None)
                else:
                    os.environ[a] = b


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="t-", dir=_SANDBOX)).resolve()


def rundir() -> Path:
    """<tmp>/runs-x/r1: a run directory the REAL store can find once config.RUNS_DIR
    points at its parent (see _server_stubs)."""
    d = Path(tempfile.mkdtemp(prefix="runs-", dir=_SANDBOX)).resolve() / "r1"
    d.mkdir()
    return d


def raises(exc, fn, *a, contains: str = "", **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        assert contains in str(e), f"{type(e).__name__} message {e!r} lacks {contains!r}"
        return e
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


# ================================================================ synthetic audio


def speechy(dur: float, rate: int, seed: int, env_out: list | None = None) -> np.ndarray:
    """Aperiodic bursty noise: an envelope that decorrelates within ~150 ms like speech
    (a periodic one would correlate with itself at every offset and defeat
    `prominence`)."""
    rng = np.random.default_rng(seed)
    n = int(dur * rate)
    t = np.arange(n) / rate
    k = int(dur * 12) + 2
    env = np.clip(np.interp(t, np.arange(k) / 12, rng.standard_normal(k)), 0, None) ** 1.5
    x = env * rng.standard_normal(n) + 0.01 * rng.standard_normal(n)
    x = (0.7 * x / np.abs(x).max()).astype(np.float32)
    if env_out is not None:
        env_out.append(env)
    return x


def bursty(dur: float, rate: int, seed: int, env_out: list | None = None) -> np.ndarray:
    """Speech-like and APERIODIC: random 80-300 ms bursts separated by random 80-400 ms
    gaps, the first one starting at 20 ms. A strictly periodic envelope cannot be
    aligned on a 2 s input by any method."""
    rng = np.random.default_rng(seed)
    n = int(dur * rate)
    env = np.zeros(n)
    t, first = int(0.02 * rate), True
    while t < n:
        length = int(rng.uniform(0.08, 0.3) * rate)
        amp = rng.uniform(0.5 if first else 0.2, 0.9)
        first = False
        seg = np.sqrt(np.hanning(length)) * amp
        m = min(length, n - t)
        env[t : t + m] = seg[:m]
        t += length + int(rng.uniform(0.08, 0.4) * rate)
    x = env * rng.standard_normal(n) + 0.003 * rng.standard_normal(n)
    x = (0.7 * x / np.abs(x).max()).astype(np.float32)
    if env_out is not None:
        env_out.append(env)
    return x


class Fix:
    """One synthetic batch on disk: a 48 kHz input.wav and the 24 kHz streams a
    subscriber would have saved (3 s of digital silence first, like the real preroll)."""

    def __init__(self, dur: float = 16.0, seed: int = 3, gen=None):
        self.tmp = tmpdir()
        self.work, self.out = self.tmp / "work", self.tmp / "out"
        self.work.mkdir()
        self.out.mkdir()
        self.x48 = (gen or speechy)(dur, 48_000, seed)
        self.input = self.tmp / "input.wav"
        sf.write(self.input, self.x48, 48_000)
        self.x24 = soxr.resample(self.x48, 48_000, 24_000).astype(np.float32)
        self.x16 = cr._mono(self.input, 16_000)
        self.dur = len(self.x16) / 16_000

    def pcm(self, gain: float = 1.0, lead_s: float = 3.0, tail_s: float = 1.0, x24=None) -> np.ndarray:
        x = self.x24 if x24 is None else x24
        y = np.concatenate([np.zeros(int(lead_s * 24_000), np.float32), x * gain,
                            np.zeros(int(tail_s * 24_000), np.float32)])
        return np.clip(y * 32768, -32768, 32767).astype(np.int16)

    @staticmethod
    def speech_flags(pcm: np.ndarray) -> np.ndarray:
        n = len(pcm) // 1200
        fr = pcm[: n * 1200].astype(np.float64).reshape(n, 1200)
        return (np.sqrt((fr ** 2).mean(axis=1)) > 300).astype(np.int8)

    def save(self, name: str, pcm: np.ndarray, flags=None) -> None:
        if flags is None:
            flags = -np.ones(len(pcm) // 1200, np.int8)
        np.save(self.work / f"{name}.pcm.npy", pcm)
        np.save(self.work / f"{name}.flags.npy", np.asarray(flags, np.int8))

    def meta(self, names, starved: float = 0.0, batch: int = 1, batches: int = 1, open_ms=None) -> dict:
        m = {
            "room": "r", "batch": batch, "batches": batches,
            "pub": {"wall_s": 20.0, "audio_s": 20.0, "starved_ms": starved,
                    "starved_events": 0 if starved == 0 else 5},
            "sub": {"streams": [{"name": n, "frames": 400, "audio_s": 20.0, "late_s": 0.0,
                                 "cpu_per_audio_s": 0.01 * (i + 1)}
                                for i, n in enumerate(names)],
                    "max_backlog_s": 0.1},
        }
        for entry in m["sub"]["streams"]:
            if open_ms and entry["name"] in open_ms:
                entry["open_ms"] = open_ms[entry["name"]]
        return m

    def finish(self, streams, starved: float = 0.0, raw_ids=("cloud-raw",), open_ms=None) -> dict:
        return cr._finish_batch(self.input, streams, self.work, self.out,
                                self.meta([s["name"] for s in streams], starved, open_ms=open_ms),
                                list(raw_ids))


def stream(name, cid, model, opts=None) -> dict:
    return {"name": name, "id": cid, "lk_model": model, "lk_options": opts or {}}


CONTROL = stream("0", None, lk_cloud.RAW)
_FIX: dict = {}


def shared_fix() -> Fix:
    """Built once: the input, not a stream, is what costs."""
    if "f" not in _FIX:
        _FIX["f"] = Fix()
    return _FIX["f"]


# ====================================================================== aic-sdk


def fake_aic_sdk(build_id="quail-vf-2.2-l-16khz-horgwub0-v14", rate=16_000, block=160,
                 delay=320, level=1.0, out2d=False):
    calls = types.SimpleNamespace(from_file=[], optimal=[], processor=[], set_param=[],
                                  process=[], terminated=0, downloads=[], fail_terminate=False)
    m = types.ModuleType("aic_sdk")
    m.calls = calls

    class ProcessorParameter:
        EnhancementLevel = "EnhancementLevel"

    class Model:
        @staticmethod
        def from_file(path):
            calls.from_file.append(path)
            return Model()

        @staticmethod
        def download(model_id, d):
            calls.downloads.append((model_id, d))
            p = Path(d) / (model_id.replace("-", "_").replace(".", "_") + "_dl_v1.aicmodel")
            p.write_bytes(b"x")
            return str(p)

        def get_id(self):
            return build_id

        def get_optimal_sample_rate(self):
            return rate

    class ProcessorConfig:
        def __init__(self):
            self.block_size = block

        @staticmethod
        def optimal(model):
            calls.optimal.append(model)
            return ProcessorConfig()

    class Ctx:
        def __init__(self):
            self.level = level

        def set_parameter(self, p, v):
            calls.set_param.append((p, v))
            self.level = v

        def get_parameter(self, p):
            return self.level

        def get_audio_delay(self):
            return delay

    class Processor:
        def __init__(self, model, key, cfg):
            calls.processor.append((model, key, cfg))
            self.ctx = Ctx()

        def get_context(self):
            return self.ctx

        def process(self, blk):
            assert blk.ndim == 1 and blk.dtype == np.float32, (blk.shape, blk.dtype)
            assert blk.flags["C_CONTIGUOUS"] and len(blk) == block, len(blk)
            calls.process.append(len(blk))
            out = blk * 0.5
            return out.reshape(1, -1) if out2d else out

        def terminate_session(self):
            calls.terminated += 1
            if calls.fail_terminate:
                raise RuntimeError("telemetry down")

    m.ProcessorParameter, m.Model, m.ProcessorConfig, m.Processor = (
        ProcessorParameter, Model, ProcessorConfig, Processor)
    return m


def _touch(d: Path, name: str, mtime: float | None = None) -> Path:
    p = d / name
    p.write_bytes(b"x")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def check_aic_model_file(env: Env):
    d = tmpdir()
    env.attr(config, "AIC_MODEL_DIR", d)
    assert aicmod.model_file("quail-ms-l-16khz") is None
    _touch(d, "quail_ms_l_8khz_zzz_v1.aicmodel")
    assert aicmod.model_file("quail-ms-l-16khz") is None, "16 kHz id matched the 8 kHz file"
    assert aicmod.model_file("quail-ms-l-8khz").name == "quail_ms_l_8khz_zzz_v1.aicmodel"
    _touch(d, "quail_ms_l_16khz_old_v1.aicmodel", 1_000_000)
    new = _touch(d, "quail_ms_l_16khz_new_v2.aicmodel", 2_000_000)
    assert aicmod.model_file("quail-ms-l-16khz") == new, "newest mtime must win"
    _touch(d, "quail_vf_2_2_l_16khz_horgwub0_v14.aicmodel")
    assert aicmod.model_file("quail-vf-2.2-l-16khz").name.startswith("quail_vf_2_2_l_16khz_")
    assert aicmod.model_file("quail-vf-2.2-s-16khz") is None


def check_aic_fetch_model(env: Env):
    d = tmpdir() / "not-yet"
    env.attr(config, "AIC_MODEL_DIR", d)
    sdk = fake_aic_sdk()
    env.module("aic_sdk", sdk)
    p = aicmod.fetch_model("quail-ms-s-16khz")
    assert d.is_dir() and p.exists() and sdk.calls.downloads == [("quail-ms-s-16khz", str(d))]
    again = aicmod.fetch_model("quail-ms-s-16khz")
    assert again == p and len(sdk.calls.downloads) == 1, "cached model was downloaded again"


def check_aic_available(env: Env):
    env.attr(config, "AIC_MODEL_ID", "")
    env.attr(config, "AIC_LICENSE_KEY", "")
    ok, why = aicmod.aic_available({"model": "x"})
    assert not ok and "AIC_LICENSE_KEY" in why, why
    env.attr(config, "AIC_LICENSE_KEY", "bogus")
    env.module("aic_sdk", None)  # `import aic_sdk` -> ImportError
    ok, why = aicmod.aic_available({"model": "x"})
    assert not ok and "not installed" in why, why
    env.module("aic_sdk", fake_aic_sdk())
    ok, why = aicmod.aic_available({})
    assert not ok and "model id" in why, why
    assert aicmod.aic_available({"model": "quail-ms-l-16khz"}) == (True, "")
    env.attr(config, "AIC_MODEL_ID", "from-env")
    assert aicmod.aic_available(None) == (True, "")


def _aic_setup(env: Env, **kw):
    d = tmpdir()
    env.attr(config, "AIC_MODEL_DIR", d)
    env.attr(config, "AIC_LICENSE_KEY", "bogus-key")
    _touch(d, "quail_vf_2_2_l_16khz_horgwub0_v14.aicmodel")
    sdk = fake_aic_sdk(**kw)
    env.module("aic_sdk", sdk)
    return d, sdk


def check_aic_processor_bookkeeping(env: Env):
    d, sdk = _aic_setup(env, rate=8_000, block=80, delay=96, level=0.8)
    p = aicmod.AICProcessor("quail-vf-2.2-l-16khz")
    c = sdk.calls
    assert c.from_file == [str(d / "quail_vf_2_2_l_16khz_horgwub0_v14.aicmodel")], c.from_file
    assert len(c.optimal) == 1 and len(c.processor) == 1
    model, key, cfg = c.processor[0]
    assert key == "bogus-key" and model is c.optimal[0] and cfg.block_size == 80
    assert p.rate == 8_000 and isinstance(p.rate, int)
    assert p.output_delay_ms == 96 / 8_000 * 1000 == 12.0, p.output_delay_ms
    assert abs(p.algo_delay_ms - (80 / 8_000 * 1000 + 12.0)) < 1e-9, p.algo_delay_ms
    assert p.enhancement_level == 0.8 and c.set_param == [], "level touched without being asked"
    assert p.name == "aic:quail-vf-2.2-l-16khz-horgwub0-v14", p.name
    p.close()
    assert c.terminated == 1
    p2 = aicmod.AICProcessor("quail-vf-2.2-l-16khz", enhancement_level=0.5)
    assert c.set_param == [("EnhancementLevel", 0.5)], c.set_param
    assert p2.enhancement_level == 0.5
    assert p2.name == "aic:quail-vf-2.2-l-16khz-horgwub0-v14@0.5", p2.name
    p3 = aicmod.AICProcessor("quail-vf-2.2-l-16khz", enhancement_level=0)
    assert c.set_param[-1] == ("EnhancementLevel", 0.0) and p3.name.endswith("@0"), p3.name


def check_aic_build_id_guard(env: Env):
    _aic_setup(env, build_id="quail-vf-2.2-s-16khz-abc-v1")
    raises(RuntimeError, aicmod.AICProcessor, "quail-vf-2.2-l-16khz", contains="holds model")
    env.module("aic_sdk", fake_aic_sdk(build_id="quail-vf-2.2-l-16khz"))  # exact id is fine
    aicmod.AICProcessor("quail-vf-2.2-l-16khz")
    # "id-" is the boundary: an id that merely starts with ours is another model
    env.module("aic_sdk", fake_aic_sdk(build_id="quail-vf-2.2-l-16khzX-abc"))
    raises(RuntimeError, aicmod.AICProcessor, "quail-vf-2.2-l-16khz")


def check_aic_streaming_and_flush(env: Env):
    for out2d in (False, True):
        _, sdk = _aic_setup(env, block=160, out2d=out2d)
        p = aicmod.AICProcessor("quail-vf-2.2-l-16khz")
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000).astype(np.float32)
        a = p.process_block(x[:100])
        assert a.shape == (0,) and a.dtype == np.float32, (a.shape, a.dtype)
        b = p.process_block(x[100:500])  # 500 buffered -> 3 blocks of 160, 20 left
        assert b.shape == (480,) and np.allclose(b, 0.5 * x[:480]), b.shape
        c = p.process_block(x[500:1000])  # 520 pending -> 3 blocks, 40 left
        assert c.shape == (480,) and np.allclose(c, 0.5 * x[480:960])
        tail = p.flush()
        assert tail.shape == (40,), f"flush returned {tail.shape}, the tail is 40 samples"
        assert np.allclose(tail, 0.5 * x[960:1000])
        assert p.flush().shape == (0,)
        assert set(sdk.calls.process) == {160} and len(sdk.calls.process) == 7
    # a tail that is an exact multiple of the block returns exactly that
    _, sdk = _aic_setup(env, block=160)
    p = aicmod.AICProcessor("quail-vf-2.2-l-16khz")
    p.process_block(np.ones(159, np.float32))
    assert p.flush().shape == (159,)
    p.process_block(np.ones(160, np.float32))  # already drained by process_block
    assert p.flush().shape == (0,)


def check_aic_close_swallows(env: Env):
    _, sdk = _aic_setup(env)
    p = aicmod.AICProcessor("quail-vf-2.2-l-16khz")
    sdk.calls.fail_terminate = True
    p.close()  # must not raise
    assert sdk.calls.terminated == 1


def check_aic_registry(env: Env):
    assert procs._REGISTRY["aic"][0] is aicmod.aic_available, "registry 'aic' availability checker"
    seen = {}

    class Spy:
        def __init__(self, model_id=None, enhancement_level=None):
            seen.update(model_id=model_id, enhancement_level=enhancement_level)

    env.attr(procs, "AICProcessor", Spy)
    procs.build_chain([{"proc": "aic", "model": "rook-ms-s-8khz", "enhancement_level": 0.3}])
    assert seen == {"model_id": "rook-ms-s-8khz", "enhancement_level": 0.3}, seen
    env.attr(config, "AIC_LICENSE_KEY", "")
    ok, why = procs.proc_available({"proc": "aic", "model": "rook-ms-s-8khz"})
    assert not ok and "AIC_LICENSE_KEY" in why


# ==================================================================== pipeline


class _Stage(Processor):
    def __init__(self, rate=16_000, delay=None, algo=0.0):
        self.rate, self.output_delay_ms, self.algo_delay_ms = rate, delay, algo

    def process_block(self, x):
        return x


def check_pipeline_output_delay(env: Env):
    assert Processor.output_delay_ms is None and Passthrough().output_delay_ms is None
    d = tmpdir()
    sf.write(d / "in.wav", np.zeros(8000, np.float32) + 0.01, 16_000)

    def run(stages):
        env.attr(pipeline, "build_chain", lambda spec: stages)
        return pipeline.run_chain(d / "in.wav", [{}], d / "out.wav")["latency"]

    lat = run([_Stage(16_000, 12.0, 20.0), _Stage(8_000, 5.5, 10.0)])
    assert lat["output_delay_ms"] == 17.5 and lat["algo_delay_ms"] == 30.0, lat
    lat = run([_Stage(16_000, 12.0, 20.0), _Stage(16_000, None, 10.0)])
    assert lat["output_delay_ms"] is None, "a partial sum must be reported as None"
    assert run([Passthrough()])["output_delay_ms"] is None
    assert run([_Stage(16_000, 0.0)])["output_delay_ms"] == 0.0, "0 ms is a value, not 'unknown'"
    assert sf.info(d / "out.wav").samplerate == config.PIPELINE_RATE


# ==================================================================== lk_cloud


class _Enh(enum.Enum):
    QUAIL_L = 1
    QUAIL_VF_L = 2
    QUAIL_VF_S = 3
    ROOK_S = 4


def fake_plugins(env: Env):
    aic = types.ModuleType("livekit.plugins.ai_coustics")
    aic.EnhancerModel = _Enh

    @dataclass
    class ModelParameters:
        bypass: float | None = None
        enhancement_level: float | None = None

    @dataclass
    class VadSettings:
        speech_hold_duration: float | None = None
        sensitivity: float | None = None
        minimum_speech_duration: float | None = None

    aic.ModelParameters, aic.VadSettings = ModelParameters, VadSettings
    aic.audio_enhancement = lambda **kw: ("aic-filter", kw)
    krisp = types.ModuleType("livekit.plugins.krisp")
    krisp.voice_isolation = lambda: "viva"
    krisp.voice_isolation_telephony = lambda: "viva-tel"
    nc = types.ModuleType("livekit.plugins.noise_cancellation")
    nc.NC, nc.BVC, nc.BVCTelephony = (lambda: "nc"), (lambda: "bvc"), (lambda: "bvc-tel")
    for name, mod in (("ai_coustics", aic), ("krisp", krisp), ("noise_cancellation", nc)):
        env.plugin(name, mod)
    return aic


def cloud_ready(env: Env):
    env.attr(config, "LIVEKIT_URL", "wss://x.livekit.cloud")
    env.attr(config, "LIVEKIT_API_KEY", "k")
    env.attr(config, "LIVEKIT_API_SECRET", "s")


def check_lk_helpers():
    assert lk_cloud.RAW == "RAW" and lk_cloud.AIC_VAD_KEY == "lk.aic-vad"
    assert lk_cloud._plugin("RAW") is None
    assert lk_cloud._plugin("AIC:QUAIL_L") == "ai_coustics"
    assert lk_cloud._plugin("KRISP:VIVA") == "krisp"
    assert lk_cloud._plugin("BVC") == "noise_cancellation"
    assert lk_cloud.is_aic("AIC:ROOK_S") and not lk_cloud.is_aic("RAW") and not lk_cloud.is_aic("BVC")
    table = [
        ("RAW", None, True), ("RAW", {"bypass": 0}, True),
        ("AIC:QUAIL_L", None, False), ("AIC:QUAIL_L", {}, False),
        ("AIC:QUAIL_L", {"bypass": 1.0}, True), ("AIC:QUAIL_L", {"bypass": 0.5}, False),
        ("AIC:QUAIL_L", {"bypass": 0.0}, False), ("AIC:QUAIL_L", {"bypass": None}, False),
        ("AIC:QUAIL_L", {"enhancement_level": 0.0}, True),
        ("AIC:QUAIL_L", {"enhancement_level": 0.5}, False),
        ("AIC:QUAIL_L", {"vad_sensitivity": 5}, False),
        ("KRISP:VIVA", {"bypass": 1.0}, False), ("BVC", None, False),
    ]
    for model, opts, want in table:
        assert lk_cloud.expects_raw(model, opts) is want, (model, opts, want)
        assert lk_cloud.changes_audio(model, opts) is (not want), (model, opts)


def check_lk_check_options():
    co = lk_cloud._check_options
    assert co("AIC:QUAIL_L", None) == "" and co("AIC:QUAIL_L", {}) == "" and co("BVC", {}) == ""
    assert "only supported for ai-coustics" in co("BVC", {"bypass": 1.0})
    assert "only supported for ai-coustics" in co("RAW", {"bypass": 1.0})
    assert "unknown lk_options" in co("AIC:QUAIL_L", {"nope": 1})
    for ok in (1.0, 15.0, 7):
        assert co("AIC:QUAIL_L", {"vad_sensitivity": ok}) == "", ok
    for bad in (0.99, 15.01, 0, 100):
        assert "vad_sensitivity" in co("AIC:QUAIL_L", {"vad_sensitivity": bad}), bad
    for key in ("bypass", "enhancement_level"):
        assert co("AIC:QUAIL_L", {key: 0.0}) == "" and co("AIC:QUAIL_L", {key: 1.0}) == ""
        assert key in co("AIC:QUAIL_L", {key: 1.01}) and key in co("AIC:QUAIL_L", {key: -0.01})


def check_lk_available(env: Env):
    fake_plugins(env)
    env.attr(config, "LIVEKIT_URL", "wss://self-hosted.example.com")
    env.attr(config, "LIVEKIT_API_KEY", "k")
    env.attr(config, "LIVEKIT_API_SECRET", "s")
    ok, why = lk_cloud.available("RAW")
    assert not ok and "not a LiveKit Cloud" in why, why
    cloud_ready(env)
    env.attr(config, "LIVEKIT_API_SECRET", "")
    ok, why = lk_cloud.available("RAW")
    assert not ok and "LIVEKIT_API_KEY" in why
    env.attr(config, "LIVEKIT_API_SECRET", "s")
    assert lk_cloud.available("RAW") == (True, "")
    assert lk_cloud.available("AIC:QUAIL_L") == (True, "")
    assert lk_cloud.available("AIC:ROOK_S", {"bypass": 1.0}) == (True, "")
    ok, why = lk_cloud.available("AIC:QUAIL_L", {"vad_sensitivity": 99})
    assert not ok and "vad_sensitivity" in why
    ok, why = lk_cloud.available("AIC:NOPE")
    assert not ok and "no model 'NOPE'" in why, why
    assert lk_cloud.available("KRISP:VIVA") == (True, "")
    assert lk_cloud.available("KRISP:VIVA_TEL") == (True, "")
    ok, why = lk_cloud.available("KRISP:FOO")
    assert not ok and "unknown Krisp VIVA" in why, why
    assert lk_cloud.available("BVC") == (True, "")
    ok, why = lk_cloud.available("NOPE")
    assert not ok and "not exposed" in why, why
    ok, why = lk_cloud.available("BVC", {"bypass": 1})
    assert not ok and "only supported for ai-coustics" in why
    env.plugin("krisp", None)  # sys.modules[name] = None -> ImportError
    ok, why = lk_cloud.available("KRISP:VIVA")
    assert not ok and "plugin not installed" in why, why


def check_lk_build(env: Env):
    fake_plugins(env)
    assert lk_cloud.build("RAW") is None
    tag, kw = lk_cloud.build("AIC:QUAIL_L")
    assert tag == "aic-filter" and kw["model"] is _Enh.QUAIL_L
    assert kw["model_parameters"] is None
    assert kw["vad_settings"].sensitivity is None
    _, kw = lk_cloud.build("AIC:ROOK_S", {"bypass": 1.0})
    assert kw["model"] is _Enh.ROOK_S
    assert (kw["model_parameters"].bypass, kw["model_parameters"].enhancement_level) == (1.0, None)
    _, kw = lk_cloud.build("AIC:QUAIL_VF_L", {"enhancement_level": 0})
    assert (kw["model_parameters"].bypass, kw["model_parameters"].enhancement_level) == (None, 0.0), \
        "enhancement_level 0.0 must be passed, not treated as unset"
    _, kw = lk_cloud.build("AIC:QUAIL_L", {"vad_sensitivity": 8})
    assert kw["vad_settings"].sensitivity == 8.0 and isinstance(kw["vad_settings"].sensitivity, float)
    assert kw["model_parameters"] is None, "sensitivity alone must not create ModelParameters"
    raises(RuntimeError, lk_cloud.build, "AIC:NOPE", contains="unknown ai-coustics model")
    raises(RuntimeError, lk_cloud.build, "AIC:QUAIL_L", {"bypass": 5}, contains="bypass")
    raises(RuntimeError, lk_cloud.build, "BVC", {"bypass": 1}, contains="only supported")
    assert lk_cloud.build("KRISP:VIVA") == "viva"
    assert lk_cloud.build("KRISP:VIVA_TEL") == "viva-tel"
    raises(RuntimeError, lk_cloud.build, "KRISP:NOPE", contains="unknown Krisp VIVA")
    assert (lk_cloud.build("NC"), lk_cloud.build("BVC"), lk_cloud.build("BVCTelephony")) == \
        ("nc", "bvc", "bvc-tel")
    raises(RuntimeError, lk_cloud.build, "Nope", contains="not exposed")


def check_lk_plugin_versions(env: Env):
    from importlib.metadata import PackageNotFoundError

    have = {"livekit": "1.1.18", "livekit-plugins-ai-coustics": "0.3.2"}

    def fake_version(pkg):
        if pkg in have:
            return have[pkg]
        raise PackageNotFoundError(pkg)

    env.attr(lk_cloud, "version", fake_version)
    assert lk_cloud.plugin_versions() == have


# ============================================================ cloud_replay: pure


def check_plan_batches():
    def cands(spec):
        return [{"id": f"{m}{i}", "lk_model": m} for m, cnt in spec for i in range(cnt)]

    assert cr.plan_batches([]) == []
    cs = cands([("A", 3), ("B", 3), ("C", 2), ("D", 1)])
    for max_streams in (2, 3, 4, 5, 9):
        per = max(1, max_streams - 1)
        batches = cr.plan_batches(cs, max_streams)
        flat = [c["id"] for b in batches for c in b]
        assert sorted(flat) == sorted(c["id"] for c in cs), "every candidate exactly once"
        assert all(0 < len(b) <= per for b in batches), [len(b) for b in batches]
        idx = {c["id"]: i for i, c in enumerate(cs)}
        for b in batches:
            order = [idx[c["id"]] for c in b]
            assert order == sorted(order), "given order lost inside a batch"
        for m, cnt in (("A", 3), ("B", 3), ("C", 2), ("D", 1)):
            where = {k for k, b in enumerate(batches) for c in b if c["lk_model"] == m}
            if cnt <= per:
                assert len(where) == 1, f"model {m} split over batches {where} at max_streams={max_streams}"
    assert len(cr.plan_batches(cs, 9)) == 2  # 9 candidates, 8 per batch
    # a bypass variant stays with its base model even when the list interleaves them
    lk = [{"id": "q", "lk_model": "AIC:QUAIL_L"}, {"id": "k", "lk_model": "KRISP:VIVA"},
          {"id": "r", "lk_model": "AIC:ROOK_S"}, {"id": "qb", "lk_model": "AIC:QUAIL_L"},
          {"id": "rb", "lk_model": "AIC:ROOK_S"}]
    b = cr.plan_batches(lk, 3)  # 2 per batch
    home = {c["id"]: k for k, bb in enumerate(b) for c in bb}
    assert home["q"] == home["qb"] and home["r"] == home["rb"], b
    assert [c["id"] for c in b[home["q"]]] == ["q", "qb"]
    # a group larger than a batch splits, max_streams=1 still makes progress
    assert [len(x) for x in cr.plan_batches(cands([("A", 5)]), 3)] == [2, 2, 1]
    assert [len(x) for x in cr.plan_batches(cands([("A", 2)]), 1)] == [1, 1]


def check_best_lag():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(2000)
    sig = np.concatenate([1e-3 * rng.standard_normal(300), 0.3 * ref, 1e-3 * rng.standard_normal(500)])
    lag, c = cr._best_lag(ref, sig, 600)
    assert lag == 300 and c > 0.99, (lag, c)
    lag2, c2 = cr._best_lag(ref, 40 * sig, 600)
    assert lag2 == 300 and abs(c2 - c) < 1e-6, "must be scale invariant"
    lag3, c3 = cr._best_lag(ref, sig, 100)  # the true lag is out of range
    assert 0 <= lag3 <= 100 and c3 < 0.5, (lag3, c3)
    short = np.concatenate([np.zeros(300), 0.3 * ref[:1500]])  # sig runs out: zero padded
    lag4, c4 = cr._best_lag(ref, short, 600)
    assert lag4 == 300 and 0.5 < c4 < 1.0, (lag4, c4)
    # a quiet stretch cannot win on scale alone
    loud_then_quiet = np.concatenate([np.zeros(50), 0.001 * ref, 100 * rng.standard_normal(2000)])
    lag5, _ = cr._best_lag(ref, loud_then_quiet, 200)
    assert lag5 == 50, lag5


def _x16(dur, seed=5):
    env = []
    return speechy(dur, 16_000, seed, env), env[0]


def check_lead_in():
    fs = 16_000
    x, env = _x16(16)
    lead = int(1.234 * fs)
    y = np.concatenate([np.zeros(lead, np.float32), 0.8 * x, np.zeros(fs, np.float32)])
    assert cr.lead_in(x, y) == lead
    assert cr.lead_in(x, np.concatenate([0.3 * x, np.zeros(fs, np.float32)])) == 0
    y_noisy = y + 0.01 * np.random.default_rng(2).standard_normal(len(y)).astype(np.float32)
    assert cr.lead_in(x, y_noisy) == lead
    # same envelope, unrelated fine structure (a voice-isolation filter that lost the
    # waveform): the coarse 10 ms-grid answer is kept
    weak = 0.3 * env * np.random.default_rng(3).standard_normal(len(x)).astype(np.float32)
    lead2 = int(1.23 * fs)
    got = cr.lead_in(x, np.concatenate([np.zeros(lead2, np.float32), weak, np.zeros(fs, np.float32)]))
    assert got % 160 == 0 and abs(got - lead2) <= 2 * 160, (got, lead2)


def check_lag_drift():
    fs = 16_000
    x, _ = _x16(30, seed=6)
    lead = int(2.345 * fs)
    y = np.concatenate([np.zeros(lead, np.float32), 0.8 * x, np.zeros(fs, np.float32)])
    got = cr.lead_in(x, y)
    al = y[got : got + len(x)]
    lags = cr.local_lags_ms(x, al)
    assert len(lags) >= 4 and set(lags) == {0.0}, lags
    assert cr.lag_drift_ms(x, al) == (0.0, len(lags))
    k = lead + 15 * fs
    y2 = np.concatenate([y[:k], np.zeros(int(0.03 * fs), np.float32), y[k:]])  # 30 ms gap
    got2 = cr.lead_in(x, y2)
    lags2 = cr.local_lags_ms(x, y2[got2 : got2 + len(x)])
    drift, used = cr.lag_drift_ms(x, y2[got2 : got2 + len(x)])
    assert abs(drift - 30) <= 1.5 and used == len(lags2), (drift, used)
    assert drift == round(max(lags2) - min(lags2), 1), "drift is the spread max-min, not the max"
    # too short for one window: no votes, no drift
    assert cr.local_lags_ms(x[:fs], al[:fs]) == [] and cr.lag_drift_ms(x[:fs], al[:fs]) == (0.0, 0)
    # a silent input has no windows above the floor worth voting
    assert cr.lag_drift_ms(np.zeros(20 * fs, np.float32), np.zeros(20 * fs, np.float32)) == (0.0, 0)


def check_prominence():
    x, _ = _x16(30, seed=8)
    other, _ = _x16(30, seed=99)
    ex, eo = cr._envelope(x), cr._envelope(other)
    same = cr.prominence(ex, cr._envelope(0.5 * x))
    assert same > 0.3, f"a scaled copy should have a clear peak, got {same}"
    assert cr.prominence(ex, eo) < cr.MIN_PROMINENCE, "unrelated recording must fail the gate"
    shifted = np.concatenate([np.zeros(30), ex[:-30]])  # 300 ms off
    assert cr.prominence(ex, shifted) < cr.MIN_PROMINENCE, "misaligned output must fail the gate"
    assert cr.prominence(ex, np.full_like(ex, 0.1)) < cr.MIN_PROMINENCE, "flat output must fail"


def check_aicvad():
    flags = np.ones(1000, np.int8)
    m, info = cr._aicvad(flags, 2.345, 40.0)
    assert m.dtype == np.int8 and len(m) == 801 and info["frames"] == 801
    assert info["dtype"] == "int8" and info["encoding"] == lk_cloud.AICVAD_ENCODING
    assert "-1" in info["encoding"] and "no flag" in info["encoding"]
    assert info["t0_s"] == -0.045 and info["frame_s"] == 0.05
    assert info["speech_frames"] == 801 and info["missing_frames"] == 0
    # lead-in on a frame boundary: t0 is exactly 0 and frame 0 is flag `start`
    f = np.array([0, 0, 1, 1, 0, -1, 1], np.int8)
    m, info = cr._aicvad(f, 0.1, 0.2)  # start=2, n=4
    assert m.dtype == np.int8 and m.tolist() == [1, 1, 0, -1] and info["t0_s"] == 0.0, \
        "-1 (no flag on that frame) must survive as -1, never 0"
    assert info["frames"] == 4 and info["speech_frames"] == 2 and info["missing_frames"] == 1
    # the flags run out before the input does: the rest is "missing", not speech
    m, info = cr._aicvad(f, 0.1, 0.35)  # n=7 but only 5 flags after start -> 2 padded
    assert m.tolist() == [1, 1, 0, -1, 1, -1, -1] and info["missing_frames"] == 3, (m.tolist(), info)
    assert info["speech_frames"] == 3
    # t0 always lands in (-frame_s, 0]
    for lead in (0.0, 0.001, 0.049, 0.05, 1.2345, 2.999):
        _, i = cr._aicvad(np.ones(400, np.int8), lead, 10.0)
        assert -0.05 < i["t0_s"] <= 1e-9, (lead, i)
        assert abs(i["t0_s"] + lead - round((lead + i["t0_s"]) / 0.05) * 0.05) < 1e-3


def check_timed_wrapper():
    try:
        from livekit import rtc  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"SKIP _timed: livekit.rtc not importable ({e})")
        return

    class Inner:
        def __init__(self):
            self.enabled = True
            self.log = []

        def _process(self, frame):
            end = time.thread_time() + 0.01
            while time.thread_time() < end:
                pass
            if frame == "boom":
                raise ValueError("inner failed")
            return ("processed", frame)

        def _on_stream_info_updated(self, **kw):
            self.log.append(("info", kw))

        def _on_credentials_updated(self, **kw):
            self.log.append(("creds", kw))

        def _close(self):
            self.log.append("closed")

    inner = Inner()
    t = cr._timed(inner)
    assert t.enabled is True
    t.enabled = False
    assert inner.enabled is False and t.enabled is False
    assert t._process("f") == ("processed", "f")
    first = t.cpu_s
    assert first >= 0.008, f"cpu not billed: {first}"
    raises(ValueError, t._process, "boom", contains="inner failed")
    assert t.cpu_s >= first + 0.008, "a failing frame is still billed"
    t._on_stream_info_updated(room_name="r", participant_identity="p")
    t._on_credentials_updated(token="t", url="u")
    t._on_stream_info_cleared()  # inner has no *_cleared: must not raise
    t._on_credentials_cleared()
    t._close()
    assert inner.log == [("info", {"room_name": "r", "participant_identity": "p"}),
                         ("creds", {"token": "t", "url": "u"}), "closed"], inner.log
    assert cr._timed(Inner()).cpu_s == 0.0


# ======================================================= cloud_replay: _finish_batch


def check_finish_batch_gates(env: Env):
    env.attr(lk_cloud, "plugin_versions", lambda: {"livekit-plugins-ai-coustics": "0.3.2"})
    f = shared_fix()
    ctl = f.pcm(0.8)
    f.save("0", ctl)
    half = f.pcm(0.5)
    fl_half = f.speech_flags(half)
    fl_half[100:110] = -1                                        # 10 frames the plugin left unflagged
    f.save("1", half, fl_half)                                   # good AIC filter
    f.save("2", ctl)                                             # Krisp that never ran
    f.save("3", ctl, f.speech_flags(ctl))                        # bypass, honoured
    f.save("4", half, fl_half)                                   # bypass NOT honoured
    f.save("5", half)                                            # AIC, flags all -1
    other = soxr.resample(speechy(f.dur, 48_000, 99), 48_000, 24_000).astype(np.float32)
    f.save("6", f.pcm(0.5, x24=other))                           # unrelated audio
    f.save("7", half[: int(3 * 24_000 + 0.9 * len(f.x24))])      # truncated
    f.save("8", np.zeros(0, np.int16), np.zeros(0, np.int8))     # nothing received
    fl_short = f.speech_flags(ctl)[:200]                         # flags stop 10 s in
    f.save("9", ctl, fl_short)                                   # enhancement_level 0
    streams = [
        CONTROL,
        stream("1", "c-ok", "AIC:QUAIL_L"),
        stream("2", "c-same", "KRISP:VIVA"),
        stream("3", "c-bypass", "AIC:ROOK_S", {"bypass": 1.0}),
        stream("4", "c-bypass-enh", "AIC:QUAIL_VF_L", {"bypass": 1.0}),
        stream("5", "c-noflags", "AIC:QUAIL_L"),
        stream("6", "c-garbage", "BVC"),
        stream("7", "c-short", "KRISP:VIVA_TEL"),
        stream("8", "c-empty", "BVCTelephony"),
        stream("9", "c-level0", "AIC:QUAIL_L", {"enhancement_level": 0.0}),
    ]
    res = f.finish(streams)
    assert set(res) == {s["id"] for s in streams[1:]} | {"cloud-raw"}, sorted(res)

    ok = {"c-ok", "c-bypass", "c-level0", "cloud-raw"}
    for cid in ok:
        assert "error" not in res[cid], (cid, res[cid])
        wav = f.out / res[cid]["output"]
        assert res[cid]["output"] == f"{cid}.wav" and wav.exists()
        info = sf.info(wav)
        assert info.samplerate == config.PIPELINE_RATE == 16_000 and info.channels == 1
        assert info.subtype == "PCM_16" and info.frames == len(f.x16), (info.frames, len(f.x16))
        assert res[cid]["duration_s"] == round(f.dur, 2)
        assert res[cid]["plugins"] == {"livekit-plugins-ai-coustics": "0.3.2"}
        rec = res[cid]["cloud_replay"]
        assert rec["rail"] == "cloud_replay" and rec["bitrate_kbps"] == 256 == cr.BITRATE_KBPS
        assert rec["red"] is False and rec["dtx"] is False
        assert (rec["room"], rec["batch"], rec["batches"], rec["streams_in_batch"]) == ("r", 1, 1, 10)
        assert abs(rec["lead_in_s"] - 3.0) < 0.005, rec["lead_in_s"]
        assert rec["env_corr"] > 0.9 and rec["env_prominence"] > cr.MIN_PROMINENCE, rec
        assert rec["publisher_starved_ms"] == 0.0 and rec["subscriber_max_backlog_s"] == 0.1
        assert rec["control_lag_drift_ms"] < cr.MAX_DRIFT_MS
        assert rec["short_s"] == 0.0 and rec["input_s"] == round(f.dur, 3)
    # per-stream numbers come from that stream's own summary
    assert res["c-ok"]["cloud_replay"]["cpu_per_audio_s"] == 0.02
    assert res["c-ok"]["cloud_replay"]["frames"] == 400
    assert -12 < res["c-ok"]["cloud_replay"]["residual_vs_control_db"] < -5
    assert res["c-ok"]["cloud_replay"]["level_db_vs_input"] < res["c-bypass"]["cloud_replay"]["level_db_vs_input"]
    assert res["c-bypass"]["cloud_replay"]["residual_vs_control_db"] < cr.PASSTHROUGH_DB
    assert "residual_vs_control_db" not in res["cloud-raw"]["cloud_replay"]

    errs = {
        "c-same": "identical to the unfiltered control",
        "c-bypass-enh": "was not honoured",
        "c-noflags": "no lk.aic-vad flag",
        "c-garbage": "does not line up",
        "c-short": "truncated",
        "c-empty": "no audio received",
    }
    for cid, want in errs.items():
        assert want in res[cid].get("error", ""), (cid, res[cid])
        assert not (f.out / f"{cid}.wav").exists(), f"{cid} failed but left a wav"
        assert not (f.out / f"{cid}.aicvad.npy").exists()
    assert res["c-short"]["cloud_replay"]["short_s"] > cr.MAX_SHORT_S
    assert "cloud_replay" in res["c-same"] and "cloud_replay" not in res["c-empty"]

    # the ai-coustics flag: bool, on the input's own timeline, only for AIC streams
    for cid, fl in (("c-ok", fl_half), ("c-bypass", f.speech_flags(ctl)), ("c-level0", fl_short)):
        av = res[cid]["aicvad"]
        mask = np.load(f.out / f"{cid}.aicvad.npy")
        assert av["file"] == f"{cid}.aicvad.npy" and mask.dtype == np.int8 and mask.ndim == 1
        assert av["dtype"] == "int8" and av["encoding"] == lk_cloud.AICVAD_ENCODING
        assert len(mask) == av["frames"] and av["frame_s"] == 0.05
        assert -0.05 < av["t0_s"] <= 0 and av["speech_frames"] == int((mask == 1).sum()) > 10
        assert av["missing_frames"] == int((mask == -1).sum())
        assert res[cid]["aicvad_missing_frames"] == av["missing_frames"], "top-level count must match"
        assert "0.3.2" in av["source"] and "lk.aic-vad" in av["source"]
        start = math.floor(res[cid]["cloud_replay"]["lead_in_s"] / 0.05 + 1e-9)
        want = fl[start : start + len(mask)]
        want = np.concatenate([want, -np.ones(len(mask) - len(want), np.int8)])  # past the stream's end
        assert (mask == want).all(), f"{cid} flags misaligned"
    # -1 frames round-trip as -1 and are counted, never turned into 0
    assert res["c-ok"]["aicvad"]["missing_frames"] == 10
    assert res["c-bypass"]["aicvad"]["missing_frames"] == 0 and res["c-bypass"]["aicvad_missing_frames"] == 0
    assert res["c-level0"]["aicvad"]["missing_frames"] > 100 and \
        (np.load(f.out / "c-level0.aicvad.npy")[-50:] == -1).all(), "frames past the flags' end are -1"
    for cid in ("c-same", "c-garbage"):
        assert "aicvad" not in res[cid]
    assert "aicvad" not in res["cloud-raw"] and not (f.out / "cloud-raw.aicvad.npy").exists()
    assert not (f.out / "c-same.aicvad.npy").exists()
    # only the four winners left files behind (+3 masks)
    assert sorted(p.name for p in f.out.iterdir()) == sorted(
        [f"{c}.wav" for c in ok] + [f"{c}.aicvad.npy" for c in ("c-ok", "c-bypass", "c-level0")])


def _pair(f: Fix, ctl_pcm, cand_gain=0.5, flagged=True):
    f.save("0", ctl_pcm)
    pcm = f.pcm(cand_gain)
    f.save("1", pcm, f.speech_flags(pcm) if flagged else None)
    return [CONTROL, stream("1", "c", "AIC:QUAIL_L")]


def check_finish_batch_starved(env: Env):
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = Fix(dur=12.0, seed=11)
    streams = _pair(f, f.pcm(0.8))
    res = f.finish(streams, starved=50.0)
    assert set(res) == {"c", "cloud-raw"}
    for cid in res:
        assert "publisher stalled" in res[cid]["error"] and "batch 1/1" in res[cid]["error"], res[cid]
        rec = res[cid]["cloud_replay"]
        assert rec["publisher_starved_ms"] == 50.0 and rec["env_corr"] > 0.9, "control record must ride along"
    assert list(f.out.iterdir()) == [], "a failed batch wrote outputs"
    # the gate is "more than one 10 ms frame": exactly 10 ms passes
    res = f.finish(streams, starved=cr.MAX_STARVED_MS)
    assert "error" not in res["c"] and "error" not in res["cloud-raw"], res
    # a control with no audio at all is not a gate, it is an exception
    f.save("0", np.zeros(0, np.int16), np.zeros(0, np.int8))
    raises(RuntimeError, f.finish, streams, contains="control stream received no audio")


def check_finish_batch_drift(env: Env):
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = Fix(dur=24.0, seed=12)
    # jitter-buffer steps in the control: +50 ms at 5 s and again at 17 s, so the
    # 12 s middle (the alignment) sits between windows at -50 and +50 ms
    base = f.pcm(0.8)
    a, b, c = 3 * 24_000 + 5 * 24_000, 3 * 24_000 + 17 * 24_000, None
    gap = np.zeros(int(0.05 * 24_000), np.int16)
    stepped = np.concatenate([base[:a], gap, base[a:b], gap, base[b:]])
    streams = _pair(f, stepped)
    res = f.finish(streams)
    rec = res["c"]["cloud_replay"]
    assert rec["control_lag_drift_ms"] > cr.MAX_DRIFT_MS, rec["control_lag_drift_ms"]
    for cid in ("c", "cloud-raw"):
        assert "jitter buffer" in res[cid]["error"], res[cid]
    assert list(f.out.iterdir()) == []
    # a single 50 ms step is under the gate: still a result, with the drift reported
    one = np.concatenate([base[:a], gap, base[a:]])
    res = f.finish(_pair(f, one))
    assert "error" not in res["c"], res["c"]
    assert 30 <= res["c"]["cloud_replay"]["control_lag_drift_ms"] <= cr.MAX_DRIFT_MS


def check_finish_batch_control_env_corr(env: Env):
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = Fix(dur=16.0, seed=13)
    noise = 0.12 * np.random.default_rng(4).standard_normal(len(f.pcm())).astype(np.float32)
    mangled = np.clip(f.pcm(0.3).astype(np.float32) + noise * 32768, -32768, 32767).astype(np.int16)
    res = f.finish(_pair(f, mangled))
    rec = res["c"]["cloud_replay"]
    assert rec["env_corr"] < cr.CONTROL_MIN_ENV_CORR, rec["env_corr"]
    assert rec["control_lag_drift_ms"] <= cr.MAX_DRIFT_MS, "this case must reach the env_corr gate"
    for cid in ("c", "cloud-raw"):
        assert "only correlates" in res[cid]["error"], res[cid]


def check_finish_batch_env_corr_gate(env: Env):
    """The env_corr floor on its own: the prominence gate is switched off, so what
    is left is 'output is silence or garbage'."""
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    env.attr(cr, "MIN_PROMINENCE", -10.0)
    f = Fix(dur=12.0, seed=14)
    f.save("0", f.pcm(0.8))
    flat = (0.002 * np.random.default_rng(6).standard_normal(len(f.pcm()))).astype(np.float32)
    f.save("1", np.clip(flat * 32768, -32768, 32767).astype(np.int16))
    res = f.finish([CONTROL, stream("1", "c", "KRISP:VIVA")], raw_ids=())
    assert "barely follows the input" in res["c"]["error"], res["c"]
    assert res["c"]["cloud_replay"]["env_corr"] < cr.MIN_ENV_CORR


def check_finish_batch_cloud_raw_gate(env: Env):
    """cloud-raw is gated like any stream: a control that does not line up is an error."""
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    env.attr(cr, "CONTROL_MIN_ENV_CORR", -10.0)   # get past the batch-level gates
    env.attr(cr, "MAX_DRIFT_MS", 1e9)
    f = Fix(dur=12.0, seed=15)
    other = soxr.resample(speechy(f.dur, 48_000, 77), 48_000, 24_000).astype(np.float32)
    f.save("0", f.pcm(0.8, x24=other))
    f.save("1", f.pcm(0.5))
    res = f.finish([CONTROL, stream("1", "c", "KRISP:VIVA")])
    assert res["cloud-raw"]["error"].startswith("cloud-raw: "), res["cloud-raw"]
    assert not (f.out / "cloud-raw.wav").exists()


# ========================================================== cloud_replay: replay()


def check_replay_orchestration(env: Env):
    env.attr(cr, "plan_batches", lambda c: _orig_plan(c, 3))  # 2 candidates per batch
    env.env("CLOUD_REPLAY_KEEP_WORK", None)
    d = tmpdir()
    inp, out = d / "input.wav", d / "out"
    out.mkdir()
    sf.write(inp, np.zeros(1600, np.float32), 16_000)
    cands = [{"id": "cloud-raw", "lk_model": lk_cloud.RAW},
             {"id": "a", "lk_model": "A"}, {"id": "b", "lk_model": "A", "lk_options": {"bypass": 1.0}},
             {"id": "c", "lk_model": "B"}, {"id": "d", "lk_model": "C"}]
    run_calls, finish_calls, works = [], [], []
    prog = object()

    async def fake_run(input_wav, streams, work, k, n, progress, ids):
        assert work.is_dir(), "replay() must hand over an existing work dir"
        works.append(work)
        run_calls.append({"k": k, "n": n, "streams": streams, "ids": ids, "progress": progress})
        if k == 1:
            raise RuntimeError("room went away")
        return {"batch": k + 1, "batches": n}

    def fake_finish(input_wav, streams, work, out_dir, meta, raw_ids):
        finish_calls.append({"streams": streams, "raw_ids": raw_ids, "out": out_dir, "meta": meta})
        return {s["id"]: {"output": f"{s['id']}.wav"} for s in streams[1:] if s["id"]} | {r: {"output": f"{r}.wav"} for r in raw_ids}

    env.attr(cr, "_run_batch", fake_run)
    env.attr(cr, "_finish_batch", fake_finish)
    res = asyncio.run(cr.replay(inp, cands, out, prog))

    assert len(run_calls) == 2, "the failed batch must not stop the next; and no extra batches"
    b0, b1 = run_calls
    assert [s["name"] for s in b0["streams"]] == ["0", "1", "1t", "2", "2t"], \
        "control, then each candidate followed by its raw twin"
    assert b0["streams"][0]["lk_model"] == "RAW" and b0["streams"][0]["id"] is None
    assert [s["id"] for s in b0["streams"][1:]] == ["a", None, "b", None]
    for tw, of in ((b0["streams"][2], "1"), (b0["streams"][4], "2")):
        assert tw["lk_model"] == "RAW" and tw["twin_of"] == of and tw["id"] is None, tw
    assert "twin_of" not in b0["streams"][1] and "twin_of" not in b0["streams"][3]
    assert b0["streams"][3]["lk_options"] == {"bypass": 1.0} and b0["streams"][1]["lk_options"] == {}
    assert b0["ids"] == ["a", "b", "cloud-raw"], "raw ids ride on batch 1 only"
    assert b1["ids"] == ["c", "d"] and b0["progress"] is prog and (b0["k"], b0["n"]) == (0, 2)
    assert len(finish_calls) == 1 and finish_calls[0]["raw_ids"] == ["cloud-raw"]
    assert finish_calls[0]["out"] == out
    assert res["a"] == {"output": "a.wav"} and res["cloud-raw"] == {"output": "cloud-raw.wav"}
    for cid in ("c", "d"):
        assert "batch 2/2" in res[cid]["error"] and "room went away" in res[cid]["error"], res[cid]
    assert set(res) == {"a", "b", "c", "d", "cloud-raw"}
    assert all(not w.exists() for w in works), "temp work dirs must be removed, failures included"

    # a failure in the aligning step is also per-candidate, and cleans up
    def bad_finish(*a):
        raise RuntimeError("control stream received no audio")

    run_calls.clear()
    works.clear()
    env.attr(cr, "_finish_batch", bad_finish)

    async def fine_run(input_wav, streams, work, k, n, progress, ids):
        works.append(work)
        return {}

    env.attr(cr, "_run_batch", fine_run)
    res = asyncio.run(cr.replay(inp, cands, out, None))
    assert all("no audio" in r["error"] for r in res.values()) and len(res) == 5
    assert all(not w.exists() for w in works)

    # only a RAW candidate: one batch holding just the control, served from it
    finish_calls.clear()
    env.attr(cr, "_finish_batch", fake_finish)
    res = asyncio.run(cr.replay(inp, [{"id": "cloud-raw", "lk_model": lk_cloud.RAW}], out, None))
    assert res == {"cloud-raw": {"output": "cloud-raw.wav"}}
    assert [s["name"] for s in finish_calls[0]["streams"]] == ["0"]

    # KEEP_WORK keeps the raw streams for debugging
    env.env("CLOUD_REPLAY_KEEP_WORK", "1")
    works.clear()
    asyncio.run(cr.replay(inp, [{"id": "x", "lk_model": "A"}], out, None))
    assert works and all(w.exists() for w in works)
    for w in works:
        shutil.rmtree(w, ignore_errors=True)


_orig_plan = cr.plan_batches


def check_replay_end_to_end(env: Env):
    """replay() with only the network step stubbed: the real _finish_batch aligns
    what the fake subscriber 'received'."""
    env.attr(lk_cloud, "plugin_versions", lambda: {"livekit-plugins-ai-coustics": "0.3.2"})
    env.env("CLOUD_REPLAY_KEEP_WORK", None)
    f = Fix(dur=12.0, seed=21)
    seen = {}

    async def fake_run(input_wav, streams, work, k, n, progress, ids):
        seen["work"] = work
        f.work = work
        f.save("0", f.pcm(0.8))
        pcm = f.pcm(0.5)
        f.save("1", pcm, f.speech_flags(pcm))
        assert [x["name"] for x in streams] == ["0", "1", "1t"] and streams[2]["twin_of"] == "1"
        f.save("1t", f.pcm(0.8))    # the raw twin: same length and start as its candidate
        if progress is not None:
            await progress({"batch": k + 1, "batches": n, "t": 1.0, "total": 2.0, "ids": ids})
        return f.meta(["0", "1", "1t"], batch=k + 1, batches=n)

    env.attr(cr, "_run_batch", fake_run)
    events = []

    async def progress(ev):
        events.append(ev)

    res = asyncio.run(cr.replay(
        f.input, [{"id": "cloud-raw", "lk_model": "RAW"}, {"id": "lkaic-q", "lk_model": "AIC:QUAIL_L"}],
        f.out, progress))
    assert events and events[0]["ids"] == ["lkaic-q", "cloud-raw"]
    assert set(res) == {"cloud-raw", "lkaic-q"} and all("error" not in r for r in res.values()), res
    assert (f.out / "cloud-raw.wav").exists() and (f.out / "lkaic-q.wav").exists()
    assert (f.out / "lkaic-q.aicvad.npy").exists() and not (f.out / "cloud-raw.aicvad.npy").exists()
    assert res["lkaic-q"]["aicvad"]["file"] == "lkaic-q.aicvad.npy"
    rec = res["lkaic-q"]["cloud_replay"]
    assert rec["twin_len_diff_ms"] == 0.0 and rec["onset_delay_ms"] == 0.0, rec
    assert "1t" not in [k for k in res] and set(res) == {"cloud-raw", "lkaic-q"}, "twins are not candidates"
    assert not seen["work"].exists()


# ================================================================ server wiring


def _server():
    from nc_bench import server

    return server


def check_server_load_candidates(env: Env):
    server = _server()
    d = tmpdir()
    entries = [
        {"id": "off", "label": "o", "chain": [{"proc": "dtln"}]},
        {"id": "cl", "label": "c", "lk_model": "BVC"},
        {"id": "cl-bad", "label": "cb", "lk_model": "BAD"},
        {"id": "hist", "label": "h", "lk_model": "AIC:QUAIL_L", "historical": True},
        {"id": "hist-off", "label": "ho", "chain": [], "historical": True, "live_only": True},
        {"id": "lo", "label": "lo", "lk_model": "BVC", "live_only": True},
    ]
    (d / "c.json").write_text(json.dumps(entries))
    env.attr(config, "CANDIDATES_FILE", d / "c.json")
    env.attr(lk_cloud, "available",
             lambda model="", options=None: (False, "not enabled") if model == "BAD" else (True, ""))
    env.attr(server, "chain_available", lambda chain: (True, ""))
    got = {c["id"]: c for c in server._load_candidates()}
    assert got["off"]["rail"] == "offline" and got["off"]["available"] is True
    assert got["cl"]["rail"] == "cloud" and got["cl"]["available"] is True
    assert got["cl-bad"]["available"] is False and got["cl-bad"]["unavailable_reason"] == "not enabled"
    assert got["hist"]["available"] is False and "historical" in got["hist"]["unavailable_reason"]
    assert got["hist-off"]["available"] is False, "historical wins over an available chain"
    assert all(c["live_only"] is False for c in got.values()), "live_only must always be False"
    assert server._candidates_by_id().keys() == got.keys()

    # the real file, with the availability probes stubbed out
    env.attr(config, "CANDIDATES_FILE", ROOT / "candidates.json")
    real = server._load_candidates()
    assert all(c["rail"] == ("cloud" if "lk_model" in c else "offline") for c in real)
    hist = [c for c in real if c.get("historical")]
    assert {c["id"] for c in hist} == {"aic-quail-l", "aic-vf-s", "aic-vf-l"}
    assert all(not c["available"] for c in hist) and all(not c["live_only"] for c in real)


def check_server_check_runnable():
    server = _server()
    HTTPException = server.HTTPException
    known = {
        "off": {"rail": "offline", "available": False, "unavailable_reason": "no model"},
        "cl": {"rail": "cloud", "available": True, "unavailable_reason": ""},
        "cl-bad": {"rail": "cloud", "available": False, "unavailable_reason": "no Cloud"},
        "hist": {"rail": "cloud", "available": False, "historical": True},
    }
    server._check_runnable(["off", "cl"], known)  # unavailable offline chains error per candidate
    server._check_runnable([], known)
    for ids, want in ((["cl", "hist"], "hist"), (["cl-bad"], "no Cloud"), (["cl-bad", "cl"], "cl-bad")):
        e = raises(HTTPException, server._check_runnable, ids, known, contains=want)
        assert e.status_code == 400, e.status_code
    e = raises(HTTPException, server._check_runnable, ["cl-bad", "hist"], known)
    assert "historical" in e.detail, "historical is reported first"


class _Store:
    """The run's meta.json on disk, seen the way a test wants to: `saved` is what the
    REAL store (nc_bench/store.py, locked_meta and all) has committed so far."""

    def __init__(self, run_dir: Path, meta: dict | None = None):
        self.rd = run_dir
        (run_dir / "meta.json").write_text(json.dumps(
            {"id": run_dir.name, "source": "upload", **(meta or {})}, indent=2))

    @property
    def saved(self) -> dict:
        return json.loads((self.rd / "meta.json").read_text())

    def edit(self, fn) -> None:
        """Change the stored meta the way another writer (a note, a re-transcription,
        the VAD bench CLI) would: under the run's lock."""
        from nc_bench import store

        with store.locked_meta(self.rd.name) as m:
            fn(m)


GOOD_INPUT = {"input_scores": {"dnsmos": {"sig": 3.0}}, "input_vad": {"segments": []},
              "vad_params": {"kept": True}}


def _server_stubs(env: Env, run_dir: Path, known: dict, meta: dict | None = None):
    """Everything heavy in `_process_run` stubbed. `run_chain` writes a small
    distinct wav (the server now checks that every staged file exists); the
    callables the checks care about are recorded on `calls`."""
    server = _server()
    ev: list = []
    bcast: list = []
    env.attr(config, "RUNS_DIR", run_dir.parent)   # the real store, on a tmp dir
    st = _Store(run_dir, meta)
    calls = types.SimpleNamespace(score_input=0, analyze=[], score_run=[], chain=[], stt=[])

    async def broadcast(msg):
        bcast.append(msg)

    async def transcribe(wav):
        calls.stt.append(Path(wav))
        return {"text": "hello"}

    async def analyze(wav, p=None, source=None):
        calls.analyze.append(Path(wav))
        return {"segments": []}

    def score_input(p):
        calls.score_input += 1
        return {"dnsmos": {"sig": 3.0}}

    def run_chain(input_wav, chain, out_wav, measure_rss=False):
        ev.append(f"chain:{Path(out_wav).stem}")
        calls.chain.append(Path(out_wav))
        write_wav(out_wav, seed=100 + len(calls.chain))  # NEW audio: never equals a stored old file
        return {"proc_ms": 1.0}

    def score_run(shim):
        ev.append("score_run")
        calls.score_run.append(shim)

    env.attr(server, "broadcast", broadcast)
    env.attr(server, "run_chain", run_chain)
    env.attr(server, "_candidates_by_id", lambda: known)
    env.attr(server.scoring, "score_input", score_input)
    env.attr(server.scoring, "score_output", lambda out, ins, script, text: {"ok": True, "text": text})
    env.attr(server.stt, "transcribe", transcribe)
    env.attr(server.vad, "analyze", analyze)
    env.attr(server.vad, "defaults", lambda source=None: {"stub": True})
    env.attr(server.vad, "score_run", score_run)
    return server, st, ev, bcast, calls


def write_wav(path: Path, seed: int) -> None:
    sf.write(path, (0.1 * np.random.default_rng(seed).standard_normal(1600)).astype(np.float32),
             16_000, subtype="PCM_16")


def snapshot(d: Path) -> dict:
    """name -> bytes of every plain file directly in d, meta.json and the store's
    lock/temp files aside (dot-dirs and dirs excluded)."""
    return {p.name: p.read_bytes() for p in d.iterdir()
            if p.is_file() and p.name != "meta.json" and not p.name.startswith(".meta.")}


def no_attempt_dirs(d: Path) -> bool:
    """No .attempt-* dir and no store temp file left behind (.meta.lock is expected)."""
    return not [p for p in d.iterdir()
                if p.name.startswith(".attempt-") or (p.name.startswith(".meta.") and p.name.endswith(".tmp"))]


def good_record(rd: Path, cid: str, seed: int, *, seg=False, flags=False, cut=False, **extra) -> dict:
    """A stored, good candidate record and the files it points at, written into rd."""
    write_wav(rd / f"{cid}.wav", seed)
    rec = {"id": cid, "label": f"{cid} é", "status": "done", "output": f"{cid}.wav",
           "stt": {"text": f"old text {seed}"}, "scores": {"dnsmos": {"sig": 1.0 + seed / 100}},
           "vad": {"n": seed}, "processed_at": "2026-08-04T11:25:11", **extra}
    if seg:
        write_wav(rd / f"{cid}.seg.wav", seed + 1000)
        rec["output_seg"] = f"{cid}.seg.wav"
    if flags:
        np.save(rd / f"{cid}.aicvad.npy", np.array([True, False] * seed, bool))
        rec["aicvad"] = {"file": f"{cid}.aicvad.npy", "frame_s": 0.05, "frames": 2 * seed}
    if cut:
        write_wav(rd / f"{cid}.vadcut.wav", seed + 2000)
    return rec


def _known():
    def cloud(cid, model, opts=None):
        c = {"id": cid, "label": cid.upper(), "rail": "cloud", "lk_model": model,
             "available": True, "unavailable_reason": "", "live_only": False}
        if opts:
            c["lk_options"] = opts
        return c

    def offline(cid):
        return {"id": cid, "label": cid.upper(), "rail": "offline", "chain": [{"proc": "specnc"}],
                "available": True, "unavailable_reason": ""}

    return {
        "gtcrn": offline("gtcrn"), "dtln": offline("dtln"), "fastenhancer-t": offline("fastenhancer-t"),
        "hush": offline("hush"),
        "krisp-viva": cloud("krisp-viva", "KRISP:VIVA"),
        "lkaic-quail-l": cloud("lkaic-quail-l", "AIC:QUAIL_L", {"bypass": 1.0}),
        "lkaic-vf22-l": cloud("lkaic-vf22-l", "AIC:QUAIL_VF_L"),
        "cloud-raw": cloud("cloud-raw", "RAW"),
    }


def _replay_writer(seen: dict, results_for):
    """A stand-in for cloud_replay.replay that, like the real one, writes its outputs
    into the out_dir it is given. results_for(out_dir) -> the dict to return."""
    async def replay(input_wav, cands, out_dir, progress):
        seen["args"] = (input_wav, cands, out_dir)
        return results_for(Path(out_dir))

    return replay


def check_server_process_run_replay(env: Env):
    d = rundir()
    server, st, ev, bcast, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT))
    seen = {}

    async def replay(input_wav, cands, out_dir, progress):
        ev.append("replay:start")
        seen["args"] = (input_wav, cands, out_dir)
        seen["dir_existed"] = Path(out_dir).is_dir()
        await progress({"batch": 1, "batches": 1, "t": 2.0, "total": 4.0, "ids": ["krisp-viva"]})
        await asyncio.sleep(0.05)
        out = Path(out_dir)
        write_wav(out / "krisp-viva.wav", 1)
        write_wav(out / "lkaic-vf22-l.wav", 2)
        np.save(out / "lkaic-vf22-l.aicvad.npy", np.array([1, 0], np.int8))  # the replay's own flags
        ev.append("replay:end")
        return {
            "krisp-viva": {"output": "krisp-viva.wav", "duration_s": 1.0, "plugins": {"p": "1"},
                           "cloud_replay": {"lead_in_s": 3.0, "rail": "cloud_replay"}},
            "lkaic-quail-l": {"error": "was not honoured", "cloud_replay": {"env_corr": 0.9}},
            "lkaic-vf22-l": {"output": "lkaic-vf22-l.wav", "duration_s": 1.0,
                             "aicvad": {"file": "lkaic-vf22-l.aicvad.npy"}, "aicvad_missing_frames": 3},
            # cloud-raw is missing from the result on purpose
        }

    env.attr(server.cloud_replay, "replay", replay)
    ids = ["gtcrn", "krisp-viva", "lkaic-quail-l", "lkaic-vf22-l", "cloud-raw"]
    asyncio.run(server._process_run("r1", ids, None, "the script", 2))

    assert ev.count("replay:start") == 1, "the replay must run exactly once"
    first_chain = min(i for i, e in enumerate(ev) if e.startswith("chain:"))
    assert ev.index("replay:end") < first_chain, f"a chain started before the replay ended: {ev}"
    assert [e for e in ev if e.startswith("chain:")] == ["chain:gtcrn"], \
        "cloud candidates must not go through run_chain"
    input_wav, cands, out_dir = seen["args"]
    assert input_wav == d / "input.wav" and seen["dir_existed"]
    assert out_dir.parent.name.startswith(".attempt-") and out_dir.name == "replay" and out_dir.parent.parent == d, \
        f"replay must write into the attempt dir, got {out_dir}"
    assert cands == [{"id": "krisp-viva", "lk_model": "KRISP:VIVA", "lk_options": {}},
                     {"id": "lkaic-quail-l", "lk_model": "AIC:QUAIL_L", "lk_options": {"bypass": 1.0}},
                     {"id": "lkaic-vf22-l", "lk_model": "AIC:QUAIL_VF_L", "lk_options": {}},
                     {"id": "cloud-raw", "lk_model": "RAW", "lk_options": {}}], cands
    assert no_attempt_dirs(d) and not out_dir.exists(), "the attempt dir must be removed"
    # promoted files are in run_dir, byte for byte what was staged; failures left none
    files = snapshot(d)
    assert {"krisp-viva.wav", "lkaic-vf22-l.wav", "lkaic-vf22-l.aicvad.npy", "gtcrn.wav"} <= set(files)
    assert not {"lkaic-quail-l.wav", "cloud-raw.wav", "lkaic-quail-l.aicvad.npy"} & set(files)
    assert np.load(d / "lkaic-vf22-l.aicvad.npy").tolist() == [1, 0]
    assert sf.info(d / "krisp-viva.wav").frames == 1600

    meta = st.saved
    assert [c["id"] for c in meta["candidates"]] == ids, "ticked order"
    by = {c["id"]: c for c in meta["candidates"]}
    assert meta["status"] == "done" and meta["script"] == "the script" and meta["concurrency"] == 2
    assert by["gtcrn"]["status"] == "done" and by["gtcrn"]["chain_spec"] == [{"proc": "specnc"}]
    assert "rail" not in by["gtcrn"] and by["gtcrn"]["output"] == "gtcrn.wav"
    assert by["gtcrn"]["concurrency"] == 2 and by["gtcrn"]["processed_at"]
    k = by["krisp-viva"]
    assert k["rail"] == "cloud_replay" and k["status"] == "done"
    assert k["chain"] == ["livekit-cloud:KRISP:VIVA (cloud replay)"] and "lk_options" not in k
    assert k["output"] == "krisp-viva.wav" and k["cloud_replay"]["lead_in_s"] == 3.0
    assert k["plugins"] == {"p": "1"} and k["stt"] == {"text": "hello"} and k["scores"]["text"] == "hello"
    assert k["vad"] == {"segments": []} and k["processed_at"] and k["concurrency"] == 2
    q = by["lkaic-quail-l"]
    assert q["rail"] == "cloud_replay" and q["status"] == "error" and q["error"] == "was not honoured"
    assert q["lk_options"] == {"bypass": 1.0} and q["cloud_replay"] == {"env_corr": 0.9}
    assert "output" not in q and "stt" not in q, "an errored candidate is not scored"
    assert "last_attempt" not in q, "no good record before, so the error IS the record"
    assert by["lkaic-vf22-l"]["aicvad"] == {"file": "lkaic-vf22-l.aicvad.npy"}
    assert by["lkaic-vf22-l"]["aicvad_missing_frames"] == 3, "the replay's top-level count must reach the record"
    r = by["cloud-raw"]
    assert r["status"] == "error" and "returned nothing" in r["error"] and r["rail"] == "cloud_replay"
    assert bcast[-1] == {"type": "run_done", "run_id": "r1"}
    assert any(m.get("stage") == "cloud_replay" and m.get("detail") == "starting" for m in bcast)
    assert any(m.get("detail") == "batch 1/1 · 2/4 s" and m["candidate"] == "krisp-viva" for m in bcast)
    # STT / VAD read the STAGED file, and the input analysis was reused, not redone
    assert all(".attempt-" in str(p) for p in calls.stt), calls.stt
    assert calls.score_input == 0 and d / "input.wav" not in calls.analyze
    assert meta["input_scores"] == GOOD_INPUT["input_scores"] and meta["vad_params"] == {"kept": True}
    assert len(calls.score_run) == 1
    assert sorted(c["id"] for c in calls.score_run[0]["candidates"]) == sorted(
        ["gtcrn", "krisp-viva", "lkaic-quail-l", "lkaic-vf22-l", "cloud-raw"]), "score_run gets the fresh records"


def check_server_process_run_replay_crash(env: Env):
    d = rundir()
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT))

    async def replay(*a, **k):
        raise RuntimeError("livekit exploded")

    env.attr(server.cloud_replay, "replay", replay)
    asyncio.run(server._process_run("r1", ["gtcrn", "krisp-viva", "lkaic-vf22-l"], None, "", 1))
    by = {c["id"]: c for c in st.saved["candidates"]}
    assert by["gtcrn"]["status"] == "done"
    for cid in ("krisp-viva", "lkaic-vf22-l"):
        assert by[cid]["status"] == "error" and "cloud replay: livekit exploded" in by[cid]["error"], by[cid]
        assert not (d / f"{cid}.wav").exists()
    assert st.saved["status"] == "done" and no_attempt_dirs(d)


def check_server_process_run_no_cloud(env: Env):
    d = rundir()
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT))

    async def replay(*a, **k):
        raise AssertionError("replay must not run without cloud candidates")

    env.attr(server.cloud_replay, "replay", replay)
    asyncio.run(server._process_run("r1", ["gtcrn", "gtcrn", "dtln"], None, "", 2))
    cs = st.saved["candidates"]
    assert [c["id"] for c in cs] == ["gtcrn", "dtln"], "duplicate ids must collapse"
    assert all(c["status"] == "done" for c in cs) and len(calls.chain) == 2
    assert (d / "gtcrn.wav").is_file() and (d / "dtln.wav").is_file() and no_attempt_dirs(d)
    assert all(".attempt-" in str(p) for p in calls.chain), "chains must write into the attempt dir"


def check_server_process_run_live(env: Env):
    d = rundir()
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT))
    stale = d / "krisp-viva.aicvad.npy"
    np.save(stale, np.ones(9, bool))   # derived from an earlier attempt of this candidate

    async def replay(*a, **k):
        raise AssertionError("live results must bypass the replay rail")

    env.attr(server.cloud_replay, "replay", replay)
    audio = (np.random.default_rng(0).standard_normal(int(server.RECORD_RATE * 0.3)) * 3000).astype(np.int16)
    live = {
        "lkaic-quail-l": {"audio": audio, "vad_flags": np.array([1, 0, 1, -1], np.int8), "frame_s": 0.05},
        "krisp-viva": {"audio": audio},
        "lkaic-vf22-l": {"error": "the stream died"},
        # cloud-raw: nothing recorded
    }
    asyncio.run(server._process_run("r1", ["krisp-viva", "lkaic-quail-l", "lkaic-vf22-l", "cloud-raw"],
                                    live, "", 2))
    by = {c["id"]: c for c in st.saved["candidates"]}
    q = by["lkaic-quail-l"]
    assert q["rail"] == "live" and q["status"] == "done" and q["chain"] == ["livekit-cloud:AIC:QUAIL_L (live rail)"]
    assert q["lk_options"] == {"bypass": 1.0} and isinstance(q["plugins"], dict)
    info = sf.info(d / "lkaic-quail-l.wav")
    assert info.samplerate == config.PIPELINE_RATE and info.subtype == "PCM_16"
    assert abs(info.duration - 0.3) < 0.01, info.duration
    mask = np.load(d / "lkaic-quail-l.aicvad.npy")
    assert mask.dtype == np.int8 and mask.tolist() == [1, 0, 1, -1], "-1 must survive on the live rail too"
    assert q["aicvad"]["dtype"] == "int8" and q["aicvad"]["encoding"] == lk_cloud.AICVAD_ENCODING
    assert q["aicvad_missing_frames"] == 1 == q["aicvad"]["missing_frames"]
    assert q["aicvad"]["file"] == "lkaic-quail-l.aicvad.npy" and q["aicvad"]["t0_s"] == 0.0
    assert (q["aicvad"]["frames"], q["aicvad"]["speech_frames"], q["aicvad"]["missing_frames"]) == (4, 2, 1)
    assert q["aicvad"]["frame_s"] == 0.05 and "live rail" in q["aicvad"]["source"]
    k = by["krisp-viva"]
    assert k["rail"] == "live" and k["status"] == "done" and "aicvad" not in k
    assert not stale.exists(), "flags derived from an earlier attempt must go when the new record has none"
    assert by["lkaic-vf22-l"]["status"] == "error" and by["lkaic-vf22-l"]["error"] == "the stream died"
    assert not (d / "lkaic-vf22-l.wav").exists()
    assert by["cloud-raw"]["status"] == "error" and "recorded nothing" in by["cloud-raw"]["error"]
    assert st.saved["status"] == "done" and no_attempt_dirs(d)


# ---------------------------------------------------------------- merge (P0)


def check_merge_keeps_others(env: Env):
    d = rundir()
    prior = [good_record(d, f"old-{i:02d}", i + 1, seg=i % 3 == 0, flags=i % 4 == 0, cut=i % 5 == 0,
                         extra_field={"nested": [i, {"k": "ü"}]}) for i in range(30)]
    prior.insert(7, good_record(d, "gtcrn", 50, seg=True, flags=True, cut=True))
    prior.append(good_record(d, "hush", 51))
    prior.append({"id": "legacy-error", "label": "L", "status": "error", "error": "old failure"})
    assert len(prior) == 33
    meta = dict(GOOD_INPUT, id="r1", source="upload", candidates=prior)
    originals = {r["id"]: json.dumps(r) for r in prior}
    ids_before = [r["id"] for r in prior]
    files_before = snapshot(d)
    server, st, ev, _, calls = _server_stubs(env, d, _known(), meta)
    asyncio.run(server._process_run("r1", ["dtln", "gtcrn", "fastenhancer-t"], None, "", 1))

    out = st.saved["candidates"]
    assert [c["id"] for c in out] == ids_before + ["dtln", "fastenhancer-t"], \
        "stored order kept, new ids appended in ticked order"
    by = {c["id"]: c for c in out}
    kept = [i for i in ids_before if i != "gtcrn"]
    assert len(kept) == 32
    for cid in kept:
        assert json.dumps(by[cid]) == originals[cid], f"{cid} was touched"
        assert by[cid] == json.loads(originals[cid])
    g = by["gtcrn"]
    assert g["status"] == "done" and g["stt"] == {"text": "hello"} and "last_attempt" not in g
    assert g["output"] == "gtcrn.wav" and g["processed_at"] != "2026-08-04T11:25:11" and g["concurrency"] == 1
    assert "aicvad" not in g and "output_seg" not in g, "the new record does not inherit old fields"
    files_after = snapshot(d)
    for name, data in files_before.items():
        if name.startswith("gtcrn."):
            continue
        assert files_after.get(name) == data, f"{name} changed or vanished"
    assert files_after["gtcrn.wav"] != files_before["gtcrn.wav"]
    assert not {"gtcrn.aicvad.npy", "gtcrn.seg.wav", "gtcrn.vadcut.wav"} & set(files_after), \
        "stale files derived from gtcrn's old audio must be gone"
    assert {"dtln.wav", "fastenhancer-t.wav"} <= set(files_after) and no_attempt_dirs(d)
    # the input analysis is reused when it exists and is clean ...
    assert calls.score_input == 0 and d / "input.wav" not in calls.analyze
    assert st.saved["input_scores"] == GOOD_INPUT["input_scores"] and st.saved["input_vad"] == GOOD_INPUT["input_vad"]
    # ... truth scores are computed for the replaced records only
    assert len(calls.score_run) == 1
    assert sorted(c["id"] for c in calls.score_run[0]["candidates"]) == ["dtln", "fastenhancer-t", "gtcrn"]

    # recomputed when missing or when it carries an error
    for bad in ({"error": "dnsmos crashed"}, None):
        m2 = dict(GOOD_INPUT, id="r1", candidates=[])
        if bad is None:
            del m2["input_scores"]
            del m2["input_vad"]
        else:
            m2["input_scores"], m2["input_vad"] = dict(bad), dict(bad)
        d2 = rundir()
        server, st2, _, _, c2 = _server_stubs(env, d2, _known(), m2)
        asyncio.run(server._process_run("r1", ["gtcrn"], None, "", 1))
        assert c2.score_input == 1 and d2 / "input.wav" in c2.analyze, (bad, c2.score_input, c2.analyze)
        assert "error" not in st2.saved["input_scores"] and st2.saved["input_vad"] == {"segments": []}


def check_merge_error_keeps_good_record(env: Env):
    d = rundir()
    old = good_record(d, "gtcrn", 5, seg=True, flags=True, cut=True,
                      last_attempt={"status": "error", "error": "an older failure", "at": "x"})
    other = good_record(d, "hush", 6)
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[old, other]))

    def boom(input_wav, chain, out_wav, measure_rss=False):
        write_wav(out_wav, 9)  # a half-written output the failed attempt leaves behind
        raise RuntimeError("chain died")

    env.attr(server, "run_chain", boom)
    before = snapshot(d)
    asyncio.run(server._process_run("r1", ["gtcrn"], None, "", 1))
    rec = {c["id"]: c for c in st.saved["candidates"]}["gtcrn"]
    la = rec["last_attempt"]
    assert la["status"] == "error" and la["error"] == "chain died" and isinstance(la["at"], str) and la["at"]
    assert la["at"] != "x", "last_attempt is the latest failure, not the carried-over one"
    assert {k: v for k, v in rec.items() if k != "last_attempt"} == \
        {k: v for k, v in old.items() if k != "last_attempt"}, "the old record must be kept as it was"
    assert rec["status"] == "done" and rec["output"] == "gtcrn.wav"
    assert snapshot(d) == before, "no file may be added, changed or removed by a failed attempt"
    assert no_attempt_dirs(d)
    assert [c["id"] for c in st.saved["candidates"]] == ["gtcrn", "hush"]
    assert {c["id"]: c for c in st.saved["candidates"]}["hush"] == other
    assert st.saved["status"] == "done"


def check_merge_error_replaces_error_record(env: Env):
    d = rundir()
    old = {"id": "gtcrn", "label": "GTCRN", "status": "error", "error": "old failure"}
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[old]))

    def boom(input_wav, chain, out_wav, measure_rss=False):
        raise RuntimeError("chain died again")

    env.attr(server, "run_chain", boom)
    asyncio.run(server._process_run("r1", ["gtcrn"], None, "", 1))
    (rec,) = st.saved["candidates"]
    assert rec["status"] == "error" and rec["error"] == "chain died again", rec
    assert "last_attempt" not in rec, "an error record has nothing good to preserve"
    assert snapshot(d) == {} and no_attempt_dirs(d)


def check_merge_success_replaces(env: Env):
    d = rundir()
    old = good_record(d, "gtcrn", 5, seg=True, flags=True, cut=True,
                      last_attempt={"status": "error", "error": "older failure", "at": "x"})
    keep = good_record(d, "hush", 6, seg=True, flags=True, cut=True)
    old_bytes = (d / "gtcrn.wav").read_bytes()
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[old, keep]))
    keep_before = snapshot(d)
    asyncio.run(server._process_run("r1", ["gtcrn"], None, "", 1))
    rec = {c["id"]: c for c in st.saved["candidates"]}["gtcrn"]
    assert rec["status"] == "done" and "last_attempt" not in rec, "a success carries nothing over"
    new_bytes = (d / "gtcrn.wav").read_bytes()
    assert new_bytes != old_bytes, "the new audio must replace the old file"
    for stale in ("gtcrn.aicvad.npy", "gtcrn.seg.wav", "gtcrn.vadcut.wav"):
        assert not (d / stale).exists(), f"{stale} belongs to the replaced audio and must be removed"
    assert "aicvad" not in rec and "output_seg" not in rec
    for name, data in keep_before.items():
        if not name.startswith("gtcrn."):
            assert snapshot(d)[name] == data, f"{name}: another candidate's file changed"
    assert {c["id"]: c for c in st.saved["candidates"]}["hush"] == keep
    assert no_attempt_dirs(d)

    # a cloud success that HAS flags: they are promoted, replacing the old flags
    d = rundir()
    old = good_record(d, "lkaic-vf22-l", 5, flags=True, cut=True)
    old_flags = (d / "lkaic-vf22-l.aicvad.npy").read_bytes()
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[old]))

    def results(out):
        write_wav(out / "lkaic-vf22-l.wav", 77)
        np.save(out / "lkaic-vf22-l.aicvad.npy", np.array([1, 1, -1], np.int8))
        return {"lkaic-vf22-l": {"output": "lkaic-vf22-l.wav", "duration_s": 1.0,
                                 "aicvad": {"file": "lkaic-vf22-l.aicvad.npy", "frames": 3}}}

    env.attr(server.cloud_replay, "replay", _replay_writer({}, results))
    asyncio.run(server._process_run("r1", ["lkaic-vf22-l"], None, "", 1))
    rec = st.saved["candidates"][0]
    assert rec["status"] == "done" and rec["aicvad"] == {"file": "lkaic-vf22-l.aicvad.npy", "frames": 3}
    assert np.load(d / "lkaic-vf22-l.aicvad.npy").tolist() == [1, 1, -1]
    assert (d / "lkaic-vf22-l.aicvad.npy").read_bytes() != old_flags
    assert not (d / "lkaic-vf22-l.vadcut.wav").exists(), "the cut of the old audio must go"
    assert "last_attempt" not in rec and no_attempt_dirs(d)


# ------------------------------------------------- failed attempts (P2), per rail


def check_p2_failed_replay_after_existing_wav(env: Env):
    d = rundir()
    old = good_record(d, "lkaic-vf22-l", 5, seg=True, flags=True, cut=True, rail="cloud_replay",
                      chain=["livekit-cloud:AIC:QUAIL_VF_L (cloud replay)"])
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[old]))
    before = snapshot(d)

    def results(out):
        return {"lkaic-vf22-l": {"error": "was not honoured", "cloud_replay": {"env_corr": 0.9}}}

    env.attr(server.cloud_replay, "replay", _replay_writer({}, results))
    asyncio.run(server._process_run("r1", ["lkaic-vf22-l"], None, "", 1))
    (rec,) = st.saved["candidates"]
    assert snapshot(d) == before, "old wav / flags / cut must be byte-identical, and nothing new written"
    la = rec["last_attempt"]
    assert la["error"] == "was not honoured" and la["rail"] == "cloud_replay" and la["cloud_replay"] == {"env_corr": 0.9}
    assert la["chain"] == ["livekit-cloud:AIC:QUAIL_VF_L (cloud replay)"]
    assert {k: v for k, v in rec.items() if k != "last_attempt"} == old
    for name in server._referenced(rec):
        assert (d / name).read_bytes() == before[name], f"{name} no longer the stored one"
    assert no_attempt_dirs(d)

    # the same, when the replay dies outright
    async def crash(*a, **k):
        raise RuntimeError("livekit exploded")

    env.attr(server.cloud_replay, "replay", crash)
    asyncio.run(server._process_run("r1", ["lkaic-vf22-l"], None, "", 1))
    (rec,) = st.saved["candidates"]  # the stub store re-serves the original meta each run
    assert snapshot(d) == before and "cloud replay: livekit exploded" in rec["last_attempt"]["error"]

    # offline rail: the chain writes part of an output, then raises
    d = rundir()
    old = good_record(d, "gtcrn", 5, flags=True, cut=True)
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[old]))

    def boom(input_wav, chain, out_wav, measure_rss=False):
        write_wav(out_wav, 9)
        raise RuntimeError("chain died")

    env.attr(server, "run_chain", boom)
    before = snapshot(d)
    asyncio.run(server._process_run("r1", ["gtcrn"], None, "", 1))
    (rec,) = st.saved["candidates"]
    assert snapshot(d) == before and rec["last_attempt"]["error"] == "chain died" and no_attempt_dirs(d)

    # live rail: the live result is an error
    d = rundir()
    old = good_record(d, "lkaic-vf22-l", 5, flags=True)
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[old]))
    before = snapshot(d)
    asyncio.run(server._process_run("r1", ["lkaic-vf22-l"], {"lkaic-vf22-l": {"error": "the stream died"}}, "", 1))
    (rec,) = st.saved["candidates"]
    assert snapshot(d) == before and rec["last_attempt"]["error"] == "the stream died"
    assert rec["last_attempt"]["rail"] == "live" and no_attempt_dirs(d)
    assert {k: v for k, v in rec.items() if k != "last_attempt"} == old


def check_p2_missing_staged_file(env: Env):
    d = rundir()
    old = good_record(d, "gtcrn", 5, flags=True)
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[old]))
    before = snapshot(d)
    env.attr(server, "run_chain", lambda input_wav, chain, out_wav, measure_rss=False: {"proc_ms": 1.0})
    asyncio.run(server._process_run("r1", ["gtcrn"], None, "", 1))
    (rec,) = st.saved["candidates"]
    assert "no output was written" in rec["last_attempt"]["error"], rec
    assert {k: v for k, v in rec.items() if k != "last_attempt"} == old and snapshot(d) == before
    assert calls.stt == [], "a candidate without an output must not reach STT"

    # no old record: the missing file is simply the candidate's error
    d = rundir()
    server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT))
    env.attr(server, "run_chain", lambda input_wav, chain, out_wav, measure_rss=False: {"proc_ms": 1.0})
    asyncio.run(server._process_run("r1", ["gtcrn"], None, "", 1))
    (rec,) = st.saved["candidates"]
    assert rec["status"] == "error" and "no output was written" in rec["error"] and "last_attempt" not in rec
    assert snapshot(d) == {}

    # a replay that reports an output (and flags) it never wrote
    for result in ({"output": "krisp-viva.wav", "duration_s": 1.0},
                   {"output": "krisp-viva.wav", "duration_s": 1.0, "aicvad": {"file": "krisp-viva.aicvad.npy"}}):
        d = rundir()
        server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT))

        def results(out, result=result):
            write_wav(out / "krisp-viva.wav", 1)  # the wav exists, the flags file does not
            if "aicvad" not in result:
                (out / "krisp-viva.wav").unlink()
            return {"krisp-viva": result}

        env.attr(server.cloud_replay, "replay", _replay_writer({}, results))
        asyncio.run(server._process_run("r1", ["krisp-viva"], None, "", 1))
        (rec,) = st.saved["candidates"]
        assert rec["status"] == "error" and "no output was written" in rec["error"], rec
        assert snapshot(d) == {}, "a partly staged attempt must not be promoted"


# ----------------------------------------------------- replay timeline gates (P1)

_FIX30: dict = {}


def fix30() -> Fix:
    if "f" not in _FIX30:
        _FIX30["f"] = Fix(dur=30.0, seed=31)
    return _FIX30["f"]


def _gap(pcm: np.ndarray, at_s: float, ms: float, lead_s: float = 3.2) -> np.ndarray:
    k = int((lead_s + at_s) * 24_000)
    return np.concatenate([pcm[:k], np.zeros(int(ms * 24), np.int16), pcm[k:]])


def _drop(pcm: np.ndarray, at_s: float, ms: float, lead_s: float = 3.2) -> np.ndarray:
    k = int((lead_s + at_s) * 24_000)
    return np.concatenate([pcm[:k], pcm[k + int(ms * 24):]])


def _heavy(f: Fix, lead_s: float = 3.2) -> np.ndarray:
    """What aggressive NC leaves: the input's energy envelope, no usable waveform."""
    env = np.convolve(np.abs(f.x24), np.ones(240) / 240, mode="same")
    noise = np.random.default_rng(5).standard_normal(len(f.x24)).astype(np.float32)
    return f.pcm(1.0, lead_s=lead_s, tail_s=2.7, x24=0.9 * env * noise)


def check_p1_gap120_fails(env: Env):
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = fix30()
    ctl = f.pcm(0.8, lead_s=3.2, tail_s=2.7)
    cand = f.pcm(0.5, lead_s=3.2, tail_s=2.7)
    heavy = _heavy(f)
    f.save("0", ctl)
    f.save("1", cand)                               # clean, NC-like
    f.save("2", heavy)                              # clean, waveform destroyed
    f.save("3", _gap(cand, 15.0, 120))              # the reviewer's case: 120 ms gap mid-file
    f.save("4", _gap(cand, f.dur - 1.0, 150))       # 150 ms gap inside the last ~1.5 s
    f.save("5", _drop(cand, 15.0, 120))             # 120 ms dropped mid-file
    f.save("6", _gap(heavy, 15.0, 120))             # the gap in a heavy-NC stream
    streams = [CONTROL,
               stream("1", "clean", "KRISP:VIVA"), stream("2", "clean-heavy", "BVC"),
               stream("3", "gap120", "KRISP:VIVA_TEL"), stream("4", "gap150-end", "KRISP:VIVA"),
               stream("5", "drop120", "BVCTelephony"), stream("6", "gap120-heavy", "NC")]
    res = f.finish(streams)
    for cid in ("clean", "clean-heavy"):
        assert "error" not in res[cid], (cid, res[cid].get("error"))
        rec = res[cid]["cloud_replay"]
        assert rec["lag_drift_ms"] is not None and rec["lag_drift_ms"] <= 4, (cid, rec["lag_drift_ms"])
        assert abs(rec["excess_ms"]) <= 5, (cid, rec["excess_ms"])
        assert rec["lag_votes"] >= 1 and rec["lag_voiced"] >= rec["lag_votes"]
        assert isinstance(rec["lag_local_ms"], list) and len(rec["lag_local_ms"]) == rec["lag_votes"]
        assert (f.out / f"{cid}.wav").exists()
    assert res["cloud-raw"]["cloud_replay"]["control_env_lag_drift_ms"] is not None
    assert res["clean"]["cloud_replay"]["control_env_lag_drift_ms"] <= 4
    assert res["clean"]["cloud_replay"]["control_env_corr"] > 0.9

    bad = {"gap120": ("inserted audio", "own timeline drifts"),
           "gap150-end": ("inserted audio",),
           "drop120": ("own timeline drifts", "lost audio"),
           "gap120-heavy": ("inserted audio", "own timeline drifts")}
    for cid, wants in bad.items():
        err = res[cid].get("error", "")
        assert any(w in err for w in wants), (cid, res[cid])
        assert "cloud_replay" in res[cid], "a gate failure must carry the record"
        assert not (f.out / f"{cid}.wav").exists(), f"{cid} failed but wrote a wav"
    assert res["gap120"]["cloud_replay"]["excess_ms"] > cr.MAX_EXCESS_MS or \
        res["gap120"]["cloud_replay"]["lag_drift_ms"] > 100, res["gap120"]["cloud_replay"]
    assert res["gap150-end"]["cloud_replay"]["excess_ms"] > 100
    assert res["drop120"]["cloud_replay"]["lag_drift_ms"] > 100
    assert sorted(p.name for p in f.out.iterdir()) == ["clean-heavy.wav", "clean.wav", "cloud-raw.wav"]


def check_p1_control_step_passes(env: Env):
    """A jitter-buffer step the control shares with every stream is the transport,
    not the candidate: a candidate's drift equal to the control's is fine."""
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = fix30()
    ctl = _gap(f.pcm(0.8, lead_s=3.2, tail_s=2.7), 15.0, 30)
    a = _gap(f.pcm(0.5, lead_s=3.2, tail_s=2.7), 15.0, 30)
    b = _gap(_heavy(f), 15.0, 30)
    c = _gap(f.pcm(0.5, lead_s=3.2, tail_s=2.7), 15.0, 30 + 120)  # ... plus its OWN 120 ms tear
    for name, pcm in (("0", ctl), ("1", a), ("2", b), ("3", c)):
        f.save(name, pcm)
    res = f.finish([CONTROL, stream("1", "shared", "KRISP:VIVA"), stream("2", "shared-heavy", "BVC"),
                    stream("3", "own-tear", "KRISP:VIVA_TEL")], raw_ids=())
    ctl_drift = res["shared"]["cloud_replay"]["control_env_lag_drift_ms"]
    assert 20 <= ctl_drift <= 40, f"the control's own step should read ~30 ms, got {ctl_drift}"
    for cid in ("shared", "shared-heavy"):
        assert "error" not in res[cid], (cid, res[cid].get("error"), res[cid]["cloud_replay"])
        # the heavy stream's lead-in is on the 10 ms envelope grid, so it reads up to 10 ms off
        assert abs(res[cid]["cloud_replay"]["excess_ms"]) <= 15
    assert abs(res["shared"]["cloud_replay"]["lag_drift_ms"] - ctl_drift) <= 6
    assert "error" in res["own-tear"], "a tear beyond the control's step must still fail"


def check_env_timeline():
    fs = 16_000
    x, _ = _x16(30, seed=41)
    tl = cr.env_timeline(x, 0.8 * x)
    assert tl["established"] is True and tl["votes"] >= 1 and tl["voiced"] >= tl["votes"], tl
    assert tl["drift_ms"] is not None and tl["drift_ms"] <= 4 and len(tl["lags_ms"]) == tl["votes"], tl
    assert tl["votes"] >= cr.ENV_LAG_MIN_SHARE * tl["voiced"]
    # the envelope survives what destroys the waveform
    env = np.convolve(np.abs(x), np.ones(240) / 240, mode="same")
    weak = (0.9 * env * np.random.default_rng(8).standard_normal(len(x))).astype(np.float32)
    tl2 = cr.env_timeline(x, weak)
    assert tl2["established"] and tl2["drift_ms"] <= 4, tl2
    # unrelated speech-like audio can match some windows by chance (bursty envelopes),
    # but never at one consistent lag: it must not read as a clean timeline
    other, _ = _x16(30, seed=97)
    tl3 = cr.env_timeline(x, other)
    assert not tl3["established"] or tl3["drift_ms"] > 100, tl3
    flat = (0.01 * np.random.default_rng(3).standard_normal(len(x))).astype(np.float32)
    tl4 = cr.env_timeline(x, flat)
    assert tl4["established"] is False and tl4["votes"] == 0 and tl4["drift_ms"] is None, tl4
    # an inserted gap reads as its own length (2 ms hops), whichever way the window falls
    for gap_ms in (40, 100, 200):
        k = int(14.3 * fs)
        y = np.concatenate([x[:k], np.zeros(int(gap_ms * fs / 1000), np.float32), x[k:]])[: len(x)]
        t = cr.env_timeline(x, y)
        assert abs(t["drift_ms"] - gap_ms) <= 4, (gap_ms, t["drift_ms"], t["lags_ms"])
        assert t["established"]
    # a dropped stretch reads the same magnitude
    k = int(14.3 * fs)
    y = np.concatenate([x[:k], x[k + int(0.1 * fs):], np.zeros(int(0.1 * fs), np.float32)])
    assert abs(cr.env_timeline(x, y)["drift_ms"] - 100) <= 4
    # too short for one window, or silent: nothing to say
    for xx in (x[: 2 * fs], np.zeros(20 * fs, np.float32)):
        t = cr.env_timeline(xx, xx)
        assert t["established"] is False and t["votes"] == 0 and t["drift_ms"] is None, t



# ------------------------------------------------ locked meta / concurrent writers


def check_put_placement():
    server = _server()
    put = server._put
    ids = lambda c: [x["id"] for x in c]  # noqa: E731
    order = ["a", "b", "c", "d"]
    cands = [{"id": "a"}, {"id": "b", "v": 0}, {"id": "c"}]
    put(cands, {"id": "b", "v": 1}, order)
    assert ids(cands) == ["a", "b", "c"] and cands[1]["v"] == 1, "in place when present"
    cands = [{"id": "a"}, {"id": "c"}]
    put(cands, {"id": "b"}, order)
    assert ids(cands) == ["a", "b", "c"], "before the next id of `order` that is present"
    put(cands, {"id": "d"}, order)
    assert ids(cands) == ["a", "b", "c", "d"]
    cands = [{"id": "a"}, {"id": "d"}]
    put(cands, {"id": "b"}, order)
    assert ids(cands) == ["a", "b", "d"], "skips absent ids to the next present one"
    # completion order 3, 1, 2 still yields the ticked order 1, 2, 3
    cands, order = [], ["1", "2", "3"]
    for cid in ("3", "1", "2"):
        put(cands, {"id": cid}, order)
    assert ids(cands) == ["1", "2", "3"], ids(cands)
    # records this pass knows nothing about stay where they are
    cands = [{"id": "x"}, {"id": "a"}, {"id": "y"}, {"id": "c"}]
    put(cands, {"id": "b"}, ["a", "b", "c"])
    assert ids(cands) == ["x", "a", "y", "b", "c"], ids(cands)
    put(cands, {"id": "z"}, ["a", "b", "c"])       # not in `order` at all
    assert ids(cands) == ["x", "a", "y", "b", "c", "z"]
    put(cands, {"id": "x", "v": 2}, ["a", "b", "c"])
    assert cands[0] == {"id": "x", "v": 2} and len(cands) == 6


def _slow_known(n: int) -> dict:
    k = _known()
    for i in range(1, n + 1):
        k[f"n{i}"] = {"id": f"n{i}", "label": f"N{i}", "rail": "offline", "chain": [{"proc": "specnc"}],
                      "available": True, "unavailable_reason": ""}
    return k


def check_locked_meta_concurrent_writer(env: Env):
    """A second writer (the VAD-bench CLI, a note) saves meta continuously while a
    6-chain pass runs at concurrency 3; none of its updates may be lost and the
    candidates must land in ticked order however they finish."""
    import threading

    d = rundir()
    prior = [good_record(d, "hush", 3), good_record(d, "gtcrn", 4)]
    server, st, ev, _, calls = _server_stubs(
        env, d, _slow_known(6), dict(GOOD_INPUT, candidates=prior, note="start", vad_bench={}))
    from nc_bench import store

    # n1 finishes last, n3 first: completion order is not ticked order
    delays = {"n1": 0.35, "n2": 0.2, "n3": 0.02, "n4": 0.3, "n5": 0.1, "n6": 0.05}

    def slow(input_wav, chain, out_wav, measure_rss=False):
        time.sleep(delays[Path(out_wav).stem])
        write_wav(out_wav, hash(out_wav.stem) % 1000)
        return {"proc_ms": 1.0}

    env.attr(server, "run_chain", slow)
    stop, written = threading.Event(), []

    def writer():
        k = 0
        while not stop.is_set():
            with store.locked_meta("r1") as m:
                m.setdefault("vad_bench", {})[f"k{k}"] = {"v": k}
                m["note"] = f"note {k}"
            written.append(k)
            k += 1
            time.sleep(0.004)

    th = threading.Thread(target=writer)
    th.start()
    try:
        time.sleep(0.03)
        asyncio.run(server._process_run("r1", [f"n{i}" for i in range(1, 7)], None, "s", 3))
    finally:
        stop.set()
        th.join(10)
    final = st.saved
    assert len(written) >= 20, f"the writer barely ran ({len(written)} updates)"
    assert sorted(final["vad_bench"], key=lambda k: int(k[1:])) == [f"k{i}" for i in written], \
        "a writer update was lost"
    assert final["note"] == f"note {written[-1]}"
    assert [c["id"] for c in final["candidates"]] == ["hush", "gtcrn"] + [f"n{i}" for i in range(1, 7)], \
        "ticked order, after the stored ones"
    assert all(c["status"] == "done" for c in final["candidates"][2:])
    assert final["candidates"][0] == prior[0] and final["candidates"][1] == prior[1]
    assert final["status"] == "done" and final["script"] == "s" and final["concurrency"] == 3
    assert final["input_scores"] == GOOD_INPUT["input_scores"]
    assert no_attempt_dirs(d), sorted(p.name for p in d.iterdir())
    assert all((d / f"n{i}.wav").is_file() for i in range(1, 7))


def check_locked_meta_old_record_from_disk(env: Env):
    """`old` is the CURRENT stored record, not the snapshot taken at the start."""
    for fail in (False, True):
        d = rundir()
        gt = good_record(d, "gtcrn", 5)
        hush = good_record(d, "hush", 6)
        server, st, ev, _, calls = _server_stubs(env, d, _known(), dict(GOOD_INPUT, candidates=[gt, hush]))

        def edit(m):
            for c in m["candidates"]:
                c["stt"] = {"text": f"EDITED {c['id']}"}
                c["scores"] = {**c["scores"], "wer_seg": 0.25}

        def chain(input_wav, chain_spec, out_wav, measure_rss=False, fail=fail):
            st.edit(edit)   # another writer saves while this attempt is in flight
            write_wav(out_wav, 1)
            if fail:
                raise RuntimeError("chain died")
            return {"proc_ms": 1.0}

        env.attr(server, "run_chain", chain)
        asyncio.run(server._process_run("r1", ["gtcrn"], None, "", 1))
        by = {c["id"]: c for c in st.saved["candidates"]}
        assert by["hush"]["stt"] == {"text": "EDITED hush"} and by["hush"]["scores"]["wer_seg"] == 0.25, \
            "a kept record lost an edit saved mid-pass"
        if fail:
            g = by["gtcrn"]
            assert g["stt"] == {"text": "EDITED gtcrn"} and g["scores"]["wer_seg"] == 0.25, \
                "a failed attempt must keep the on-disk record, not the start-of-pass snapshot"
            assert g["last_attempt"]["error"] == "chain died" and g["output"] == "gtcrn.wav"
            assert snapshot(d)["gtcrn.wav"] == (d / "gtcrn.wav").read_bytes()
        else:
            g = by["gtcrn"]
            assert g["stt"] == {"text": "hello"} and "last_attempt" not in g and g["status"] == "done"
        assert no_attempt_dirs(d)


# ------------------------------------------------ onset: raw twins (P1 follow-up)

_TWIN = {"name": "%dt", "id": None, "lk_model": lk_cloud.RAW}


def _twin_stream(i: int) -> dict:
    return {"name": f"{i}t", "id": None, "lk_model": lk_cloud.RAW, "twin_of": str(i)}


def _onset_streams(f: Fix, cases: list[dict]) -> list[dict]:
    """Save a control (lead 3.2 s, x0.8) and, per case, a filtered candidate (x0.5) plus
    its raw twin (x0.8) that share a start. A case sets `start` (the lead of both: the
    preroll they missed, a Krisp-like start skew, is 3.2 - start), `delay` (the
    filter's own delay: the candidate is shifted by it and ends earlier by it, so its
    sample count still equals the twin's) and `edit` (what happens to the candidate)."""
    f.save("0", f.pcm(0.8, lead_s=3.2, tail_s=2.7))
    streams = [CONTROL]
    for i, c in enumerate(cases, 1):
        start, delay = c.get("start", 3.2), c.get("delay", 0.0)
        cand = f.pcm(0.5, lead_s=start + delay, tail_s=2.7 - delay)
        if c.get("edit"):
            cand = c["edit"](cand, start + delay)
        f.save(str(i), cand)
        f.save(f"{i}t", f.pcm(0.8, lead_s=start, tail_s=2.7))
        streams += [stream(str(i), c["cid"], "KRISP:VIVA"), _twin_stream(i)]
    return streams


def check_p1_onset_gap_fails(env: Env):
    """The reviewer's hole: a candidate-only 120 ms zero gap 150 ms after the audio
    starts used to be aligned away as extra lead-in. Through the real _finish_batch,
    with twins, on 2/4/12/30 s of aperiodic speech-like audio."""
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    onset_gap = lambda pcm, lead: _gap(pcm, 0.15, 120, lead_s=lead)    # noqa: E731
    onset_drop = lambda pcm, lead: _drop(pcm, 0.15, 120, lead_s=lead)  # noqa: E731
    cases = [
        {"cid": "gap", "edit": onset_gap},
        {"cid": "skew-gap", "start": 3.08, "edit": onset_gap},              # Krisp-like +120 ms start skew
        {"cid": "drop", "edit": onset_drop},
        {"cid": "late", "delay": 0.2},                                      # the opening shifted, length intact
        {"cid": "early", "delay": -0.10},                                   # 100 ms early: outside -60..+90
        {"cid": "clean", "delay": 0.03},                                    # the filter's own delay
        {"cid": "smoke", "start": 3.08, "delay": 0.03},                     # measured Krisp skew + delay
    ]
    for dur, seed in ((2.0, 1), (4.0, 2), (12.0, 3), (30.0, 4)):
        f = Fix(dur=dur, seed=seed, gen=bursty)
        res = f.finish(_onset_streams(f, cases), raw_ids=())
        assert set(res) == {c["cid"] for c in cases}, (dur, sorted(res))
        for cid, want in (("gap", "inserted audio"), ("skew-gap", "inserted audio"),
                          ("drop", "lost audio"), ("late", "opening was shifted"),
                          ("early", "opening was shifted")):
            err = res[cid].get("error", "")
            # on a 2 s input the prominence gate already rejects the onset gap (its
            # envelope has almost nothing else to match); either way it is refused
            alt = "does not line up" if dur == 2.0 and cid in ("gap", "skew-gap") else want
            assert want in err or alt in err, (dur, cid, res[cid])
            if dur >= 4.0:
                assert want in err, (dur, cid, "the twin gate must be what catches it", err)
            assert not (f.out / f"{cid}.wav").exists(), f"{dur}s {cid}: failed but wrote a wav"
            assert "cloud_replay" in res[cid]
        assert res["gap"]["cloud_replay"]["twin_len_diff_ms"] == 120.0, (dur, res["gap"]["cloud_replay"])
        assert res["drop"]["cloud_replay"]["twin_len_diff_ms"] == -120.0
        assert res["late"]["cloud_replay"]["onset_delay_ms"] > cr.MAX_ONSET_DELAY_MS
        assert res["early"]["cloud_replay"]["onset_delay_ms"] < cr.MIN_ONSET_DELAY_MS
        for cid in ("clean", "smoke"):
            assert "error" not in res[cid], (dur, cid, res[cid].get("error"), res[cid].get("cloud_replay"))
            rec = res[cid]["cloud_replay"]
            assert rec["twin_len_diff_ms"] == 0.0, (dur, cid, rec["twin_len_diff_ms"])
            assert abs(rec["onset_delay_ms"] - 30) <= 5, (dur, cid, rec["onset_delay_ms"])
            assert (f.out / f"{cid}.wav").is_file() and sf.info(f.out / f"{cid}.wav").frames == len(f.x16)
        assert sorted(p.name for p in f.out.iterdir()) == ["clean.wav", "smoke.wav"], (dur, list(f.out.iterdir()))


def check_p1_midfile_with_twins(env: Env):
    """The mid-file cases on the twin layout, plus one batch with NO twins so the
    control-relative excess_ms fallback stays covered (check_p1_gap120_fails does too)."""
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = fix30()
    for old in f.out.iterdir():   # the shared fixture's out dir still holds other checks' files
        old.unlink()
    cases = [
        {"cid": "clean"},
        {"cid": "gap120", "edit": lambda pcm, lead: _gap(pcm, 15.0, 120, lead_s=lead)},
        {"cid": "gap150-end", "edit": lambda pcm, lead: _gap(pcm, f.dur - 1.0, 150, lead_s=lead)},
        {"cid": "drop120", "edit": lambda pcm, lead: _drop(pcm, 15.0, 120, lead_s=lead)},
    ]
    res = f.finish(_onset_streams(f, cases), raw_ids=())
    assert "error" not in res["clean"], res["clean"].get("error")
    assert res["clean"]["cloud_replay"]["twin_len_diff_ms"] == 0.0
    assert res["clean"]["cloud_replay"]["lag_drift_ms"] <= 4
    assert "inserted audio" in res["gap120"]["error"] and "inserted audio" in res["gap150-end"]["error"]
    assert "lost audio" in res["drop120"]["error"]
    assert sorted(p.name for p in f.out.iterdir()) == ["clean.wav"]

    # the same candidates with no twins: the excess_ms fallback
    f2 = Fix(dur=30.0, seed=31)
    f2.save("0", f2.pcm(0.8, lead_s=3.2, tail_s=2.7))
    f2.save("1", f2.pcm(0.5, lead_s=3.2, tail_s=2.7))
    f2.save("2", _gap(f2.pcm(0.5, lead_s=3.2, tail_s=2.7), f2.dur - 1.0, 150))
    res = f2.finish([CONTROL, stream("1", "clean", "KRISP:VIVA"), stream("2", "gap150-end", "KRISP:VIVA")],
                    raw_ids=())
    assert "error" not in res["clean"] and "twin_len_diff_ms" not in res["clean"]["cloud_replay"]
    assert "inserted audio" in res["gap150-end"]["error"] and "longer than the control" in res["gap150-end"]["error"]


def check_twin_missing_or_empty(env: Env):
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = Fix(dur=12.0, seed=51, gen=bursty)
    f.save("0", f.pcm(0.8))
    f.save("1", f.pcm(0.5))
    f.save("1t", np.zeros(0, np.int16), np.zeros(0, np.int8))
    f.save("2", f.pcm(0.5))
    f.save("2t", f.pcm(0.8))
    res = f.finish([CONTROL, stream("1", "no-twin-audio", "KRISP:VIVA"), _twin_stream(1),
                    stream("2", "ok", "KRISP:VIVA"), _twin_stream(2)], raw_ids=())
    assert set(res) == {"no-twin-audio", "ok"}, "twins are never results"
    assert "raw twin received no audio" in res["no-twin-audio"]["error"], res["no-twin-audio"]
    assert "cloud_replay" in res["no-twin-audio"] and "error" not in res["ok"]
    # a batch-level failure lists candidates and raw ids only, not twins
    res = f.finish([CONTROL, stream("1", "a", "KRISP:VIVA"), _twin_stream(1)], starved=50.0, raw_ids=("cloud-raw",))
    assert set(res) == {"a", "cloud-raw"} and all("publisher stalled" in r["error"] for r in res.values())


def check_best_lag_silent_tail():
    """On a 4 s input with a 3.2 s lead and a 2.7 s silent tail, near-zero-variance
    windows used to score "correlations" far above 1 and the lead-in came out 7.2 s."""
    fs = 16_000
    x = bursty(4.0, fs, 7)
    lead = int(3.2 * fs)
    y = np.concatenate([np.zeros(lead, np.float32), x, np.zeros(int(2.7 * fs), np.float32)])
    assert cr.lead_in(x, y) == lead, cr.lead_in(x, y) / fs
    y2 = np.concatenate([np.zeros(lead, np.float32), 0.5 * x, np.zeros(int(2.7 * fs), np.float32)])
    assert cr.lead_in(x, y2) == lead
    # the primitive: exact lag, correlation never above 1
    lag, c = cr._best_lag(x, y, len(y) - len(x))
    assert lag == lead and 0.999 < c <= 1 + 1e-6, (lag / fs, c)
    env = cr._envelope(x)
    envy = cr._envelope(y)
    lag_e, c_e = cr._best_lag(env, envy, len(envy) - len(env))
    assert lag_e * cr._ENV_HOP == lead and c_e <= 1 + 1e-6, (lag_e, c_e)
    # nothing but silence: no match, and no division by ~0
    assert cr._best_lag(x, np.zeros(len(x) * 3), len(x) * 2) == (0, 0.0)
    # a genuinely quiet match (100 dB under the loud neighbour) still counts
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(2000)
    sig = np.concatenate([np.zeros(50), 1e-3 * ref, 100 * rng.standard_normal(2000)])
    assert cr._best_lag(ref, sig, 200)[0] == 50


# ------------------------------------------- the twin anchored to the control (P1 follow-up)


def _delay(pcm: np.ndarray, ms: float) -> np.ndarray:
    """A filter's own delay: same sample count, the audio arrives `ms` later."""
    k = int(ms * 24)
    return np.concatenate([np.zeros(k, np.int16), pcm[: len(pcm) - k]]) if k else pcm


def _anchored_streams(f: Fix, cases: list[dict]) -> tuple[list[dict], dict]:
    """A control (x0.8, lead 3.2 s, tail 2.7 s) and, per case, a candidate (x0.5, delayed
    by the filter's own delay) plus its raw twin. As in the real recording the twin is
    the control's EXACT samples from where it opened (`skew` ms in): a twin built from
    separately generated audio would (rightly) fail the anchor. A case may `edit` the
    twin and the candidate alike (a fault the pair shares), and gives `open` = whether
    the subscriber's opening times are reported."""
    ctl = f.pcm(0.8, lead_s=3.2, tail_s=2.7)
    f.save("0", ctl)
    half = f.pcm(0.5, lead_s=3.2, tail_s=2.7)
    streams, open_ms = [CONTROL], {"0": [1000.0, 1000.0]}
    for i, c in enumerate(cases, 1):
        k = int(c.get("skew", 0) * 24)
        twin, cand = ctl[k:], _delay(half[k:], c.get("delay", 0.0))
        lead = 3.2 - c.get("skew", 0) / 1000     # the audio starts here in both
        if c.get("edit"):
            twin, cand = c["edit"](twin, lead), c["edit"](cand, lead + c.get("delay", 0.0) / 1000)
        if c.get("edit_cand"):                   # a fault only the candidate has
            cand = c["edit_cand"](cand, lead + c.get("delay", 0.0) / 1000)
        if c.get("tail_ms"):                     # a shorter/longer END, as a dropped partial frame makes
            n = int(c["tail_ms"] * 24)
            cut = (lambda a: a[:len(a) + n]) if n < 0 else (lambda a: np.concatenate([a, np.zeros(n, np.int16)]))
            twin, cand = cut(twin), cut(cand)
        f.save(str(i), cand)
        f.save(f"{i}t", twin)
        streams += [stream(str(i), c["cid"], "KRISP:VIVA"), _twin_stream(i)]
        if c.get("open", True):
            t = 1000.0 + c.get("skew", 0)
            open_ms[str(i)] = open_ms[f"{i}t"] = [t - 5, t]
    return streams, open_ms


def check_p1_shared_onset_gap_fails(env: Env):
    """The reviewer's exact case: the candidate AND its twin both carry the same 120 ms
    zero gap 150 ms after onset. Candidate-vs-twin agrees perfectly, so only the twin
    against the control can see it."""
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    gap = lambda pcm, lead: _gap(pcm, 0.15, 120, lead_s=lead)   # noqa: E731
    cases = [
        {"cid": "shared-gap", "edit": gap},
        {"cid": "shared-gap-skew", "skew": 120, "edit": gap},       # Krisp-like start skew, open_ms says so
        {"cid": "shared-gap-noopen", "skew": 60, "edit": gap, "open": False},
        {"cid": "clean", "delay": 30},
        {"cid": "smoke", "skew": 120, "delay": 30},                 # measured skew and filter delay
        {"cid": "noopen", "skew": 60, "delay": 30, "open": False},  # skew check skipped, still passes
    ]
    for dur, seed in ((2.0, 11), (4.0, 12), (12.0, 13), (30.0, 14)):
        f = Fix(dur=dur, seed=seed, gen=bursty)
        streams, open_ms = _anchored_streams(f, cases)
        res = f.finish(streams, raw_ids=(), open_ms=open_ms)
        for cid in ("shared-gap", "shared-gap-skew", "shared-gap-noopen"):
            err = res[cid].get("error", "")
            assert err, (dur, cid, "accepted a shared onset gap", res[cid].get("cloud_replay"))
            if dur > 2.0:   # a 2 s envelope may already be refused by the prominence gate
                assert "its raw twin" in err, (dur, cid, err)
            assert not (f.out / f"{cid}.wav").exists(), f"{dur}s {cid}: failed but wrote a wav"
        a = res["shared-gap"]["cloud_replay"]
        if dur > 2.0:
            anchor = a["twin_vs_control"]
            assert anchor["residual_db"] > cr.MAX_TWIN_RESIDUAL_DB or anchor["len_vs_control_ms"] > 60, anchor
            assert a["twin_len_diff_ms"] == 0.0, "the pair agrees with itself: only the control can tell"
            sk = res["shared-gap-skew"]["cloud_replay"]["twin_vs_control"]
            assert sk["predicted_offset_ms"] == 120.0
        for cid, skew in (("clean", 0), ("smoke", 120), ("noopen", 60)):
            assert "error" not in res[cid], (dur, cid, res[cid].get("error"))
            rec = res[cid]["cloud_replay"]
            anchor = rec["twin_vs_control"]
            assert anchor["residual_db"] < -200, (dur, cid, anchor)
            assert anchor["offset_ms"] == float(skew), (dur, cid, anchor)
            assert anchor["len_vs_control_ms"] == 0.0 and rec["twin_len_diff_ms"] == 0.0
            assert abs(rec["onset_delay_ms"] - 30) <= 5
            if cid == "noopen":
                assert anchor["predicted_offset_ms"] is None and anchor["offset_err_ms"] is None
            else:
                assert anchor["predicted_offset_ms"] == float(skew) and abs(anchor["offset_err_ms"]) <= 3
            assert (f.out / f"{cid}.wav").is_file()
        assert sorted(p.name for p in f.out.iterdir()) == ["clean.wav", "noopen.wav", "smoke.wav"], \
            (dur, sorted(p.name for p in f.out.iterdir()))
    # a shared gap mid-file, and a start skew the opening times contradict
    f = Fix(dur=30.0, seed=15, gen=bursty)
    streams, open_ms = _anchored_streams(f, [
        {"cid": "shared-mid", "edit": lambda pcm, lead: _gap(pcm, 15.0, 120, lead_s=lead)},
        {"cid": "wrong-skew", "skew": 120, "delay": 30},
        {"cid": "clean", "delay": 30}])
    open_ms["2"] = open_ms["2t"] = [1000.0, 1000.0]      # says "opened with the control": the audio says +120 ms
    res = f.finish(streams, raw_ids=(), open_ms=open_ms)
    assert "its raw twin" in res["shared-mid"]["error"], res["shared-mid"]
    assert "from where its opening time puts it" in res["wrong-skew"]["error"], res["wrong-skew"]
    assert res["wrong-skew"]["cloud_replay"]["twin_vs_control"]["offset_err_ms"] == 120.0
    assert "error" not in res["clean"] and sorted(p.name for p in f.out.iterdir()) == ["clean.wav"]


def check_p1_partial_frame_length(env: Env):
    """A sound twin's length differs from the control's by less than one 50 ms frame
    (each stream drops its own last partial frame): +-49.9 ms is legit, more is not."""
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = Fix(dur=12.0, seed=71, gen=bursty)
    streams, open_ms = _anchored_streams(f, [
        {"cid": "cut40", "delay": 30, "tail_ms": -40},      # END cut from the silent tail: samples still match
        {"cid": "pad40", "delay": 30, "tail_ms": 40},
        {"cid": "cut45", "delay": 30, "tail_ms": -45},
        {"cid": "cut50", "delay": 30, "tail_ms": -50},
        {"cid": "cut60", "delay": 30, "tail_ms": -60},
        {"cid": "drop60", "delay": 30, "edit": lambda pcm, lead: _drop(pcm, 6.0, 60, lead_s=lead)},  # 60 ms of AUDIO
    ])
    res = f.finish(streams, raw_ids=(), open_ms=open_ms)
    for cid, lvc in (("cut40", -40.0), ("pad40", 40.0), ("cut45", -45.0)):
        assert "error" not in res[cid], (cid, res[cid].get("error"))
        a = res[cid]["cloud_replay"]["twin_vs_control"]
        assert a["len_vs_control_ms"] == lvc and a["residual_db"] < -200, (cid, a)
        assert res[cid]["cloud_replay"]["twin_len_diff_ms"] == 0.0
    for cid in ("cut50", "cut60"):
        assert "raw twin's length is" in res[cid]["error"], (cid, res[cid])
        assert res[cid]["cloud_replay"]["twin_vs_control"]["len_vs_control_ms"] <= -50
    assert "its raw twin" in res["drop60"]["error"], res["drop60"]
    assert sorted(p.name for p in f.out.iterdir()) == ["cut40.wav", "cut45.wav", "pad40.wav"]


def _scatter(shifts_ms: list[int]):
    """Edit for a stream: chunk k of the input (from 0.4 s, 1.5 s each, one per env window)
    is delayed by shifts_ms[k % n] without changing the stream's length."""
    def edit(pcm: np.ndarray, lead: float) -> np.ndarray:
        out = pcm.copy()
        k, t = 0, 0.4
        while True:
            c0, c1 = int((lead + t) * 24_000), int((lead + t + 1.5) * 24_000)
            if c1 > len(pcm) - 24_000:
                break
            sh = int(shifts_ms[k % len(shifts_ms)] * 24)
            out[c0:c1] = pcm[c0 - sh : c1 - sh]
            k, t = k + 1, t + 1.5
        return out
    return edit


def check_confirmed_drift(env: Env):
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    f = Fix(dur=30.0, seed=81, gen=bursty)
    fs = 16_000
    x = f.x16
    # (a) scattered lags: nothing confirms anything, so there is no confirmed drift
    shifts = [(((7 * i) % 20) - 10) * 20 for i in range(20)]   # -200..180 ms, 20 ms apart: all distinct
    y = _scatter(shifts)(f.pcm(1.0, lead_s=0.0, tail_s=0.0), 0.0)
    y16 = soxr.resample(y.astype(np.float32) / 32768, 24_000, fs).astype(np.float32)[: len(x)]
    tl = cr.env_timeline(x, y16)
    assert tl["votes"] >= 4 and tl["drift_ms"] is not None and tl["drift_ms"] >= 60, tl
    assert tl["confirmed_drift_ms"] is None and tl["confirmed"] < 2, tl
    # (b) a real tear: every window after it moves by the same 120 ms
    k = int(14.3 * fs)
    torn = np.concatenate([x[:k], np.zeros(int(0.12 * fs), np.float32), x[k:]])[: len(x)]
    tb = cr.env_timeline(x, torn)
    assert tb["confirmed"] >= 4 and abs(tb["confirmed_drift_ms"] - 120) <= 4, tb
    assert abs(tb["drift_ms"] - 120) <= 4
    # the empty/short case carries the new keys too
    e = cr.env_timeline(x[: 2 * fs], x[: 2 * fs])
    assert e["confirmed_drift_ms"] is None and e["confirmed"] == 0 and e["votes"] == 0, e

    # through _finish_batch
    scatter = _scatter([(((7 * i) % 20) - 10) * 14 for i in range(20)])   # milder: still a valid stream

    def drop_then_insert(pcm, lead):     # the middle stretch moves 120 ms earlier; the length is unchanged
        a, b, n = int((lead + 8.0) * 24_000), int((lead + 22.0) * 24_000), int(0.12 * 24_000)
        return np.concatenate([pcm[:a], pcm[a + n : b], np.zeros(n, np.int16), pcm[b:]])

    streams, open_ms = _anchored_streams(f, [
        {"cid": "clean", "delay": 30},
        {"cid": "scattered", "delay": 30, "edit_cand": scatter},     # samples proven intact by the twin
        {"cid": "same-length-tear", "delay": 30, "edit_cand": drop_then_insert},
    ])
    res = f.finish(streams, raw_ids=(), open_ms=open_ms)
    assert "error" not in res["clean"], res["clean"].get("error")
    sc = res["scattered"]
    assert "error" not in sc, ("a with-twin candidate whose lags do not confirm must abstain", sc.get("error"))
    rec = sc["cloud_replay"]
    assert rec["lag_confirmed_drift_ms"] is None or rec["lag_confirmed_drift_ms"] <= cr.MAX_CAND_DRIFT_OVER_CONTROL_MS, rec
    assert rec["twin_len_diff_ms"] == 0.0
    assert rec["lag_drift_ms"] >= 60, "the strict all-votes drift is still recorded"
    tear = res["same-length-tear"]
    assert "timeline drifts" in tear["error"] and "confirmed windows" in tear["error"], tear
    assert abs(tear["cloud_replay"]["lag_confirmed_drift_ms"] - 120) <= 6, tear["cloud_replay"]
    assert tear["cloud_replay"]["twin_len_diff_ms"] == 0.0, "only the envelope drift can see this one"
    assert sorted(p.name for p in f.out.iterdir()) == ["clean.wav", "scattered.wav"]

    # a batch with NO twin keeps the old strict gate for the same scattered stream
    f2 = Fix(dur=30.0, seed=81, gen=bursty)
    f2.save("0", f2.pcm(0.8, lead_s=3.2, tail_s=2.7))
    f2.save("1", scatter(f2.pcm(0.5, lead_s=3.2, tail_s=2.7), 3.2))
    res = f2.finish([CONTROL, stream("1", "scattered-notwin", "KRISP:VIVA")], raw_ids=())
    err = res["scattered-notwin"].get("error", "")
    assert "timeline drifts" in err and "confirmed windows" not in err, res["scattered-notwin"]
    assert "twin_len_diff_ms" not in res["scattered-notwin"]["cloud_replay"]


def check_missing_twin_file_is_local(env: Env):
    env.attr(lk_cloud, "plugin_versions", lambda: {})
    cases = [{"cid": "a", "delay": 30}, {"cid": "b", "delay": 30}, {"cid": "c", "delay": 30}]

    def build(seed):
        f = Fix(dur=12.0, seed=seed, gen=bursty)
        streams, open_ms = _anchored_streams(f, cases)
        return f, streams, open_ms

    # 1. the middle twin's recording is absent
    f, streams, open_ms = build(21)
    (f.work / "2t.pcm.npy").unlink()
    res = f.finish(streams, raw_ids=(), open_ms=open_ms)
    assert "raw twin's recording is missing or unreadable" in res["b"]["error"], res["b"]
    assert "cloud_replay" in res["b"]
    for cid in ("a", "c"):
        assert "error" not in res[cid], (cid, res[cid].get("error"))
    assert sorted(p.name for p in f.out.iterdir()) == ["a.wav", "c.wav"]

    # 2. a torn twin: garbage bytes, and an .npy cut short
    for damage in ("garbage", "truncated"):
        f, streams, open_ms = build(22)
        path = f.work / "2t.pcm.npy"
        if damage == "garbage":
            path.write_bytes(b"this is not an npy file")
        else:
            data = path.read_bytes()
            path.write_bytes(data[: 128 + (len(data) - 128) // 2])
        res = f.finish(streams, raw_ids=(), open_ms=open_ms)
        assert "raw twin's recording is missing or unreadable" in res["b"]["error"], (damage, res["b"])
        assert "error" not in res["a"] and "error" not in res["c"], (damage, res["a"], res["c"])
        assert sorted(p.name for p in f.out.iterdir()) == ["a.wav", "c.wav"]

    # 3. the candidate's own recording is absent
    f, streams, open_ms = build(23)
    (f.work / "1.pcm.npy").unlink()
    res = f.finish(streams, raw_ids=(), open_ms=open_ms)
    assert "this stream's recording is missing or unreadable" in res["a"]["error"], res["a"]
    assert "error" not in res["b"] and "error" not in res["c"]
    assert sorted(p.name for p in f.out.iterdir()) == ["b.wav", "c.wav"]


def check_twin_vs_control():
    f = Fix(dur=16.0, seed=61, gen=bursty)
    ctl = f.pcm(0.8, lead_s=3.2, tail_s=2.7)
    k = int(0.12 * 24_000)
    twin = ctl[k:]
    onset = int((3.2 - 0.12) * 24_000)
    r = cr.twin_vs_control(ctl, twin, onset=onset)
    assert r["offset_ms"] == 120.0 and r["len_vs_control_ms"] == 0.0, r
    assert r["residual_db"] <= -200 and r["onset_residual_db"] <= -200, r
    assert cr.twin_vs_control(ctl, ctl)["offset_ms"] == 0.0
    assert cr.twin_vs_control(ctl, twin)["onset_residual_db"] is None, "no onset given, none reported"
    # a 120 ms gap 150 ms after the twin's audio starts
    g = onset + int(0.15 * 24_000)
    torn = np.concatenate([twin[:g], np.zeros(int(0.12 * 24_000), np.int16), twin[g:]])
    r = cr.twin_vs_control(ctl, torn, onset=onset)
    assert r["residual_db"] > cr.MAX_TWIN_RESIDUAL_DB and r["onset_residual_db"] > cr.MAX_TWIN_RESIDUAL_DB, r
    # the offset lands on the longer (post-gap) stretch, so the length looks fine: the residual is what tells
    assert r["offset_ms"] in (0.0, 120.0), r
    # a shortened twin reads negative
    assert cr.twin_vs_control(ctl, twin[: len(twin) - 24 * 100])["len_vs_control_ms"] == -100.0
    # empty on either side: nothing to say
    empty = {"offset_ms": None, "residual_db": None, "onset_residual_db": None, "len_vs_control_ms": None}
    assert cr.twin_vs_control(ctl, np.zeros(0, np.int16)) == empty
    assert cr.twin_vs_control(np.zeros(0, np.int16), twin, onset=0) == empty


# ================================================================ fetch_models


def check_fetch_aic(env: Env):
    import fetch_models

    env.module("aic_sdk", fake_aic_sdk())
    want = sorted({s["model"] for c in json.loads((ROOT / "candidates.json").read_text())
                   for s in c.get("chain") or [] if s.get("proc") == "aic"})
    assert len(want) == 12, want
    d = tmpdir()
    have_file = _touch(d, "quail_ms_l_8khz_x_v1.aicmodel")
    fetched: list = []

    def fake_model_file(mid):
        return have_file if mid == "quail-ms-l-8khz" else None

    def fake_fetch(mid):
        fetched.append(mid)
        if mid in ("rook-ms-l-48khz", "quail-vf-2.2-s-16khz"):
            raise RuntimeError("CDN 503")
        return _touch(d, f"{mid}.aicmodel")

    env.attr(aicmod, "model_file", fake_model_file)
    env.attr(aicmod, "fetch_model", fake_fetch)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bad = fetch_models.fetch_aic(False)
    assert sorted(fetched) == sorted(set(want) - {"quail-ms-l-8khz"}), fetched
    assert sorted(bad) == ["quail-vf-2.2-s-16khz", "rook-ms-l-48khz"], "failures are collected, not raised"
    out = buf.getvalue()
    assert "FAILED aic/rook-ms-l-48khz: CDN 503" in out and "aic/quail_ms_l_8khz_x_v1.aicmodel" in out and "have" in out
    fetched.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        bad = fetch_models.fetch_aic(True)
    assert fetched == [], "--verify must not download"
    assert sorted(bad) == sorted(set(want) - {"quail-ms-l-8khz"})
    # without the SDK the step is skipped, not failed
    env.module("aic_sdk", None)
    with contextlib.redirect_stdout(io.StringIO()):
        assert fetch_models.fetch_aic(False) == []


# ============================================================= candidates.json


def check_candidates_json():
    cands = json.loads((ROOT / "candidates.json").read_text())
    ids = [c["id"] for c in cands]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    by = {c["id"]: c for c in cands}
    assert "aic-sdk" not in by, "the old generic entry must be gone"
    sdk = {
        "aicsdk-quail-ms-l-16k": "quail-ms-l-16khz", "aicsdk-quail-ms-s-16k": "quail-ms-s-16khz",
        "aicsdk-quail-ms-l-8k": "quail-ms-l-8khz", "aicsdk-quail-ms-s-8k": "quail-ms-s-8khz",
        "aicsdk-vf22-l": "quail-vf-2.2-l-16khz", "aicsdk-vf22-s": "quail-vf-2.2-s-16khz",
        "aicsdk-rook-ms-l-48k": "rook-ms-l-48khz", "aicsdk-rook-ms-s-48k": "rook-ms-s-48khz",
        "aicsdk-rook-ms-l-16k": "rook-ms-l-16khz", "aicsdk-rook-ms-s-16k": "rook-ms-s-16khz",
        "aicsdk-rook-ms-l-8k": "rook-ms-l-8khz", "aicsdk-rook-ms-s-8k": "rook-ms-s-8khz",
    }
    assert {i for i in ids if i.startswith("aicsdk-")} == set(sdk), "aicsdk-* set differs from the ticket"
    for cid, model in sdk.items():
        c = by[cid]
        assert c["chain"] == [{"proc": "aic", "model": model}], (cid, c["chain"])
        assert "ai-coustics SDK 3.2 (standalone, own key)" in c["label"], c["label"]
        assert "lk_model" not in c and not c.get("historical")
    lk = {
        "lkaic-quail-l": ("AIC:QUAIL_L", None), "lkaic-vf22-l": ("AIC:QUAIL_VF_L", None),
        "lkaic-vf22-s": ("AIC:QUAIL_VF_S", None), "lkaic-rook-s": ("AIC:ROOK_S", None),
        "lkaic-quail-l-bypass": ("AIC:QUAIL_L", {"bypass": 1.0}),
        "lkaic-rook-s-bypass": ("AIC:ROOK_S", {"bypass": 1.0}),
    }
    assert {i for i in ids if i.startswith("lkaic-")} == set(lk)
    assert "lkaic-vf22-l-bypass" not in by, "dropped on purpose: plugin 0.3.2 ignores bypass on Voice Focus"
    for cid, (model, opts) in lk.items():
        assert by[cid]["lk_model"] == model and by[cid].get("lk_options") == opts, (cid, by[cid])
        assert not by[cid].get("historical")
    assert by["krisp-viva"]["lk_model"] == "KRISP:VIVA"
    assert by["krisp-viva-tel"]["lk_model"] == "KRISP:VIVA_TEL"
    assert by["cloud-raw"]["lk_model"] == lk_cloud.RAW and not by["cloud-raw"].get("lk_options")
    assert "Cloud replay" in by["cloud-raw"]["label"]
    for cid in ("krisp-nc", "krisp-bvc", "krisp-bvc-tel"):
        assert cid in by and "lk_model" in by[cid], f"{cid} must be unchanged"
    for cid in ("aic-quail-l", "aic-vf-s", "aic-vf-l"):
        assert by[cid].get("historical") is True and "lk_model" in by[cid], cid
        assert "historical" in by[cid]["label"]
    assert {c["id"] for c in cands if c.get("historical")} == {"aic-quail-l", "aic-vf-s", "aic-vf-l"}
    for c in cands:
        assert ("chain" in c) != ("lk_model" in c), f"{c['id']} needs exactly one of chain / lk_model"
        assert c.get("label"), c["id"]
        if "lk_model" in c and not c.get("historical"):
            why = lk_cloud._check_options(c["lk_model"], c.get("lk_options"))
            assert why == "", (c["id"], why)
            if "lk_options" in c:
                assert lk_cloud.is_aic(c["lk_model"]), f"{c['id']}: lk_options on a non-AIC model"


# ====================================================================== runner


def main() -> None:
    checks = [(n, f) for n, f in globals().items() if n.startswith("check_") and callable(f)]
    started = time.perf_counter()
    failed, known = [], []
    try:
        for name, fn in checks:
            env = Env()
            t0 = time.perf_counter()
            try:
                fn(env) if fn.__code__.co_varnames[: fn.__code__.co_argcount] == ("env",) else fn()
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                if name in KNOWN_BUGS:
                    known.append(name)
                    print(f"KNOWN-BUG {name} ({time.perf_counter() - t0:.1f}s)\n{tb}")
                else:
                    failed.append(name)
                    print(f"FAIL {name} ({time.perf_counter() - t0:.1f}s)\n{tb}")
            else:
                print(f"OK {name} ({time.perf_counter() - t0:.1f}s)")
            finally:
                env.close()
    finally:
        shutil.rmtree(_SANDBOX, ignore_errors=True)
    total = time.perf_counter() - started
    print(f"\n{len(checks) - len(failed) - len(known)}/{len(checks)} passed, "
          f"{len(failed)} failed, {len(known)} known-bug, {total:.1f}s")
    if failed:
        sys.exit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
