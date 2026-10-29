"""One loader per VAD family: mono int16 audio in, per-window speech
probabilities out.

Every loader returns `(probs, hop_s, t0_s)`: float32 probabilities (or 0/1
flags for a binary detector), the window hop in seconds, and where window 0
starts in file time (see segmenter.py). They take int16 PCM plus its rate and
resample themselves, because the conversion to float is part of each
reference implementation and differs between them: the livekit Silero plugin
divides by 32767, FireRed and FSMN feed kaldi fbank int16-scaled values, and
the rest take [-1, 1) floats.

Ported from the PR-42 bake-off's scripts (vad_bench.py, probe_loaders.py). Where a loader departs from it, the docstring says so; the
parity check (scripts/check_vad_parity.py) measures how far that moves the
probabilities.
"""

from __future__ import annotations

import asyncio

import numpy as np

from .. import config


RESAMPLER = "soxr-QQ"


def _i16(x_i16: np.ndarray, src: int, dst: int) -> np.ndarray:
    """int16 at `src` -> int16 at `dst`: libsoxr's quick (QQ) filter, which is
    the filter livekit's own VAD streams resample with, minus livekit's jitter.

    The resampler is not a detail on phone audio. Both livekit paths (the
    Silero plugin and inference.VAD) resample with rtc.AudioResampler QUICK
    before inference; soxr HQ instead moved Silero v6.2 8k probabilities by up
    to 0.147 against the plugin (20260803-164743/hush.wav), enough to flip
    threshold crossings. So every model here gets that same quick filter, which
    also keeps the resampler from being a hidden variable between rows.

    Why python-soxr QQ and not rtc.AudioResampler itself: livekit's QUICK is not
    a pure function. On identical input it returned 5 different outputs in 30
    calls, about 41% of samples off by up to 2 LSB, varying with unrelated
    allocations in the process (an ONNX session, a soxr call); measured
    2026-09-29, rtc 1.1.18. That jitter is what makes the plugin and
    inference.VAD non-repeatable, and a stored-probability bench cannot carry
    it. python-soxr QQ gave 1 output in 20 calls under the same churn, and it
    is within 1 LSB of QUICK (mean 0.25 LSB, r 0.999999, zero lag), inside
    QUICK's own run-to-run spread. No delay either way.
    """
    x = np.ascontiguousarray(x_i16, dtype=np.int16)
    if src == dst:
        return x
    import soxr

    return np.ascontiguousarray(soxr.resample(x, src, dst, quality="QQ"), dtype=np.int16)


def _f32(x_i16: np.ndarray, src: int, dst: int) -> np.ndarray:
    """int16 at `src` -> float32 [-1, 1) at `dst` (see _i16 for the resampler)."""
    return _i16(x_i16, src, dst).astype(np.float32) / 32768.0


def _ort_session(path):
    """Single-threaded, as livekit's own Silero session is: CPU cost is then a
    per-core number, comparable across models and with a production worker."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])


# ------------------------------------------------------------------ Silero


class SileroOnnx:
    """Silero ONNX per release tag, driven exactly as livekit-plugins-silero
    1.8.3's OnnxModel does (32 ms windows: 512 samples at 16 kHz, 256 at 8 kHz).

    v5/v6 graphs take a `state` tensor and expect the previous window's tail
    (64/32 samples) prepended as context. v4.0 has the older `h`/`c` state and
    no context, so the plugin's `onnx_file_path` cannot load it; it is driven
    the way the bake-off drove it.
    """

    def __init__(self, path, rate: int, state_api: str):
        self.sess = _ort_session(path)
        self.rate = rate
        self.v4 = state_api == "v4"
        self.win, self.ctx = (512, 64) if rate == 16000 else (256, 32)

    def run(self, x_i16, src_rate):
        # the plugin's own scaling: int16 / np.iinfo(np.int16).max
        x = _f32(x_i16, src_rate, self.rate) * (32768.0 / 32767.0)
        win, ctx, sr = self.win, self.ctx, np.array(self.rate, dtype=np.int64)
        n = len(x) // win
        out = np.empty(n, dtype=np.float32)
        if self.v4:
            h = np.zeros((2, 1, 64), np.float32)
            c = np.zeros((2, 1, 64), np.float32)
            for i in range(n):
                p, h, c = self.sess.run(None, {"input": x[i * win:(i + 1) * win][None, :],
                                               "sr": sr, "h": h, "c": c})
                out[i] = float(np.asarray(p).reshape(-1)[0])
        else:
            state = np.zeros((2, 1, 128), np.float32)
            buf = np.zeros((1, ctx + win), np.float32)
            for i in range(n):
                buf[0, ctx:] = x[i * win:(i + 1) * win]
                p, state = self.sess.run(None, {"input": buf, "state": state, "sr": sr})
                out[i] = float(np.asarray(p).reshape(-1)[0])
                buf[0, :ctx] = buf[0, -ctx:]
        return out, win / self.rate, 0.0


class LkInference:
    """Production's VAD: `livekit.agents.inference.VAD`, the native Silero
    v6.2 engine from livekit-local-inference, unsmoothed.

    Driven through the framework's own stream (frames in, INFERENCE_DONE
    probabilities out) rather than by calling the native library directly.
    Two reasons. livekit-local-inference's MODEL_LICENSE allows its models
    only together with the LiveKit Agents framework, not standalone. And the
    framework path is production's path: its windowing, its event loop, its
    probabilities. Runs offline; no Cloud session.

    Frames go in at 16 kHz, already resampled with the quick filter the stream
    would use (_i16). Handed 48 kHz frames, the stream resamples internally with
    rtc.AudioResampler QUICK, whose output jitters by up to 2 LSB from call to
    call (see _i16): five runs over the same input.wav (20260804-112511) gave
    three different probability tracks, every window off by up to 0.035,
    because the LSTM carries the jitter forward. At 16 kHz the stream does no
    resampling and repeats exactly, which a stored-probability bench needs.
    """

    PUSH_MS = 100

    def __init__(self):
        from livekit.agents import inference  # noqa: F401  (import check)

    def run(self, x_i16, src_rate):
        return asyncio.run(self._run(_i16(x_i16, src_rate, 16000), 16000))

    async def _run(self, x, rate):
        from livekit import rtc
        from livekit.agents import inference
        from livekit.agents import vad as agents_vad

        stream = inference.VAD().stream()
        probs: list[float] = []
        hop = 0.032
        step = max(1, int(rate * self.PUSH_MS / 1000))
        try:
            for i in range(0, len(x), step):
                chunk = x[i:i + step]
                stream.push_frame(rtc.AudioFrame(data=chunk.tobytes(), sample_rate=rate,
                                                 num_channels=1,
                                                 samples_per_channel=len(chunk)))
            stream.end_input()
            async for ev in stream:
                if ev.type is agents_vad.VADEventType.INFERENCE_DONE:
                    probs.append(float(ev.probability))
        finally:
            await stream.aclose()
        return np.asarray(probs, dtype=np.float32), hop, 0.0


# ----------------------------------------------------------------- FireRed


def _kaldi_cmvn(path):
    """Kaldi CMVN stats -> (means, inverse std), as FireRedVAD's CMVN class
    computes them (fireredvad/core/audio_feat.py @71352f2)."""
    import kaldiio

    stats = kaldiio.load_mat(str(path))
    dim = stats.shape[1] - 1
    count = stats[0, dim]
    mean = stats[0, :dim] / count
    var = np.maximum(stats[1, :dim] / count - mean * mean, 1e-20)
    return mean, 1.0 / np.sqrt(var)


def _fbank(x_i16, rate, **frame_opts):
    import kaldi_native_fbank as knf

    o = knf.FbankOptions()
    o.frame_opts.samp_freq = rate
    o.frame_opts.frame_length_ms = 25
    o.frame_opts.frame_shift_ms = 10
    o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = True
    o.mel_opts.num_bins = 80
    for k, v in frame_opts.items():
        setattr(o.frame_opts if hasattr(o.frame_opts, k) else o, k, v)
    fb = knf.OnlineFbank(o)
    fb.accept_waveform(rate, x_i16.astype(np.float32).tolist())
    n = fb.num_frames_ready
    if not n:
        return np.zeros((0, 80), dtype=np.float32)
    return np.vstack([fb.get_frame(i) for i in range(n)]).astype(np.float32)


class FireRed:
    """FireRedVAD (offline or streaming ONNX), kaldi fbank + CMVN, 10 ms frames.

    Features exactly as the publisher's AudioFeat (25/10 ms, 80 bins, no dither,
    snip_edges, int16-valued input). The streaming model is run over the whole
    sequence in one call, as the bake-off did. That is what live streaming
    gives, bit for bit: 20/50 ms waveform chunks into one OnlineFbank and
    10/20/50 ms of model input per call through the publisher's _with_cache
    graph, carrying its caches, match it with max |Δp| = 0 on three stored
    files, while resetting the caches each chunk moves it by up to 0.95
    (scripts/check_vad_parity.py --only firered, 2026-09-29). So there is no
    need to chunk here.
    """

    def __init__(self, onnx_path, cmvn_path):
        self.sess = _ort_session(onnx_path)
        self.mean, self.istd = _kaldi_cmvn(cmvn_path)

    def run(self, x_i16, src_rate):
        x = _i16(x_i16, src_rate, 16000)
        f = _fbank(x, 16000)
        if not len(f):
            return np.zeros(0, np.float32), 0.01, 0.0
        f = ((f - self.mean) * self.istd).astype(np.float32)[None]
        p = self.sess.run(None, {self.sess.get_inputs()[0].name: f})[0].reshape(-1)
        return np.asarray(p, dtype=np.float32), 0.01, 0.0


# -------------------------------------------------------- TEN / Earshot / WebRTC


class Ten:
    """TEN VAD (Agora) through its own pip package, 256-sample (16 ms) hops."""

    HOP = 256

    def __init__(self):
        from ten_vad import TenVad  # noqa: F401

    def run(self, x_i16, src_rate):
        from ten_vad import TenVad

        x = _i16(x_i16, src_rate, 16000)
        v = TenVad(self.HOP, 0.5)
        n = len(x) // self.HOP
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = float(v.process(np.ascontiguousarray(x[i * self.HOP:(i + 1) * self.HOP]))[0])
        return out, self.HOP / 16000, 0.0


class Earshot:
    """Earshot (pykeio), 256-sample (16 ms) frames at 16 kHz."""

    HOP = 256

    def __init__(self):
        import earshot  # noqa: F401

    def run(self, x_i16, src_rate):
        import earshot

        x = _i16(x_i16, src_rate, 16000)
        d = earshot.Detector()
        n = len(x) // self.HOP
        out = np.array([d.predict_i16(np.ascontiguousarray(x[i * self.HOP:(i + 1) * self.HOP]))
                        for i in range(n)], dtype=np.float32)
        return out, self.HOP / 16000, 0.0


class WebRtc:
    """WebRTC VAD: a binary decision per 30 ms frame, no probability."""

    def __init__(self, mode: int, rate: int):
        import webrtcvad  # noqa: F401

        self.mode, self.rate = int(mode), int(rate)

    def run(self, x_i16, src_rate):
        import webrtcvad

        x = _i16(x_i16, src_rate, self.rate)
        v = webrtcvad.Vad(self.mode)
        n_s = int(self.rate * 0.03)
        n = len(x) // n_s
        out = np.array([1.0 if v.is_speech(x[i * n_s:(i + 1) * n_s].tobytes(), self.rate)
                        else 0.0 for i in range(n)], dtype=np.float32)
        return out, 0.03, 0.0


# ---------------------------------------------------------------- vadonnx


class VadOnnx:
    """MarbleNet v2.0, FSMN-16k and pyannote segmentation-3.0, through the
    `vadonnx` package's own frontends (as the bake-off ran them) but with the
    ONNX file from models/vad/ — the same pinned revision vadonnx 0.1.0 would
    download, fetched and hash-checked by scripts/fetch_vads.py instead of
    landing unverified in a HuggingFace cache.

    The hop is the backend's own `frame_duration`; the bake-off derived it as
    duration / n_frames instead, which is the same number to within one
    partial frame.
    """

    def __init__(self, name: str, onnx_path):
        from vadonnx.registry import get_backend, get_spec

        spec = get_spec(name)
        self.model = get_backend(spec.backend)(str(onnx_path), spec.signature)

    def run(self, x_i16, src_rate):
        x = _f32(x_i16, src_rate, self.model.sample_rate)
        p = np.asarray(self.model.probabilities(x, sample_rate=self.model.sample_rate),
                       dtype=np.float32).reshape(-1)
        return np.clip(np.nan_to_num(p), 0, 1), float(self.model.frame_duration), 0.0


def _am_mvn(path):
    """FunASR's am.mvn (a kaldi nnet1 text: <AddShift> then <Rescale>)."""
    txt = open(path).read()

    def vec(tag):
        body = txt.split(tag, 1)[1]
        body = body.split("[", 1)[1].split("]", 1)[0]
        return np.array(body.split(), dtype=np.float32)

    return vec("<AddShift>"), vec("<Rescale>")


class Fsmn8k:
    """FunASR FSMN-VAD 8 kHz (publisher's quantised ONNX on ModelScope).

    Same frontend as the 16 kHz model vadonnx wraps (config.yaml: hamming,
    80 mels, 25/10 ms, LFR m=5 n=1), at fs 8000, with the model's own am.mvn.
    Speech probability = 1 - p(silence pdf 0), as vadonnx defines it for 16k.
    """

    def __init__(self, onnx_path, mvn_path):
        from vadonnx.backends import fsmn as vfsmn

        self._lfr = vfsmn._apply_lfr
        self.sess = _ort_session(onnx_path)
        self.shift, self.scale = _am_mvn(mvn_path)
        self.state_in = [i.name for i in self.sess.get_inputs()][1:]
        self.feat_in = self.sess.get_inputs()[0].name

    def run(self, x_i16, src_rate):
        x = _i16(x_i16, src_rate, 8000)
        f = _fbank(x, 8000, window_type="hamming", energy_floor=0.0)
        if not len(f):
            return np.zeros(0, np.float32), 0.01, 0.0
        f = ((self._lfr(f) + self.shift) * self.scale).astype(np.float32)
        feed = {self.feat_in: f[None]}
        for name in self.state_in:
            feed[name] = np.zeros((1, 128, 19, 1), dtype=np.float32)
        out = self.sess.run(None, feed)[0][0]
        return (1.0 - out[:, 0]).astype(np.float32), 0.01, 0.0


# ------------------------------------------------------------ ai-coustics


class AicVad:
    """ai-coustics' STANDALONE VAD model (aic-sdk 3.2, `aic_sdk.Vad`), own key.

    Stores the model's raw per-block probability, not the SDK's post-processed
    is_speech flag: the flag bakes in the SDK's own hold/min-duration logic,
    and the bench segments every model with one shared state machine instead.
    The published prediction lags its input by get_prediction_delay() samples
    (480 = 30 ms at 16 kHz for both models, measured 2026-09-29), so t0 is
    negative by that much.

    ai-coustics says to feed this VAD the original audio. The bench runs it on
    whatever source is ticked, so a row on an NC output is an experiment the
    vendor does not recommend.
    """

    def __init__(self, model_path):
        import aic_sdk as aic

        if not config.AIC_LICENSE_KEY:
            raise RuntimeError("AIC_LICENSE_KEY is not set in .env")
        self.model = aic.Model.from_file(str(model_path))
        self.cfg = aic.ProcessorConfig.optimal(self.model)

    def run(self, x_i16, src_rate):
        import aic_sdk as aic

        rate, bs = self.cfg.sample_rate, self.cfg.block_size
        x = _f32(x_i16, src_rate, rate)
        vad = aic.Vad(self.model, config.AIC_LICENSE_KEY, self.cfg)
        ctx = vad.get_context()
        delay = ctx.get_prediction_delay()
        n = len(x) // bs
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            vad.process(np.ascontiguousarray(x[i * bs:(i + 1) * bs]))
            out[i] = ctx.raw_vad_probability()
        return out, bs / rate, -delay / rate
