"""Reference-free scoring for candidate outputs.

- DNSMOS P.835 (Microsoft DNS-Challenge sig_bak_ovr.onnx): SIG (speech
  quality), BAK (background intrusiveness), OVRL — no clean reference needed.
- Gap-RMS: silero-VAD (conservative gating) finds confident no-speech windows
  on the RAW input; every candidate is measured in those same windows. The
  delta vs input, in dB, is "how much noise died in the pauses".
- WER vs an optional user-provided reference script (the only exact metric;
  see README on scoring reliability).

All scoring runs at PIPELINE_RATE (16 kHz mono float32).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import config

_DNSMOS_MODEL = config.PROJECT_ROOT / "models" / "dnsmos" / "sig_bak_ovr.onnx"
_SILERO_MODEL = (
    Path(__import__("livekit.plugins.silero", fromlist=["__file__"]).__file__).parent
    / "resources"
    / "silero_vad.onnx"
)

# ------------------------------------------------------------------ DNSMOS

_INPUT_LEN_S = 9.01
_FS = 16_000
# polynomial raw->MOS mappings from microsoft/DNS-Challenge dnsmos_local.py
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

_dnsmos_session = None


def _dnsmos(audio: np.ndarray) -> dict:
    """DNSMOS P.835 on 16 kHz float32 mono; averaged over 1 s hops."""
    global _dnsmos_session
    if _dnsmos_session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _dnsmos_session = ort.InferenceSession(str(_DNSMOS_MODEL), opts)

    seg_len = int(_INPUT_LEN_S * _FS)
    while len(audio) < seg_len:
        audio = np.concatenate([audio, audio])
    num_hops = max(1, int(np.floor(len(audio) / _FS) - _INPUT_LEN_S) + 1)
    sig, bak, ovr = [], [], []
    for idx in range(num_hops):
        seg = audio[idx * _FS : idx * _FS + seg_len]
        if len(seg) < seg_len:
            break
        raw = _dnsmos_session.run(
            None, {"input_1": seg.astype(np.float32)[np.newaxis, :]}
        )[0][0]
        sig.append(float(_P_SIG(raw[0])))
        bak.append(float(_P_BAK(raw[1])))
        ovr.append(float(_P_OVR(raw[2])))
    return {
        "sig": round(float(np.mean(sig)), 2),
        "bak": round(float(np.mean(bak)), 2),
        "ovrl": round(float(np.mean(ovr)), 2),
    }


# ------------------------------------------------------------- silero VAD

_VAD_WINDOW = 512  # samples @16k, ~32 ms
_GAP_PROB = 0.15  # window counts as a confident gap only under this
_MIN_GAP_S = 1.0
_EDGE_TRIM_S = 0.2

_vad_session = None


def _speech_probs(audio: np.ndarray) -> np.ndarray:
    """Per-512-sample-window speech probabilities (silero v5, streaming state)."""
    global _vad_session
    if _vad_session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _vad_session = ort.InferenceSession(str(_SILERO_MODEL), opts)

    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(_FS, dtype=np.int64)
    probs = []
    for i in range(0, len(audio) - _VAD_WINDOW + 1, _VAD_WINDOW):
        chunk = audio[i : i + _VAD_WINDOW].astype(np.float32)[np.newaxis, :]
        out, state = _vad_session.run(None, {"input": chunk, "state": state, "sr": sr})
        probs.append(float(out[0][0]))
    return np.array(probs)


def gap_windows(audio: np.ndarray) -> list[tuple[float, float]]:
    """Confident no-speech windows (seconds) on 16 kHz mono float32 audio.

    Conservative: prob < 0.15 sustained >= 1 s, 200 ms trimmed off each edge.
    Loud background *speech* keeps probabilities high, so heavy-babble audio
    yields few/no gaps — the metric abstains rather than lies (see README).
    """
    probs = _speech_probs(audio)
    win_s = _VAD_WINDOW / _FS
    gaps: list[tuple[float, float]] = []
    start = None
    for i, p in enumerate(list(probs) + [1.0]):  # sentinel closes a trailing gap
        if p < _GAP_PROB and start is None:
            start = i
        elif p >= _GAP_PROB and start is not None:
            g0, g1 = start * win_s + _EDGE_TRIM_S, i * win_s - _EDGE_TRIM_S
            if g1 - g0 >= _MIN_GAP_S - 2 * _EDGE_TRIM_S:
                gaps.append((round(g0, 2), round(g1, 2)))
            start = None
    return gaps


def gap_rms_db(audio: np.ndarray, gaps: list[tuple[float, float]]) -> float | None:
    """RMS (dBFS) inside the given windows; None if no coverage."""
    parts = [audio[int(g0 * _FS) : int(g1 * _FS)] for g0, g1 in gaps]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    x = np.concatenate(parts)
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    return round(20 * np.log10(max(rms, 1e-9)), 1)


# ------------------------------------------------------------------- WER


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> dict | None:
    """Word error rate via word-level edit distance. None if reference empty."""
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    if not ref:
        return None
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    errors = int(d[len(ref), len(hyp)])
    return {"wer": round(errors / len(ref), 3), "errors": errors, "ref_words": len(ref)}


# ------------------------------------------------------- measured bandwidth


_BAND_FLOOR_DB = 40.0  # a codec cliff is 50+ dB down; mic HF hiss stays inside 35


def bandwidth_hz(audio: np.ndarray, rate: int) -> float | None:
    """Highest frequency still carrying real signal — the band's rolloff edge.

    The file's sample rate says nothing about its band: LiveKit hands every track
    over at 48 kHz, so an 8 kHz G.711 call arrives as a 48 kHz wav with ~3.4 kHz
    of content in it. This measures what's actually there — ~4 kHz for a PSTN
    call, ~7 kHz for a wideband trunk, 14 kHz+ for a browser mic — which decides
    whether a narrowband model was given a fair fight.

    Deliberately *not* "the frequency holding 99% of the energy": speech puts
    ~98% of its energy below 1 kHz, so that measure sits right on a knee and
    swings by 8 kHz on a rounding error. Instead: average the spectrum over the
    loud frames only (so room tone can't set the answer), take the 300-1000 Hz
    level as the in-band reference, and walk down from Nyquist for the first
    frequency within _BAND_FLOOR_DB of it. A codec cutoff is a 50+ dB cliff, so
    the two cases separate by a wide margin rather than a tuned threshold.
    """
    n_fft = 1024
    hop = n_fft // 2
    if len(audio) < n_fft:
        return None
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.array([audio[i : i + n_fft] for i in range(0, len(audio) - n_fft, hop)])
    if not len(frames):
        return None
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    loud = rms >= max(rms.max() * 0.2, 1e-6)  # within ~14 dB of the loudest frame
    if not loud.any():
        return None
    power = (np.abs(np.fft.rfft(frames[loud] * win, axis=1)) ** 2).mean(axis=0)
    hz = np.arange(len(power)) * rate / n_fft
    db = 10 * np.log10(np.maximum(power, 1e-20))
    # smooth over ~5 bins so a single noisy bin can't set the edge
    db = np.convolve(db, np.ones(5) / 5, mode="same")
    in_band = (hz >= 300) & (hz <= 1000)
    if not in_band.any():
        return None
    ref = float(np.median(db[in_band]))
    above = np.flatnonzero(db >= ref - _BAND_FLOOR_DB)
    return round(float(hz[above[-1]])) if len(above) else None


# ------------------------------------------------------------- entrypoints


def load_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    import soxr

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != _FS:
        mono = soxr.resample(mono, rate, _FS).astype(np.float32)
    return mono


def score_input(input_wav: Path) -> dict:
    """Once per run: DNSMOS + VAD gaps + noise floor + measured band of the raw input."""
    import soundfile as sf

    audio = load_16k(input_wav)
    gaps = gap_windows(audio)
    # measured at the file's own rate, not resampled: a 16 kHz view would clip
    # every wideband recording to 8 kHz and hide the difference we're after
    native, native_rate = sf.read(input_wav, dtype="float32", always_2d=True)
    return {
        "dnsmos": _dnsmos(audio),
        "gaps": gaps,
        "gap_total_s": round(sum(g1 - g0 for g0, g1 in gaps), 2),
        "gap_rms_db": gap_rms_db(audio, gaps),
        "bandwidth_hz": bandwidth_hz(native.mean(axis=1), native_rate),
        "file_rate": native_rate,
    }


def score_output(output_wav: Path, input_scores: dict,
                 reference_script: str = "", transcript: str = "") -> dict:
    """Per candidate: DNSMOS, gap-RMS delta vs input, WER vs script."""
    audio = load_16k(output_wav)
    # outputs are all 16 kHz, so this caps at 8 kHz — enough to catch a model
    # that band-limits what it hands to STT
    scores: dict = {"dnsmos": _dnsmos(audio), "bandwidth_hz": bandwidth_hz(audio, _FS)}
    gaps = input_scores.get("gaps") or []
    out_db = gap_rms_db(audio, gaps)
    in_db = input_scores.get("gap_rms_db")
    scores["gap_rms_db"] = out_db
    scores["noise_reduction_db"] = (
        round(in_db - out_db, 1) if (out_db is not None and in_db is not None) else None
    )
    if reference_script.strip():
        scores["wer"] = wer(reference_script, transcript)
    return scores
