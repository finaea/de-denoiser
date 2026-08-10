"""Silero VAD over a finished wav, driven through the same livekit-agents plugin
a production voice agent runs.

Why this belongs in an NC bench: production never hands a whole call to the STT.
LiveKit's VAD cuts the call into turns and each turn becomes one /recognize POST
(src/components/stt/local_stt.py). So what a candidate does to the *segmentation*
matters as much as what it does to the audio — a chain that shaves a word's onset
loses that word before the recogniser ever sees it, and one that leaves enough
residual noise to trip activation adds turns that were never spoken.

Driven through livekit.plugins.silero rather than re-implemented over the raw
ONNX. The model file is identical either way (scoring.py already reads it from
the plugin), but the *state machine* is what turns per-window probabilities into
turns — activation/deactivation hysteresis, min-speech rejection, min-silence
bridging, prefix padding — and a re-implementation that drifted from it would
answer a question nobody asked.

Segment bounds come from END_OF_SPEECH, and the correction matters:
`ev.timestamp` is when the end was *declared*, which is `ev.silence_duration`
(~min_silence_duration) AFTER the speech actually stopped. Taking it as the end
shifts every span uniformly late — measured at +0.576 s, i.e. most of a word.
`ev.speech_duration` is exactly `end - start`, so:

    end   = ev.timestamp - ev.silence_duration
    start = end - ev.speech_duration

scripts/check_vad.py exists specifically to keep that offset from creeping back:
a shifted highlight looks plausible on every waveform while sitting a fixed
distance from the audio it marks.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from livekit import rtc
from livekit.agents import vad as agents_vad

from . import config

_PUSH_MS = 100  # frame size we feed; the plugin re-windows internally

# Ranges the UI is allowed to ask for. Thresholds are probabilities; the
# durations are bounded well past anything useful so a fat-fingered value fails
# here instead of producing a plausible-looking but nonsense span set.
LIMITS = {
    "activation_threshold": (0.05, 0.95),
    "deactivation_threshold": (0.05, 0.95),
    "min_speech_duration": (0.0, 2.0),
    "min_silence_duration": (0.05, 5.0),
    "prefix_padding_duration": (0.0, 2.0),
}

# The only two rates the model has weights for — it takes `sr` as a graph input
# and uses a different window/context per rate (512/64 at 16k, 256/32 at 8k).
# Unlike the NC models this is real dual-rate support, not one graph plus
# resampling. Anything else raises inside the plugin.
SUPPORTED_RATES = (8000, 16000)

# Inference rate per recording source. A phone leg carries ~3.5 kHz of real
# content no matter what container rate it arrives in, so 8 kHz inference matches
# the band instead of feeding the model an empty top half; mic and file sources
# are genuinely wideband.
#
# NOTE this is the one place nc_bench deliberately diverges from a typical
# production config, which pins 16 kHz for every source. A phone run measured at 8 kHz is an experiment,
# not a mirror of production — the panel says so when the rate is not 16000.
SOURCE_RATES = {"phone": 8000, "web": 16000, "upload": 16000}
DEFAULT_RATE = 16000

_cache: dict[tuple, object] = {}


def defaults(source: str | None = None) -> dict:
    """.env thresholds plus the inference rate implied by `source`.

    Thresholds come from .env; see config.py for why they sit tighter than
    the livekit-agents defaults. The rate is NOT an .env value because a single setting cannot
    express "8 k for a phone leg, 16 k for a mic".
    """
    return {
        "activation_threshold": config.VAD_ACTIVATION_THRESHOLD,
        "deactivation_threshold": config.VAD_DEACTIVATION_THRESHOLD,
        "min_speech_duration": config.VAD_MIN_SPEECH_DURATION,
        "min_silence_duration": config.VAD_MIN_SILENCE_DURATION,
        "prefix_padding_duration": config.VAD_PREFIX_PADDING_DURATION,
        "sample_rate": SOURCE_RATES.get(source or "", DEFAULT_RATE),
    }


def clean(p: dict | None, source: str | None = None) -> dict:
    """Defaults overlaid with whatever the caller supplied, clamped to LIMITS.

    Unparseable or out-of-range values fall back to the default for that key
    rather than raising: a bad threshold in one field should not throw away a
    whole re-run. `sample_rate` is an enum, not a range — an unsupported value
    falls back rather than clamping, because 12000 is not "nearly 16000", it is
    a rate the model has no weights for.
    """
    out = defaults(source)
    for key, (lo, hi) in LIMITS.items():
        if not p or key not in p:
            continue
        try:
            out[key] = max(lo, min(hi, float(p[key])))
        except (TypeError, ValueError):
            pass
    if p and "sample_rate" in p:
        try:
            rate = int(p["sample_rate"])
            if rate in SUPPORTED_RATES:
                out["sample_rate"] = rate
        except (TypeError, ValueError):
            pass
    return out


# --------------------------------------------------- vs hand-marked speech

TRUTH_FRAME = 0.01  # 10 ms


def _mask(spans, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in spans or []:
        a = max(0, int(float(s) / TRUTH_FRAME))
        b = min(n, int(-(-float(e) // TRUTH_FRAME)))
        if b > a:
            m[a:b] = True
    return m


def score_spans(truth, segments, duration_s) -> dict | None:
    """Frame-level agreement between hand-marked speech and a VAD's spans.

    Masks on 10 ms frames rather than interval algebra: 12 000 frames for a
    two-minute call costs nothing, and a mask is obviously correct where the
    merge/clip edge cases of interval arithmetic are exactly where this kind of
    code goes wrong.

    `agree` is the headline — the share of the call where the VAD and the human
    say the same thing. The two error terms are kept separate because they mean
    opposite things: `miss_s` is speech a chain hid from the recogniser, `fa_s` is
    turns nobody spoke.
    """
    if not truth or segments is None or not duration_s:
        return None
    n = max(1, round(float(duration_s) / TRUTH_FRAME))
    t, v = _mask(truth, n), _mask(segments, n)
    tp = int((t & v).sum())
    fn = int((t & ~v).sum())
    fp = int((~t & v).sum())
    tn = int((~t & ~v).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "agree": round((tp + tn) / n, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "miss_s": round(fn * TRUTH_FRAME, 2),
        "fa_s": round(fp * TRUTH_FRAME, 2),
        "marked_s": round((tp + fn) * TRUTH_FRAME, 2),
    }


def score_run(meta: dict) -> None:
    """(Re)score every candidate against meta["truth_spans"], in place.

    Has to be called wherever EITHER input changes — the marks or the spans —
    since a stored score is only meaningful for the pair that produced it.
    """
    truth = meta.get("truth_spans") or []
    dur = float((meta.get("input") or {}).get("duration_s") or 0)
    if not dur:
        dur = float((meta.get("input_vad") or {}).get("duration_s") or 0)
    for c in meta.get("candidates") or []:
        v = c.get("vad") or {}
        c["truth_score"] = (
            None if v.get("error") else score_spans(truth, v.get("segments"), dur)
        )


def _load(p: dict):
    """One plugin instance per distinct parameter set, cached.

    Keyed rather than rebuilt because a re-run over ~500 files would otherwise
    pay the ~100 ms model load every time, and keyed rather than mutated because
    the options are baked in at load().
    """
    key = tuple(sorted(p.items()))
    if key not in _cache:
        from livekit.plugins import silero

        _cache[key] = silero.VAD.load(**p)
    return _cache[key]


async def analyze(wav, p: dict | None = None, source: str | None = None) -> dict | None:
    """Speech spans in `wav`, or None when VAD is switched off.

    Fed at the file's own rate and left to the plugin to resample down to
    p["sample_rate"]: a phone leg reaches production as 48 kHz frames and the
    plugin resamples there too, so feeding native rate is what production sees.
    """
    if not config.VAD_ENABLED:
        return None
    p = clean(p, source)
    data, rate = sf.read(wav, dtype="int16", always_2d=True)
    mono = np.ascontiguousarray(data[:, 0])
    duration_s = len(mono) / rate if rate else 0.0

    stream = _load(p).stream()
    step = max(1, int(rate * _PUSH_MS / 1000))
    try:
        for i in range(0, len(mono), step):
            chunk = mono[i : i + step]
            stream.push_frame(
                rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=rate,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
            )
        stream.end_input()

        segments: list[list[float]] = []
        open_start: float | None = None
        async for ev in stream:
            if ev.type is agents_vad.VADEventType.START_OF_SPEECH:
                # already min_speech_duration in when it fires, hence the subtraction
                open_start = max(0.0, float(ev.timestamp) - float(ev.speech_duration))
            elif ev.type is agents_vad.VADEventType.END_OF_SPEECH:
                end = float(ev.timestamp) - float(ev.silence_duration)
                start = max(0.0, end - float(ev.speech_duration))
                segments.append([round(start, 3), round(min(end, duration_s or end), 3)])
                open_start = None
        # A turn still open when the audio ran out: the talker was mid-word when
        # the recording stopped, so no END_OF_SPEECH ever fires. Reading only END
        # events silently drops that whole final turn — measured on a 13 s web
        # call that is 68% speech by probability and reported ZERO spans. Common
        # enough to matter: you stop the recorder while still talking.
        if open_start is not None and duration_s > open_start:
            segments.append([round(open_start, 3), round(duration_s, 3)])
    finally:
        await stream.aclose()

    speech_s = sum(e - s for s, e in segments)
    return {
        "segments": segments,
        "n": len(segments),
        "speech_s": round(speech_s, 2),
        "duration_s": round(duration_s, 2),
        # painted as a fraction of the canvas, so the viewer never needs the
        # audio element's metadata to place a span
        "coverage": round(speech_s / duration_s, 4) if duration_s else 0.0,
        # stored per file: a span set is meaningless without the thresholds that
        # produced it, and the UI can re-run with different ones at any time
        "params": p,
        "file_rate": rate,
    }
