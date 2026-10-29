"""The one state machine every VAD in the bench is segmented with.

A line-for-line replica of the livekit-agents stream loop that turns per-window
probabilities into turns: `livekit/plugins/silero/vad.py` (1.8.3,
`VADStream._main_task`) and `livekit/agents/inference/vad.py` (`_VADStream`) run
the same loop, differing only in the ExpFilter the plugin applies first. It is a
replica rather than a call because those loops are welded to an rtc frame
stream and a live model; the bench needs to re-segment STORED probabilities
with new parameters without re-running inference, which is what makes a
parameter sweep over twenty runs affordable.

Replicated exactly, including the parts that look incidental:

- speech while `p >= activation`, or `p > deactivation` once speaking
  (hysteresis; strict `>` on the way out, as livekit has it);
- durations are ACCUMULATED floats, `+= hop` per window, compared with `>=`.
  Counting windows instead is not equivalent: 10 hops of 0.01 s sum to
  0.09999999999999999, so a 0.1 s min_silence (or min_speech) needs 11
  windows in livekit where a counter would stop at 10;
- START fires on the window where accumulated speech reaches
  min_speech_duration; END on the window where accumulated silence reaches
  min_silence_duration.

Span bounds use the same correction nc_bench/vad.py documents for the plugin's
events (END's timestamp is `silence_duration` late; START's is
`speech_duration` late), so a span is the speech itself, not when it was
declared:

    start = t(START) - speech accumulated at START
    end   = t(END)   - silence accumulated at END

prefix_padding_duration does not move a span. livekit uses it only to back-date
the audio buffer it hands downstream, which here is stt.merge_turns' job.

`t0_s` shifts the whole window grid. Window k covers
[t0 + k*hop, t0 + (k+1)*hop): 0 for a model whose first window starts at the
file's first sample, negative for a detector whose output lags its input (the
ai-coustics SDK VAD reports its own delay) or whose frames were cut from a
stream that began before t=0 (ticket 1's cloud-replay flags).
"""

from __future__ import annotations

import numpy as np

# livekit's legacy plugin smooths with ExpFilter(alpha=0.35) once per 32 ms
# window. Production's inference.VAD does not smooth at all (livekit PR #5800),
# so smoothing is opt-in here and off by default.
PLUGIN_ALPHA = 0.35
PLUGIN_HOP_S = 0.032


def smooth(probs: np.ndarray, hop_s: float, alpha: float = PLUGIN_ALPHA) -> np.ndarray:
    """livekit's ExpFilter, as the Silero plugin applies it.

    `alpha` is per 32 ms window. For other hops it is scaled to the same time
    constant (alpha ** (hop / 32 ms)), as the PR-42 bake-off did, so "smoothed"
    means the same thing for a 10 ms model as for Silero. At a 32 ms hop the
    scaling is the identity and this is the plugin's filter exactly: the first
    sample passes through, then `a * prev + (1 - a) * p`.
    """
    p = np.asarray(probs, dtype=np.float64)
    if not len(p):
        return p
    a = alpha ** (float(hop_s) / PLUGIN_HOP_S)
    out = np.empty_like(p)
    f = p[0]
    out[0] = f
    for i in range(1, len(p)):
        f = a * f + (1 - a) * p[i]
        out[i] = f
    return out


def segment(
    probs,
    hop_s: float,
    *,
    activation_threshold: float,
    deactivation_threshold: float,
    min_speech_duration: float,
    min_silence_duration: float,
    duration_s: float | None = None,
    t0_s: float = 0.0,
    smoothing: float | None = None,
) -> list[list[float]]:
    """Speech spans [[start, end], ...] in seconds, rounded to the millisecond.

    `duration_s` closes a span still open when the probabilities run out (the
    talker was mid-word when the recording stopped), exactly as vad.analyze()
    does; without it the open span ends at the last window. `smoothing` is an
    ExpFilter alpha per 32 ms (see smooth()), or None for raw probabilities.
    """
    p = np.asarray(probs, dtype=np.float64)
    if smoothing:
        p = smooth(p, hop_s, smoothing)
    act, deact = float(activation_threshold), float(deactivation_threshold)
    min_sp, min_sil = float(min_speech_duration), float(min_silence_duration)
    hop = float(hop_s)

    segments: list[list[float]] = []
    speaking = False
    speech_acc = 0.0   # livekit: speech_threshold_duration
    sil_acc = 0.0      # livekit: silence_threshold_duration
    ts = 0.0           # livekit: pub_timestamp, time at the END of this window
    start = 0.0
    for v in p:
        ts += hop
        if v >= act or (speaking and v > deact):
            speech_acc += hop
            sil_acc = 0.0
            if not speaking and speech_acc >= min_sp:
                speaking = True
                start = ts - speech_acc
        else:
            sil_acc += hop
            speech_acc = 0.0
            if speaking and sil_acc >= min_sil:
                speaking = False
                segments.append([start, ts - sil_acc])

    end_all = float(duration_s) if duration_s else ts + t0_s
    if speaking:
        segments.append([start, end_all - t0_s])

    out: list[list[float]] = []
    for s, e in segments:
        s, e = s + t0_s, e + t0_s
        s = max(0.0, s)
        e = min(e, end_all)
        if e > s:
            out.append([round(s, 3), round(e, 3)])
    return out


def summary(segments, duration_s: float) -> dict:
    """The same headline numbers vad.analyze() stores for a span set."""
    speech_s = sum(e - s for s, e in segments)
    return {
        "n": len(segments),
        "speech_s": round(speech_s, 2),
        "duration_s": round(float(duration_s), 2),
        "coverage": round(speech_s / duration_s, 4) if duration_s else 0.0,
    }
