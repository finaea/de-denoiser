"""Does the VAD bench's own logic do what its docs say, on inputs small enough to check by hand?

Pure: hand-built probability tracks and spans, temp dirs, no model, no network,
no STT, no licence key, nothing under data/runs/. Covers

- the segmenter (nc_bench/vads/segmenter.py), a replica of the livekit-agents
  VAD loop: min_speech, min_silence, hysteresis, the float-accumulation trap,
  open-at-end spans, the t0_s grid offset, smoothing;
- event scoring (nc_bench/vads/events.py): PR-42 misses, PR-54 phantoms,
  fragmentation, onset lag, strict overlap, and no-truth -> None;
- the pure helpers in nc_bench/vads/bench.py and the registry expansion.

    .venv/bin/python scripts/check_vad_bench.py
"""

import json
import multiprocessing
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import vad, vads  # noqa: E402
from nc_bench.vads import bench  # noqa: E402
from nc_bench.vads.events import EVENT_KEYS, score_events  # noqa: E402
from nc_bench.vads.segmenter import PLUGIN_ALPHA, segment, smooth, summary  # noqa: E402

HOP = 0.01


def track(*runs, hop_value=None):
    """track((1, 20), (0, 5), ...) -> probabilities: `value` repeated `n` times."""
    return np.concatenate([np.full(n, v, dtype=np.float64) for v, n in runs])


def seg(probs, hop=HOP, act=0.5, deact=0.5, ms=0.045, msil=0.1, **kw):
    return segment(probs, hop, activation_threshold=act, deactivation_threshold=deact,
                   min_speech_duration=ms, min_silence_duration=msil, **kw)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def spans_eq(got, want):
    return len(got) == len(want) and all(
        approx(g[0], w[0]) and approx(g[1], w[1]) for g, w in zip(got, want))


# ---------------------------------------------------------------- segmenter


def check_segmenter() -> None:
    # min_speech: 4 windows (0.04 s) < 0.045 opens nothing, 5 (0.05 s) opens
    assert seg(track((0, 10), (1, 4), (0, 30))) == []
    assert spans_eq(seg(track((0, 10), (1, 5), (0, 30))), [[0.10, 0.15]])
    print("OK   min_speech: a burst shorter than min_speech opens nothing")

    # a rejected burst also resets: two short bursts do not add up
    assert seg(track((1, 3), (0, 1), (1, 3), (0, 30))) == []
    print("OK   min_speech: short bursts separated by silence do not accumulate")

    # min_silence: a 5-window gap (0.05 s) < 0.1 s bridges, a 15-window gap splits
    got = seg(track((1, 20), (0, 5), (1, 20), (0, 30)))
    assert spans_eq(got, [[0.0, 0.45]]), got
    got = seg(track((1, 20), (0, 15), (1, 20), (0, 30)))
    assert spans_eq(got, [[0.0, 0.20], [0.35, 0.55]]), got
    print("OK   min_silence: a short gap keeps one span, a long one splits it")

    # hysteresis: act 0.6 / deact 0.3
    kw = dict(act=0.6, deact=0.3)
    got = seg(track((0.7, 10), (0.4, 10), (0.7, 10), (0, 30)), **kw)
    assert spans_eq(got, [[0.0, 0.30]]), got                   # 0.4 keeps it open
    assert seg(track((0.4, 30), (0, 30)), **kw) == []          # ...but never opens one
    assert seg(track((0.59, 30), (0, 30)), **kw) == []         # opening is >= act
    assert spans_eq(seg(track((0.6, 30), (0, 30)), **kw), [[0.0, 0.30]])
    # on the way out it is strict: p == deact counts as silence
    got = seg(track((0.7, 10), (0.3, 20), (0, 20)), **kw)
    assert spans_eq(got, [[0.0, 0.10]]), got
    got = seg(track((0.7, 10), (0.31, 20), (0, 20)), **kw)
    assert spans_eq(got, [[0.0, 0.30]]), got
    print("OK   hysteresis: between deact and act holds a span but never opens one; >= act in, > deact out")

    # the float trap. Accumulating 0.01 with += is NOT what the docstring says for
    # 0.3 (30 hops give 0.3000000000000001, which passes >= 0.3), but it IS true
    # for 0.1: 10 hops give 0.09999999999999999, so min_silence 0.1 needs 11.
    def accumulate(n):
        acc = 0.0
        for _ in range(n):
            acc += HOP
        return acc

    assert accumulate(10) < 0.1 <= accumulate(11), (accumulate(10), accumulate(11))
    # 10 silent windows do NOT satisfy min_silence 0.1, so the gap is bridged...
    got = seg(track((1, 50), (0, 10), (1, 50), (0, 40)), msil=0.1)
    assert spans_eq(got, [[0.0, 1.10]]), got
    # ...11 do, and the span ends where the silence began, not where it was declared
    got = seg(track((1, 50), (0, 11), (1, 50), (0, 40)), msil=0.1)
    assert spans_eq(got, [[0.0, 0.50], [0.61, 1.11]]), got
    print("OK   float accumulation: hop 0.01 / min_silence 0.1 needs 11 silent windows, not 10")

    # the ticket's example (0.3 -> 31 windows) does not reproduce with sequential +=
    assert accumulate(30) >= 0.3, accumulate(30)
    got = seg(track((1, 50), (0, 30), (1, 50), (0, 40)), msil=0.3)
    assert spans_eq(got, [[0.0, 0.50], [0.80, 1.30]]), got
    print("OK   float accumulation: min_silence 0.3 at hop 0.01 splits at 30 windows (pins actual behaviour)")

    # same trap on the speech side: 10 windows of 0.01 do not reach min_speech 0.1
    assert seg(track((1, 10), (0, 40)), ms=0.1) == []
    assert spans_eq(seg(track((1, 11), (0, 40)), ms=0.1), [[0.0, 0.11]])
    print("OK   float accumulation: same off-by-one on min_speech 0.1")

    # span bounds are the speech, not the declaration: START back-dated by the
    # accumulated speech, END by the accumulated silence
    got = seg(track((0, 7), (1, 20), (0, 40)))
    assert spans_eq(got, [[0.07, 0.27]]), got
    print("OK   START/END timing correction: span is the speech itself")

    # open at end: without duration_s it ends at the last window, with it at duration_s
    p = track((0, 10), (1, 30))
    assert spans_eq(seg(p), [[0.10, 0.40]])
    assert spans_eq(seg(p, duration_s=0.5), [[0.10, 0.5]])
    assert spans_eq(seg(p, duration_s=0.37), [[0.10, 0.37]])   # clamped to the file
    print("OK   open-at-end span is closed at duration_s (and clamped to it)")

    # t0_s: a negative offset moves spans earlier; positive later
    p = track((0, 10), (1, 20), (0, 40))
    assert spans_eq(seg(p, t0_s=-0.05), [[0.05, 0.25]])
    assert spans_eq(seg(p, t0_s=0.05), [[0.15, 0.35]])
    # clamps at 0 rather than going negative
    assert spans_eq(seg(p, t0_s=-0.15), [[0.0, 0.15]])
    # a span that ends before 0 disappears
    assert seg(track((1, 10), (0, 40)), t0_s=-0.2) == []
    # an open span with t0 and duration_s still closes at duration_s
    assert spans_eq(seg(track((0, 10), (1, 30)), t0_s=-0.05, duration_s=0.5), [[0.05, 0.5]])
    # and one running past duration_s is cut back to it
    assert spans_eq(seg(p, t0_s=0.1, duration_s=0.35), [[0.2, 0.35]])
    print("OK   t0_s shifts the window grid; negative shifts earlier, clamped to [0, duration]")

    # rounding to 3 decimals
    got = seg(track((1, 3)), hop=0.0123456, ms=0.01, msil=0.1)
    assert got == [[0.0, 0.037]], got
    print("OK   spans are rounded to the millisecond")

    # degenerate input
    assert seg(np.array([])) == []
    assert seg(track((0, 50))) == []
    print("OK   empty and all-silence tracks give no spans")

    # smoothing: ExpFilter, first sample passes through then a*prev + (1-a)*p
    p = np.array([0.0, 1.0, 1.0, 0.0])
    s = smooth(p, 0.032)
    a = PLUGIN_ALPHA
    want = [0.0, (1 - a), a * (1 - a) + (1 - a), a * (a * (1 - a) + (1 - a))]
    assert all(approx(x, y) for x, y in zip(s, want)), (s, want)
    assert approx(smooth(np.array([0.7]), 0.032)[0], 0.7)
    assert len(smooth(np.array([]), 0.032)) == 0
    # alpha scaled by hop: 10 ms is alpha ** (10/32)
    a10 = PLUGIN_ALPHA ** (0.010 / 0.032)
    s10 = smooth(np.array([0.0, 1.0]), 0.010)
    assert approx(s10[1], 1 - a10), s10
    # a one-window spike above act is removed by smoothing, kept raw
    spike = np.array([0.0] * 5 + [0.6] + [0.0] * 5)
    raw = seg(spike, hop=0.032, ms=0.01, msil=0.05)
    smoothed = seg(spike, hop=0.032, ms=0.01, msil=0.05, smoothing=PLUGIN_ALPHA)
    assert len(raw) == 1 and smoothed == [], (raw, smoothed)
    print("OK   smoothing: ExpFilter alpha 0.35 per 32 ms, hop-scaled, first sample passes through")

    # summary
    sm = summary([[0.0, 1.0], [2.0, 2.5]], 10.0)
    assert sm == {"n": 2, "speech_s": 1.5, "duration_s": 10.0, "coverage": 0.15}, sm
    assert summary([], 0)["coverage"] == 0.0
    print("OK   summary: n / speech_s / duration_s / coverage")


# ------------------------------------------------------------------- events


def check_events() -> None:
    # perfect match
    ev = score_events([[1, 2], [5, 6]], [[1, 2], [5, 6]])
    assert (ev["turns"], ev["hit"], ev["turn_recall"]) == (2, 2, 1.0), ev
    assert (ev["missed_n"], ev["missed_s"], ev["phantom_n"], ev["phantom_s"]) == (0, 0, 0, 0), ev
    assert ev["onset_lag_med_s"] == 0 and ev["onset_lag_max_s"] == 0 and ev["frag"] == 1.0, ev
    print("OK   events: perfect agreement is recall 1, no misses, no phantoms, lag 0, frag 1")

    # PR-42: one quiet utterance never opened
    ev = score_events([[1, 2], [5, 5.4], [8, 9]], [[1, 2], [8, 9]])
    assert ev["hit"] == 2 and ev["missed_n"] == 1, ev
    assert ev["missed"] == [[5, 5.4]] and approx(ev["missed_s"], 0.4), ev
    assert approx(ev["turn_recall"], 0.6667, 1e-4), ev
    assert ev["phantom_n"] == 0, ev
    print("OK   events: PR-42-style missed turn is listed, summed and lowers recall")

    # PR-54: a 96 ms click that touches no marked turn
    ev = score_events([[1, 2]], [[1, 2], [3.0, 3.096]])
    assert ev["phantom_n"] == 1 and ev["phantoms"] == [[3.0, 3.096]], ev
    assert approx(ev["phantom_s"], 0.096), ev
    assert ev["missed_n"] == 0 and ev["turn_recall"] == 1.0, ev
    print("OK   events: PR-54-style phantom click is counted and summed")

    # a turn split in two -> frag 2.0; averaged over hit turns only
    ev = score_events([[1, 3]], [[1, 1.5], [2, 3]])
    assert ev["frag"] == 2.0 and ev["hit"] == 1 and ev["phantom_n"] == 0, ev
    ev = score_events([[1, 3], [5, 6], [8, 9]], [[1, 1.5], [2, 3], [5, 6]])
    assert ev["frag"] == 1.5 and ev["missed_n"] == 1, ev      # (2 + 1) / 2 hit turns, the miss excluded
    print("OK   events: a turn cut in two has frag 2.0; misses are excluded from the mean")

    # late onset is a positive lag, median/max over hit turns only
    ev = score_events([[1, 2]], [[1.3, 2]])
    assert approx(ev["onset_lag_med_s"], 0.3) and approx(ev["onset_lag_max_s"], 0.3), ev
    ev = score_events([[1, 2], [4, 5], [7, 8], [10, 11]], [[1.1, 2], [4.3, 5], [7.2, 8]])
    assert approx(ev["onset_lag_med_s"], 0.2) and approx(ev["onset_lag_max_s"], 0.3), ev
    assert ev["missed_n"] == 1, ev
    # lag is the first overlapping segment's start when the turn is split
    ev = score_events([[1, 3]], [[1.2, 1.5], [2, 3]])
    assert approx(ev["onset_lag_med_s"], 0.2), ev
    print("OK   events: late onset is a positive lag (median / max over hit turns)")

    # strict overlap: a segment that only touches an edge does not count
    ev = score_events([[1, 2]], [[0.5, 1.0]])
    assert ev["hit"] == 0 and ev["missed_n"] == 1 and ev["phantom_n"] == 1, ev
    ev = score_events([[1, 2]], [[2.0, 2.5]])
    assert ev["hit"] == 0 and ev["missed_n"] == 1 and ev["phantom_n"] == 1, ev
    ev = score_events([[1, 2]], [[0.5, 1.001]])
    assert ev["hit"] == 1 and ev["phantom_n"] == 0, ev            # 1 ms of overlap does
    print("OK   events: touching edges do not overlap (strict), 1 ms of real overlap does")

    # marked speech but nothing detected: recall 0, and lag / frag are None, not 0
    ev = score_events([[1, 2]], [])
    assert ev["turn_recall"] == 0.0 and ev["missed_n"] == 1 and ev["phantom_n"] == 0, ev
    assert ev["onset_lag_med_s"] is None and ev["onset_lag_max_s"] is None and ev["frag"] is None, ev
    print("OK   events: nothing detected -> recall 0; onset lag / frag are None (no onset to be late on)")

    # no truth -> every metric None, never 0
    for truth, segs in (([], [[1, 2]]), (None, [[1, 2]]), ([], []), ([[1, 2]], None)):
        ev = score_events(truth, segs)
        assert set(ev) == set(EVENT_KEYS), ev
        assert all(v is None for v in ev.values()), (truth, segs, ev)
    print("OK   events: no truth spans -> every metric None, never 0")

    # inputs are not required to be sorted
    ev = score_events([[5, 6], [1, 2]], [[5, 6], [1, 2]])
    assert ev["hit"] == 2 and ev["frag"] == 1.0, ev
    print("OK   events: unsorted truth / segments are handled")


# -------------------------------------------------------------------- bench


def check_params() -> None:
    meta = {"vad_params": {"activation_threshold": 0.5, "deactivation_threshold": 0.35,
                           "min_speech_duration": 0.05, "min_silence_duration": 0.55,
                           "prefix_padding_duration": 0.5}}
    params = bench.clean_params(None, meta)
    assert params["activation_threshold"] == 0.5 and params["deactivation_threshold"] == 0.35
    assert params["smoothing"] is False

    # clamped to vad.LIMITS, junk ignored, smoothing coerced
    p = bench.clean_params({"activation_threshold": 5, "min_silence_duration": 0,
                            "min_speech_duration": "abc", "smoothing": "false"}, meta)
    lo, hi = vad.LIMITS["activation_threshold"]
    assert p["activation_threshold"] == hi
    assert p["min_silence_duration"] == vad.LIMITS["min_silence_duration"][0]
    assert p["min_speech_duration"] == 0.05                      # unparseable -> kept
    assert p["smoothing"] is False
    assert bench.clean_params({"smoothing": True}, meta)["smoothing"] is True
    assert bench.clean_params({"smoothing": "0"}, meta)["smoothing"] is False
    # a stored bench run's params are the base for the next call
    meta2 = dict(meta, vad_bench={"params": dict(params, min_silence_duration=0.9)})
    assert bench.clean_params({"activation_threshold": 0.6}, meta2)["min_silence_duration"] == 0.9
    print("OK   clean_params: defaults overlaid, clamped to vad.LIMITS, junk ignored")

    # clean_thresholds
    t = bench.clean_thresholds({"a": "0.7", "b": None, "c": "", "d": "x", "e": 2.0,
                                "f": {"activation": 0.8, "deactivation": 0.6, "bogus": 1},
                                "g": 0.01})
    assert t == {"a": 0.7, "e": hi, "f": {"activation": 0.8, "deactivation": 0.6}, "g": lo}, t
    assert bench.clean_thresholds(None) == {}
    print("OK   clean_thresholds: coerced, clamped, blanks / junk dropped, dict form filtered")

    # effective_params
    entry = {"id": "silero-v4.0-8k", "family": "silero-v4.0", "binary": False}
    gapped = dict(params, activation_threshold=0.5, deactivation_threshold=0.35)
    e = bench.effective_params(gapped, entry, None)
    assert (e["activation_threshold"], e["deactivation_threshold"]) == (0.5, 0.35)
    assert e["smoothing"] is None and e["threshold_override"] is False
    # override keeps the act-deact gap (0.15)
    e = bench.effective_params(gapped, entry, {"silero-v4.0-8k": 0.7})
    assert (e["activation_threshold"], e["deactivation_threshold"]) == (0.7, 0.55), e
    assert e["threshold_override"] is True
    # keyed by family: both rate variants get it
    for vid in ("silero-v4.0-8k", "silero-v4.0-16k"):
        e = bench.effective_params(gapped, dict(entry, id=vid), {"silero-v4.0": 0.7})
        assert approx(e["activation_threshold"], 0.7) and approx(e["deactivation_threshold"], 0.55), e
    # the variant id beats the family id
    e = bench.effective_params(gapped, entry, {"silero-v4.0": 0.9, "silero-v4.0-8k": 0.6})
    assert e["activation_threshold"] == 0.6, e
    # an override for some other VAD does not apply
    e = bench.effective_params(gapped, entry, {"other": 0.9})
    assert e["activation_threshold"] == 0.5 and e["threshold_override"] is False
    # equal pair 0.5/0.5 + override 0.7 -> 0.7/0.7
    eq = dict(params, activation_threshold=0.5, deactivation_threshold=0.5)
    e = bench.effective_params(eq, entry, {"silero-v4.0-8k": 0.7})
    assert (e["activation_threshold"], e["deactivation_threshold"]) == (0.7, 0.7), e
    # the deactivation never goes below 0.01
    e = bench.effective_params(gapped, entry, {"silero-v4.0-8k": 0.1})
    assert approx(e["deactivation_threshold"], max(0.01, 0.1 - 0.15)), e
    # dict form: explicit pair, or activation only (gap kept)
    e = bench.effective_params(gapped, entry, {"silero-v4.0": {"activation": 0.8, "deactivation": 0.2}})
    assert (e["activation_threshold"], e["deactivation_threshold"]) == (0.8, 0.2), e
    e = bench.effective_params(gapped, entry, {"silero-v4.0": {"activation": 0.8}})
    assert (e["activation_threshold"], e["deactivation_threshold"]) == (0.8, 0.65), e
    # smoothing flag -> the plugin alpha
    e = bench.effective_params(dict(gapped, smoothing=True), entry, None)
    assert e["smoothing"] == PLUGIN_ALPHA
    # binary: forced 0.5/0.5, no smoothing, override ignored
    bin_entry = {"id": "webrtc", "family": "webrtc", "binary": True}
    e = bench.effective_params(dict(gapped, smoothing=True), bin_entry, {"webrtc": 0.9})
    assert (e["activation_threshold"], e["deactivation_threshold"]) == (0.5, 0.5), e
    assert e["smoothing"] is None and e["threshold_override"] is False, e
    # the other params pass through untouched
    assert e["min_silence_duration"] == gapped["min_silence_duration"]
    print("OK   effective_params: override keeps the gap; variant or family key; dict form; binary forced to 0.5/0.5")


def check_resolve() -> None:
    meta = {"id": "r1", "input": {"file": "input.wav"},
            "candidates": [{"id": "a", "output": "a.wav"}, {"id": "b", "output": "b.wav"},
                           {"id": "noout"}]}
    assert bench.source_files(meta) == {"input": "input.wav", "a": "a.wav", "b": "b.wav"}
    assert bench.resolve_sources(meta, "all") == ["input", "a", "b"]
    assert bench.resolve_sources(meta, None) == ["input", "a", "b"]
    assert bench.resolve_sources(meta, ["all"]) == ["input", "a", "b"]
    assert bench.resolve_sources(meta, "ticked", ["b"]) == ["input", "b"]
    assert bench.resolve_sources(meta, "ticked", ["input", "b", "ghost", "noout"]) == ["input", "b"]
    assert bench.resolve_sources(meta, "ticked") == ["input"]
    assert bench.resolve_sources(meta, "a, b") == ["a", "b"]
    assert bench.resolve_sources(meta, ["b", "a", "b"]) == ["b", "a"]
    for bad in (["a", "zzz"], "zzz", ["noout"]):
        try:
            bench.resolve_sources(meta, bad)
        except ValueError as ex:
            assert "zzz" in str(ex) or "noout" in str(ex), ex
        else:
            raise AssertionError(f"{bad} should have raised")
    print("OK   resolve_sources: all / ticked / explicit; unknown ids raise ValueError")


def check_registry_and_vads() -> None:
    reg_json = [
        {"id": "fam", "label": "Family", "loader": "silero_onnx", "rates": [8000, 16000],
         "files": {"vad/does-not-exist.onnx": {"sha256": "x"}}},
        {"id": "solo", "label": "Solo", "loader": "webrtc", "rates": [16000], "files": {}},
        {"id": "norate", "label": "No rate", "loader": "lkaic_flag", "files": {}},
        {"id": "weird", "label": "Weird", "loader": "no-such-loader", "rates": [16000]},
        {"id": "gated", "label": "Gated", "loader": "silero_onnx", "rates": [16000], "binary": True,
         "files": {}},
    ]
    old = vads.REGISTRY_FILE
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "vad_candidates.json"
        f.write_text(json.dumps(reg_json))
        vads.REGISTRY_FILE = f
        try:
            reg = vads.registry()
            ids = [e["id"] for e in reg]
            assert ids == ["fam-8k", "fam-16k", "solo", "norate", "weird", "gated"], ids
            by = {e["id"]: e for e in reg}
            assert (by["fam-8k"]["family"], by["fam-8k"]["rate"]) == ("fam", 8000)
            assert (by["fam-16k"]["family"], by["fam-16k"]["rate"]) == ("fam", 16000)
            assert by["fam-8k"]["label"] == "Family @ 8 kHz" and by["fam-16k"]["label"] == "Family @ 16 kHz"
            assert (by["solo"]["family"], by["solo"]["rate"]) == ("solo", 16000)
            assert by["norate"]["rate"] is None and by["norate"]["family"] == "norate"
            # unavailable entries say why, available ones say nothing
            assert by["fam-8k"]["available"] is False and by["fam-8k"]["unavailable_reason"]
            assert by["fam-8k"]["unavailable_reason"] == by["fam-16k"]["unavailable_reason"]
            assert by["weird"]["available"] is False and "unknown loader" in by["weird"]["unavailable_reason"]
            assert by["norate"]["available"] is True and by["norate"]["unavailable_reason"] is None
            # binary defaults to False when absent
            assert by["solo"]["binary"] is False and by["gated"]["binary"] is True
            print("OK   registry: multi-rate entries expand to <id>-8k / -16k with family + rate; unavailable carry a reason")

            # resolve_vads: None / all -> everything
            assert [e["id"] for e in bench.resolve_vads(None)] == ids
            assert [e["id"] for e in bench.resolve_vads("all")] == ids
            # a family id expands to both rate variants
            assert [e["id"] for e in bench.resolve_vads(["fam"])] == ["fam-8k", "fam-16k"]
            assert [e["id"] for e in bench.resolve_vads("fam")] == ["fam-8k", "fam-16k"]
            # a variant id picks just that one; duplicates collapse
            assert [e["id"] for e in bench.resolve_vads("fam-8k, fam")] == ["fam-8k", "fam-16k"]
            assert [e["id"] for e in bench.resolve_vads(["solo", "fam-16k"])] == ["solo", "fam-16k"]
            # unknown raises and names the culprit
            for bad in (["fam", "nope"], "nope"):
                try:
                    bench.resolve_vads(bad)
                except ValueError as ex:
                    assert "nope" in str(ex), ex
                else:
                    raise AssertionError(f"{bad} should have raised")
            print("OK   resolve_vads: family id expands to both rates; variant id picks one; unknown raises ValueError")
        finally:
            vads.REGISTRY_FILE = old

    # and the shipped registry parses, with unique ids and every multi-rate entry expanded
    real = vads.registry()
    rids = [e["id"] for e in real]
    assert len(rids) == len(set(rids)), "duplicate VAD ids in vad_candidates.json"
    for e in real:
        assert e["family"] and (e["rate"] is None or isinstance(e["rate"], int))
        assert e["available"] or e["unavailable_reason"], e["id"]
    print(f"OK   shipped vad_candidates.json: {len(real)} unique VAD ids, every unavailable one has a reason")


class patched:
    """Temporarily set obj.name = value (restored on exit, even on failure)."""

    def __init__(self, obj, name, value):
        self.obj, self.name, self.value = obj, name, value

    def __enter__(self):
        self.old = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self.old)


def make_wav(path: Path, seed: int = 0, n: int = 16000, rate: int = 16000) -> None:
    rng = np.random.default_rng(seed)
    sf.write(path, (rng.standard_normal(n) * 3000).astype(np.int16), rate, subtype="PCM_16")


def check_probs_and_flags() -> None:
    entry = {"id": "v-8k", "rate": 8000, "loader": "webrtc", "mode": 3, "files": {}}
    probs = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    res = {"probs": probs, "hop_s": 0.032, "t0_s": -0.01, "cpu_ms_per_s": 3.5}
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        wav = run / "input.wav"
        make_wav(wav)

        # nothing stored yet
        assert bench.load_probs(run, "input", entry, wav) is None

        bench.save_probs(run, "input", entry, res, wav, 1.0)
        npy, js = bench.probs_paths(run, "input", "v-8k")
        assert npy.name == "input.v-8k.vadp.npy" and js.name == "input.v-8k.vadp.json", (npy, js)
        other = bench.probs_paths(run, "input", "silero-v4.0-8k")
        assert other[0].name == "input.silero-v4.0-8k.vadp.npy", other
        assert bench.probs_paths(run, "cand", "v-8k")[0] != npy
        got = bench.load_probs(run, "input", entry, wav)
        assert got is not None and np.allclose(got["probs"], probs)
        assert got["hop_s"] == 0.032 and got["t0_s"] == -0.01 and got["rate"] == 8000
        assert got["source_fp"]["sha256"] == bench.file_sha256(wav)
        assert got["inference_key"] == bench.fp_key(bench.inference_fp(entry))
        assert "weights" not in got and "src_size" not in got
        assert bench.load_probs(run, "input", entry) is not None       # wav=None skips the source check
        print("OK   load_probs: a fresh store round-trips probabilities and the new sidecar schema")

        js.rename(run / "side.bak")
        assert bench.load_probs(run, "input", entry, wav) is None
        (run / "side.bak").rename(js)
        good = js.read_text()
        js.write_text("{not json")
        assert bench.load_probs(run, "input", entry, wav) is None
        js.write_text(good)
        assert bench.load_probs(run, "input", entry, wav) is not None
        print("OK   load_probs: a missing half of the pair or a corrupt sidecar is 'absent'")

        # source changed: size -> stale; same size, other content -> stale; touch -> NOT stale
        st = wav.stat()
        os.utime(wav, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        assert bench.load_probs(run, "input", entry, wav) is not None
        wav.write_bytes(wav.read_bytes() + b"\0")
        assert bench.load_probs(run, "input", entry, wav) is None
        make_wav(wav)
        bench.save_probs(run, "input", entry, res, wav, 1.0)
        raw = bytearray(wav.read_bytes())
        raw[-1] ^= 0xFF
        wav.write_bytes(bytes(raw))
        assert wav.stat().st_size == len(raw)
        assert bench.load_probs(run, "input", entry, wav) is None
        print("OK   load_probs: size change or same-size rewrite is stale; a touch alone is not")

        # lkaic_flags: the ticket-1 contract
        np.save(run / "lk.aicvad.npy", np.array([0, 1, 1, 0], dtype=np.int8))
        meta = {"candidates": [
            {"id": "lk", "aicvad": {"file": "lk.aicvad.npy", "frame_s": 0.01, "t0_s": -0.02}},
            {"id": "nofile", "aicvad": {"file": "gone.npy", "frame_s": 0.01}},
            {"id": "noinfo"},
            {"id": "nofilekey", "aicvad": {"frame_s": 0.01}},
            {"id": "not0", "aicvad": {"file": "lk.aicvad.npy", "frame_s": 0.02}},
        ]}
        r = bench.lkaic_flags(meta, run, "lk")
        assert r["hop_s"] == 0.01 and r["t0_s"] == -0.02, r
        assert r["probs_file"] == "lk.aicvad.npy" and r["rate"] is None
        assert r["probs"].dtype == np.float32 and r["probs"].tolist() == [0, 1, 1, 0]
        assert r["flag_coverage"] == 1.0 and r["missing_spans"] == []
        assert r["aux_fp"]["sha256"] == bench.file_sha256(run / "lk.aicvad.npy")
        assert bench.lkaic_flags(meta, run, "not0")["t0_s"] == 0.0
        for sid in ("nofile", "noinfo", "nofilekey", "input", "ghost"):
            assert bench.lkaic_flags(meta, run, sid) is None, sid
        assert bench.lkaic_flags({}, run, "lk") is None
        spans = segment(r["probs"], r["hop_s"], t0_s=r["t0_s"], activation_threshold=0.5,
                        deactivation_threshold=0.5, min_speech_duration=0.0,
                        min_silence_duration=0.01)
        assert spans_eq(spans, [[0.0, 0.01]]), spans
        print("OK   lkaic_flags: reads candidate['aicvad'] {file, frame_s, t0_s}; None when candidate or file is absent")


def check_cache_fingerprint() -> None:
    """P1: the inference fingerprint covers more than the registry's `files`."""
    import importlib.metadata

    from nc_bench.vads import loaders

    entry = {"id": "w", "rate": 16000, "loader": "webrtc", "mode": 3, "files": {}}
    res = {"probs": np.array([0.1, 0.9], dtype=np.float32), "hop_s": 0.03, "t0_s": 0.0,
           "cpu_ms_per_s": 1.0}
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        wav = run / "input.wav"
        make_wav(wav)
        bench.save_probs(run, "input", entry, res, wav, 0.1)
        assert bench.load_probs(run, "input", entry, wav) is not None
        k0 = bench.fp_key(bench.inference_fp(entry))

        # loader config (WebRTC aggressiveness): files stays {}
        assert bench.load_probs(run, "input", dict(entry, mode=2), wav) is None
        assert bench.load_probs(run, "input", dict(entry, rate=8000), wav) is None
        # a package version (real call, faked answer)
        with patched(importlib.metadata, "version", lambda d: "999.0"):
            assert bench.load_probs(run, "input", entry, wav) is None
        # embedded native resource hash
        blob = run / "native.so"
        blob.write_bytes(b"one")
        with patched(bench, "_resource_files", lambda kind: [blob]):
            bench.save_probs(run, "input", entry, res, wav, 0.1)
            assert bench.load_probs(run, "input", entry, wav) is not None
            blob.write_bytes(b"two!")
            assert bench.load_probs(run, "input", entry, wav) is None
        # the resampler and the bench schema
        bench.save_probs(run, "input", entry, res, wav, 0.1)
        with patched(loaders, "RESAMPLER", "something-else"):
            assert bench.load_probs(run, "input", entry, wav) is None
        with patched(bench, "BENCH_SCHEMA", bench.BENCH_SCHEMA + 1):
            assert bench.load_probs(run, "input", entry, wav) is None
        assert bench.load_probs(run, "input", entry, wav) is not None
        assert bench.fp_key(bench.inference_fp(entry)) == k0          # deterministic
        print("OK   load_probs fingerprint: package version, loader config, embedded-resource hash, "
              "resampler and schema each invalidate the cache with files == {}")

        # a real model file: its ACTUAL content counts, not the registry's declared hash
        models = run / "models"
        (models / "vad").mkdir(parents=True)
        (models / "vad/m.onnx").write_bytes(b"weights-1")
        e2 = dict(entry, id="m", files={"vad/m.onnx": {"sha256": "declared-and-never-changes"}})
        with patched(bench, "MODELS_DIR", models):
            bench.save_probs(run, "input", e2, res, wav, 0.1)
            assert bench.load_probs(run, "input", e2, wav) is not None
            (models / "vad/m.onnx").write_bytes(b"weights-2")
            assert bench.load_probs(run, "input", e2, wav) is None
        print("OK   load_probs fingerprint: a changed model file invalidates it though the declared sha is unchanged")


# ------------------------------------------------- lock / stale / resegment

FAKE_REG = [
    {"id": "lkf", "label": "lkaic flag", "loader": "lkaic_flag", "rates": [], "binary": True,
     "family": "lkf", "rate": None, "files": {}, "available": True, "unavailable_reason": None},
    {"id": "fake", "label": "Fake prob VAD", "loader": "silero_onnx", "rates": [16000],
     "binary": False, "family": "fake", "rate": 16000, "files": {}, "available": True,
     "unavailable_reason": None},
]
DUR = 1.0


def install_fake_registry() -> None:
    bench.registry = lambda: [dict(e) for e in FAKE_REG]


def build_run(runs_dir: Path, run_id: str = "r1") -> Path:
    """A fake run: input + two lkaic candidates with small int8 flag files."""
    d = runs_dir / run_id
    d.mkdir(parents=True)
    make_wav(d / "input.wav", 0)
    cands = []
    for i, cid in enumerate(("lk1", "lk2")):
        make_wav(d / f"{cid}.wav", i + 1)
        flags = np.zeros(100, dtype=np.int8)
        flags[20:60] = 1
        np.save(d / f"{cid}.aicvad.npy", flags)
        cands.append({"id": cid, "output": f"{cid}.wav",
                      "aicvad": {"file": f"{cid}.aicvad.npy", "frame_s": 0.01, "t0_s": 0.0}})
    meta = {"id": run_id, "input": {"file": "input.wav", "duration_s": DUR}, "candidates": cands,
            "vad_params": {"activation_threshold": 0.5, "deactivation_threshold": 0.5,
                           "min_speech_duration": 0.05, "min_silence_duration": 0.1,
                           "prefix_padding_duration": 0.5},
            "truth_spans": [[0.2, 0.6]]}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def worker_counter(runs_dir: str, run_id: str, tag: str, n: int) -> None:
    from nc_bench import config as cfg, store

    cfg.RUNS_DIR = Path(runs_dir)
    for i in range(n):
        with store.locked_meta(run_id) as meta:
            c = meta.get("counter", 0)
            time.sleep(0.001)                       # widen the race a lost update needs
            meta["counter"] = c + 1
            meta[f"{tag}_{i}"] = i


def worker_bench(runs_dir: str, run_id: str, cand: str, n: int) -> None:
    from nc_bench import config as cfg

    cfg.RUNS_DIR = Path(runs_dir)
    install_fake_registry()
    for _ in range(n):
        bench.run_bench(run_id, vads="lkf", sources=[cand])


def run_procs(target, arglists) -> None:
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=target, args=a) for a in arglists]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]


def check_lock_and_stale_and_reseg() -> None:
    from nc_bench import config as cfg, store

    old_runs = cfg.RUNS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp).resolve()             # macOS /var -> /private/var; run_dir() resolves
        assert "data/runs" not in str(runs)
        cfg.RUNS_DIR = runs
        try:
            install_fake_registry_local = bench.registry
            install_fake_registry()
            try:
                _lock_tests(runs, store)
                _stale_tests(runs, store)
                _reseg_tests(runs, store)
            finally:
                bench.registry = install_fake_registry_local
        finally:
            cfg.RUNS_DIR = old_runs
    assert cfg.RUNS_DIR == old_runs


def _lock_tests(runs: Path, store) -> None:
    d = build_run(runs, "lock")

    build_run(runs, "lock2")

    # save_meta is atomic: no temp files left behind, valid JSON after
    m0 = json.loads((runs / "lock2/meta.json").read_text())
    store.save_meta("lock2", dict(m0, x=1))
    assert json.loads((runs / "lock2/meta.json").read_text())["x"] == 1
    assert not list((runs / "lock2").glob(".meta.*.tmp"))

    # re-entrancy: a nested block yields the same dict; only the outermost saves
    with store.locked_meta("lock2") as m1:
        m1["outer"] = 1
        with store.locked_meta("lock2") as m2:
            assert m2 is m1
            m2["inner"] = 2
        assert json.loads((runs / "lock2/meta.json").read_text()).get("inner") is None  # not saved yet
    got = json.loads((runs / "lock2/meta.json").read_text())
    assert got["outer"] == 1 and got["inner"] == 2
    # ...and the thread is free again afterwards (no leaked hold)
    with store.locked_meta("lock2") as m3:
        assert m3 is not m1 and m3["outer"] == 1
    print("OK   locked_meta: re-entrant, same dict, only the outermost block saves")

    # exception -> nothing saved, lock released, held-state cleaned
    try:
        with store.locked_meta("lock2") as m:
            m["boom"] = True
            raise RuntimeError("no save")
    except RuntimeError:
        pass
    assert "boom" not in json.loads((runs / "lock2/meta.json").read_text())
    with store.locked_meta("lock2", save=False) as m:
        m["ignored"] = 1
    assert "ignored" not in json.loads((runs / "lock2/meta.json").read_text())
    with store.locked_meta("lock2") as m:
        assert "boom" not in m and "ignored" not in m
    print("OK   locked_meta: an exception saves nothing; save=False is a read")

    # threads in this process
    def bump(n):
        for _ in range(n):
            with store.locked_meta("lock2") as m:
                c = m.get("tc", 0)
                time.sleep(0.0005)
                m["tc"] = c + 1

    ts = [threading.Thread(target=bump, args=(15,)) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert json.loads((runs / "lock2/meta.json").read_text())["tc"] == 45
    print("OK   locked_meta: 3 threads x 15 read-modify-writes lose nothing (45)")

    # P0: two processes, disjoint keys + a shared counter
    run_procs(worker_counter, [(str(runs), "lock", "a", 40), (str(runs), "lock", "b", 40)])
    m = json.loads((d / "meta.json").read_text())         # valid JSON, or this raises
    assert m["counter"] == 80, m["counter"]
    assert all(m.get(f"a_{i}") == i and m.get(f"b_{i}") == i for i in range(40))
    assert m["id"] == "lock" and "candidates" in m and "input" in m   # nothing else damaged
    assert not list(d.glob(".meta.*.tmp"))
    print("OK   P0 lock: two processes x 40 read-merge-writes lose nothing (counter 80, all 80 keys)")

    # P0: two processes running run_bench (lkaic-flag) on disjoint pairs
    run_procs(worker_bench, [(str(runs), "lock", "lk1", 4), (str(runs), "lock", "lk2", 4)])
    m = json.loads((d / "meta.json").read_text())
    res = m["vad_bench"]["results"]
    assert set(res) == {"lk1", "lk2"}, list(res)
    for c in ("lk1", "lk2"):
        r = res[c]["lkf"]
        assert r["segments"] and not r.get("stale") and r["flag_coverage"] == 1.0, r
        assert spans_eq(r["segments"], [[0.2, 0.6]]), r["segments"]
    assert len(m["vad_bench"]["runs"]) == 8, len(m["vad_bench"]["runs"])
    by_src = sorted(tuple(r["sources"]) for r in m["vad_bench"]["runs"])
    assert by_src == [("lk1",)] * 4 + [("lk2",)] * 4, by_src
    assert m["counter"] == 80 and (d / "meta.pre-vadbench.json").is_file()
    print("OK   P0 lock: two processes x 4 run_bench on disjoint pairs keep both results and all 8 run logs")


def _stale_tests(runs: Path, store) -> None:
    d = build_run(runs, "stale")
    bench.run_bench("stale", vads="lkf", sources=["lk1"])
    view = bench.fresh_view("stale")
    row = view["results"]["lk1"]["lkf"]
    assert not row.get("stale") and row["source_fp"]["sha256"] and row["aux_fp"]["file"] == "lk1.aicvad.npy"

    # a touch (mtime only) is not a change
    st = (d / "lk1.wav").stat()
    os.utime(d / "lk1.wav", ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))
    assert not bench.fresh_view("stale")["results"]["lk1"]["lkf"].get("stale")
    # fresh_view saves nothing
    before = (d / "meta.json").read_bytes()
    bench.fresh_view("stale")
    assert (d / "meta.json").read_bytes() == before

    # same size, different content -> stale
    raw = bytearray((d / "lk1.wav").read_bytes())
    raw[-2] ^= 0x55
    (d / "lk1.wav").write_bytes(bytes(raw))
    assert (d / "lk1.wav").stat().st_size == len(raw)
    row = bench.fresh_view("stale")["results"]["lk1"]["lkf"]
    assert row["stale"] is True and "source audio changed" in row["stale_reason"], row
    assert row["segments"] is not None                      # numbers kept for inspection
    assert (d / "meta.json").read_bytes() == before         # ...and still nothing saved
    print("OK   P1 stale source: same-size rewrite marks the row stale; a touch does not; fresh_view saves nothing")

    # restore the audio, then the flag file (aux_fp) changes instead
    raw[-2] ^= 0x55
    (d / "lk1.wav").write_bytes(bytes(raw))
    assert not bench.fresh_view("stale")["results"]["lk1"]["lkf"].get("stale")
    flags = np.load(d / "lk1.aicvad.npy")
    flags[70] = 1
    np.save(d / "lk1.aicvad.npy", flags)
    row = bench.fresh_view("stale")["results"]["lk1"]["lkf"]
    assert row["stale"] and "flag file" in row["stale_reason"], row
    np.save(d / "lk1.aicvad.npy", np.where(np.arange(100) == 70, 0, flags).astype(np.int8))
    assert not bench.fresh_view("stale")["results"]["lk1"]["lkf"].get("stale")
    print("OK   P1 stale source: a changed ai-coustics flag file marks the row stale")

    # STT records: read_fp mismatch is stale even with an identical seg_key
    meta = store.load_meta("stale")
    r = meta["vad_bench"]["results"]["lk1"]["lkf"]
    key = bench._seg_key(r["segments"], meta["vad_bench"]["params"]["prefix_padding_duration"])
    good = bench.source_fp(d / "lk1.wav")
    r["stt"] = {
        "same": {"seg_key": key, "read_fp": good, "text": "x"},
        "raw": {"seg_key": key, "read_fp": dict(good, sha256="0" * 64), "text": "x"},
    }
    meta["vad_bench"]["results"]["lk2"] = {"lkf": dict(r, stt={
        "same": {"seg_key": "other-cuts", "read_fp": bench.source_fp(d / "lk2.wav")},
        "raw": {"seg_key": None},
    })}
    meta["candidates"][1]["output"] = "lk2.wav"
    bench.mark_stale(meta, d)
    s1 = meta["vad_bench"]["results"]["lk1"]["lkf"]["stt"]
    assert not s1["same"].get("stale"), s1["same"]
    assert s1["raw"]["stale"] is True and "STT read changed" in s1["raw"]["stale_reason"], s1["raw"]
    s2 = meta["vad_bench"]["results"]["lk2"]["lkf"]["stt"]
    assert s2["same"]["stale"] and "cut list" in s2["same"]["stale_reason"], s2
    assert s2["raw"]["stale"], s2
    # and it un-stales when the fingerprint matches again
    s1["raw"]["read_fp"] = bench.source_fp(d / "input.wav")   # `raw` reads input.wav here
    bench.mark_stale(meta, d)
    assert not meta["vad_bench"]["results"]["lk1"]["lkf"]["stt"]["raw"].get("stale")
    print("OK   P1 stale STT: a mismatched read_fp (same seg_key), a moved cut list or a missing seg_key is stale")


def _reseg_tests(runs: Path, store) -> None:
    d = build_run(runs, "reseg")
    entry = dict(FAKE_REG[1])
    wav = d / "input.wav"
    # hand-written probabilities for the prob VAD on `input`: no model runs
    fp = np.zeros(100, dtype=np.float32)
    fp[10:50] = 0.9
    bench.save_probs(d, "input", entry, {"probs": fp, "hop_s": 0.01, "t0_s": 0.0,
                                         "cpu_ms_per_s": 1.0}, wav, DUR)
    A = {"min_silence_duration": 0.1}
    log = bench.run_bench("reseg", vads="fake", sources=["input"], params=A)
    assert log["cached"] == 1 and log["inferred"] == 0, log
    bench.run_bench("reseg", vads="lkf", sources=["lk1", "lk2"])
    meta = store.load_meta("reseg")
    row = meta["vad_bench"]["results"]["input"]["fake"]
    assert row["segments"] and row["params"]["min_silence_duration"] == 0.1, row

    # delete only that pair's .vadp files, then re-segment at other params
    npy, js = bench.probs_paths(d, "input", "fake")
    npy.unlink()
    js.unlink()
    B = {"min_silence_duration": 0.9}
    log = bench.run_bench("reseg", vads="fake", sources=["input"], params=B, resegment_only=True)
    assert log["skipped"] == 1, log
    meta = store.load_meta("reseg")
    vb = meta["vad_bench"]
    row = vb["results"]["input"]["fake"]
    assert row["segments"] is None and row["stale"] is True and row["stale_reason"], row
    assert vb["params"]["min_silence_duration"] == 0.9
    # the untouched pairs followed the new parameters (no old-param spans left behind)
    assert vb["results"]["lk1"]["lkf"]["params"]["min_silence_duration"] == 0.9
    assert vb["results"]["lk2"]["lkf"]["params"]["min_silence_duration"] == 0.9
    reg = {e["id"]: e for e in FAKE_REG}
    bad = []
    for sid, per in vb["results"].items():
        for vid, r in per.items():
            if r.get("segments") is None or r.get("stale"):
                continue
            want = bench.effective_params(vb["params"], reg[vid], vb["thresholds"])
            if r["params"] != want:
                bad.append((sid, vid, r["params"], want))
    assert not bad, bad
    print("OK   P1 resegment-only: pair without probabilities becomes a stale row (segments None); "
          "no saved non-stale row carries other params than effective_params()")

    # a threshold override also re-segments the rest, and rows still match
    bench.run_bench("reseg", vads="lkf", sources=["lk1"], params=B, thresholds={"fake": 0.8})
    vb = store.load_meta("reseg")["vad_bench"]
    for sid, per in vb["results"].items():
        for vid, r in per.items():
            if r.get("segments") is not None and not r.get("stale"):
                assert r["params"] == bench.effective_params(vb["params"], reg[vid], vb["thresholds"])
    print("OK   P1 resegment-only: rows still match effective_params after a threshold override")


def check_missing_flags() -> None:
    hop = 0.01
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)

        def flags_for(arr, name="f.npy", **info):
            np.save(run / name, arr)
            cand = {"id": "c", "aicvad": {"file": name, "frame_s": hop, "t0_s": 0.0, **info}}
            return bench.lkaic_flags({"candidates": [cand]}, run, "c")

        # a would-be phantom: speech flags, then -1 frames, then silence
        arr = np.array([0] * 20 + [1] * 10 + [-1] * 10 + [0] * 60, dtype=np.int8)
        r = flags_for(arr)
        assert r["probs"].dtype == np.float32
        # the missing frames HOLD the previous decision (1), they are not zeros
        assert r["probs"][30:40].tolist() == [1.0] * 10, r["probs"][28:42]
        assert r["probs"][40:].max() == 0.0
        assert r["flag_coverage"] == 0.9 and r["missing_spans"] == [[0.3, 0.4]], r
        kw = dict(activation_threshold=0.5, deactivation_threshold=0.5,
                  min_speech_duration=0.05, min_silence_duration=0.05, duration_s=1.0)
        held = segment(r["probs"], hop, **kw)
        as_zeros = segment((arr == 1).astype(np.float32), hop, **kw)
        assert spans_eq(held, [[0.2, 0.4]]) and spans_eq(as_zeros, [[0.2, 0.3]]), (held, as_zeros)
        # ... and a pair with a 10% hole is out of the rankings
        res = {"segments": held, "flag_coverage": r["flag_coverage"], "missing_spans": r["missing_spans"]}
        sc = bench.score(res, [[5.0, 6.0]], 10.0)
        assert sc["rank_excluded"] and "cover" in sc["rank_excluded"], sc
        print("OK   P1 missing flags: -1 frames hold the previous decision (never read as 0); 90% coverage is rank_excluded")

        # a gap at the start takes the first known decision; all-missing is coverage 0
        r = flags_for(np.array([-1, -1, 1, 1, 0], dtype=np.int8))
        assert r["probs"].tolist() == [1, 1, 1, 1, 0] and r["missing_spans"] == [[0.0, 0.02]], r
        r = flags_for(np.full(10, -1, dtype=np.int8))
        assert r["flag_coverage"] == 0.0 and r["missing_spans"] == [[0.0, 0.1]]

        # coverage >= 99% but a missing frame inside a truth span still excludes it
        arr = np.zeros(1000, dtype=np.int8)
        arr[300:305] = -1                                   # 0.5% missing -> 99.5% covered
        r = flags_for(arr)
        assert r["flag_coverage"] == 0.995 and r["missing_spans"] == [[3.0, 3.05]], r
        base = {"segments": [], "flag_coverage": r["flag_coverage"], "missing_spans": r["missing_spans"]}
        assert bench.score(base, [[10.0, 11.0]], 10.0)["rank_excluded"] is None   # gap far from any truth
        sc = bench.score(base, [[2.9, 3.02]], 10.0)
        assert sc["rank_excluded"] and "inside marked speech" in sc["rank_excluded"], sc
        sc = bench.score(base, [[5.0, 6.0]], 10.0, known_nonspeech=[[3.04, 3.5]])
        assert sc["rank_excluded"] and "non-speech" in sc["rank_excluded"], sc
        assert bench.score(base, [[5.0, 6.0]], 10.0, known_nonspeech=[[3.05, 3.5]])["rank_excluded"] is None  # touching edge
        assert bench.score(base, [], 10.0)["rank_excluded"] is None
        # rescore uses meta['known_nonspeech'] too
        meta = {"input": {"duration_s": 10.0}, "truth_spans": [[5.0, 6.0]], "known_nonspeech": [[3.0, 3.1]],
                "vad_bench": {"results": {"c": {"lkf": dict(base)}}}}
        bench.rescore(meta)
        assert meta["vad_bench"]["results"]["c"]["lkf"]["rank_excluded"]
        # coverage exactly at the threshold is fine; just under is not
        assert bench.score({"segments": [], "flag_coverage": 0.99, "missing_spans": []}, [[1, 2]], 10.0)["rank_excluded"] is None
        assert bench.score({"segments": [], "flag_coverage": 0.9899, "missing_spans": []}, [[1, 2]], 10.0)["rank_excluded"]
        # a non-flag row has no coverage key and is never excluded
        assert bench.score({"segments": [[1, 2]]}, [[1, 2]], 10.0)["rank_excluded"] is None
        print("OK   P1 missing flags: <99% excludes; a missing frame inside truth or known_nonspeech excludes even at 99.5%")

        # legacy bool file: every frame valid, coverage 1.0
        r = flags_for(np.array([False, True, True, False]))
        assert r["flag_coverage"] == 1.0 and r["missing_spans"] == [] and r["probs"].tolist() == [0, 1, 1, 0], r
        assert r["probs"].dtype == np.float32
        print("OK   P1 missing flags: a legacy bool file is all-valid, coverage 1.0")


def check_overreach() -> None:
    # the giant span: hits every turn, is never a phantom, so recall/phantoms look perfect
    ev = score_events([[10, 11], [30, 31]], [[0, 40]])
    assert ev["turn_recall"] == 1.0 and ev["missed_n"] == 0 and ev["phantom_n"] == 0, ev
    assert ev["merged_turns"] == 2 and approx(ev["overreach_s"], 38.0), ev
    # clean: nothing merged, nothing overreaching
    ev = score_events([[10, 11], [30, 31]], [[10, 11], [30, 31]])
    assert ev["merged_turns"] == 0 and ev["overreach_s"] == 0, ev
    # small overreach: 50 ms each side of one turn
    ev = score_events([[10, 11], [30, 31]], [[9.95, 11.05], [30, 31]])
    assert ev["merged_turns"] == 0 and approx(ev["overreach_s"], 0.1), ev
    # only turns that share a segment count as merged
    ev = score_events([[1, 2], [3, 4], [8, 9]], [[0.5, 4.5], [8, 9]])
    assert ev["merged_turns"] == 2 and approx(ev["overreach_s"], 0.5 + 1.0 + 0.5), ev
    # a phantom is NOT overreach (it touches no turn), it stays a phantom
    ev = score_events([[1, 2]], [[1, 2], [5, 6]])
    assert ev["overreach_s"] == 0 and ev["phantom_n"] == 1, ev
    # segments overlapping each other do not double-count the marked speech inside them
    ev = score_events([[1, 2]], [[0.9, 2.0], [1.0, 2.1]])
    assert approx(ev["overreach_s"], 0.1 + 0.1), ev
    # no truth -> both None
    for truth in ([], None):
        ev = score_events(truth, [[0, 40]])
        assert ev["merged_turns"] is None and ev["overreach_s"] is None, ev
    print("OK   P2 overreach: the giant span is recall 1.0 / 0 phantoms but merged_turns 2 and overreach 38 s; "
          "clean = 0; no truth = None")



def check_score_rescore() -> None:
    truth = [[1.0, 2.0], [5.0, 6.0]]
    res = {"segments": [[1.0, 2.0], [8.0, 8.1]]}
    sc = bench.score(res, truth, 10.0)
    assert sc["frame"] is not None and sc["events"]["hit"] == 1 and sc["events"]["phantom_n"] == 1, sc
    assert sc["events"]["missed"] == [[5.0, 6.0]]
    sc = bench.score(res, [], 10.0)
    assert sc["frame"] is None and all(v is None for v in sc["events"].values()), sc
    print("OK   score: frame + events for a result; no truth -> frame None, events all None")

    def meta_with(truth_spans):
        return {"input": {"duration_s": 10.0}, "truth_spans": truth_spans,
                "vad_bench": {"results": {
                    "input": {"v": {"vad": "v", "segments": [[1.0, 2.0], [8.0, 8.1]]},
                              "broken": {"vad": "broken", "error": "boom"},
                              "na": {"vad": "na", "unavailable": "no model"}},
                    "cand": {"v": {"vad": "v", "segments": []}}}}}

    m = meta_with(truth)
    bench.rescore(m)
    r = m["vad_bench"]["results"]["input"]["v"]
    assert r["events"]["hit"] == 1 and r["events"]["missed_n"] == 1 and r["frame"] is not None, r
    assert r["segments"] == [[1.0, 2.0], [8.0, 8.1]]                  # spans untouched
    assert m["vad_bench"]["results"]["cand"]["v"]["events"]["turn_recall"] == 0.0
    # error / unavailable results are left alone
    assert "events" not in m["vad_bench"]["results"]["input"]["broken"]
    assert "events" not in m["vad_bench"]["results"]["input"]["na"]
    # re-marking: the same results re-score against the new truth
    m["truth_spans"] = [[8.0, 8.1]]
    bench.rescore(m)
    r = m["vad_bench"]["results"]["input"]["v"]
    assert r["events"]["turns"] == 1 and r["events"]["phantom_n"] == 1, r
    # clearing the marks: frame None, events all None (not 0)
    m["truth_spans"] = []
    bench.rescore(m)
    r = m["vad_bench"]["results"]["input"]["v"]
    assert r["frame"] is None and all(v is None for v in r["events"].values()), r
    # duration falls back to input_vad when input has none
    m2 = meta_with(truth)
    m2["input"] = {}
    m2["input_vad"] = {"duration_s": 10.0}
    bench.rescore(m2)
    assert m2["vad_bench"]["results"]["input"]["v"]["frame"] is not None
    # no vad_bench at all is a no-op
    bench.rescore({"truth_spans": truth})
    print("OK   rescore: re-scores stored segments against meta['truth_spans']; no truth -> None; errors untouched")


def check_coalescing() -> None:
    from nc_bench.vads.events import coalesce_truth

    def row(min_sil, segs):
        return {"segments": segs, "params": {"min_silence_duration": min_sil}}

    # a 51 ms gap at min_silence 0.3 is one turn; a 0.5 s gap is two
    r = bench.score(row(0.3, [[1.0, 3.0]]), [[1.0, 2.0], [2.051, 3.0]], 10.0)
    assert r["truth_turns"] == 1 and r["truth_turns_raw"] == 2, r
    assert r["events"]["turns"] == 1 and r["events"]["merged_turns"] == 0, r["events"]
    r = bench.score(row(0.3, [[1.0, 3.0]]), [[1.0, 2.0], [2.5, 3.0]], 10.0)
    assert r["truth_turns"] == 2 and r["truth_turns_raw"] == 2, r
    assert r["events"]["merged_turns"] == 2, r["events"]          # 0.5 s pause: really merged
    # the bound is strict: a gap of exactly min_gap is two turns
    assert len(coalesce_truth([[1, 2], [2.5, 3]], 0.5)) == 2
    assert len(coalesce_truth([[1, 2], [2.49, 3]], 0.5)) == 1
    assert coalesce_truth([[3, 4], [1, 2], [2.1, 2.5]], 0.3) == [[1, 2.5], [3, 4]]      # unsorted, chain
    assert coalesce_truth([[1, 2], [2.1, 3], [3.1, 4]], 0.3) == [[1, 4]]                # chained merge
    assert coalesce_truth([], 0.3) == [] and coalesce_truth([[1, 2]], 0.0) == [[1, 2]]
    print("OK   coalescing: 51 ms gap at min_silence 0.3 is one turn, 0.5 s is two; strict bound; chains")

    # the same stored row re-scored at other params changes truth_turns, never the frame metrics
    truth = [[1.0, 2.0], [2.051, 3.0]]
    segs = [[1.0, 3.0]]
    meta = {"input": {"duration_s": 10.0}, "truth_spans": truth,
            "vad_bench": {"results": {"input": {"v": row(0.03, segs)}}}}
    bench.rescore(meta)
    lo = dict(meta["vad_bench"]["results"]["input"]["v"])
    assert lo["truth_turns"] == 2 and lo["events"]["merged_turns"] == 2, lo
    meta["vad_bench"]["results"]["input"]["v"]["params"]["min_silence_duration"] = 0.3
    bench.rescore(meta)
    hi = meta["vad_bench"]["results"]["input"]["v"]
    assert hi["truth_turns"] == 1 and hi["events"]["merged_turns"] == 0 and hi["truth_turns_raw"] == 2, hi
    assert hi["frame"] == lo["frame"] and hi["frame"]["marked_s"] == 1.95, (hi["frame"], lo["frame"])
    # a row with no params (older / hand-built) coalesces nothing
    r = bench.score({"segments": segs}, truth, 10.0)
    assert r["truth_turns"] == 2, r
    print("OK   coalescing: re-scoring the same row at min_silence 0.03 vs 0.3 changes truth_turns, not the frame metrics")

    # a gap that holds a known_nonspeech window is never bridged
    truth = [[1.0, 2.0], [2.5, 3.0]]
    kns = [{"name": "click", "start": 2.1, "end": 2.2}]
    r = bench.score(row(1.0, [[1.0, 2.0], [2.5, 3.0]]), truth, 10.0, kns)
    assert r["truth_turns"] == 2, r
    assert bench.score(row(1.0, [[1.0, 3.0]]), truth, 10.0)["truth_turns"] == 1
    # ...but a window elsewhere does not block it
    far = [{"name": "far", "start": 7.0, "end": 7.1}]
    assert bench.score(row(1.0, [[1.0, 3.0]]), truth, 10.0, far)["truth_turns"] == 1
    assert len(coalesce_truth(truth, 1.0, kns)) == 2 and len(coalesce_truth(truth, 1.0, [[2.1, 2.2]])) == 2
    print("OK   coalescing: a gap containing a known_nonspeech window is not bridged")

    # the 112511 case
    r = bench.score(row(0.4, [[1.8, 5.8]]), [[1.7, 3.076], [3.127, 5.931]], 10.0)
    assert r["events"]["merged_turns"] == 0 and r["truth_turns"] == 1 and r["truth_turns_raw"] == 2, r
    assert r["events"]["turn_recall"] == 1.0 and r["events"]["phantom_n"] == 0, r["events"]
    # no truth: turns None, not 0
    r = bench.score(row(0.4, [[1.8, 5.8]]), [], 10.0)
    assert r["truth_turns"] is None and r["truth_turns_raw"] is None and r["frame"] is None, r
    print("OK   coalescing: 112511 (one segment over two marks 51 ms apart) is merged_turns 0, truth_turns 1, raw 2")


def check_ignore_and_nonspeech() -> None:
    # a segment on ignore-only audio is neither a phantom nor a hit
    ev = score_events([[1, 2]], [[1, 2], [5.0, 5.2]], ignore=[[4.9, 5.5]])
    assert ev["phantom_n"] == 0 and ev["phantoms"] == [] and ev["ignored_segments"] == 1, ev
    ev = score_events([[1, 2]], [[1, 2], [5.0, 5.2]])
    assert ev["phantom_n"] == 1 and ev["ignored_segments"] == 0, ev
    # touching an ignore span by an edge only does not count
    ev = score_events([[1, 2]], [[1, 2], [5.0, 5.2]], ignore=[[5.2, 5.5]])
    assert ev["phantom_n"] == 1 and ev["ignored_segments"] == 0, ev
    # ...and one that also touches a known non-speech sound stays a phantom
    kns = [{"name": "click", "start": 5.05, "end": 5.1}]
    ev = score_events([[1, 2]], [[1, 2], [5.0, 5.2]], ignore=[[4.9, 5.5]], known_nonspeech=kns)
    assert ev["phantom_n"] == 1 and ev["ignored_segments"] == 0 and ev["nonspeech_hits"] == {"click": True}, ev
    # a segment that touches a turn is never ignored, whatever ignore says
    ev = score_events([[1, 2]], [[1.5, 2.5]], ignore=[[2.0, 3.0]])
    assert ev["phantom_n"] == 0 and ev["ignored_segments"] == 0 and ev["hit"] == 1, ev
    print("OK   ignore spans: ignore-only segments are ignored_segments, not phantoms; a known click keeps them phantoms")

    # frame: ignore frames leave every term, so fa_s drops by exactly the ignored overlap
    truth, segs = [[1.0, 2.0]], [[1.0, 2.0], [5.0, 5.2]]
    base = vad.score_spans(truth, segs, 10.0)
    # (the 10 ms frame mask rounds float edges, so 5.0-5.2 is 20 or 21 frames; derive the
    # expected numbers from the same mask instead of hard-coding 0.2)
    m_seg, m_ign = vad._mask([[5.0, 5.2]], 1000), vad._mask([[5.0, 5.1]], 1000)
    overlap = int((m_seg & m_ign).sum())
    assert approx(base["fa_s"], m_seg.sum() * 0.01) and base["fa_s"] > 0.19, base
    ign = vad.score_spans(truth, segs, 10.0, ignore=[[5.0, 5.1]])
    assert approx(base["fa_s"] - ign["fa_s"], overlap * 0.01) and overlap >= 10, (base, ign, overlap)
    kept = 1000 - int(m_ign.sum())
    wrong = int(round(base["fa_s"] / 0.01)) - overlap
    assert approx(ign["agree"], (kept - wrong) / kept, 1e-3), ign         # ignored frames leave the denominator
    assert ign["marked_s"] == base["marked_s"] == 1.0 and ign["miss_s"] == 0
    ign_all = vad.score_spans(truth, segs, 10.0, ignore=[[4.9, 5.5]])
    assert ign_all["fa_s"] == 0 and approx(ign_all["agree"], 1.0), ign_all
    # dict-form ignore windows work too
    assert vad.score_spans(truth, segs, 10.0, ignore=[{"name": "e", "start": 5.0, "end": 5.1}])["fa_s"] == ign["fa_s"]
    # ignore covering marked speech removes that speech from marked_s as well
    part = vad.score_spans(truth, [[1.0, 1.5]], 10.0, ignore=[[1.5, 2.0]])
    assert part["miss_s"] == 0 and approx(part["marked_s"], 0.5), part
    r = bench.score({"segments": segs, "params": {}}, truth, 10.0, None, [[5.0, 5.1]])
    assert r["frame"] == ign and r["events"]["ignored_segments"] == 1 == 1 - r["events"]["phantom_n"], r
    print("OK   ignore spans: ignore frames are left out of the frame scores (fa_s drops by exactly the overlap)")

    # overreach: ignore time inside a turn-touching segment is not overreach
    assert approx(score_events([[1, 2]], [[0.5, 2.5]])["overreach_s"], 1.0)
    assert approx(score_events([[1, 2]], [[0.5, 2.5]], ignore=[[0.5, 1.0]])["overreach_s"], 0.5)
    assert approx(score_events([[1, 2]], [[0.5, 2.5]], ignore=[[0.5, 1.0], [2.0, 2.5]])["overreach_s"], 0.0)
    assert approx(score_events([[1, 2]], [[0.5, 2.5]], ignore=[[0.7, 0.9], [0.8, 1.0]])["overreach_s"], 0.7)  # overlapping ignores: union 0.7-1.0
    print("OK   ignore spans: ignore time inside a turn-touching segment is not overreach")

    # nonspeech_hits
    truth = [[1, 2]]
    kns = [{"name": "a", "start": 5.0, "end": 5.1}, {"name": "b", "start": 7.0, "end": 7.1}]
    ev = score_events(truth, [[1, 2], [5.05, 5.3]], known_nonspeech=kns)
    assert ev["nonspeech_hits"] == {"a": True, "b": False}, ev
    ev = score_events(truth, [[1, 2]], known_nonspeech=kns)
    assert ev["nonspeech_hits"] == {"a": False, "b": False}, ev
    ev = score_events(truth, [[1, 2], [4.9, 5.0]], known_nonspeech=kns)       # edge touch is not a hit
    assert ev["nonspeech_hits"]["a"] is False, ev
    ev = score_events(truth, [[1, 2], [7.0, 7.05]], known_nonspeech=[[5.0, 5.1], [7.0, 7.1]])
    assert ev["nonspeech_hits"] == {"nonspeech@5.00": False, "nonspeech@7.00": True}, ev
    assert score_events(truth, [[1, 2]])["nonspeech_hits"] == {}
    # unsorted dict-form input is keyed by name all the same
    ev = score_events(truth, [[7.0, 7.05]], known_nonspeech=list(reversed(kns)))
    assert ev["nonspeech_hits"] == {"a": False, "b": True}, ev
    # a PR-54 style click: reported both as a phantom and by name
    ev = score_events(truth, [[1, 2], [5.0, 5.096]], known_nonspeech=kns)
    assert ev["phantom_n"] == 1 and ev["nonspeech_hits"]["a"] is True, ev
    # no truth: the new keys are None too
    ev = score_events([], [[1, 2]], ignore=[[3, 4]], known_nonspeech=kns)
    assert ev["ignored_segments"] is None and ev["nonspeech_hits"] is None, ev
    print("OK   nonspeech_hits: names map True / False correctly; dict form, unnamed pairs and edge touches")


def check_label_overlaps() -> None:
    from nc_bench.vads.events import label_overlaps, spans_of, windows

    kns = [{"name": "click", "start": 5.0, "end": 5.2}]
    assert label_overlaps([[1, 2]], kns, [[8, 9]]) == []
    assert label_overlaps([], None, None) == [] and label_overlaps(None) == []
    # touching edges are not an overlap (same strictness as scoring)
    assert label_overlaps([[1, 2]], [{"name": "c", "start": 2.0, "end": 2.5}], [[2.5, 3]]) == []
    o = label_overlaps([[4.9, 5.05]], kns, [[8, 9]])
    assert len(o) == 1 and "speech" in o[0] and "known_nonspeech" in o[0] and "click" in o[0], o
    o = label_overlaps([[1, 2]], [], [[1.5, 3]])
    assert len(o) == 1 and "speech" in o[0] and "ignore" in o[0], o
    o = label_overlaps([[1, 2]], kns, [[5.1, 6]])
    assert len(o) == 1 and "known_nonspeech" in o[0] and "ignore" in o[0] and "click" in o[0], o
    o = label_overlaps([[5.0, 5.5]], kns, [[5.1, 6]])
    assert len(o) == 3, o                                           # all three pairs
    # both label forms
    assert spans_of([{"name": "x", "start": 2, "end": 3}, [0, 1]]) == [[0.0, 1.0], [2.0, 3.0]]
    assert windows([[2, 3], {"name": "x", "start": 0, "end": 1}]) == [("x", 0.0, 1.0), (None, 2.0, 3.0)]
    print("OK   label_overlaps: each pair of classes is detected, disjoint or merely touching gives []")


def check_set_truth_http() -> None:
    from starlette.testclient import TestClient

    from nc_bench import config as cfg
    from nc_bench.server import app

    old = cfg.RUNS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp).resolve()
        cfg.RUNS_DIR = runs
        try:
            d = runs / "t1"
            d.mkdir()
            (d / "meta.json").write_text(json.dumps(
                {"id": "t1", "input": {"file": "input.wav", "duration_s": 10.0}, "candidates": []}))
            client = TestClient(app)          # no `with`: the lifespan (LiveKit worker) never starts
            url = "/api/runs/t1/truth"
            raw = lambda: (d / "meta.json").read_bytes()  # noqa: E731
            meta = lambda: json.loads(raw())  # noqa: E731
            click = {"name": "click", "start": 8.0, "end": 8.2}

            # a valid body stores all three
            r = client.post(url, json={"spans": [[1, 2], [6, 7], [1.5, 1.8]], "ignore_spans": [[4, 5], [4.5, 5.2]],
                                       "known_nonspeech": [click]})
            assert r.status_code == 200, r.text
            m = meta()
            assert m["truth_spans"] == [[1.0, 2.0], [6.0, 7.0]], m["truth_spans"]
            assert m["ignore_spans"] == [[4.0, 5.2]] and m["known_nonspeech"] == [click], m
            j = r.json()
            assert j["n"] == 2 and j["ignore_spans"] == [[4.0, 5.2]] and j["known_nonspeech"] == [click], j
            after_valid = raw()
            print("OK   set_truth: a valid body stores spans, ignore_spans (merged) and known_nonspeech")

            # a body with only `spans` keeps the other two classes
            r = client.post(url, json={"spans": [[1, 2.5]]})
            assert r.status_code == 200, r.text
            m = meta()
            assert m["truth_spans"] == [[1.0, 2.5]] and m["ignore_spans"] == [[4.0, 5.2]] and m["known_nonspeech"] == [click], m
            r = client.post(url, json={"ignore_spans": []})                  # an explicit empty clears just that class
            assert r.status_code == 200 and meta()["ignore_spans"] == [] and meta()["known_nonspeech"] == [click]
            client.post(url, json={"ignore_spans": [[4, 5]]})
            print("OK   set_truth: a later body with only `spans` keeps the existing ignore_spans / known_nonspeech")

            # overlapping classes -> 400 and nothing saved
            before = raw()
            bad = [
                {"ignore_spans": [[1.5, 1.8]]},                                    # over speech
                {"spans": [[4.5, 4.8]]},                                           # new speech over existing ignore
                {"known_nonspeech": [{"name": "x", "start": 1.2, "end": 1.4}]},    # over speech
                {"known_nonspeech": [{"name": "x", "start": 4.5, "end": 4.7}]},    # over ignore
                {"spans": [[8.1, 8.5]]},                                           # new speech over the click
            ]
            for body in bad:
                r = client.post(url, json=body)
                assert r.status_code == 400, (body, r.status_code, r.text)
                assert "overlap" in r.json()["detail"], r.json()
                assert raw() == before, f"meta.json changed by a refused body {body}"
            print("OK   set_truth: overlapping classes give 400 and meta.json is byte-identical")

            # malformed known_nonspeech -> 400, unchanged
            for body in (
                {"known_nonspeech": [click, dict(click, start=9.0, end=9.1)]},           # duplicate name
                {"known_nonspeech": [{"name": "x", "start": 9.0}]},                      # missing end
                {"known_nonspeech": [{"start": 9.0, "end": 9.1}]},                       # missing name
                {"known_nonspeech": [{"name": "x", "start": "a", "end": 9.1}]},          # not a number
                {"known_nonspeech": [{"name": "x", "start": 9.0, "end": 9.0}]},          # empty window
                {"known_nonspeech": [{"name": "  ", "start": 9.0, "end": 9.1}]},         # blank name
            ):
                r = client.post(url, json=body)
                assert r.status_code == 400, (body, r.status_code, r.text)
                assert raw() == before, body
            print("OK   set_truth: duplicate name, missing field, empty window and blank name each give 400, nothing saved")

            # unknown run -> 404; the touched-only-on-success run is otherwise unchanged
            assert client.post("/api/runs/nope/truth", json={"spans": [[1, 2]]}).status_code == 404
            assert raw() == before
            # and a good write still works after all the refusals (lock released)
            r = client.post(url, json={"spans": [[1, 2.5]], "known_nonspeech": [click, dict(click, name="clunk", start=9.0, end=9.1)]})
            assert r.status_code == 200 and [w["name"] for w in meta()["known_nonspeech"]] == ["click", "clunk"]
            assert after_valid != raw()
            print("OK   set_truth: unknown run is 404, and the run is still writable after refused bodies")
        finally:
            cfg.RUNS_DIR = old
    assert cfg.RUNS_DIR == old


def _hash16(probs) -> str:
    import hashlib
    import io

    buf = io.BytesIO()
    np.save(buf, np.asarray(probs).astype(np.float16))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def worker_save(run_dir: str, entry: dict, tag: str, n: int) -> None:
    """One of two writers racing on one pair: A is (0.25 x 50, hop 0.01), B is (0.75 x 80, hop 0.02)."""
    run = Path(run_dir)
    wav = run / "input.wav"
    arr, hop = (np.full(50, 0.25), 0.01) if tag == "A" else (np.full(80, 0.75), 0.02)
    for _ in range(n):
        bench.save_probs(run, "input", entry, {"probs": arr, "hop_s": hop, "t0_s": 0.0,
                                               "cpu_ms_per_s": 1.0}, wav, 1.0,
                         src_fp=bench.source_fp(wav))


def check_cache_publication() -> None:
    from nc_bench import config as cfg, store

    entry = {"id": "pv", "rate": 16000, "loader": "webrtc", "mode": 3, "files": {}}

    def res(vals, hop=0.01):
        return {"probs": np.asarray(vals, dtype=np.float32), "hop_s": hop, "t0_s": 0.0, "cpu_ms_per_s": 1.0}

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp).resolve()
        wav = run / "input.wav"
        make_wav(wav)
        npy, js = bench.probs_paths(run, "input", "pv")

        # a consistent pair is a hit, and the accepted probs hash to the sidecar's payload_sha256
        side = bench.save_probs(run, "input", entry, res([0.8, 0.9]), wav, 1.0)
        assert side["payload_sha256"] and side["generation"]
        got = bench.load_probs(run, "input", entry, wav)
        assert got is not None and np.allclose(got["probs"], [0.8, 0.9], atol=1e-3)
        assert _hash16(got["probs"]) == got["payload_sha256"] == json.loads(js.read_text())["payload_sha256"]
        g1 = got["generation"]
        bench.save_probs(run, "input", entry, res([0.8, 0.9]), wav, 1.0)
        assert bench.load_probs(run, "input", entry, wav)["generation"] != g1      # a new publish is a new generation
        assert not [p for p in run.iterdir() if p.name.endswith(".tmp")]
        assert (run / ".input.pv.vadp.lock").is_file()
        print("OK   publication: a consistent pair is a hit; accepted probs hash to the sidecar's payload_sha256")

        # the reviewer's interleave: sidecar of [0.8, 0.9], npy holding [0.1, 0.2] -> a MISS
        bench.save_probs(run, "input", entry, res([0.8, 0.9]), wav, 1.0)
        import io
        buf = io.BytesIO()
        np.save(buf, np.array([0.1, 0.2], dtype=np.float16))
        npy.write_bytes(buf.getvalue())
        assert bench.load_probs(run, "input", entry, wav) is None
        # a truncated npy, or one that is not an npy at all, is a miss too (not a crash)
        npy.write_bytes(buf.getvalue()[:20])
        assert bench.load_probs(run, "input", entry, wav) is None
        npy.write_bytes(b"junk")
        assert bench.load_probs(run, "input", entry, wav) is None
        print("OK   publication: a sidecar published for [0.8, 0.9] over an npy holding [0.1, 0.2] is a miss")

        # a sidecar without payload_sha256 (older) is a miss, even over a fine npy
        bench.save_probs(run, "input", entry, res([0.8, 0.9]), wav, 1.0)
        assert bench.load_probs(run, "input", entry, wav) is not None
        old = json.loads(js.read_text())
        del old["payload_sha256"]
        js.write_text(json.dumps(old))
        assert bench.load_probs(run, "input", entry, wav) is None
        old["payload_sha256"] = ""
        js.write_text(json.dumps(old))
        assert bench.load_probs(run, "input", entry, wav) is None
        print("OK   publication: a sidecar without payload_sha256 is a miss")

        # a stale older job cannot publish over a newer pair
        bench.save_probs(run, "input", entry, res([0.8, 0.9]), wav, 1.0)
        old_fp = bench.source_fp(wav)
        make_wav(wav, seed=99)                                  # the audio is rewritten since old_fp
        bench.save_probs(run, "input", entry, res([0.3, 0.4, 0.5]), wav, 1.0)      # a newer job publishes
        b_npy, b_js = npy.read_bytes(), js.read_bytes()
        try:
            bench.save_probs(run, "input", entry, res([0.1, 0.1]), wav, 1.0, src_fp=old_fp)
        except bench.StaleSource:
            pass
        else:
            raise AssertionError("an older job published over newer audio")
        assert npy.read_bytes() == b_npy and js.read_bytes() == b_js
        assert not [p for p in run.iterdir() if p.name.endswith(".tmp")]
        assert np.allclose(bench.load_probs(run, "input", entry, wav)["probs"], [0.3, 0.4, 0.5], atol=1e-3)
        # ...and one with a current fingerprint still publishes
        bench.save_probs(run, "input", entry, res([0.6]), wav, 1.0, src_fp=bench.source_fp(wav))
        assert len(bench.load_probs(run, "input", entry, wav)["probs"]) == 1
        print("OK   publication: a StaleSource job raises and leaves the existing pair byte-identical")

    # run_bench on top: miss -> re-infer / stale row; a change during inference or the read -> stale rows
    old_runs = cfg.RUNS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp).resolve()
        cfg.RUNS_DIR = runs
        keep = bench.registry
        try:
            install_fake_registry()
            d = build_run(runs, "pub")
            fake = dict(FAKE_REG[1])
            bench.save_probs(d, "input", fake, res([0.8, 0.9]), d / "input.wav", DUR)
            buf = io.BytesIO()
            np.save(buf, np.array([0.1, 0.2], dtype=np.float16))
            bench.probs_paths(d, "input", "fake")[0].write_bytes(buf.getvalue())      # crossed pair

            log = bench.run_bench("pub", vads="fake", sources=["input"], resegment_only=True)
            row = store.load_meta("pub")["vad_bench"]["results"]["input"]["fake"]
            assert log["skipped"] == 1 and row["segments"] is None and row["stale"] is True, (log, row)

            track = np.zeros(100, dtype=np.float32)
            track[10:60] = 0.9
            calls = []

            def stub_infer(e, pcm, rate):
                calls.append(e["id"])
                return {"probs": track, "hop_s": 0.01, "t0_s": 0.0, "cpu_ms_per_s": 2.0}

            with patched(bench, "infer", stub_infer):
                log = bench.run_bench("pub", vads="fake", sources=["input"])
            row = store.load_meta("pub")["vad_bench"]["results"]["input"]["fake"]
            assert log["inferred"] == 1 and calls == ["fake"], (log, calls)
            assert row["n_frames"] == 100 and spans_eq(row["segments"], [[0.1, 0.6]]) and not row.get("stale"), row
            assert bench.load_probs(d, "input", fake, d / "input.wav") is not None      # pair repaired
            print("OK   publication: run_bench on a crossed pair re-infers (or, resegment-only, gives a stale row), never the old track")

            # the wav is rewritten DURING inference: save_probs refuses, the row is stale
            def infer_and_rewrite(e, pcm, rate):
                make_wav(d / "input.wav", seed=7)
                return {"probs": track, "hop_s": 0.01, "t0_s": 0.0, "cpu_ms_per_s": 2.0}

            before = (bench.probs_paths(d, "input", "fake")[0].read_bytes(),
                      bench.probs_paths(d, "input", "fake")[1].read_bytes())
            with patched(bench, "infer", infer_and_rewrite):
                log = bench.run_bench("pub", vads="fake", sources=["input"], force=True)
            row = store.load_meta("pub")["vad_bench"]["results"]["input"]["fake"]
            assert log["stale"] == 1 and row["segments"] is None and row["stale"], (log, row)
            assert (bench.probs_paths(d, "input", "fake")[0].read_bytes(),
                    bench.probs_paths(d, "input", "fake")[1].read_bytes()) == before
            print("OK   publication: audio rewritten during inference -> stale row, existing pair untouched")

            # ...and DURING the read: every pair for that source becomes a stale row
            real_read = bench._read_mono

            def read_then_rewrite(w):
                out = real_read(w)
                make_wav(w, seed=8)
                return out

            with patched(bench, "_read_mono", read_then_rewrite), patched(bench, "infer", stub_infer):
                log = bench.run_bench("pub", vads="fake", sources=["input"], force=True)
            row = store.load_meta("pub")["vad_bench"]["results"]["input"]["fake"]
            assert log["stale"] == 1 and row["segments"] is None and row["stale"], (log, row)
            assert "while it was being read" in row["stale_reason"], row
            print("OK   publication: audio rewritten during the read -> stale row")
        finally:
            bench.registry = keep
            cfg.RUNS_DIR = old_runs


def check_cache_interleave_processes() -> None:
    """Two writers race on one pair while this process keeps reading it."""
    entry = {"id": "race", "rate": 16000, "loader": "webrtc", "mode": 3, "files": {}}
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp).resolve()
        wav = run / "input.wav"
        make_wav(wav)
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=worker_save, args=(str(run), entry, tag, 250)) for tag in ("A", "B")]
        for p in procs:
            p.start()
        hits = misses = 0
        seen = set()
        t_end = time.time() + 90
        while any(p.is_alive() for p in procs) and time.time() < t_end:
            got = bench.load_probs(run, "input", entry, wav)
            if got is None:
                misses += 1
                time.sleep(0.0005)          # the machine is shared: don't spin
                continue
            hits += 1
            probs = got["probs"]
            # whichever writer it is, npy and sidecar must be the SAME writer's
            if got["hop_s"] == 0.01:
                ok = got["n"] == 50 and len(probs) == 50 and np.allclose(probs, 0.25)
            else:
                ok = got["hop_s"] == 0.02 and got["n"] == 80 and len(probs) == 80 and np.allclose(probs, 0.75)
            assert ok, f"crossed pair: hop {got['hop_s']} n {got['n']} probs {probs[:3]}"
            assert _hash16(probs) == got["payload_sha256"]
            seen.add(got["hop_s"])
        for p in procs:
            p.join(60)
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        final = bench.load_probs(run, "input", entry, wav)
        assert final is not None and _hash16(final["probs"]) == final["payload_sha256"]
        assert not [p for p in run.iterdir() if p.name.endswith(".tmp")]
        assert hits > 0, (hits, misses)
        print(f"OK   interleave: 2 writers x 250 save_probs vs {hits + misses} loads ({hits} hits, {misses} misses, "
              f"writers seen {sorted(seen)}): every hit is one writer's own pair, never crossed")


def check_coverage_boundary() -> None:
    ok = bench._coverage_ok
    assert not ok({"valid_frames": 24749, "total_frames": 25000, "flag_coverage": 0.99})
    assert ok({"valid_frames": 24750, "total_frames": 25000, "flag_coverage": 0.99})
    assert ok({"valid_frames": 25000, "total_frames": 25000})
    assert not ok({"valid_frames": 0, "total_frames": 0, "flag_coverage": 1.0})
    # legacy rows (no counts) fall back to the ratio
    assert ok({"flag_coverage": 0.99}) and not ok({"flag_coverage": 0.9899}) and not ok({})
    for v in (24749, 24750):
        r = {"segments": [], "valid_frames": v, "total_frames": 25000, "flag_coverage": 0.99, "missing_spans": []}
        sc = bench.score(r, [[1.0, 2.0]], 250.0)
        if v == 24749:
            assert sc["rank_excluded"] and "cover" in sc["rank_excluded"], sc
        else:
            assert sc["rank_excluded"] is None, sc
    legacy = bench.score({"segments": [], "flag_coverage": 0.99, "missing_spans": []}, [[1.0, 2.0]], 250.0)
    assert legacy["rank_excluded"] is None
    assert bench.score({"segments": [], "flag_coverage": 0.9899, "missing_spans": []}, [[1.0, 2.0]], 250.0)["rank_excluded"]
    print("OK   coverage boundary: 24,749 / 25,000 is excluded and 24,750 / 25,000 is included, in _coverage_ok and score(); "
          "legacy rows use the ratio")

    # a real int8 fixture gives the exact counts
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        for n_missing, want_ok in ((251, False), (250, True), (0, True)):
            arr = np.zeros(25000, dtype=np.int8)
            arr[25000 - n_missing:] = -1                          # far from the marked speech at 1-2 s
            np.save(run / "f.npy", arr)
            meta = {"candidates": [{"id": "c", "aicvad": {"file": "f.npy", "frame_s": 0.01, "t0_s": 0.0}}]}
            r = bench.lkaic_flags(meta, run, "c")
            assert (r["valid_frames"], r["total_frames"]) == (25000 - n_missing, 25000), r
            assert isinstance(r["valid_frames"], int) and isinstance(r["total_frames"], int)
            if n_missing == 251:
                assert r["flag_coverage"] == 0.99, r["flag_coverage"]       # the rounded display value hides it
            assert bench._coverage_ok(r) is want_ok, (n_missing, r["flag_coverage"])
            sc = bench.score(dict(r, segments=[]), [[1.0, 2.0]], 250.0)
            assert (sc["rank_excluded"] is None) is want_ok, (n_missing, sc["rank_excluded"])
        # a legacy bool file: counts are all-valid
        np.save(run / "b.npy", np.array([True, False, True]))
        meta = {"candidates": [{"id": "c", "aicvad": {"file": "b.npy", "frame_s": 0.01}}]}
        r = bench.lkaic_flags(meta, run, "c")
        assert (r["valid_frames"], r["total_frames"]) == (3, 3) and bench._coverage_ok(r)
    print("OK   coverage boundary: a real int8 fixture gives exact valid / total counts (251 missing -> excluded, 250 -> included)")


def main() -> None:
    check_segmenter()
    print()
    check_events()
    print()
    check_params()
    check_resolve()
    check_registry_and_vads()
    print()
    check_probs_and_flags()
    check_cache_fingerprint()
    check_score_rescore()
    check_missing_flags()
    check_overreach()
    check_coalescing()
    check_ignore_and_nonspeech()
    check_label_overlaps()
    check_set_truth_http()
    check_coverage_boundary()
    check_cache_publication()
    check_cache_interleave_processes()
    print()
    check_lock_and_stale_and_reseg()
    print("\ncheck_vad_bench passed")


if __name__ == "__main__":
    main()
