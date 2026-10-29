"""Event-level scoring of a VAD's spans against hand-marked speech.

Frame agreement (vad.score_spans) answers "how much of the call did the VAD
label the way the human did", and that number hides both of the failures this
bench exists for. PR-42's missed turn is one quiet utterance, a few hundred ms
of a two-minute call, so it barely moves agreement. PR-54's phantom click held
96 ms above threshold, too little to register as false-alarm seconds, but it
still opened a turn that the recogniser filled with four invented words. Both
are EVENTS, so they are counted as events:

- turn recall: marked turns that any segment overlaps at all;
- missed turns: marked turns no segment touches (PR-42), listed and summed;
- phantom segments: segments that touch no marked turn (PR-54), counted and summed;
- onset lag: segment start minus marked start, over the turns that were hit.
  A late onset costs the first syllable even when the turn is "found";
- fragmentation: segments per hit turn. One marked utterance cut into three
  segments means three STT requests, each with less context;
- merged turns: hit turns that share a segment with another marked turn
  (their audio goes to STT fused into one request);
- overreach: seconds of unmarked audio inside segments that do touch a
  marked turn. With merged turns, this is what stops an always-open VAD (one
  [0, end] segment: 100% recall, zero phantoms) from ranking as perfect.

Overlap is strict: a segment that ends exactly where a turn starts does not
touch it. With no marked speech every metric is None, never 0: "no truth" and
"a perfect score" must not print the same.
"""

from __future__ import annotations

import statistics


def _overlaps(a, b) -> bool:
    return float(a[0]) < float(b[1]) and float(a[1]) > float(b[0])


EVENT_KEYS = (
    "turns", "hit", "turn_recall", "missed_n", "missed_s", "missed",
    "phantom_n", "phantom_s", "phantoms", "onset_lag_med_s", "onset_lag_max_s",
    "frag", "merged_turns", "overreach_s", "ignored_segments", "nonspeech_hits",
)


# ------------------------------------------------------------- label classes
# A run can carry three label classes, and they must not overlap:
#
#   truth_spans      [[s, e], ...]                  marked speech
#   known_nonspeech  [{"name", "start", "end"}]     sounds known NOT to be
#                                                   speech (a click, a clunk)
#   ignore_spans     [[s, e], ...]                  sounds whose true label is
#                                                   uncertain (a quiet echo, a
#                                                   distant voice); scored as
#                                                   neither speech nor silence


def windows(x) -> list[tuple[str | None, float, float]]:
    """(name, start, end) for either label form: [s, e] pairs or
    {name, start, end} dicts. Sorted by start."""
    out = []
    for w in x or []:
        if isinstance(w, dict):
            out.append((w.get("name"), float(w["start"]), float(w["end"])))
        else:
            out.append((None, float(w[0]), float(w[1])))
    return sorted(out, key=lambda t: (t[1], t[2]))


def spans_of(x) -> list[list[float]]:
    return [[s, e] for _, s, e in windows(x)]


def label_overlaps(truth, known_nonspeech=None, ignore=None) -> list[str]:
    """Human-readable overlaps between the three label classes; [] if none."""
    classes = {"speech": spans_of(truth), "known_nonspeech": windows(known_nonspeech),
               "ignore": spans_of(ignore)}
    classes["known_nonspeech"] = [[s, e, n] for n, s, e in classes["known_nonspeech"]]
    out = []
    names = list(classes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for x in classes[a]:
                for y in classes[b]:
                    if _overlaps(x, y):
                        tag = lambda cls, w: f"{cls} {w[0]:.2f}-{w[1]:.2f}" + (  # noqa: E731
                            f" ({w[2]})" if len(w) > 2 and w[2] else "")
                        out.append(f"{tag(a, x)} overlaps {tag(b, y)}")
    return out


def coalesce_truth(truth, min_gap: float, known_nonspeech=None) -> list[list[float]]:
    """Marked turns closer than the segmenter can split, merged into one.

    A VAD run with min_silence_duration = X cannot end a turn on a pause
    shorter than X, so two marks separated by less than that are one turn to
    every model at that setting. Scoring them as two would charge every VAD a
    "merged turn" (and deny it the incident gate) for a property of the marks.
    Event scoring therefore runs on these coalesced turns; frame scoring keeps
    the raw marks.

    A gap that touches a known_nonspeech window is never bridged: that window
    was marked as not speech on purpose, and merging across it would turn it
    into speech.
    """
    kns = spans_of(known_nonspeech)
    out: list[list[float]] = []
    for s, e in sorted([float(a), float(b)] for a, b in spans_of(truth)):
        if out and s - out[-1][1] < float(min_gap) and not any(
                c < s and d > out[-1][1] for c, d in kns):
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _union(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _inside(seg, union) -> float:
    return sum(max(0.0, min(seg[1], e) - max(seg[0], s)) for s, e in union)


def score_events(truth, segments, ignore=None, known_nonspeech=None) -> dict:
    """Event metrics for one span set. See the module docstring.

    `ignore` spans are uncertain audio: a segment that touches no marked turn
    but does touch an ignore span, and touches no known non-speech, is counted
    in ignored_segments instead of as a phantom (it may have opened on real,
    unmarked speech), and ignore time is left out of overreach.
    `known_nonspeech` windows are reported one by one: nonspeech_hits[name] is
    True when any segment opened on that sound (a PR-54 click, say).
    """
    if not truth or segments is None:
        return {k: None for k in EVENT_KEYS}
    truth = sorted([float(s), float(e)] for s, e in spans_of(truth))
    segs = sorted([float(s), float(e)] for s, e in segments)
    ign = _union(spans_of(ignore))
    kns = windows(known_nonspeech)

    missed, lags, frags = [], [], []
    for t in truth:
        over = [s for s in segs if _overlaps(s, t)]
        if not over:
            missed.append([round(t[0], 3), round(t[1], 3)])
            continue
        lags.append(over[0][0] - t[0])
        frags.append(len(over))
    phantoms, ignored = [], []
    for s in segs:
        if any(_overlaps(s, t) for t in truth):
            continue
        on_ignore = any(_overlaps(s, w) for w in ign)
        on_nonspeech = any(_overlaps(s, (a, b)) for _, a, b in kns)
        (ignored if on_ignore and not on_nonspeech else phantoms).append(
            [round(s[0], 3), round(s[1], 3)])
    # The two failures recall and phantoms cannot see: one segment spanning
    # several turns (merged: one STT request, turns fused), and a segment that
    # touches a real turn but runs on through non-speech (overreach). A single
    # [0, end] span hits every turn and is never a phantom, so without these it
    # would look perfect on both incidents.
    turns_of = [[i for i, t in enumerate(truth) if _overlaps(s, t)] for s in segs]
    merged = {i for ts in turns_of if len(ts) > 1 for i in ts}
    known = _union(truth + ign)          # speech or uncertain: not overreach
    overreach = sum((s[1] - s[0]) - _inside(s, known) for s, ts in zip(segs, turns_of) if ts)

    hit = len(truth) - len(missed)
    hits: dict[str, bool] = {}
    for i, (name, a, b) in enumerate(kns):
        key = name or f"nonspeech@{a:.2f}"
        hits[key if key not in hits else f"{key}#{i}"] = any(_overlaps(s, (a, b)) for s in segs)
    return {
        "turns": len(truth),
        "hit": hit,
        "turn_recall": round(hit / len(truth), 4),
        "missed_n": len(missed),
        "missed_s": round(sum(e - s for s, e in missed), 3),
        "missed": missed,
        "phantom_n": len(phantoms),
        "phantom_s": round(sum(e - s for s, e in phantoms), 3),
        "phantoms": phantoms,
        # None, not 0, when nothing was hit: there is no onset to be late on
        "onset_lag_med_s": round(statistics.median(lags), 3) if lags else None,
        "onset_lag_max_s": round(max(lags), 3) if lags else None,
        "frag": round(sum(frags) / len(frags), 3) if frags else None,
        # hit turns that share a segment with another turn
        "merged_turns": len(merged),
        # seconds of unmarked, non-ignored audio inside segments that touch a turn
        "overreach_s": round(overreach, 3),
        # segments on uncertain audio only: neither phantom nor hit
        "ignored_segments": len(ignored),
        # per known non-speech sound: did any segment open on it?
        "nonspeech_hits": hits,
    }
