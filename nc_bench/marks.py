"""Hand-marked speech with ids and references, and the run's full reference.

A mark is one violet region drawn on the input card:

    meta["truth_marks"] = [{"id", "start", "end", "ref", "ref_empty"?, "check"?}, ...]
    meta["truth_spans"] = [[start, end], ...]      # derived, for every scorer

`ref` is what was actually said inside that mark. The run's reference script
is derived from them, never typed: the references joined in time order, and
only when EVERY mark has one. A partly referenced run has no WER at all, since
a reference missing words scores the STT for words it was never given.

`ref_empty` is a reference too: nothing should be transcribed there (a VAD is
expected to open on it, and the STT to judge it noise and return no text). It
adds no reference words, so anything the STT does return there is scored as
inserted words. The API sends it as a null reference ({id: null}); "" means
the mark has no reference yet.

`check` flags a mark whose reference may no longer match its audio: two
referenced marks merged by an overlapping drag (their references are joined in
time order), or a referenced mark whose bounds changed. Saving that mark's
reference clears it.

Runs marked before marks had ids only have truth_spans. ensure() gives those
marks ids derived from their bounds, so every read of the same run names the
same mark the same way, persisted or not.
"""

from __future__ import annotations

import hashlib
import uuid

from . import scoring

MIN_MARK_S = 0.02    # a 20 ms drag is a mis-click, not a region
SOURCES = ("marks", "marks-incomplete")


class UnknownMarks(Exception):
    """References were sent for marks the run no longer has."""

    def __init__(self, ids: list[str]):
        super().__init__(", ".join(ids))
        self.ids = ids


def _bound_id(start: float, end: float) -> str:
    return "m" + hashlib.sha1(f"{start:.3f}-{end:.3f}".encode()).hexdigest()[:8]


def _new_id() -> str:
    return "m" + uuid.uuid4().hex[:8]


def ensure(meta: dict) -> list[dict]:
    """The run's marks, reconciled with truth_spans (which other writers, e.g.
    the import scripts, may have set on their own): a span keeps the mark with
    exactly its bounds, and a span with no such mark gets an empty one. Does
    not write to meta."""
    spans = [[round(float(a), 3), round(float(b), 3)] for a, b in meta.get("truth_spans") or []]
    stored = {(round(float(m["start"]), 3), round(float(m["end"]), 3)): m
              for m in meta.get("truth_marks") or []}
    out = []
    for a, b in spans:
        m = stored.get((a, b))
        out.append(dict(m, start=a, end=b) if m else
                   {"id": _bound_id(a, b), "start": a, "end": b, "ref": ""})
    return out


def _referenced(m: dict) -> bool:
    return bool(m.get("ref_empty") or (m.get("ref") or "").strip())


def _set_ref(m: dict, value) -> dict:
    """A reference from the API onto mark `m`, in place: null = expected empty,
    a string = that text ("" = no reference yet). Saving one clears `check`."""
    if value is None:
        m["ref"], m["ref_empty"] = "", True
    else:
        m["ref"] = str(value).strip()
        m.pop("ref_empty", None)
    m.pop("check", None)
    return m


def _store(meta: dict, marks: list[dict]) -> None:
    marks = sorted(marks, key=lambda m: (m["start"], m["end"]))
    meta["truth_marks"] = marks
    meta["truth_spans"] = [[m["start"], m["end"]] for m in marks]


def apply_spans(meta: dict, items, limit: float, refs: dict | None = None) -> list[dict]:
    """Replace the run's marks with `items` (from the page: {id?, start, end};
    from older callers: [start, end] pairs), keeping every reference.

    Clamped to the recording, sorted, overlaps merged. A merged region keeps
    the earliest id and the references of every mark it swallowed, joined in
    time order (marks expected empty add no text; a region made only of them
    stays expected empty). `refs` ({id: text or null}) are applied to the
    current marks first, so text typed before a re-mark is never lost to it."""
    old = {m["id"]: m for m in ensure(meta)}
    for mid, text in (refs or {}).items():
        if mid in old:
            old[mid] = _set_ref(dict(old[mid]), text)
    by_bounds = {(m["start"], m["end"]): mid for mid, m in old.items()}

    parts = []
    for it in items or []:
        try:
            if isinstance(it, dict):
                a, b, mid = float(it["start"]), float(it["end"]), it.get("id")
            else:
                a, b, mid = float(it[0]), float(it[1]), None
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        lo, hi = sorted((a, b))
        lo, hi = round(max(0.0, min(lo, limit)), 3), round(max(0.0, min(hi, limit)), 3)
        if hi - lo < MIN_MARK_S:
            continue
        if mid not in old:
            mid = by_bounds.get((lo, hi))
        parts.append({"start": lo, "end": hi, "id": mid})
    parts.sort(key=lambda p: (p["start"], p["end"]))

    groups: list[list[dict]] = []
    for p in parts:
        if groups and p["start"] <= max(q["end"] for q in groups[-1]):
            groups[-1].append(p)
        else:
            groups.append([p])

    out, used = [], set()
    for g in groups:
        lo, hi = g[0]["start"], max(p["end"] for p in g)
        ids = [p["id"] for p in g if p["id"] and p["id"] not in used]
        ids = sorted(dict.fromkeys(ids), key=lambda i: old[i]["start"])
        if not ids:
            out.append({"id": _new_id(), "start": lo, "end": hi, "ref": ""})
            continue
        prim = old[ids[0]]
        refs_in = [old[i]["ref"] for i in ids if old[i].get("ref")]
        mark = {"id": ids[0], "start": lo, "end": hi, "ref": " ".join(refs_in)}
        if not refs_in and any(old[i].get("ref_empty") for i in ids):
            mark["ref_empty"] = True
        moved = (lo, hi) != (prim["start"], prim["end"])
        had_ref = any(_referenced(old[i]) for i in ids)
        if any(old[i].get("check") for i in ids) or (had_ref and (len(ids) > 1 or moved)):
            mark["check"] = True
        used.update(ids)
        out.append(mark)
    _store(meta, out)
    return meta["truth_marks"]


def apply_refs(meta: dict, refs: dict) -> list[dict]:
    """Set the reference of each mark named in `refs` ({id: text, or null for
    expected empty}) and clear its check flag. An id the run no longer has
    raises UnknownMarks: the marks changed under the page, and silently
    dropping typed text would lose it."""
    marks = ensure(meta)
    known = {m["id"] for m in marks}
    unknown = [mid for mid in refs or {} if mid not in known]
    if unknown:
        raise UnknownMarks(unknown)
    for m in marks:
        if m["id"] in (refs or {}):
            _set_ref(m, refs[m["id"]])
    _store(meta, marks)
    return meta["truth_marks"]


def reference_state(meta: dict) -> dict:
    marks = ensure(meta)
    n_ref = sum(1 for m in marks if _referenced(m))
    return {"marks": len(marks), "referenced": n_ref,
            "empty": sum(1 for m in marks if m.get("ref_empty")),
            "complete": bool(marks) and n_ref == len(marks),
            "checks": sum(1 for m in marks if m.get("check"))}


def sync_reference(meta: dict) -> dict:
    """Derive meta["script"] from the marks and re-score every WER that reads it.

    - every mark referenced: script = the references joined in time order
      (marks expected empty add nothing; if every mark is, there are no
      reference words and so no WER);
    - some referenced: no script, so no WER anywhere, until the rest are filled;
    - none referenced: a script recorded with the run (typed at upload or call
      start, before marks had references) stays as it is.

    The first time references take over, a recorded script is kept in
    meta["script_legacy"], so it is never lost. Whole-file WER is re-scored from
    the transcript already stored on each candidate (no STT call), and the VAD
    bench's VAD-cut WER from its stored cuts. In place; returns the state."""
    from .vads import bench   # lazy: keeps this module free of the VAD stack's imports

    st = reference_state(meta)
    marks = ensure(meta)
    source = meta.get("script_source")
    script = (meta.get("script") or "").strip()
    if st["referenced"]:
        if source not in SOURCES and script:
            meta["script_legacy"] = script
        if st["complete"]:
            script = " ".join(m["ref"].strip() for m in marks if m["ref"].strip())
            source = "marks"
        else:
            script, source = "", "marks-incomplete"
    elif source in SOURCES:
        script, source = "", None          # references removed: nothing to score against
    else:
        source = "legacy" if script else None
    meta["script"] = script
    if source:
        meta["script_source"] = source
    else:
        meta.pop("script_source", None)

    for c in meta.get("candidates") or []:
        if not isinstance(c.get("scores"), dict):
            continue
        if script:
            c["scores"]["wer"] = scoring.wer(script, (c.get("stt") or {}).get("text", ""))
        else:
            c["scores"].pop("wer", None)
    bench.rescore_wer(meta)
    return dict(st, source=source, words=len(scoring._tokens(script)))
