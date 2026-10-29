"""Round-2 parameter sweeps and VAD-cut WER over stored VAD-bench probabilities.

    .venv/bin/python scripts/sweep_round2.py sweep --runs all --gates hush,none,... --vads lk-inference,...
    .venv/bin/python scripts/sweep_round2.py stt --runs all --combo hush:lk-inference:0.6:0.05:0.3 ...
    .venv/bin/python scripts/sweep_round2.py sweep ... --dry-run

**sweep** re-segments the stored probabilities (`<source>.<vad>.vadp.npy`, or
the ai-coustics flags for `lkaic-flag`) of every (gate, VAD) over a grid, in
memory, and scores each result with the VAD bench's own score() against the
run's labels. No model runs. It never touches `meta["vad_bench"]`: that stays
at the run's one parameter set (production's). Results go into an additive key,
one compact cell per (gate, VAD, activation, min_speech, min_silence):

    meta["vad_sweep_round2"] = {"at", "grid", "cells": {"[gate, vad, act, ms, msil]": {
        turns, hit, missed, phantom, merged, overreach_s, ignored, agree,
        nonspeech_hits, n_segments, segments (incident runs only)}}}

Deactivation follows activation (production runs them equal, 0.6 / 0.6).
min_silence changes the truth coalescing too, as the bench's score() does. A
binary detector ignores the activation, as effective_params() says.

**stt** runs the VAD-cut WER for named combos, one /recognize request per turn,
in both read modes (`same`: the gate's audio; `raw`: `none`, production's split
gate). Each segment is one request, as production's gated STT sends it: 0.5 s of
prefix pad + the segment + the min_silence the VAD waited out before ending it,
not merged with its neighbours even when their slices overlap.
Stored in `meta["vad_cut_wer_round2"][combo]`, only on runs with a script.

Both are resumable: a cell / combo already stored for the same grid is kept
unless --force.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import itertools
import json
import sys
import tempfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import scoring, store, stt  # noqa: E402
from nc_bench.vads import bench, by_id  # noqa: E402
from nc_bench.vads.segmenter import segment  # noqa: E402

ACT = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MIN_SPEECH = (0.05, 0.10, 0.15, 0.25)
MIN_SILENCE = (0.3, 0.4, 0.55)
# the old bench's set, for continuity with 08-04 (act 0.5 = deact 0.5, min_speech 0.05, min_silence 0.4)
CONTINUITY = (0.5, 0.05, 0.4)
PAD = 0.5
KEY = "vad_sweep_round2"
WER_KEY = "vad_cut_wer_round2"


def resolve_runs(spec: str) -> list[str]:
    from nc_bench import config

    if spec == "all":
        return sorted(d.name for d in config.RUNS_DIR.iterdir()
                      if (d / "meta.json").is_file() and not d.name.startswith("_"))
    return [r.strip() for r in spec.split(",") if r.strip()]


def ckey(gate: str, vid: str, act: float, ms: float, msil: float) -> str:
    return json.dumps([gate, vid, act, ms, msil])


def _probs(meta: dict, run_dir: Path, sid: str, entry: dict, wav: Path) -> dict | None:
    if entry["loader"] == "lkaic_flag":
        rec = bench.lkaic_flags(meta, run_dir, sid)
        if rec is not None and not bench._coverage_ok(rec):
            return None                  # incomplete flags never rank; see bench.score
        return rec
    rec = bench.load_probs(run_dir, sid, entry, wav)
    return rec if rec is not None else _probs_other_platform(run_dir, sid, entry, wav)


def _probs_other_platform(run_dir: Path, sid: str, entry: dict, wav: Path) -> dict | None:
    """Stored probabilities made by the same model on another OS.

    The round-2 probabilities were inferred on Linux x86_64. For
    loaders that ship a native extension (lk-inference, Earshot, WebRTC), the
    bench's inference fingerprint includes that binary's sha256, which differs
    per platform build of the same package version, so load_probs() calls them
    stale on this Mac. Re-segmenting is still sound when everything else in
    the fingerprint is equal (schema, loader, config, model files, package
    versions, resampler) and the audio and payload hashes check out. Only the
    `resources` key may differ. vad_bench's own cache rule is not changed.
    """
    import hashlib
    import io

    import numpy as np

    npy, js = bench.probs_paths(run_dir, sid, entry["id"])
    if not npy.is_file() or not js.is_file():
        return None
    side = json.loads(js.read_text())
    if not bench.fp_current(side.get("source_fp"), wav):
        return None
    now = bench.inference_fp(entry)
    old = side.get("inference_fp") or {}
    if {k: v for k, v in old.items() if k != "resources"} != {k: v for k, v in now.items() if k != "resources"}:
        return None
    if set(old.get("resources") or {}) == set(now.get("resources") or {}):
        return None   # same files, different bytes: a real change on THIS platform
    payload = npy.read_bytes()
    if hashlib.sha256(payload).hexdigest() != side.get("payload_sha256"):
        return None
    side["probs"] = np.load(io.BytesIO(payload)).astype(np.float32)
    side["other_platform"] = sorted(old.get("resources") or {})
    return side


def cell(meta: dict, rec: dict, entry: dict, act: float, ms: float, msil: float,
         file_dur: float, keep_segments: bool) -> dict:
    params = {"activation_threshold": act, "deactivation_threshold": act,
              "min_speech_duration": ms, "min_silence_duration": msil,
              "prefix_padding_duration": PAD, "smoothing": False}
    p = bench.effective_params(params, entry, {})
    segs = segment(rec["probs"], rec["hop_s"], t0_s=rec.get("t0_s") or 0.0, duration_s=file_dur,
                   smoothing=p["smoothing"],
                   **{k: p[k] for k in ("activation_threshold", "deactivation_threshold",
                                        "min_speech_duration", "min_silence_duration")})
    res = {"segments": segs, "params": p, **{k: rec[k] for k in (
        "flag_coverage", "valid_frames", "total_frames", "missing_spans") if k in rec}}
    sc = bench.score(res, meta.get("truth_spans") or [],
                     bench._truth_duration(meta), meta.get("known_nonspeech") or [],
                     meta.get("ignore_spans") or [])
    e, f = sc.get("events") or {}, sc.get("frame") or {}
    # known non-speech hits straight from the segments: they don't need marked
    # speech (PR-54 B has none, and score_events returns null without it)
    hits = {w["name"]: any(s[0] < w["end"] and s[1] > w["start"] for s in segs)
            for w in meta.get("known_nonspeech") or []}
    out = {"turns": e.get("turns"), "hit": e.get("hit"), "missed": e.get("missed_n"),
           "phantom": e.get("phantom_n"), "merged": e.get("merged_turns"),
           "overreach_s": e.get("overreach_s"), "ignored": e.get("ignored_segments"),
           "agree": f.get("agree"), "miss_s": f.get("miss_s"), "fa_s": f.get("fa_s"),
           "nonspeech_hits": hits, "n_segments": len(segs),
           # which (coalesced) marked turns were missed: the short-turn check needs
           # their lengths, not only the count
           "missed_spans": e.get("missed"),
           "rank_excluded": sc.get("rank_excluded")}
    if keep_segments:
        out["segments"] = segs
    return out


def sweep(runs: list[str], gates: list[str], vads: list[str], force: bool, dry: bool) -> None:
    reg = by_id()
    entries = [reg[v] for v in vads]
    grid = sorted(set(itertools.product(ACT, MIN_SPEECH, MIN_SILENCE)) | {CONTINUITY})
    for rid in runs:
        meta = store.load_meta(rid)
        run_dir = store.run_dir(rid)
        files = bench.source_files(meta)
        old = ((meta.get(KEY) or {}).get("cells") or {}) if not force else {}
        keep_segs = bool(meta.get("incident"))
        new, missing = {}, []
        for g in gates:
            if g not in files:
                missing.append(f"{g}: no audio")
                continue
            wav = run_dir / files[g]
            info = sf.info(wav)
            dur = info.frames / info.samplerate
            for e in entries:
                todo = [t for t in grid if ckey(g, e["id"], *t) not in old]
                if not todo:
                    continue
                rec = _probs(meta, run_dir, g, e, wav)
                if rec is None:
                    missing.append(f"{g} x {e['id']}: no fresh probabilities")
                    continue
                for t in todo:
                    new[ckey(g, e["id"], *t)] = cell(meta, rec, e, *t, dur, keep_segs)
        print(json.dumps({"run": rid, "new_cells": len(new), "missing": missing}), flush=True)
        if dry or not new:
            continue
        with store.locked_meta(rid) as m:
            sw = m.setdefault(KEY, {})
            sw["at"] = datetime.datetime.now().isoformat(timespec="seconds")
            sw["grid"] = {"activation": ACT, "min_speech": MIN_SPEECH, "min_silence": MIN_SILENCE,
                          "continuity": CONTINUITY, "deactivation": "= activation", "pad": PAD}
            sw.setdefault("cells", {}).update(new)
            if missing:
                sw["missing"] = sorted(set(sw.get("missing") or []) | set(missing))


async def _wer_combo(meta: dict, run_dir: Path, gate: str, vid: str, act: float, ms: float,
                     msil: float, tmpdir: Path, modes=("same", "raw")) -> dict:
    reg = by_id()
    e = reg[vid]
    files = bench.source_files(meta)
    wav = run_dir / files[gate]
    info = sf.info(wav)
    rec = _probs(meta, run_dir, gate, e, wav)
    if rec is None:
        return {"error": "no fresh probabilities"}
    dur = info.frames / info.samplerate
    c = cell(meta, rec, e, act, ms, msil, dur, True)
    segs = c.pop("segments")
    # production's STT slice is prefix pad + speech + the min_silence it waited
    # out before END fired (PR-54: "0.5 s pad + speech + 0.3 s"); the bench's
    # segment end is corrected back to the speech, so the tail is added here
    # one request per segment, NOT merged: each END is its own gated-STT request,
    # and two close segments send overlapping audio (the next one's prefix pad
    # reaches back into the last one's tail), as production does
    cuts = [[max(0.0, s - PAD), min(dur, e + msil)] for s, e in segs]
    script = (meta.get("script") or "").strip()
    out = {"cell": c, "segments_n": len(segs), "pad_s": PAD, "trail_s": msil, "merged": False}
    for mode in modes:
        name = bench._stt_read_name(files, gate, mode)
        try:
            got = await _per_segment(bench._wav16(run_dir / name, tmpdir), cuts)
            got["wer"] = scoring.wer(script, got["text"])
            got["read"] = name
            out[mode] = got
        except Exception as ex:  # noqa: BLE001 - recorded, never silently skipped
            out[mode] = {"error": f"{type(ex).__name__}: {ex}", "read": name}
    return out


async def _per_segment(wav16: Path, cuts: list[list[float]]) -> dict:
    """stt.transcribe_turns without its overlap merge: one /recognize per cut,
    texts joined in time order, confidence weighted by cut length."""
    mono, rate = stt._load_mono(wav16)
    texts, langs, confs = [], [], []
    for s, e in cuts:
        piece = mono[int(s * rate): int(e * rate)]
        if len(piece) < int(0.02 * rate):
            continue
        r = await stt._recognize(piece, rate)
        if r["text"]:
            texts.append(r["text"])
            langs.append(r.get("language"))
            if r.get("confidence") is not None:
                confs.append((e - s, float(r["confidence"])))
    total = sum(d for d, _ in confs)
    return {"text": " ".join(texts), "language": next((x for x in langs if x), None),
            "confidence": round(sum(d * c for d, c in confs) / total, 4) if total else None,
            "mode": "per-segment", "turns": len(cuts)}


def run_wer(runs: list[str], combos: list[str], force: bool, dry: bool,
            modes=("same", "raw"), concurrency: int = 1) -> None:
    """VAD-cut WER for the named combos. `concurrency` combos of one run are
    transcribed at once (each sends its requests one after another), so that is
    also the number of STT requests in flight."""
    parsed = []
    for c in combos:
        g, v, a, ms, msil = c.split(":")
        parsed.append((g, v, float(a), float(ms), float(msil)))
    for rid in runs:
        meta = store.load_meta(rid)
        if not (meta.get("script") or "").strip():
            continue
        run_dir = store.run_dir(rid)
        files = bench.source_files(meta)
        old = (meta.get(WER_KEY) or {}) if not force else {}
        todo = [p for p in parsed if ckey(*p) not in old and p[0] in files]
        print(json.dumps({"run": rid, "combos": len(todo)}), flush=True)
        if dry or not todo:
            continue

        async def _all(tmpdir: Path) -> dict:
            sem = asyncio.Semaphore(max(1, concurrency))

            async def one(p):
                async with sem:
                    return ckey(*p), await _wer_combo(meta, run_dir, *p, tmpdir, modes)
            # _wav16 caches converted inputs in tmpdir; do that once, serially,
            # so concurrent combos never write the same file
            for mode in modes:
                for p in todo:
                    bench._wav16(run_dir / bench._stt_read_name(files, p[0], mode), tmpdir)
            return dict(await asyncio.gather(*(one(p) for p in todo)))

        with tempfile.TemporaryDirectory(prefix="sweep-stt-") as tmp:
            got = asyncio.run(_all(Path(tmp)))
        with store.locked_meta(rid) as m:
            wk = m.setdefault(WER_KEY, {})
            for k, v in got.items():
                # a modes subset only replaces the modes it ran
                wk[k] = {**(wk.get(k) or {}), **v} if not force else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--runs", required=True)
    s.add_argument("--gates", required=True)
    s.add_argument("--vads", required=True)
    s.add_argument("--force", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    w = sub.add_parser("stt")
    w.add_argument("--runs", required=True)
    w.add_argument("--combo", action="append", required=True, help="gate:vad:act:min_speech:min_silence")
    w.add_argument("--force", action="store_true")
    w.add_argument("--dry-run", action="store_true")
    w.add_argument("--modes", default="same,raw", help="read modes: same, raw or both")
    w.add_argument("--concurrency", type=int, default=1, help="STT requests in flight (combos at once)")
    a = ap.parse_args()
    if a.cmd == "sweep":
        sweep(resolve_runs(a.runs), a.gates.split(","), a.vads.split(","), a.force, a.dry_run)
    else:
        run_wer(resolve_runs(a.runs), a.combo, a.force, a.dry_run,
                tuple(m.strip() for m in a.modes.split(",") if m.strip()), a.concurrency)


if __name__ == "__main__":
    main()
