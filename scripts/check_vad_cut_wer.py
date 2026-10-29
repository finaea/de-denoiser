"""Assert-based check of the VAD-cut WER flow: per-mark references, the derived
reference script, WER re-derived from stored cuts, and legacy-silero.

    .venv/bin/python scripts/check_vad_cut_wer.py [--run 20260802-203822]

Works on a copy of one recorded run in a temporary DATA_DIR (its meta.json and
wavs only), with the STT faked, so it touches no recorded data and needs no
STT server. The source run must have spans from the old per-run Silero pass.
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_RUNS = ROOT / "data" / "runs"
TMP = Path(tempfile.mkdtemp(prefix="check-vadcut-"))
os.environ["DATA_DIR"] = str(TMP)          # before nc_bench reads its config
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nc_bench import marks, scoring, server, store, stt  # noqa: E402
from nc_bench.vads import bench  # noqa: E402


def ok(msg: str) -> None:
    print("OK  ", msg)


# ------------------------------------------------------------------ pure parts


def check_wer() -> None:
    def cells(reference, hypothesis):      # the cell-by-cell loop it replaced
        ref, hyp = scoring._tokens(reference), scoring._tokens(hypothesis)
        if not ref:
            return None
        d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
        d[:, 0] = np.arange(len(ref) + 1)
        d[0, :] = np.arange(len(hyp) + 1)
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
        e = int(d[len(ref), len(hyp)])
        return {"wer": round(e / len(ref), 3), "errors": e, "ref_words": len(ref)}

    rng = random.Random(7)
    words = "saya nak tanya pasal akaun boleh tak lah ok bank".split()
    for _ in range(2000):
        a = " ".join(rng.choice(words) for _ in range(rng.randint(0, 30)))
        b = " ".join(rng.choice(words) for _ in range(rng.randint(0, 30)))
        assert scoring.wer(a, b) == cells(a, b), (a, b)
    ok("vectorised WER equals the cell-by-cell edit distance on 2000 random pairs")


def check_cuts() -> None:
    cuts = stt.cut_list([[5.0, 8.0], [8.4, 10.0], [20.0, 20.005], [25.0, 26.0]], 0.5, 30.0)
    assert [c["segments"] for c in cuts] == [2, 1, 1]
    assert cuts[0]["span"] == [5.0, 10.0] and cuts[0]["start"] == 4.5
    assert stt.merge_turns([[5.0, 8.0], [8.4, 10.0]], 0.5, 30.0) == [[4.5, 10.0]]
    ok("cut_list keeps each request's own VAD span and how many segments it fused")


def check_marks() -> None:
    meta = {"truth_spans": [[1.0, 2.0], [5.0, 6.0]]}
    m1 = marks.ensure(meta)
    assert [m["id"] for m in m1] == [m["id"] for m in marks.ensure(meta)], "ids not stable"
    assert all(m["ref"] == "" for m in m1)
    a, b = m1[0]["id"], m1[1]["id"]
    ok("marks recorded before ids get the same id from their bounds on every read")

    marks.apply_refs(meta, {a: "hello there", b: "general kenobi"})
    # a new drag swallowing both, plus a new region elsewhere
    out = marks.apply_spans(meta, [{"id": a, "start": 1.0, "end": 2.0},
                                   {"id": b, "start": 5.0, "end": 6.0},
                                   {"start": 1.5, "end": 5.5}, {"start": 9.0, "end": 9.5}], 30.0)
    assert len(out) == 2, out
    assert out[0]["id"] == a and out[0]["ref"] == "hello there general kenobi" and out[0]["check"]
    assert out[1]["ref"] == "" and "check" not in out[1]
    assert meta["truth_spans"] == [[1.0, 6.0], [9.0, 9.5]]
    ok("a merge keeps the earliest id and both references in time order, flagged check")

    marks.apply_refs(meta, {a: "hello there general kenobi"})
    assert "check" not in meta["truth_marks"][0]
    out = marks.apply_spans(meta, [{"id": a, "start": 1.0, "end": 6.0}, {"start": 5.8, "end": 7.0},
                                   {"id": meta["truth_marks"][1]["id"], "start": 9.0, "end": 9.5}], 30.0)
    assert out[0]["check"] and out[0]["end"] == 7.0, out[0]
    ok("extending a referenced mark keeps its text and flags it; saving its text clears that")

    out = marks.apply_spans(meta, [[9.0, 9.5]], 30.0)   # an older caller: plain spans
    assert len(out) == 1 and out[0]["id"] == meta["truth_marks"][0]["id"]
    try:
        marks.apply_refs(meta, {a: "x"})
        raise AssertionError("a reference for a deleted mark was accepted")
    except marks.UnknownMarks:
        pass
    ok("plain [start, end] bodies keep matching marks; text for a deleted mark is refused")

    meta = {"truth_spans": [[1.0, 2.0], [3.0, 4.0], [6.0, 7.0]]}
    x, y, z = (m["id"] for m in marks.ensure(meta))
    marks.apply_refs(meta, {x: None, y: None, z: "hi"})
    out = marks.apply_spans(meta, [{"id": x, "start": 1.0, "end": 2.0}, {"id": y, "start": 3.0, "end": 4.0},
                                   {"start": 1.5, "end": 3.5}, {"id": z, "start": 6.0, "end": 7.0}], 30.0)
    assert out[0]["ref_empty"] and out[0]["ref"] == "" and out[0]["check"], out[0]
    assert out[1]["ref"] == "hi" and "ref_empty" not in out[1] and "check" not in out[1], out[1]
    out = marks.apply_spans(meta, [{"id": x, "start": 1.0, "end": 4.0}, {"start": 3.5, "end": 6.5},
                                   {"id": z, "start": 6.0, "end": 7.0}], 30.0)
    assert len(out) == 1 and out[0]["ref"] == "hi" and "ref_empty" not in out[0] and out[0]["check"]
    ok("merging marks expected empty stays expected empty; merged with text, the text wins; both flag check")


def check_reference() -> None:
    meta = {"truth_spans": [[1.0, 2.0], [5.0, 6.0]], "script": "typed at upload",
            "candidates": [{"id": "none", "stt": {"text": "hello there"}, "scores": {}}]}
    st = marks.sync_reference(meta)
    assert meta["script"] == "typed at upload" and st["source"] == "legacy"
    assert meta["candidates"][0]["scores"]["wer"]["ref_words"] == 3
    ok("no references yet: the script recorded with the run stays, and is scored")

    ids = [m["id"] for m in marks.ensure(meta)]
    marks.apply_refs(meta, {ids[1]: "general kenobi"})
    st = marks.sync_reference(meta)
    assert meta["script"] == "" and st["source"] == "marks-incomplete"
    assert "wer" not in meta["candidates"][0]["scores"]
    assert meta["script_legacy"] == "typed at upload"
    ok("partly referenced: no script and no WER anywhere; the old script kept aside")

    marks.apply_refs(meta, {ids[0]: "hello there"})
    st = marks.sync_reference(meta)
    assert meta["script"] == "hello there general kenobi" and st["complete"] and st["words"] == 4
    assert meta["candidates"][0]["scores"]["wer"] == scoring.wer("hello there general kenobi",
                                                                "hello there")
    ok("every mark referenced: script = references in time order, card WER re-scored")

    marks.apply_refs(meta, {ids[0]: None})
    st = marks.sync_reference(meta)
    assert meta["truth_marks"][0]["ref_empty"] and st["complete"] and st["empty"] == 1
    assert meta["script"] == "general kenobi" and st["words"] == 2
    assert meta["candidates"][0]["scores"]["wer"] == scoring.wer("general kenobi", "hello there")
    ok("a mark expected empty completes the reference with no words; text heard there scores as extra")

    marks.apply_refs(meta, {ids[1]: None})
    st = marks.sync_reference(meta)
    assert st["complete"] and st["source"] == "marks" and meta["script"] == "" and st["words"] == 0
    assert "wer" not in meta["candidates"][0]["scores"]
    ok("every mark expected empty: complete, but no reference words, so no WER")

    marks.apply_refs(meta, {ids[0]: "", ids[1]: ""})
    marks.sync_reference(meta)
    assert meta["script"] == "" and "script_source" not in meta
    assert not any(m.get("ref_empty") for m in meta["truth_marks"])
    ok("references cleared (expected-empty ones too): no script (the legacy one is not resurrected)")


def check_hyp() -> None:
    rec = {"text": "a b c", "cuts": [
        {"start": 0.5, "end": 2.0, "span": [1.0, 2.0], "text": "a"},
        {"start": 4.5, "end": 6.0, "span": [5.0, 6.0], "text": "b"},
        {"start": 8.5, "end": 9.0, "span": [9.0, 9.0], "text": ""},
        {"start": 9.5, "end": 11.0, "span": [10.0, 11.0], "text": "c"}]}
    assert bench.hyp_text(rec) == "a b c"
    assert bench.hyp_text(rec, [[4.8, 6.2]]) == "a c"
    assert bench.hyp_text(rec, [[4.5, 5.5]]) == "a b c"     # only partly inside
    assert bench.hyp_text({"text": "old join"}, [[0, 99]]) == "old join"
    ok("a cut wholly inside an ignore span is left out of the hypothesis; older joins scored as is")


# --------------------------------------------------------------- a real run


def copy_run(run_id: str) -> str:
    src = REAL_RUNS / run_id
    dst = store.config.RUNS_DIR / run_id
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name == "meta.json" or f.suffix == ".wav":
            shutil.copy2(f, dst / f.name)
    return run_id


async def check_run(run_id: str) -> None:
    server.broadcast = lambda ev: asyncio.sleep(0)   # no websocket in a script
    meta = store.load_meta(run_id)
    recorded = {"input": meta["input_vad"]} | {c["id"]: c["vad"] for c in meta["candidates"]
                                               if (c.get("vad") or {}).get("segments") is not None}
    before = (meta.get("vad_bench") or {}).get("results") or {}

    log = bench.run_bench(run_id, vads=["legacy-silero"], sources="all")
    vb = store.load_meta(run_id)["vad_bench"]
    rows = {sid: per["legacy-silero"] for sid, per in vb["results"].items() if "legacy-silero" in per}
    for sid, v in recorded.items():
        r = rows[sid]
        assert r["segments"] == [[float(a), float(b)] for a, b in v["segments"]], sid
        assert r["fixed"] and r["rank_excluded"] == bench.LEGACY_FIXED_WHY
        assert r["params"]["min_silence_duration"] == v["params"]["min_silence_duration"]
    for sid, per in before.items():
        for vid, r in per.items():
            assert vb["results"][sid][vid].get("segments") == r.get("segments"), (sid, vid)
    ok(f"legacy-silero: {len(recorded)} sources carry exactly the recorded spans, fixed and "
       f"unranked; {log['fixed']} rows written, every other row untouched")

    bench.run_bench(run_id, vads=["legacy-silero"], sources="all",
                    params={"min_silence_duration": 0.9})
    vb2 = store.load_meta(run_id)["vad_bench"]
    for sid in recorded:
        assert vb2["results"][sid]["legacy-silero"]["segments"] == rows[sid]["segments"], sid
    ok("a bench parameter change does not move the legacy-silero rows")

    # VAD-cut STT with a fake recogniser: each cut "hears" one word per second
    calls = {"n": 0}

    async def fake(piece, rate):
        calls["n"] += 1
        n = max(1, round(len(piece) / rate))
        return {"text": " ".join(["kata"] * n), "language": "Malay", "confidence": 0.9}
    stt._recognize = fake
    pick = dict(vads=["legacy-silero"], sources=["input", "none"], modes=["same", "raw"])
    slog = await bench.run_stt(run_id, **pick)
    m = store.load_meta(run_id)
    row = m["vad_bench"]["results"]["none"]["legacy-silero"]
    cuts = row["stt"]["same"]["cuts"]
    dur = row["duration_s"]
    expect = stt.cut_list(row["segments"], row["params"]["prefix_padding_duration"], dur)
    assert len(cuts) == len(expect) and all(c["text"] for c in cuts)
    assert slog["errors"] == 0 and slog["pairs"] == 2
    script = (m.get("script") or "").strip()
    want = scoring.wer(script, bench.hyp_text(row["stt"]["same"])) if script else None
    assert row["stt"]["same"]["wer"] == want
    ok(f"VAD-cut STT stores {len(cuts)} cuts with their own text; WER scored at merge")

    # the cut cache
    total = sum(len(m["vad_bench"]["results"][s]["legacy-silero"]["stt"][md]["cuts"])
                for s in ("input", "none") for md in ("same", "raw"))
    assert slog["sent"] == calls["n"] and slog["sent"] + slog["reused"] == total, (slog, total)
    assert slog["reused"] >= len(cuts), slog    # none read as same, then raw: one request each
    ok(f"first run: {slog['sent']} cuts sent, {slog['reused']} reused inside the job "
       "(`none` read as same and as raw is one request)")

    n0 = calls["n"]
    slog2 = await bench.run_stt(run_id, **pick)
    m2 = store.load_meta(run_id)
    assert calls["n"] == n0 and slog2["sent"] == 0 and slog2["reused"] == total, slog2
    assert m2["vad_bench"]["results"]["none"]["legacy-silero"]["stt"]["same"]["cuts"] == cuts
    ok(f"running it again on unchanged cuts sends nothing: all {total} cuts come from the cache")

    slog3 = await bench.run_stt(run_id, **pick, force=True)
    assert slog3["sent"] == total and slog3["reused"] == 0 and calls["n"] == n0 + total, slog3
    ok("force sends every cut again")

    # a moved cut is sent, the rest reused: one extra VAD segment far from the others
    # padded by 0.5 s, a 0.6 s segment needs ~2.2 s of gap to touch no other cut
    seg = [list(x) for x in row["segments"]]
    gap = next(((a, b) for (_s, a), (b, _e) in zip(seg, seg[1:]) if b - a > 2.4), None)
    if gap:
        extra = [(gap[0] + gap[1]) / 2, (gap[0] + gap[1]) / 2 + 0.6]
    elif seg and dur - seg[-1][1] > 1.8:
        extra = [seg[-1][1] + 1.0, seg[-1][1] + 1.6]
    else:
        extra = None
    name = row["stt"]["same"]["read"]
    cache = bench.cut_cache(m2["vad_bench"]["results"], store.run_dir(run_id), [name])
    if extra and name in cache:
        n1 = calls["n"]
        got = await stt.transcribe_turns(store.run_dir(run_id) / name, sorted(seg + [extra]),
                                         row["params"]["prefix_padding_duration"], cache=cache[name])
        assert got["sent"] == 1 and got["reused"] == len(cuts) and calls["n"] == n1 + 1, got
        ok("a re-segment that adds one cut sends that one cut; every unchanged cut is reused")
    else:
        print("SKIP the moved-cut check: no room for an extra cut between `none`'s segments")

    # what makes a cached cut invalid: other audio, another STT endpoint
    res = {s: {"legacy-silero": json.loads(json.dumps(m2["vad_bench"]["results"][s]["legacy-silero"]))}
           for s in ("input", "none")}
    reading = [r for s in res for r in res[s]["legacy-silero"]["stt"].values() if r["read"] == name]
    assert len(reading) == 3, [r["read"] for s in res for r in res[s]["legacy-silero"]["stt"].values()]
    reading[0]["read_fp"]["sha256"] = "0" * 64
    for r in reading[1:]:
        r["stt_key"] = "http://elsewhere/recognize|boost=true"
    full = bench.cut_cache(m2["vad_bench"]["results"], store.run_dir(run_id), [name])[name]
    left = bench.cut_cache(res, store.run_dir(run_id), [name])[name]
    assert full and not left, f"{len(left)} cuts survived a changed file and a changed endpoint"
    ok("cuts from audio that changed since, or from another STT endpoint, are never reused")

    # marks + references through the endpoints
    r = await server.set_truth(run_id, {"marks": [{"start": 1.0, "end": 3.0},
                                                  {"start": 5.0, "end": 7.5}]})
    ids = [x["id"] for x in r["marks"]]
    assert r["reference"]["marks"] == 2 and r["reference"]["referenced"] == 0
    await server.set_refs(run_id, {"refs": {ids[0]: "kata kata kata"}})
    m = store.load_meta(run_id)
    assert m["script"] == "" and m["script_source"] == "marks-incomplete"
    assert all("wer" not in (c.get("scores") or {}) for c in m["candidates"])
    assert m["vad_bench"]["results"]["none"]["legacy-silero"]["stt"]["same"]["wer"] is None
    ok("one of two marks referenced: every WER on the run is withdrawn")

    await server.set_refs(run_id, {"refs": {ids[1]: "kata kata"}})
    m = store.load_meta(run_id)
    assert m["script"] == "kata kata kata kata kata" and m["script_source"] == "marks"
    rec = m["vad_bench"]["results"]["none"]["legacy-silero"]["stt"]["same"]
    assert rec["wer"] == scoring.wer(m["script"], bench.hyp_text(rec))
    c = next(c for c in m["candidates"] if c["id"] == "none")
    assert c["scores"]["wer"] == scoring.wer(m["script"], c["stt"]["text"])
    ok("both referenced: script derived, card and VAD-cut WER re-scored without STT")

    await server.set_refs(run_id, {"refs": {ids[1]: None}})
    m = store.load_meta(run_id)
    assert m["truth_marks"][1]["ref_empty"] and m["truth_marks"][1]["ref"] == ""
    assert m["script"] == "kata kata kata" and m["script_source"] == "marks"
    rec = m["vad_bench"]["results"]["none"]["legacy-silero"]["stt"]["same"]
    assert rec["wer"] == scoring.wer("kata kata kata", bench.hyp_text(rec))
    st = (await server.api_run(run_id))["reference_state"]
    assert st["complete"] and st["empty"] == 1 and st["referenced"] == 2, st
    ok("a mark expected empty (null) completes the reference and adds no words; WER re-scored")

    # an ignore span may not overlap marked speech: mark the first cut's span,
    # ignore the last cut's (cuts are disjoint, so the two cannot touch)
    if len(rec["cuts"]) >= 2:
        first, last = rec["cuts"][0], rec["cuts"][-1]
        words_before = len(scoring._tokens(bench.hyp_text(rec)))
        r = await server.set_truth(run_id, {
            "marks": [{"start": first["span"][0], "end": first["span"][1]}],
            "ignore_spans": [[last["span"][0] - 0.01, last["span"][1] + 0.01]]})
        await server.set_refs(run_id, {"refs": {r["marks"][0]["id"]: "kata kata"}})
        m = store.load_meta(run_id)
        rec2 = m["vad_bench"]["results"]["none"]["legacy-silero"]["stt"]["same"]
        hyp2 = bench.hyp_text(rec2, m["ignore_spans"])
        assert len(scoring._tokens(hyp2)) == words_before - len(scoring._tokens(last["text"]))
        assert rec2["wer"] == scoring.wer(m["script"], hyp2) and m["script"] == "kata kata"
        ok("an ignore span over a cut re-scores the VAD-cut WER without that cut")
        ids = [x["id"] for x in r["marks"]]
    else:
        print(f"SKIP the ignore-span check: `none` has {len(rec['cuts'])} cut(s) on this run, needs 2")

    try:
        await server.set_refs(run_id, {"refs": {"mdeadbeef": "x"}})
        raise AssertionError("text for an unknown mark was accepted")
    except Exception as e:
        assert getattr(e, "status_code", None) == 409, e
    ok("references for a mark the run no longer has are refused with 409")

    v = await server.api_run(run_id)
    assert [x["id"] for x in v["truth_marks"]] == ids and v["reference_state"]["complete"]
    ok("GET /api/runs/<id> returns the marks with their ids and the reference state")

    # an STT that judges every cut noise: empty cuts are results, not errors
    async def silent(piece, rate):
        return {"text": "", "language": None, "confidence": 0.12}
    stt._recognize = silent
    # force: these cuts are cached from the fake above, and a cache hit is the point
    slog = await bench.run_stt(run_id, vads=["legacy-silero"], sources=["none"], modes=["same"],
                               force=True)
    m = store.load_meta(run_id)
    rec = m["vad_bench"]["results"]["none"]["legacy-silero"]["stt"]["same"]
    assert slog["errors"] == 0 and rec["cuts"] and not rec.get("error")
    assert rec["text"] == "" and all(c["text"] == "" for c in rec["cuts"])
    words = len(scoring._tokens(m["script"]))
    assert rec["wer"]["errors"] == rec["wer"]["ref_words"] == words, rec["wer"]
    ok(f"an STT returning nothing for all {len(rec['cuts'])} cuts is stored as empty cuts, "
       "and every reference word scores as deleted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260802-203822")
    args = ap.parse_args()
    try:
        check_wer()
        check_cuts()
        check_marks()
        check_reference()
        check_hyp()
        if not (REAL_RUNS / args.run / "meta.json").is_file():
            print(f"SKIP the run-level checks: no data/runs/{args.run}")
        else:
            src = json.loads((REAL_RUNS / args.run / "meta.json").read_text())
            if (src.get("input_vad") or {}).get("segments") is None:
                print(f"SKIP the run-level checks: {args.run} has no recorded Silero spans")
            else:
                asyncio.run(check_run(copy_run(args.run)))
        print("\ncheck_vad_cut_wer passed")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
