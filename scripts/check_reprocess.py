"""Does reprocessing a run keep everything that was NOT the candidate list?

Worth a script because _process_run was written for fresh runs and overwrites
meta keys on purpose — meta["script"] is assigned from its argument. Reprocess
routes around that, and the script is now derived from the marks' references,
which must survive too. Nothing about a dropped note or a lost reference is
visible in the UI afterwards: the run just looks finished, with every WER gone.

Runs against a throwaway copy of a real run's input.wav, so it never touches
recorded data.

    .venv/bin/python scripts/check_reprocess.py
"""

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import scoring, server, store  # noqa: E402

FIRST = ["none"]
THEN = ["none", "hecttor-coda-vi", "gtcrn"]
SCRIPT = "so this is a mic test on how well the noise is working"
NOTE = "conditions worth remembering"
MARKS = [[1.0, 4.0], [6.0, 9.0]]
REFS = ["so this is a mic test", "on how well the noise is working"]


def _find_input() -> Path | None:
    for d in sorted(store.config.RUNS_DIR.iterdir(), reverse=True):
        wav = d / "input.wav"
        if wav.is_file() and wav.stat().st_size > 200_000:
            return wav
    return None


async def main() -> None:
    src = _find_input()
    if src is None:
        print("SKIP: no recorded run to copy an input.wav from")
        return
    server.broadcast = lambda ev: asyncio.sleep(0)  # no websocket in a script

    run_id, run_dir = store.new_run("phone", NOTE)
    made = [run_id]
    try:
        shutil.copy(src, run_dir / "input.wav")
        import soundfile as sf

        info = sf.info(run_dir / "input.wav")
        meta = store.load_meta(run_id)
        meta["input"] = {"file": "input.wav", "duration_s": round(info.duration, 2),
                         "sample_rate": info.samplerate}
        store.save_meta(run_id, meta)

        await server._process_run(run_id, FIRST, None, SCRIPT, 1)
        await server.set_truth(run_id, {"spans": MARKS})
        before = store.load_meta(run_id)
        assert [c["id"] for c in before["candidates"]] == FIRST
        assert before["script"] == SCRIPT and before.get("script_source") == "legacy", before
        print(f"seeded {run_id}: {FIRST} · {len(before['truth_spans'])} marks · a typed script")

        # references on both marks take the script over
        ids = [m["id"] for m in (await server.api_run(run_id))["truth_marks"]]
        await server.set_refs(run_id, {"refs": dict(zip(ids, REFS))})
        mid = store.load_meta(run_id)
        assert mid["script"] == " ".join(REFS) and mid["script_legacy"] == SCRIPT, mid["script"]
        print("       then referenced both marks: the script is now built from them")

        r = await server.reprocess_run(run_id, {"candidates": THEN, "concurrency": 2})
        assert r["status"] == "processing"
        while run_id in server._reprocessing:
            await asyncio.sleep(0.5)

        after = store.load_meta(run_id)
        # merged by id: the stored order first, new ids after in ticked order —
        # here FIRST is a prefix of THEN, so the order equals THEN
        assert [c["id"] for c in after["candidates"]] == THEN, after["candidates"]
        print(f"OK   candidate set merged: {[c['id'] for c in after['candidates']]}")
        assert after["script"] == " ".join(REFS), f"script lost: {after['script']!r}"
        assert [m["ref"] for m in after["truth_marks"]] == REFS, after["truth_marks"]
        print("OK   references and the script built from them survived (the fragile one)")
        assert after["note"] == NOTE, f"note lost: {after['note']!r}"
        print("OK   note survived")
        assert after["truth_spans"] == [[float(a), float(b)] for a, b in MARKS], after["truth_spans"]
        print("OK   hand marks survived")
        assert after["source"] == "phone", after["source"]
        print("OK   source unchanged (an upload copy would have become 'upload')")

        # the new candidates must be transcribed and scored against the
        # references, not just listed. An unavailable one (Hecttor without its
        # wheel/key) is an error record — that is right, not a failure
        known = server._candidates_by_id()
        done = [c for c in after["candidates"] if c.get("status") == "done"]
        for c in after["candidates"]:
            if not known[c["id"]]["available"]:
                assert c.get("status") == "error", f"{c['id']} is unavailable but not an error"
        assert len(done) >= 2, [c["id"] for c in done]
        for c in done:
            assert c["scores"]["wer"] == scoring.wer(after["script"], c["stt"]["text"]), c["id"]
            assert "vad" not in c, f"{c['id']} got per-run VAD spans; that pass is gone"
        print(f"OK   every new candidate that ran ({[c['id'] for c in done]}) is scored against "
              "the references, with no per-run VAD; unavailable ones are error records")

        # and the guards
        for label, rid, body, code in [
            ("unknown run", "nope", {"candidates": FIRST}, 404),
            ("no candidates", run_id, {}, 400),
            ("unknown candidate", run_id, {"candidates": ["bogus"]}, 400),
            # cloud candidates now replay a stored wav (cloud_replay), so the
            # candidate a reprocess must refuse is a historical one
            ("historical candidate", run_id, {"candidates": ["aic-vf-l"]}, 400),
        ]:
            try:
                await server.reprocess_run(rid, body)
                raise AssertionError(f"{label} was accepted")
            except Exception as e:
                got = getattr(e, "status_code", None)
                assert got == code, f"{label}: expected {code}, got {got} ({e})"
        print("OK   guards reject unknown run / empty set / unknown id / historical")

        print("\ncheck_reprocess passed")
    finally:
        for rid in made:
            shutil.rmtree(store.run_dir(rid), ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
