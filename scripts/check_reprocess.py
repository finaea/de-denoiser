"""Does reprocessing a run keep everything that was NOT the candidate list?

Worth a script because _process_run was written for fresh runs and overwrites
meta keys on purpose — meta["script"] is assigned from its argument, and
meta["vad_params"] used to be reset to defaults every time. Reprocess routes
around both. Nothing about a dropped note or a silently reset VAD rate is visible
in the UI afterwards: the run just looks finished, every WER is gone or every span
was measured at a rate that no longer matches the marks drawn against it.

Runs against a throwaway copy of a real run's input.wav, so it never touches
recorded data.

    .venv/bin/python scripts/check_reprocess.py
"""

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import server, store  # noqa: E402

FIRST = ["none"]
THEN = ["none", "hecttor-coda-vi", "gtcrn"]
SCRIPT = "so this is a mic test on how well the noise is working"
NOTE = "conditions worth remembering"
MARKS = [[1.0, 4.0], [6.0, 9.0]]


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

    # a phone-source run, so the 8 kHz VAD default is in play and a reset to the
    # global default would be visible
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
        assert before["vad_params"]["sample_rate"] == 8000, before["vad_params"]
        print(f"seeded {run_id}: {FIRST} · rate {before['vad_params']['sample_rate']} · "
              f"{len(before['truth_spans'])} marks")

        # deliberately diverge the rate, so a reset to defaults is detectable
        await server.rerun_vad(run_id, {"params": {"sample_rate": 16000,
                                                   "min_silence_duration": 0.9}})
        mid = store.load_meta(run_id)
        assert mid["vad_params"]["sample_rate"] == 16000
        print("       then forced rate 16000 / min_silence 0.9 (a reset would undo this)")

        r = await server.reprocess_run(run_id, {"candidates": THEN, "concurrency": 2})
        assert r["status"] == "processing"
        while run_id in server._reprocessing:
            await asyncio.sleep(0.5)

        after = store.load_meta(run_id)
        assert [c["id"] for c in after["candidates"]] == THEN, after["candidates"]
        print(f"OK   candidate set replaced: {[c['id'] for c in after['candidates']]}")
        assert after["script"] == SCRIPT, f"script lost: {after['script']!r}"
        print("OK   script survived (it is a _process_run argument, so this is the fragile one)")
        assert after["note"] == NOTE, f"note lost: {after['note']!r}"
        print("OK   note survived")
        assert after["truth_spans"] == [[float(a), float(b)] for a, b in MARKS], after["truth_spans"]
        print("OK   hand marks survived")
        assert after["source"] == "phone", after["source"]
        print("OK   source unchanged (an upload copy would have become 'upload')")
        assert after["vad_params"] == mid["vad_params"], (
            f"vad params reset to defaults: {after['vad_params']} != {mid['vad_params']}"
        )
        print(f"OK   VAD params kept the run's own values, not defaults "
              f"(rate {after['vad_params']['sample_rate']}, "
              f"min_silence {after['vad_params']['min_silence_duration']})")

        # the new candidates must be measured and scored, not just listed
        assert all(c.get("vad") for c in after["candidates"]), "a candidate has no spans"
        assert all(c.get("truth_score") for c in after["candidates"]), "a candidate is unscored"
        assert all(c["vad"]["params"] == after["vad_params"] for c in after["candidates"])
        print("OK   every new candidate got spans + a truth score at the run's own params")

        # and the guards
        for label, rid, body, code in [
            ("unknown run", "nope", {"candidates": FIRST}, 404),
            ("no candidates", run_id, {}, 400),
            ("unknown candidate", run_id, {"candidates": ["bogus"]}, 400),
            ("live-only candidate", run_id, {"candidates": ["krisp-nc"]}, 400),
        ]:
            try:
                await server.reprocess_run(rid, body)
                raise AssertionError(f"{label} was accepted")
            except Exception as e:
                got = getattr(e, "status_code", None)
                assert got == code, f"{label}: expected {code}, got {got} ({e})"
        print("OK   guards reject unknown run / empty set / unknown id / live-only")

        print("\ncheck_reprocess passed")
    finally:
        for rid in made:
            shutil.rmtree(store.run_dir(rid), ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
