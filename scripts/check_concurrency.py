"""Does concurrent processing change the RESULTS, or only the wall clock?

Runs the same input through the same candidates at concurrency 1 and 4 and
requires the output audio to be byte-identical.

Worth a script because the failure would be silent and would poison every
comparison the bench exists to make: any shared or global state inside a
processor — the Hecttor SDK is proprietary, machine-licensed and opaque, and the
ONNX wrappers hold analysis buffers and caches per instance — would corrupt runs
in a way that looks like a real difference between candidates.

Also pins that meta stays in ticked order however completion interleaves, so two
runs of the same candidate set remain readable side by side.

    .venv/bin/python scripts/check_concurrency.py
"""

import asyncio
import hashlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import server, store  # noqa: E402

# A spread of chain shapes: passthrough, the commercial SDK, ONNX spectral,
# wav2wav, a native lib, and an 8 kHz variant.
CANDS = ["none", "hecttor-coda-vi", "hecttor-coda", "gtcrn",
         "dtln", "dpdfnet2-8k", "fastenhancer-t", "hush"]
SCRIPT = "so this is an outbound phone call test can you uh hear me"


def _find_input() -> Path | None:
    for d in sorted(store.config.RUNS_DIR.iterdir(), reverse=True):
        wav = d / "input.wav"
        if wav.is_file() and wav.stat().st_size > 200_000:
            return wav
    return None


async def _once(src: Path, conc: int):
    run_id, run_dir = store.new_run("upload", f"concurrency probe {conc}")
    shutil.copy(src, run_dir / "input.wav")
    t0 = time.monotonic()
    await server._process_run(run_id, CANDS, None, SCRIPT, conc)
    elapsed = time.monotonic() - t0
    meta = store.load_meta(run_id)
    digests = {}
    for e in meta["candidates"]:
        out = run_dir / (e.get("output") or "")
        digests[e["id"]] = (
            hashlib.sha256(out.read_bytes()).hexdigest()[:16]
            if e.get("output") and out.exists()
            else f"ERR:{str(e.get('error'))[:40]}"
        )
    order = [e["id"] for e in meta["candidates"]]
    return run_id, elapsed, digests, order, meta.get("concurrency")


async def main() -> None:
    src = _find_input()
    if src is None:
        print("SKIP: no recorded run to reuse as input")
        return
    server.broadcast = lambda ev: asyncio.sleep(0)  # no websocket in a script

    made = []
    try:
        r1, t1, d1, o1, c1 = await _once(src, 1)
        made.append(r1)
        r4, t4, d4, o4, c4 = await _once(src, 4)
        made.append(r4)

        print(f"serial : {t1:6.1f}s  meta.concurrency={c1}")
        print(f"conc 4 : {t4:6.1f}s  meta.concurrency={c4}   speedup {t1 / t4:.2f}x")
        assert c1 == 1 and c4 == 4, "concurrency not recorded on the run"

        assert o1 == CANDS, f"serial order drifted: {o1}"
        assert o4 == CANDS, f"concurrent order drifted: {o4}"
        print(f"OK   meta in ticked order at both settings ({len(CANDS)} candidates)")

        diff = [k for k in d1 if d1[k] != d4.get(k)]
        for k in diff:
            print(f"  MISMATCH {k}: serial {d1[k]}  conc4 {d4.get(k)}")
        assert not diff, "concurrency changed the audio — a processor has shared state"
        print(f"OK   output byte-identical for all {len(d1)} candidates")
        print("\ncheck_concurrency passed")
    finally:
        for run_id in made:
            shutil.rmtree(store.run_dir(run_id), ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
