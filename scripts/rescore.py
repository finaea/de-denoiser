"""Recompute the scores of stored runs from the audio already on disk.

Scoring evolves (gap gating, measured band, new metrics) but runs keep whatever
numbers were current when they were recorded — so history and the rankings end
up mixing metric versions, which is exactly the comparison the tool exists to
avoid. This recomputes every score from the wavs; transcripts are reused, so no
STT call is made and no audio is reprocessed.

    .venv/bin/python scripts/rescore.py            # every run
    .venv/bin/python scripts/rescore.py 20260801-141543 ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import scoring, store  # noqa: E402


def rescore(run_id: str) -> None:
    meta = store.load_meta(run_id)
    run_dir = store.run_dir(run_id)
    inp = run_dir / ((meta.get("input") or {}).get("file") or "input.wav")
    if not inp.exists():
        print(f"skip {run_id}: no input audio")
        return

    before = (meta.get("input_scores") or {}).get("gap_rms_db")
    meta["input_scores"] = scoring.score_input(inp)
    after = meta["input_scores"].get("gap_rms_db")
    script = meta.get("script") or ""

    changed = 0
    for entry in meta.get("candidates") or []:
        out = run_dir / (entry.get("output") or "")
        if not entry.get("output") or not out.exists():
            continue
        try:
            entry["scores"] = scoring.score_output(
                out, meta["input_scores"], script, (entry.get("stt") or {}).get("text", "")
            )
            entry.pop("scores_error", None)
            changed += 1
        except Exception as e:  # keep going; one bad candidate isn't fatal
            entry["scores_error"] = str(e)
    store.save_meta(run_id, meta)
    print(f"{run_id}: floor {before} -> {after} dBFS, {changed} candidates rescored")


if __name__ == "__main__":
    ids = sys.argv[1:] or [r["id"] for r in store.list_runs()]
    for rid in ids:
        try:
            rescore(rid)
        except Exception as e:
            print(f"{rid}: FAILED {e}")
