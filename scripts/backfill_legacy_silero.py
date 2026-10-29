"""Give every recorded run its `legacy-silero` rows: the spans the old per-run
Silero pass stored with the run, read into the VAD bench as they are.

    .venv/bin/python scripts/backfill_legacy_silero.py            # every run
    .venv/bin/python scripts/backfill_legacy_silero.py --runs 20260802-203822,...
    .venv/bin/python scripts/backfill_legacy_silero.py --dry-run

No model runs and no other row changes: legacy-silero rows are fixed, so the
bench's parameters and every stored (source, VAD) result stay as they are.
Runs with no recorded spans (recorded after that pass was removed) are skipped.
Safe to run again; a run already carrying the rows is just rewritten with the
same spans. The bench's first write to a run leaves meta.pre-vadbench.json.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import store  # noqa: E402
from nc_bench.vads import bench  # noqa: E402


def recorded_sources(meta: dict) -> list[str]:
    return [sid for sid in bench.source_files(meta) if bench.legacy_spans(meta, sid) is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="comma-separated run ids (default: every run)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ids = ([r.strip() for r in args.runs.split(",") if r.strip()] if args.runs
           else sorted(d.name for d in store.config.RUNS_DIR.iterdir() if (d / "meta.json").is_file()))
    done = skipped = 0
    for rid in ids:
        meta = store.load_meta(rid)
        sids = recorded_sources(meta)
        if not sids:
            print(f"skip {rid}: no spans recorded with it")
            skipped += 1
            continue
        if args.dry_run:
            print(f"would backfill {rid}: {len(sids)} sources")
            done += 1
            continue
        log = bench.run_bench(rid, vads=["legacy-silero"], sources=sids)
        print(f"ok   {rid}: {log['fixed']} legacy-silero rows ({log['errors']} errors, "
              f"{log['stale_rows']} stale rows in the run)")
        done += 1
    print(f"\n{done} runs {'to backfill' if args.dry_run else 'backfilled'}, "
          f"{skipped} without recorded spans")


if __name__ == "__main__":
    main()
