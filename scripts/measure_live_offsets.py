"""Store each live-rail output's offset against its input, for the report.

    .venv/bin/python scripts/measure_live_offsets.py [--dry-run]

The live rail (recorder.py) opens one extra stream per cloud candidate on the
track it records. Those streams attach after the track starts, so on the stored
runs `aic-*` and `krisp-bvc-tel` outputs begin 50-230 ms INTO the input, and
every event in them sits that much early on input time; `krisp-nc` runs ~30 ms
late (its filter delay). The hand marks are on input time, so the report shows
these rows as stored and offset-corrected. The correction needs this number,
and the report reads only meta.json, so it is measured here once and stored,
additively, as `live_rail_offset` on each live-rail record:

    {lag_ms, len_diff_ms, env_corr, prominence, measured_at}

lag_ms < 0: the output starts |lag| into the input. Same measurement as
scripts/restore_orphans.py (10 ms envelopes, peak within ±1 s).
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nc_bench import config, store  # noqa: E402
from restore_orphans import measure  # noqa: E402

LIVE_RAIL_IDS = ("krisp-nc", "krisp-bvc", "krisp-bvc-tel", "aic-quail-l", "aic-vf-s", "aic-vf-l")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for d in sorted(config.RUNS_DIR.iterdir()):
        if d.name.startswith("_") or not (d / "meta.json").is_file():
            continue
        meta = store.load_meta(d.name)
        got = {}
        for c in meta.get("candidates") or []:
            if c.get("id") not in LIVE_RAIL_IDS or c.get("rail") == "cloud_replay" \
                    or not c.get("output") or not (d / c["output"]).is_file():
                continue
            g = measure(d / meta["input"]["file"], d / c["output"])
            got[c["id"]] = {k: g[k] for k in ("lag_ms", "len_diff_ms", "env_corr", "prominence")}
            got[c["id"]]["measured_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            print(f"{d.name} {c['id']:14s} lag {g['lag_ms']:+6.0f} ms  len {g['len_diff_ms']:+7.1f} ms  "
                  f"prom {g['prominence']:.3f}")
        if got and not args.dry_run:
            with store.locked_meta(d.name) as m:
                for c in m.get("candidates") or []:
                    if c.get("id") in got and c.get("output") and c.get("rail") != "cloud_replay":
                        c["live_rail_offset"] = got[c["id"]]


if __name__ == "__main__":
    main()
