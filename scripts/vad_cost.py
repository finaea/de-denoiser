"""CPU cost per VAD, measured once on an idle machine: the report's cost column.

    .venv/bin/python scripts/vad_cost.py [--run 20260804-112511] [--source none] [-n 3]

The costs stored with each VAD-bench row were measured while other jobs shared
the machine (the round-2 backfill ran 6 in parallel on a loaded Linux box), so
they are not comparable. This re-measures every available inference VAD on one
source, single-threaded as the loaders are, best of n. Read-only.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import store, vads  # noqa: E402
from nc_bench.vads.bench import _read_mono, source_files  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="20260804-112511")
    ap.add_argument("--source", default="none")
    ap.add_argument("-n", type=int, default=3)
    a = ap.parse_args()
    meta = store.load_meta(a.run)
    pcm, rate = _read_mono(store.run_dir(a.run) / source_files(meta)[a.source])
    print(f"| VAD | cpu ms / audio s ({platform.system()} {platform.machine()}, "
          f"{a.run} {a.source}, best of {a.n}) |\n|---|---|")
    for e in vads.registry():
        if not e["available"] or e["loader"] == "lkaic_flag":
            continue
        best = min(vads.infer(e, pcm, rate)["cpu_ms_per_s"] for _ in range(a.n))
        print(f"| `{e['id']}` | {best:.1f} |")


if __name__ == "__main__":
    main()
