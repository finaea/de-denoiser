"""VAD bench from the command line: the same code the UI's VAD bench runs.

    .venv/bin/python scripts/vad_bench.py --runs 20260804-112511 --vads all --sources input,none,hush
    .venv/bin/python scripts/vad_bench.py --runs all --vads lk-inference,firered-stream \\
        --sources all --params min_speech_duration=0.1 --resegment-only
    .venv/bin/python scripts/vad_bench.py --runs all --vads all --sources ticked --stt same,raw

--runs      run ids, comma separated, or `all` (every run under data/runs/ whose
            id does not start with `_`, so scratch copies are never swept by accident)
--vads      VAD ids (`silero-v4.0-8k`), family ids (`silero-v4.0` = both rates), or `all`
--sources   `input`, candidate ids, `all`, or `ticked` (input plus the candidates
            named by --ticked; without --ticked, every candidate the run has)
--params    k=v for activation_threshold, deactivation_threshold, min_speech_duration,
            min_silence_duration, prefix_padding_duration, smoothing (true/false).
            Unset keys keep the run's stored VAD-bench params, else its VAD panel values.
--threshold vad=thr, per VAD or family (FireRed and TEN were best at 0.7 on PR-42).
            Overrides are saved on the run; `--threshold none` clears them.
            Any parameter or threshold change re-segments EVERY stored pair on the
            run (no model runs), so a run never mixes two parameter sets.
--stt       also run VAD-cut WER in these read modes (same, raw). Expensive: one
            STT request per turn per pair per mode.
--resegment-only   never run a model; re-segment and re-score stored probabilities
--force     re-infer even where fresh probabilities are stored

Resumable and idempotent: a (source, VAD) whose stored probabilities are fresh
(same source wav, same pinned weights) is not re-inferred. Writes only
meta["vad_bench"] plus <source>.<vad>.vadp.{npy,json}; the first write to a run
leaves meta.pre-vadbench.json beside it. Do not run it while the web UI is
processing the same run: both would rewrite that run's meta.json.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import config, store  # noqa: E402
from nc_bench.vads import bench  # noqa: E402


def _kv(items, what):
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"{what}: expected k=v, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _runs(arg: str) -> list[str]:
    if arg == "all":
        return sorted(d.name for d in config.RUNS_DIR.iterdir()
                      if (d / "meta.json").is_file() and not d.name.startswith("_"))
    return [r.strip() for r in arg.split(",") if r.strip()]


def _fmt(v, pct=False, n=2):
    if v is None:
        return "—"
    return f"{v * 100:.0f}%" if pct else f"{v:.{n}f}"


def print_table(run_id: str, only_sources=None, only_vads=None) -> None:
    res = bench.fresh_view(run_id).get("results") or {}
    print(f"\n{run_id}")
    print(f"  {'source':<14} {'vad':<20} {'segs':>4} {'recall':>6} {'miss':>4} {'phan':>4} "
          f"{'phan_s':>6} {'merge':>5} {'over_s':>6} {'lag_md':>6} {'frag':>4} {'agree':>6} "
          f"{'fa_s':>5} {'cpu':>6}  wer same/raw")
    n = lambda v: v if v is not None else "—"  # noqa: E731
    for sid, per in res.items():
        if only_sources and sid not in only_sources:
            continue
        for vid, r in per.items():
            if only_vads and vid not in only_vads:
                continue
            if r.get("segments") is None:
                what = (f"STALE: {r.get('stale_reason')}" if r.get("stale")
                        else r.get("unavailable") or r.get("error"))
                print(f"  {sid:<14} {vid:<20} {what}")
                continue
            ev, fr = r.get("events") or {}, r.get("frame") or {}
            st = r.get("stt") or {}
            wers = "/".join("stale" if (st.get(m) or {}).get("stale")
                            else _fmt(((st.get(m) or {}).get("wer") or {}).get("wer"), pct=True)
                            for m in bench.STT_MODES)
            tag = ("  STALE: " + r["stale_reason"] if r.get("stale")
                   else "  NOT RANKED: " + r["rank_excluded"] if r.get("rank_excluded") else "")
            print(f"  {sid:<14} {vid:<20} {r['n']:>4} {_fmt(ev.get('turn_recall'), True):>6} "
                  f"{n(ev.get('missed_n')):>4} {n(ev.get('phantom_n')):>4} "
                  f"{_fmt(ev.get('phantom_s')):>6} {n(ev.get('merged_turns')):>5} "
                  f"{_fmt(ev.get('overreach_s')):>6} {_fmt(ev.get('onset_lag_med_s')):>6} "
                  f"{_fmt(ev.get('frag'), n=1):>4} {_fmt(fr.get('agree'), True):>6} "
                  f"{_fmt(fr.get('fa_s')):>5} {_fmt(r.get('cpu_ms_per_s'), n=1):>6}  {wers}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--vads", default="all")
    ap.add_argument("--sources", default="ticked")
    ap.add_argument("--ticked", default=None, help="NC candidate ids that `--sources ticked` means")
    ap.add_argument("--params", nargs="*", default=[])
    ap.add_argument("--threshold", nargs="*", default=[])
    ap.add_argument("--stt", default="", help="comma list of read modes: same,raw")
    ap.add_argument("--resegment-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="no per-pair progress lines")
    a = ap.parse_args()

    params = _kv(a.params, "--params") or None
    # overrides persist per run; `--threshold none` clears them, no flag keeps them
    thresholds = {} if a.threshold == ["none"] else (_kv(a.threshold, "--threshold") or None)
    modes = [m.strip() for m in a.stt.split(",") if m.strip()]
    bad = [m for m in modes if m not in bench.STT_MODES]
    if bad:
        sys.exit(f"--stt: unknown modes {bad}; use {list(bench.STT_MODES)}")

    def progress(ev):
        if not a.quiet and ev.get("stage") not in ("cached",):
            print(f"  [{ev['done']}/{ev['total']}] {ev['source']} × {ev['vad']}: {ev['stage']}",
                  flush=True)

    failures = 0
    for run_id in _runs(a.runs):
        try:
            meta = store.load_meta(run_id)
        except (KeyError, FileNotFoundError):
            print(f"{run_id}: no such run", file=sys.stderr)
            failures += 1
            continue
        sources = a.sources
        if sources == "ticked" and a.ticked is None:
            sources = "all"
        ticked = [t.strip() for t in (a.ticked or "").split(",") if t.strip()]
        t0 = time.time()
        try:
            log = bench.run_bench(run_id, vads=a.vads, sources=sources, ticked=ticked,
                                  params=params, thresholds=thresholds, force=a.force,
                                  resegment_only=a.resegment_only, progress=progress)
        except ValueError as e:
            print(f"{run_id}: {e}", file=sys.stderr)
            failures += 1
            continue
        print(f"{run_id}: {log['inferred']} inferred, {log['cached']} cached, "
              f"{log['resegmented']} re-segmented, {log['unavailable']} unavailable, "
              f"{log['skipped']} skipped, {log['errors']} errors  ({time.time() - t0:.1f}s)")
        if modes:
            async def aprog(ev):
                progress(ev)
            slog = asyncio.run(bench.run_stt(run_id, vads=log["vads"], sources=log["sources"],
                                             modes=modes, progress=aprog))
            print(f"{run_id}: VAD-cut STT on {slog['pairs']} pairs ({', '.join(modes)}), "
                  f"{slog['errors']} errors ({slog['elapsed_s']}s)")
        print_table(run_id, set(log["sources"]), set(log["vads"]))
        del meta
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
