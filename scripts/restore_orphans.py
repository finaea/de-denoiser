"""Re-attach candidate outputs a stored run lost from its meta.json.

    .venv/bin/python scripts/restore_orphans.py --run 20260804-131319 --before 2026-08-10 --dry-run
    .venv/bin/python scripts/restore_orphans.py --run 20260804-131319 --before 2026-08-10

Before 2026-09-29, "Re-run ticked candidates" replaced meta["candidates"]
wholesale. On 2026-08-10, 20260804-131319 was re-run with 7 candidates, and the
other 28 records vanished from its meta.json while their wavs stayed on disk. The
Hecttor and live-rail outputs among them can't be made again. This finds every
`<cid>.wav` no record points at, and attaches it only if it provably belongs to
this run's input.wav:

- its length matches the input's within 50 ms;
- its 10 ms envelope lines up with the input's at a lag within ±50 ms, with a
  clear peak (prominence >= 0.15: the cloud replay's rule, which holds for heavy
  NC as well);
- its mtime predates the reprocess (`--before`), and the input's mtime predates it.

A wav that fails a check stays an orphan, listed with the reason. A restored
record gets `restored_orphan: true`, `restored_at`, `restored_from`, the label
from candidates.json, and null for everything that can't be known now (chain,
latency, cost). Then it is scored like any new record: DNSMOS, gap and band
against the run's stored input_scores; whole-file STT with the current setting;
the legacy Silero spans at the run's vad_params; truth scores.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import cloud_replay, config, scoring, store, stt, vad  # noqa: E402

MAX_LEN_DIFF_S = 0.05
MAX_LAG_MS = 50.0
MIN_PROMINENCE = 0.15
LIVE_RAIL = ("krisp-", "aic-")   # the cloud ids that were live-rail on a phone session
PASS = "round2-2026-09-29"


def _labels() -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(config.CANDIDATES_FILE.read_text())}


def _mtime(p: Path) -> str:
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def measure(input_wav: Path, wav: Path) -> dict:
    """How one output sits against its input: {len_diff_ms, lag_ms, env_corr,
    prominence, mtime}. lag_ms > 0: the output runs late (a model's own delay);
    < 0: it starts part-way into the input (a live-rail stream that attached late)."""
    x = cloud_replay._mono(input_wav, config.PIPELINE_RATE)
    y = cloud_replay._mono(wav, config.PIPELINE_RATE)
    out = {"len_diff_ms": round((len(y) - len(x)) / config.PIPELINE_RATE * 1000, 1),
           "mtime": _mtime(wav)}
    ex, ey = cloud_replay._envelope(x), cloud_replay._envelope(y)
    n = min(len(ex), len(ey))
    ex, ey = ex[:n], ey[:n]

    def at(d: int) -> float:
        a, b = (ex[: n - d], ey[d:]) if d >= 0 else (ex[-d:], ey[: n + d])
        return cloud_replay._corr(a, b)

    corr = {d: at(d) for d in range(-100, 101)}   # 10 ms hops: ±1 s
    best = max(corr, key=corr.get)
    # prominence at the peak: the output shifted by the lag found, so a model's
    # own structural delay does not count against the peak's clarity
    a, b = (ex[: n - best], ey[best:]) if best >= 0 else (ex[-best:], ey[: n + best])
    out.update(lag_ms=best * 10.0, env_corr=round(corr[best], 4),
               prominence=cloud_replay.prominence(a, b))
    return out


def reference(cid: str, rid: str, surface: str | None) -> dict | None:
    """The same candidate's median lag and length diff on the OTHER stored runs of
    this surface, from their own records: what this candidate normally looks like."""
    lags, lens = [], []
    for d in sorted(config.RUNS_DIR.iterdir()):
        if d.name == rid or d.name.startswith("_") or not (d / "meta.json").is_file():
            continue
        m = json.loads((d / "meta.json").read_text())
        if _surface(m) != surface:
            continue
        rec = next((c for c in m.get("candidates") or [] if c.get("id") == cid), None)
        if not rec or not rec.get("output") or rec.get("restored_orphan") \
                or not (d / rec["output"]).is_file():
            continue
        g = measure(d / m["input"]["file"], d / rec["output"])
        lags.append(g["lag_ms"])
        lens.append(g["len_diff_ms"])
    if not lags:
        return None
    return {"median_lag_ms": float(np.median(lags)), "median_len_diff_ms": float(np.median(lens)),
            "runs": len(lags)}


def _surface(m: dict) -> str | None:
    if m.get("source") in ("phone", "web"):
        return m["source"]
    note = (m.get("note") or "").lower()
    return "phone" if "(phone)" in note else "web" if "(web)" in note else None


def gate(g: dict, ref: dict | None, before: str) -> dict:
    """Accept on input time (|lag| and |len diff| <= 50 ms), or, for a candidate
    whose outputs are never on input time (a chain's structural delay, a live-rail
    stream's late start), on its OWN usual offset: within 50 ms of its median lag
    and length diff on the other stored runs of the surface. Either way the peak
    must be clear (prominence >= 0.15) and the file older than the reprocess."""
    why = []
    if g["prominence"] < MIN_PROMINENCE:
        why.append(f"envelope prominence {g['prominence']} < {MIN_PROMINENCE}")
    if g["mtime"] >= before:
        why.append(f"written {g['mtime']}, not before {before}")
    lim = MAX_LEN_DIFF_S * 1000
    on_input = abs(g["lag_ms"]) <= MAX_LAG_MS and abs(g["len_diff_ms"]) <= lim
    own = ref is not None and abs(g["lag_ms"] - ref["median_lag_ms"]) <= MAX_LAG_MS \
        and abs(g["len_diff_ms"] - ref["median_len_diff_ms"]) <= lim
    if not (on_input or own):
        why.append(f"lag {g['lag_ms']} ms / length diff {g['len_diff_ms']} ms fit neither input "
                   f"time (±{MAX_LAG_MS:g} ms) nor this candidate's own offset on other runs "
                   + (f"({ref['median_lag_ms']} / {ref['median_len_diff_ms']} ms)" if ref
                      else "(no other run has it)"))
    return {"ok": not why, "rule": "input-time" if on_input else "candidate-relative" if own else None,
            "reason": "; ".join(why) or None}


async def _score(rid: str, rec: dict, wav: Path, meta: dict) -> None:
    script = (meta.get("script") or "").strip()
    try:
        rec["stt"] = await stt.transcribe(wav)
        rec["stt_setting"] = {"boost_volume": config.WHISPER_BOOST_VOLUME, "whole_file": True,
                              "pass": PASS}
    except Exception as e:  # noqa: BLE001 - recorded on the candidate, as _process_run does
        rec["stt_error"] = str(e)
    try:
        rec["scores"] = await asyncio.to_thread(
            scoring.score_output, wav, meta.get("input_scores") or {}, script,
            (rec.get("stt") or {}).get("text", ""))
    except Exception as e:  # noqa: BLE001
        rec["scores_error"] = str(e)
    try:
        rec["vad"] = await vad.analyze(wav, meta.get("vad_params"))
    except Exception as e:  # noqa: BLE001
        rec["vad_error"] = str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--before", required=True, help="ISO date the reprocess happened, e.g. 2026-08-10")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if config.WHISPER_BOOST_VOLUME:
        raise SystemExit("WHISPER_BOOST_VOLUME is true in .env; round 2 transcribes with false")

    rid, run_dir = args.run, store.run_dir(args.run)
    meta = store.load_meta(rid)
    input_wav = run_dir / meta["input"]["file"]
    if _mtime(input_wav) >= args.before:
        raise SystemExit(f"input.wav itself was written {_mtime(input_wav)}, after {args.before}")
    referenced = {meta["input"]["file"]} | {
        f for c in meta.get("candidates") or [] for f in (c.get("output"), c.get("output_seg")) if f}
    known = _labels()
    orphans = sorted(p for p in run_dir.glob("*.wav")
                     if p.name not in referenced and not p.name.endswith(".vadcut.wav")
                     and p.stem in known)
    others = sorted(p.name for p in run_dir.glob("*.wav")
                    if p.name not in referenced and not p.name.endswith(".vadcut.wav")
                    and p.stem not in known)
    report = {"run": rid, "checked": [], "restored": [], "kept_orphaned": [],
              "not_candidates": others}
    for wav in orphans:
        cid = wav.stem
        g = measure(input_wav, wav)
        ref = reference(cid, rid, _surface(meta))
        g.update(gate(g, ref, args.before))
        g["ref_median_lag_ms"] = ref and ref["median_lag_ms"]
        g["ref_median_len_diff_ms"] = ref and ref["median_len_diff_ms"]
        g["ref_runs"] = ref and ref["runs"]
        report["checked"].append({"id": cid, **g})
        print(f"{cid:34s} {'OK  ' if g['ok'] else 'FAIL'} len {g['len_diff_ms']:+7.1f} ms  lag "
              f"{g['lag_ms']:+6.1f} ms (own median {g['ref_median_lag_ms']}, n={g['ref_runs']})  "
              f"prom {g['prominence']:.3f}  {g['rule'] or ''}  {g['reason'] or ''}")
        if not g["ok"]:
            report["kept_orphaned"].append({"id": cid, "reason": g["reason"]})
            continue
        if args.dry_run:
            continue
        cand = known[cid]
        rec = {
            "id": cid, "label": cand["label"], "output": wav.name, "status": "done",
            "restored_orphan": True,
            "restored_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "restored_from": f"{wav.name} mtime {g['mtime']}",
            "restored_check": {k: g[k] for k in ("lag_ms", "len_diff_ms", "ref_median_lag_ms",
                                                 "ref_median_len_diff_ms", "ref_runs", "prominence",
                                                 "env_corr", "rule")},
            # unknowable now: the record that described how it was made is gone
            "chain": None, "chain_spec": None, "latency": None, "cost": None,
            "proc_ms": None, "rtf": None, "processed_at": None,
            "duration_s": round(sf.info(wav).duration, 2),
        }
        if cid.startswith(LIVE_RAIL):
            rec["rail"] = "live"
        asyncio.run(_score(rid, rec, wav, meta))
        with store.locked_meta(rid) as m:
            cands = m.setdefault("candidates", [])
            if any(c.get("id") == cid for c in cands):
                print(f"  {cid} gained a record meanwhile; not restored")
                continue
            cands.append(rec)
            vad.score_run({"truth_spans": m.get("truth_spans"), "ignore_spans": m.get("ignore_spans"),
                           "input": m.get("input"), "input_vad": m.get("input_vad"),
                           "candidates": [rec]})
        report["restored"].append(cid)
    if not args.dry_run:
        with store.locked_meta(rid) as m:
            # one entry per pass, appended: a later pass that restores nothing
            # must not erase the audit trail of the one that did
            log = m.setdefault("backfill_round2", {}).setdefault("restored_orphans", {"passes": []})
            log.setdefault("passes", []).append({
                "at": datetime.datetime.now().isoformat(timespec="seconds"), "before": args.before,
                "restored": report["restored"], "kept_orphaned": report["kept_orphaned"],
                "not_candidates": others,
                "checks": {c["id"]: {k: c[k] for k in ("ok", "rule", "len_diff_ms", "lag_ms",
                                                      "ref_median_lag_ms", "env_corr",
                                                      "prominence", "mtime", "reason")}
                           for c in report["checked"]}})
            log["restored"] = sorted({c for p in log["passes"] for c in p["restored"]})
            log["kept_orphaned"] = report["kept_orphaned"]
    print(json.dumps({k: v for k, v in report.items() if k != "checked"}, indent=1))


if __name__ == "__main__":
    main()
