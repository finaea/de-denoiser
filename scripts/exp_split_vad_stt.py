"""Experiment: split the pipeline — one NC model gates the VAD, another feeds
the STT. READ-ONLY against the bench: imports its helpers, never writes to
nc_bench/ or data/runs/; results go to stdout + a JSON next to this script's
invocation (path via --out).

Rationale (docs/nc-ranking-2026-08-04.md): the VAD wants aggressive suppression
(kill phantom turns), the STT wants minimal intervention (artifacts become wrong
words), and no single candidate won both columns. Here turn spans come from the
GATE candidate's stored VAD measurement, the audio cut at those spans comes from
the READ candidate's stored output, and the pair is scored with the same
turn-by-turn STT + WER as the stored wer_seg numbers — so pairs are directly
comparable with every self-gated candidate already in the report.

The gate's misses are inherited by construction: a turn the gate never opened is
never transcribed, whatever the read model preserved there.

    .venv/bin/python scripts/exp_split_vad_stt.py [--out results.json]
"""

import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import scoring, store, stt  # noqa: E402

# gate (spans from) -> read (audio from). Chosen from the free-candidate tables:
# the best gates by agreement, the best readers by transcript quality, plus
# read=none ("NC for gating only, STT hears raw") as the minimal variant.
PAIRS = {
    "phone": [
        ("hush", "dpdfnet2-8k"),        # the proposal as stated
        ("gtcrn+hush", "dtln"),         # best free gate x best free reader
        ("hush", "none"),               # gate-only NC
        ("gtcrn+hush", "dpdfnet2-8k"),
    ],
    "web": [
        ("hush-atten12", "none"),       # only Hush variant that survives web
        ("hush-atten12", "dtln"),
        ("gtcrn", "none"),              # the speech-safe gate
        ("hush-atten12", "fastenhancer-l"),  # top web reader, risky pedigree
    ],
}
REP = 5  # a 1-3gram repeated this often = decoder collapse, transcript invalid


def maxrep(t):
    w = re.findall(r"[a-z']+", (t or "").lower())
    best = 1
    for n in (1, 2, 3):
        i = 0
        while i + n <= len(w):
            g = w[i:i + n]
            k = 1
            while w[i + k * n:i + (k + 1) * n] == g:
                k += 1
            best = max(best, k)
            i += max(1, (k - 1) * n) if k > 1 else 1
    return best


def surface(m):
    note = (m.get("note") or "").lower()
    if m["source"] in ("phone", "web"):
        return m["source"]
    if "phone" in note:
        return "phone"
    if "web" in note:
        return "web"
    return None


def stored_seg_wer(c):
    return ((c.get("scores") or {}).get("wer_seg") or {}).get("wer")


async def main() -> None:
    out_path = None
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])

    results = defaultdict(list)  # (surf, gate, read) -> per-run dicts
    t0 = time.time()
    for m in store.list_runs():
        surf = surface(m)
        note = (m.get("note") or "").lower()
        if surf is None or "no reference" in note or "ignore reference" in note:
            continue
        script = (m.get("script") or "").strip()
        if not script:
            continue
        cands = {c["id"]: c for c in m.get("candidates") or []}
        base = cands.get("none")
        bw = stored_seg_wer(base) if base else None
        if bw is None or maxrep((base.get("stt_seg") or {}).get("text")) >= REP:
            continue
        run_dir = store.run_dir(m["id"])
        pad = float((m.get("vad_params") or {}).get("prefix_padding_duration", 0.5))
        for gate, read in PAIRS[surf]:
            g, r = cands.get(gate), cands.get(read)
            if not g or not r or not r.get("output"):
                continue
            if (g.get("vad") or {}).get("segments") is None:
                continue
            res = await stt.transcribe_turns(
                run_dir / r["output"], g["vad"]["segments"], pad)
            collapsed = maxrep(res["text"]) >= REP
            w = scoring.wer(script, res["text"])
            results[(surf, gate, read)].append({
                "run": m["id"], "note": m.get("note"),
                "wer": w["wer"], "delta": round(w["wer"] - bw, 3),
                "turns": res["turns"], "collapsed": collapsed,
                "base_seg": bw,
                "gate_self": stored_seg_wer(g),
                "read_self": stored_seg_wer(r),
            })
            print(f"{m['id']} {surf:<5} {gate}->{read}: wer {w['wer']:.3f} "
                  f"(base {bw:.3f}, Δ{w['wer']-bw:+.3f}, {res['turns']} turns"
                  f"{', COLLAPSED' if collapsed else ''})  {time.time()-t0:.0f}s",
                  flush=True)

    print("\n" + "=" * 78)
    summary = {}
    for (surf, gate, read), rows in sorted(results.items()):
        ok = [r for r in rows if not r["collapsed"]]
        ds = sorted(r["delta"] for r in ok)
        med = ds[len(ds) // 2] if len(ds) % 2 else (ds[len(ds)//2-1] + ds[len(ds)//2]) / 2
        gs = [r["wer"] - r["gate_self"] for r in ok if r["gate_self"] is not None]
        rs = [r["wer"] - r["read_self"] for r in ok if r["read_self"] is not None]
        summary[f"{surf}:{gate}->{read}"] = {
            "n": len(ok), "collapsed": len(rows) - len(ok),
            "median": round(med, 3), "mean": round(sum(ds) / len(ds), 3),
            "worst": round(max(ds), 3),
            "wins": sum(1 for d in ds if d < -1e-9),
            "losses": sum(1 for d in ds if d > 1e-9),
            "vs_gate_self_mean": round(sum(gs) / len(gs), 3) if gs else None,
            "vs_read_self_mean": round(sum(rs) / len(rs), 3) if rs else None,
            "rows": rows,
        }
        s = summary[f"{surf}:{gate}->{read}"]
        print(f"{surf:<6} {gate} -> {read:<16} n={s['n']:<2} "
              f"median {s['median']:+.3f}  mean {s['mean']:+.3f}  worst {s['worst']:+.3f}  "
              f"{s['wins']}W/{s['losses']}L   vs gate-self {s['vs_gate_self_mean']:+.3f}  "
              f"vs read-self {s['vs_read_self_mean']:+.3f}")
    if out_path:
        out_path.write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
