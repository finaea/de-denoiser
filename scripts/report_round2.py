"""Round-2 ranking tables, regenerated from the stored meta.json files alone.

    .venv/bin/python scripts/report_round2.py                 # every section, markdown
    .venv/bin/python scripts/report_round2.py --section nc-stt --section coverage
    .venv/bin/python scripts/report_round2.py --list

Everything docs/nc-vad-ranking-2026-09-29.md quotes as a number comes from here,
and this reads nothing but data/runs/*/meta.json: no audio, no stored
probabilities, no network. Where a number needs something audio-derived (a
live-rail output's offset, a parameter sweep), a backfill step stored it in
meta.json first (`live_rail_offset`, `vad_sweep_round2`).

Method, as in docs/nc-ranking-2026-08-04.md:

- Every number is a delta against the run's own control (`none`, and for the
  cloud replay rail also `cloud-raw`), aggregated over runs: n, mean (median
  where one run dominates), worst, and W / L (strictly better / worse; ties
  count in neither).
- A candidate in fewer than half of a table's runs is left out.
- Surfaces: `source` phone / web; an upload by its note, "(phone)" / "(web)".
  Uploads with no surface label and the incident runs are not in the
  per-surface tables. The incidents have their own section.
- STT: runs with no reference script, and runs whose `none` transcript
  collapsed into repetition, are excluded; so is any single candidate
  transcript that collapsed. Collapse = a 1-3-gram repeated >= 5x in a row.
- VAD numbers come from `vad_bench` only (the one shared segmenter, stored
  probabilities), never from the legacy `vad` / `truth_score` columns, which
  were Silero v6.0 on every run stored before 2026-09-29.
- Ranking rows for LiveKit-hosted models come from the cloud replay rail
  (`krisp-*-replay`, `krisp-viva*`, `lkaic-*`). The live-rail records
  (`krisp-nc/bvc/bvc-tel`, historical `aic-*`) have their own table: they sit
  up to 0.23 s off input time.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import config  # noqa: E402
from nc_bench.vads.events import coalesce_truth  # noqa: E402

PROD_VAD = "lk-inference"
PROD_GATE = "hush"
LIVE_RAIL_IDS = ("krisp-nc", "krisp-bvc", "krisp-bvc-tel", "aic-quail-l", "aic-vf-s", "aic-vf-l")
HISTORICAL = ("aic-quail-l", "aic-vf-s", "aic-vf-l")
REPLAY_IDS = ("cloud-raw", "krisp-nc-replay", "krisp-bvc-replay", "krisp-bvc-tel-replay",
              "krisp-viva", "krisp-viva-tel", "lkaic-quail-l", "lkaic-quail-l-bypass",
              "lkaic-vf22-l", "lkaic-vf22-s", "lkaic-rook-s", "lkaic-rook-s-bypass")
TAIL_RUN, TAIL_FROM = "20260804-112511", 23.8   # unmarked tail, flagged for a listen


# ================================================================ loading


def load_runs(runs_dir: Path | None = None) -> dict[str, dict]:
    d = runs_dir or config.RUNS_DIR
    out = {}
    for p in sorted(d.iterdir()):
        if p.name.startswith("_") or not (p / "meta.json").is_file():
            continue
        out[p.name] = json.loads((p / "meta.json").read_text())
    return out


def surface(m: dict) -> str | None:
    """phone / web for the per-surface tables; None for unlabelled uploads and
    the incident runs (which have their own section)."""
    if m.get("incident"):
        return None
    if m.get("source") in ("phone", "web"):
        return m["source"]
    note = (m.get("note") or "").lower()
    return "phone" if "(phone)" in note else "web" if "(web)" in note else None


def cands(m: dict) -> dict[str, dict]:
    return {c["id"]: c for c in m.get("candidates") or [] if c.get("id")}


def is_live_rail(cid: str, rec: dict) -> bool:
    return cid in LIVE_RAIL_IDS and rec.get("rail") != "cloud_replay"


def label(cid: str, rec: dict | None = None) -> str:
    tags = []
    if cid in HISTORICAL:
        tags.append("VF 2.1 / historical")
    if rec and rec.get("restored_orphan"):
        tags.append("restored")
    return f"`{cid}`" + (f" ({', '.join(tags)})" if tags else "")


# ================================================================ helpers


def toks(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", (text or "").lower()).split()


def collapse(text: str) -> int:
    """Longest run of one 1-3-gram repeated back to back (the 08-04 filter: >= 5
    means the decoder looped and the transcript measures that, not the NC)."""
    w, best = toks(text), 0
    for n in (1, 2, 3):
        for i in range(len(w) - n + 1):
            g, k, j = w[i:i + n], 1, i + n
            while w[j:j + n] == g:
                k, j = k + 1, j + n
            best = max(best, k)
    return best


def wer_of(rec: dict | None) -> float | None:
    if not rec or rec.get("status") != "done":
        return None
    w = ((rec.get("scores") or {}).get("wer") or {})
    return w.get("wer") if isinstance(w, dict) else None


def agg(vals: list[float], higher_better: bool = False) -> dict:
    """n, mean, median, worst, W, L of per-run deltas (W = strictly better)."""
    if not vals:
        return {"n": 0}
    worst = min(vals) if higher_better else max(vals)
    better = (lambda v: v > 0) if higher_better else (lambda v: v < 0)
    worse = (lambda v: v < 0) if higher_better else (lambda v: v > 0)
    return {"n": len(vals), "mean": statistics.mean(vals), "median": statistics.median(vals),
            "worst": worst, "W": sum(better(v) for v in vals), "L": sum(worse(v) for v in vals)}


def fmt(v, spec: str = "+.3f", none: str = "—") -> str:
    if v is None:
        return none
    return format(v, spec)


def md(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ranked(per: dict[str, list[float]], runs_n: int, key: str = "mean",
           higher_better: bool = False, top: int | None = 10) -> list[tuple[str, dict]]:
    rows = [(cid, agg(v, higher_better)) for cid, v in per.items() if len(v) >= runs_n / 2]
    rows = [r for r in rows if r[1]["n"]]
    rows.sort(key=lambda r: (-r[1][key] if higher_better else r[1][key], r[0]))
    return rows[:top] if top else rows


# ================================================================ VAD bench access


def vb_row(m: dict, sid: str, vid: str) -> dict | None:
    """One stored VAD-bench result, or None when missing, stale, errored or
    excluded from rankings."""
    r = (((m.get("vad_bench") or {}).get("results") or {}).get(sid) or {}).get(vid)
    if not r or r.get("segments") is None or r.get("stale") or r.get("error") \
            or r.get("unavailable") or r.get("rank_excluded"):
        return None
    return r


def vb_params(m: dict) -> dict:
    return (m.get("vad_bench") or {}).get("params") or {}


def ev(r: dict | None, k: str):
    return ((r or {}).get("events") or {}).get(k)


def fr(r: dict | None, k: str):
    return ((r or {}).get("frame") or {}).get(k)


def has_truth(m: dict) -> bool:
    return bool(m.get("truth_spans"))


# in-memory rescoring (the 112511 tail, the live-rail offset correction):
# the same scoring code the bench stores, on segments this module derives,
# never written back


def rescore(m: dict, segments, extra_ignore=None, params: dict | None = None) -> dict:
    from nc_bench.vads.bench import _truth_duration, score

    ign = list(m.get("ignore_spans") or []) + list(extra_ignore or [])
    res = {"segments": segments, "params": params or vb_params(m)}
    return score(res, m.get("truth_spans") or [], _truth_duration(m),
                 m.get("known_nonspeech") or [], ign)


# ================================================================ sections


SECTIONS: dict[str, callable] = {}


def section(name: str):
    def deco(fn):
        SECTIONS[name] = fn
        return fn
    return deco


@section("coverage")
def sec_coverage(runs: dict[str, dict]) -> str:
    """Per run: what it holds, what the backfill added, what's missing and why."""
    rows = []
    for rid, m in runs.items():
        cs = cands(m)
        done = {c for c, r in cs.items() if r.get("status") == "done" and r.get("output")}
        aic = sorted(c for c in done if c.startswith("aicsdk-"))
        rep = [c for c in REPLAY_IDS if c in done and cs[c].get("rail") == "cloud_replay"]
        restored = [c for c, r in cs.items() if r.get("restored_orphan")]
        bf = m.get("backfill_round2") or {}
        cr = bf.get("cloud_replay") or {}
        vb = (m.get("vad_bench") or {}).get("results") or {}
        pairs = sum(1 for per in vb.values() for r in per.values() if r.get("segments") is not None
                    and not r.get("stale"))
        stale = sum(1 for per in vb.values() for r in per.values() if r.get("stale"))
        unav = sum(1 for per in vb.values() for r in per.values() if r.get("unavailable"))
        err = sum(1 for per in vb.values() for r in per.values() if r.get("error"))
        drift = sorted({(r.get("cloud_replay") or {}).get("control_lag_drift_ms")
                        for r in cs.values() if r.get("rail") == "cloud_replay"} - {None})
        missing = []
        if len(aic) < 12:
            missing.append(f"aicsdk {12 - len(aic)} missing")
        if len(rep) < len(REPLAY_IDS):
            missing.append(f"replay {len(REPLAY_IDS) - len(rep)} missing"
                           + (f" ({', '.join(sorted(cr.get('failed') or {}))} failed)" if cr.get("failed") else ""))
        if not m.get("truth_spans"):
            missing.append("no hand marks: event metrics null")
        if not (m.get("script") or "").strip():
            missing.append("no script: no WER")
        if restored:
            missing.append(f"{len(restored)} restored orphans")
        rows.append([f"`{rid}`", surface(m) or ("incident" if m.get("incident") else "—"),
                     (m.get("incident") or {}).get("id") or (m.get("note") or "")[:28],
                     fmt((m.get("input") or {}).get("duration_s"), ".1f"), len(m.get("truth_spans") or []),
                     len(cs), f"{len(aic)}/12", f"{len(rep)}/{len(REPLAY_IDS)}",
                     cr.get("retries", "—") if cr else "—",
                     "/".join(f"{d:g}" for d in drift) or "—",
                     pairs, stale, unav, err, "; ".join(missing) or "—"])
    return md(["run", "surface", "note", "dur s", "marks", "records", "aicsdk", "replay",
               "replay retries", "control drift ms", "VAD pairs", "stale", "unavail", "errors",
               "gaps"], rows)


@section("nc-stt")
def sec_nc_stt(runs: dict[str, dict]) -> str:
    """Whole-file ΔWER vs `none` per surface, round-2 STT (boost_volume=false),
    plus the cloud replay rail against `cloud-raw`."""
    out = []
    excluded = []
    dropped = []
    for surf in ("phone", "web"):
        per: dict[str, list[float]] = {}
        per_raw: dict[str, list[float]] = {}
        valid = 0
        base_w = []
        for rid, m in runs.items():
            if surface(m) is None and surf == "phone" and not m.get("incident"):
                excluded.append((rid, m.get("note"), "upload with no surface label"))
            if surface(m) != surf:
                continue
            if not (m.get("script") or "").strip():
                excluded.append((rid, m.get("note"), "no reference script"))
                continue
            cs = cands(m)
            base = cs.get("none")
            bw = wer_of(base)
            if bw is None:
                excluded.append((rid, m.get("note"), "no passthrough WER"))
                continue
            k = collapse((base.get("stt") or {}).get("text", ""))
            if k >= 5 or bw > 1.0:
                # a junk baseline (a loop, or hallucination past WER 1.0) hands every
                # candidate a win on that run
                excluded.append((rid, m.get("note"), f"passthrough transcript collapsed (x{k})" if k >= 5
                                 else f"passthrough WER {bw:.2f} > 1.0 (junk transcript)"))
                continue
            valid += 1
            base_w.append(bw)
            rw = wer_of(cs.get("cloud-raw"))
            for cid, rec in cs.items():
                if cid == "none" or is_live_rail(cid, rec):
                    continue
                w = wer_of(rec)
                if w is None:
                    continue
                k = collapse((rec.get("stt") or {}).get("text", ""))
                if k >= 5:
                    dropped.append((rid, m.get("note"), cid, k))
                    continue
                per.setdefault(cid, []).append(w - bw)
                if rec.get("rail") == "cloud_replay" and cid != "cloud-raw" and rw is not None:
                    per_raw.setdefault(cid, []).append(w - rw)
        # ranked by median, as the VAD-cut tables are: a few-word script makes one
        # run's delta large (~1 word = 0.03-0.09 WER)
        out.append(f"### {surf.title()} — STT ({valid} valid runs · passthrough WER median "
                   f"{statistics.median(base_w):.3f}, mean {statistics.mean(base_w):.3f})\n"
                   if base_w else f"### {surf.title()} — STT\n")
        rows = [[i + 1, label(cid), a["n"], fmt(a["median"]), fmt(a["mean"]), fmt(a["worst"]), a["W"], a["L"]]
                for i, (cid, a) in enumerate(ranked(per, valid, key="median", top=12))]
        out.append(md(["#", "candidate", "n", "median ΔWER", "mean", "worst", "W", "L"], rows))
        if per_raw:
            out.append(f"\n**{surf.title()}, cloud replay rail against `cloud-raw`** (the second Opus "
                       "pass cancelled out):\n")
            rows = [[label(cid), a["n"], fmt(a["median"]), fmt(a["mean"]), fmt(a["worst"]), a["W"], a["L"]]
                    for cid, a in ranked(per_raw, max(len(v) for v in per_raw.values()), key="median",
                                         top=None)]
            out.append(md(["candidate", "n", "median ΔWER vs cloud-raw", "mean", "worst", "W", "L"], rows))
        out.append("")
    out.append("Excluded runs:\n")
    out.append(md(["run", "note", "why"], [[f"`{r}`", n, w] for r, n, w in excluded]))
    out.append("\nCandidate transcripts dropped for repetition collapse:\n")
    out.append(md(["run", "note", "candidate", "repeat"],
                  [[f"`{r}`", n, f"`{c}`", f"x{k}"] for r, n, c, k in dropped]) if dropped else "none")
    return "\n".join(out)


@section("stt-v1")
def sec_stt_v1(runs: dict[str, dict]) -> str:
    """What the STT setting change did: passthrough WER, v1 (boost on) vs v2 (off)."""
    rows = []
    for rid, m in runs.items():
        n = cands(m).get("none")
        if not n or not (m.get("script") or "").strip() or not surface(m):
            continue
        v1 = ((n.get("stt_v1") or {}).get("wer") or {}).get("wer")
        v2 = wer_of(n)
        rows.append([f"`{rid}`", surface(m), fmt(v1, ".3f"), fmt(v2, ".3f"),
                     fmt(None if v1 is None or v2 is None else v2 - v1)])
    return md(["run", "surface", "none WER v1 (boost on)", "v2 (boost off)", "Δ"], rows)


def _nc_vad(runs: dict[str, dict], surf: str, vid: str, metric, higher_better: bool,
            base_sid: str = "none") -> tuple[dict[str, list[float]], int, list[float]]:
    per: dict[str, list[float]] = {}
    valid, base_vals = 0, []
    for rid, m in runs.items():
        if surface(m) != surf or not has_truth(m):
            continue
        b = vb_row(m, base_sid, vid)
        bv = metric(b) if b else None
        if bv is None:
            continue
        valid += 1
        base_vals.append(bv)
        for sid, rec in cands(m).items():
            if sid == base_sid or is_live_rail(sid, rec):
                continue
            r = vb_row(m, sid, vid)
            v = metric(r) if r else None
            if v is not None:
                per.setdefault(sid, []).append(v - bv)
    return per, valid, base_vals


@section("nc-vad")
def sec_nc_vad(runs: dict[str, dict]) -> str:
    """NC gates under production's VAD (lk-inference, the run's params): Δ vs `none`."""
    metrics = [
        ("missed turns", lambda r: ev(r, "missed_n"), False, "+.2f", 12),
        ("missed speech s", lambda r: fr(r, "miss_s"), False, "+.2f", 12),
        ("phantom segments", lambda r: ev(r, "phantom_n"), False, "+.2f", 12),
        ("frame agreement", lambda r: fr(r, "agree"), True, "+.1%", 12),
    ]
    out = []
    for surf in ("phone", "web"):
        for title, fn, hb, spec, top in metrics:
            per, valid, base = _nc_vad(runs, surf, PROD_VAD, fn, hb)
            if not valid:
                continue
            base_mean = statistics.mean(base)
            out.append(f"### {surf.title()} — {title} under `{PROD_VAD}` ({valid} runs · `none` "
                       f"{format(base_mean, spec.lstrip('+'))} per run)\n")
            rows = [[i + 1, label(cid), a["n"], fmt(a["mean"], spec), fmt(a["median"], spec),
                     fmt(a["worst"], spec), a["W"], a["L"]]
                    for i, (cid, a) in enumerate(ranked(per, valid, higher_better=hb, top=top))]
            out.append(md(["#", "gate", "n", "mean Δ", "median", "worst", "W", "L"], rows) + "\n")
    return "\n".join(out)


def _vad_table(runs: dict[str, dict], surf: str, sid: str, tail: bool = True) -> list[list]:
    """Per VAD on one source, pooled over the surface's marked runs."""
    from nc_bench.vads import registry

    rows = []
    for e in registry():
        vid = e["id"]
        turns = hit = missed = phantom = merged = 0
        lags, agree, cost, over, nruns = [], [], [], 0.0, 0
        for rid, m in runs.items():
            if surface(m) != surf or not has_truth(m):
                continue
            r = vb_row(m, sid, vid)
            if not r or ev(r, "turns") is None:
                continue
            if not tail and rid == TAIL_RUN:
                r = {**r, **rescore(m, r["segments"], [[TAIL_FROM, 1e9]], r.get("params"))}
            nruns += 1
            turns += ev(r, "turns")
            hit += ev(r, "hit")
            missed += ev(r, "missed_n")
            phantom += ev(r, "phantom_n")
            merged += ev(r, "merged_turns") or 0
            over += ev(r, "overreach_s") or 0
            if ev(r, "onset_lag_med_s") is not None:
                lags.append(ev(r, "onset_lag_med_s"))
            if fr(r, "agree") is not None:
                agree.append(fr(r, "agree"))
            if r.get("cpu_ms_per_s") is not None:
                cost.append(r["cpu_ms_per_s"])
        if not nruns:
            continue
        rows.append([f"`{vid}`", nruns, f"{hit / turns:.1%}" if turns else "—", missed, phantom,
                     merged, f"{over:.1f}", fmt(statistics.median(lags) if lags else None, "+.3f"),
                     fmt(statistics.mean(agree) if agree else None, ".1%")])
    rows.sort(key=lambda r: (r[3] + r[4], r[3], r[0]))
    return rows


# no cost column: the stored per-row costs were measured on a shared, loaded box
# with 6 jobs in parallel. The report's cost table is scripts/vad_cost.py's.
VAD_HEAD = ["VAD", "runs", "turn recall", "missed turns", "phantom segs", "merged", "overreach s",
            "onset lag (median)", "frame agree"]


@section("vad")
def sec_vad(runs: dict[str, dict]) -> str:
    """Every VAD on production's gate (hush) and on `none`, per surface, pooled."""
    out = []
    for surf in ("phone", "web"):
        for sid in (PROD_GATE, "none"):
            rows = _vad_table(runs, surf, sid)
            if rows:
                out.append(f"### {surf.title()} — VADs on `{sid}`\n")
                out.append(md(VAD_HEAD, rows) + "\n")
    rows = _vad_table(runs, "phone", PROD_GATE, tail=False)
    if rows:
        out.append(f"### Phone — VADs on `{PROD_GATE}`, with `{TAIL_RUN}`'s unmarked tail "
                   f"(from {TAIL_FROM} s) ignored\n")
        out.append(md(VAD_HEAD, rows) + "\n")
    return "\n".join(out)


def _matrix(runs: dict[str, dict], surf: str, tail: bool = True) -> dict[tuple[str, str], dict]:
    cells: dict[tuple[str, str], dict] = {}
    for rid, m in runs.items():
        if surface(m) != surf or not has_truth(m):
            continue
        res = (m.get("vad_bench") or {}).get("results") or {}
        cs = cands(m)
        for sid, per in res.items():
            if sid != "input" and sid in cs and is_live_rail(sid, cs[sid]):
                continue
            for vid in per:
                r = vb_row(m, sid, vid)
                if not r or ev(r, "turns") is None:
                    continue
                if not tail and rid == TAIL_RUN:
                    r = {**r, **rescore(m, r["segments"], [[TAIL_FROM, 1e9]], r.get("params"))}
                c = cells.setdefault((sid, vid), {"runs": 0, "turns": 0, "missed": 0, "phantom": 0,
                                                  "merged": 0, "over": 0.0})
                c["runs"] += 1
                c["turns"] += ev(r, "turns")
                c["missed"] += ev(r, "missed_n")
                c["phantom"] += ev(r, "phantom_n")
                c["merged"] += ev(r, "merged_turns") or 0
                c["over"] += ev(r, "overreach_s") or 0
    return cells


def pareto(cells: dict, min_runs: int) -> list[tuple]:
    pts = [(k, c) for k, c in cells.items() if c["runs"] >= min_runs]
    front = []
    for k, c in pts:
        dom = any(o["missed"] <= c["missed"] and o["phantom"] <= c["phantom"]
                  and (o["missed"] < c["missed"] or o["phantom"] < c["phantom"]) for _, o in pts)
        if not dom:
            front.append((k, c))
    return sorted(front, key=lambda kc: (kc[1]["missed"], kc[1]["phantom"]))


@section("matrix")
def sec_matrix(runs: dict[str, dict]) -> str:
    """Gate x VAD, pooled per surface: the miss-phantom trade-off and its Pareto front."""
    out = []
    for surf in ("phone", "web"):
        for tail in ((True, False) if surf == "phone" else (True,)):
            cells = _matrix(runs, surf, tail)
            if not cells:
                continue
            full = max(c["runs"] for c in cells.values())
            best = sorted(((k, c) for k, c in cells.items() if c["runs"] == full),
                          key=lambda kc: (kc[1]["missed"] + kc[1]["phantom"], kc[1]["merged"],
                                          kc[1]["over"]))[:20]
            note = "" if tail else f", `{TAIL_RUN}` tail ignored"
            out.append(f"### {surf.title()} — best 20 (gate, VAD) by missed + phantom ({full} runs{note})\n")
            prod = cells.get((PROD_GATE, PROD_VAD))
            rows = [[f"`{g}` → `{v}`", c["turns"], c["missed"], c["phantom"], c["merged"],
                     f"{c['over']:.1f}"] for (g, v), c in best]
            if prod:
                rows.append([f"**production: `{PROD_GATE}` → `{PROD_VAD}`**", prod["turns"],
                             prod["missed"], prod["phantom"], prod["merged"], f"{prod['over']:.1f}"])
            out.append(md(["gate → VAD", "turns", "missed", "phantom", "merged", "overreach s"], rows))
            front = pareto(cells, full)
            out.append(f"\nPareto front ({len(front)} combos): " + ", ".join(
                f"`{g}`→`{v}` ({c['missed']}/{c['phantom']})" for (g, v), c in front[:15]) + "\n")
    return "\n".join(out)


# ------------------------------------------------------------- incidents

INCIDENT_GREETING_B = (12.5, 16.3)   # PR-54 B: the unlabelled sounds during the agent's greeting


def _incidents(runs: dict[str, dict]) -> dict[str, tuple[str, dict]]:
    return {(m.get("incident") or {}).get("id"): (rid, m) for rid, m in runs.items()
            if m.get("incident")}


def _overlaps(a, b) -> bool:
    return a[0] < b[1] and a[1] > b[0]


@section("incidents")
def sec_incidents(runs: dict[str, dict], combos: list[tuple[str, str]] | None = None) -> str:
    """Per (gate, VAD): did each incident phantom open, and was the speech kept."""
    inc = _incidents(runs)
    out = []
    a, b, c = inc.get("PR-54 A"), inc.get("PR-54 B"), inc.get("PR-42")
    keys: set[tuple[str, str]] = set()
    for x in (a, b, c):
        if x:
            for sid, per in ((x[1].get("vad_bench") or {}).get("results") or {}).items():
                keys |= {(sid, v) for v in per}
    if combos is None:
        combos = [(g, v) for g in SWEEP_GATES for v in SWEEP_VADS] + [
            ("input", PROD_VAD), ("none", PROD_VAD), (PROD_GATE, PROD_VAD)]
    keys = [k for k in dict.fromkeys(combos) if k in keys]
    # B discriminates only where production's own VAD opens on the throat-clear
    b_ref: dict[str, bool] = {}
    if b:
        tcw = next((w for w in b[1].get("known_nonspeech") or [] if w["name"] == "throat-clear-B"), None)
        for sid in {k[0] for k in keys}:
            rp = vb_row(b[1], sid, PROD_VAD)
            b_ref[sid] = bool(rp and tcw and any(_overlaps(s, (tcw["start"], tcw["end"]))
                                                 for s in rp["segments"]))
    rows = []
    for sid, vid in keys:
        row = [f"`{sid}` → `{vid}`"]
        # PR-54 A: speech kept, click / clunk opened, other phantoms
        ra = vb_row(a[1], sid, vid) if a else None
        if ra:
            hits = ev(ra, "nonspeech_hits") or {}
            kns = [(w["start"], w["end"]) for w in a[1].get("known_nonspeech") or []]
            others = sum(1 for s in ev(ra, "phantoms") or [] if not any(_overlaps(s, w) for w in kns))
            row += [f"{ev(ra, 'hit')}/{ev(ra, 'turns')}", "OPEN" if hits.get("click-A") else "no",
                    "OPEN" if hits.get("hangup-clunk-A") else "no", others,
                    ev(ra, "ignored_segments")]
        else:
            row += ["—"] * 5
        # PR-54 B: no marks, so from the segments directly
        rb = vb_row(b[1], sid, vid) if b else None
        if rb:
            tc = next((w for w in b[1].get("known_nonspeech") or [] if w["name"] == "throat-clear-B"), None)
            segs = rb["segments"]
            opened = tc is not None and any(_overlaps(s, (tc["start"], tc["end"])) for s in segs)
            outside = sum(1 for s in segs if not _overlaps(s, INCIDENT_GREETING_B))
            verdict = ("OPEN" if opened else "rejects") if b_ref.get(sid) else (
                "OPEN" if opened else "n/a: prod VAD silent here")
            row += [verdict, outside, len(segs)]
        else:
            row += ["—"] * 3
        # PR-42: the 7 loud-missed caught, the 21 detected kept, extra triggers
        rc = vb_row(c[1], sid, vid) if c else None
        if rc:
            snd = (c[1].get("incident") or {}).get("sounds") or []
            segs = rc["segments"]
            det = [s for s in snd if s["cls"] == "detected"]
            lm = [s for s in snd if s["cls"] == "loud_missed"]
            caught = lambda group: sum(any(_overlaps(g, (s["start"], s["end"])) for g in segs)  # noqa: E731
                                       for s in group)
            row += [f"{caught(det)}/{len(det)}", f"{caught(lm)}/{len(lm)}", ev(rc, "ignored_segments"),
                    ev(rc, "phantom_n")]
        else:
            row += ["—"] * 4
        if any(x != "—" for x in row[1:]):
            rows.append(row)
    out.append(md(["gate → VAD", "A turns kept", "A click", "A clunk", "A phantoms excl. click/clunk",
                   "A on ignore", "B throat-clear", "B segs outside greeting", "B segs",
                   "PR-42 detected kept", "PR-42 missed caught", "PR-42 on quiet/echo",
                   "PR-42 orphans"], rows))
    return "\n".join(out)


# ------------------------------------------------------------- live rail


@section("live-rail")
def sec_live_rail(runs: dict[str, dict]) -> str:
    """The live-rail records as recorded (08-04), as stored and offset-corrected.

    Corrected: segments moved onto input time by the output's measured lag
    (`live_rail_offset.lag_ms`, from scripts/measure_live_offsets.py), which is
    what re-segmenting the stored probabilities with t0_s = -lag gives; the head
    the output never had (the first |lag| s, for a stream that attached late)
    is ignored in both the truth and the frames."""
    out = []
    offs = []
    per: dict[str, dict[str, list]] = {}
    for rid, m in runs.items():
        for cid, rec in cands(m).items():
            if not is_live_rail(cid, rec):
                continue
            o = rec.get("live_rail_offset")
            if o:
                offs.append((cid, rid, o))
            if surface(m) is None or not has_truth(m):
                continue
            r = vb_row(m, cid, PROD_VAD)
            b = vb_row(m, "none", PROD_VAD)
            if not r or not b or not o or ev(r, "turns") is None or ev(b, "turns") is None:
                continue
            lag = o["lag_ms"] / 1000.0
            dur = float((m.get("input") or {}).get("duration_s") or 0)
            shifted = [[round(max(0.0, s - lag), 3), round(min(dur, e - lag), 3)]
                       for s, e in r["segments"] if min(dur, e - lag) > max(0.0, s - lag)]
            head = [[0.0, -lag]] if lag < 0 else []
            cor = rescore(m, shifted, head, r.get("params"))
            bcor = rescore(m, b["segments"], head, b.get("params")) if head else b
            d = per.setdefault(f"{surface(m)}:{cid}", {"stored": [], "corrected": []})
            d["stored"].append((ev(r, "missed_n") - ev(b, "missed_n"), ev(r, "phantom_n") - ev(b, "phantom_n"),
                                ev(r, "onset_lag_med_s")))
            d["corrected"].append((ev(cor, "missed_n") - ev(bcor, "missed_n"),
                                   ev(cor, "phantom_n") - ev(bcor, "phantom_n"),
                                   ev(cor, "onset_lag_med_s")))
    if offs:
        by: dict[str, list] = {}
        for cid, rid, o in offs:
            by.setdefault(cid, []).append(o["lag_ms"])
        out.append("Measured offset of each live-rail output against its input (negative: the stream "
                   "attached late and the output starts that far into the call):\n")
        out.append(md(["candidate", "runs", "lag ms min", "median", "max"],
                       [[label(c), len(v), f"{min(v):+.0f}", f"{statistics.median(v):+.0f}", f"{max(v):+.0f}"]
                        for c, v in sorted(by.items())]) + "\n")
    rows = []
    for key, d in sorted(per.items()):
        surf, cid = key.split(":")
        for kind in ("stored", "corrected"):
            vals = d[kind]
            lags = [x[2] for x in vals if x[2] is not None]
            rows.append([surf, label(cid), kind, len(vals), fmt(statistics.mean(x[0] for x in vals), "+.2f"),
                         fmt(statistics.mean(x[1] for x in vals), "+.2f"),
                         fmt(statistics.median(lags) if lags else None, "+.3f")])
    if rows:
        out.append(f"Under `{PROD_VAD}`, Δ vs `none` per run (corrected rows re-scored in memory, never "
                   "written):\n")
        out.append(md(["surface", "candidate", "scored", "runs", "Δ missed turns", "Δ phantoms",
                       "onset lag median"], rows))
    return "\n".join(out) or "(no live-rail offsets stored yet)"


@section("replay")
def sec_replay(runs: dict[str, dict]) -> str:
    """The replay rail per run: finish time, retries, control drift, twin and
    onset checks, and cloud-raw against none."""
    rows, flags = [], []
    for rid, m in runs.items():
        cs = cands(m)
        recs = [(c, r) for c, r in cs.items() if r.get("rail") == "cloud_replay" and r.get("status") == "done"]
        bf = m.get("backfill_round2") or {}
        passes = [p for p in bf.get("passes") or [] if p.get("cands") == "cloud"]
        if not recs and not passes:
            continue
        crs = {c: r.get("cloud_replay") or {} for c, r in recs}
        drift = sorted({x.get("control_lag_drift_ms") for x in crs.values()} - {None})
        onset = [x["onset_delay_ms"] for x in crs.values() if x.get("onset_delay_ms") is not None]
        tv = {c: x.get("twin_vs_control") or {} for c, x in crs.items()}
        res = [t["residual_db"] for t in tv.values() if t.get("residual_db") is not None]
        err = [abs(t["offset_err_ms"]) for t in tv.values() if t.get("offset_err_ms") is not None]
        lvc = [t["len_vs_control_ms"] for t in tv.values() if t.get("len_vs_control_ms") is not None]
        conf = [x["lag_confirmed_drift_ms"] for x in crs.values() if x.get("lag_confirmed_drift_ms") is not None]
        tld = [x.get("twin_len_diff_ms") for x in crs.values()]
        for c, x in crs.items():
            t = tv[c]
            why = []
            if t.get("residual_db") is not None and t["residual_db"] > -300:
                why.append(f"twin residual {t['residual_db']} dB")
            if x.get("twin_len_diff_ms") not in (None, 0, 0.0):
                why.append(f"twin len diff {x['twin_len_diff_ms']} ms")
            if x.get("onset_delay_ms") is not None and not -60 <= x["onset_delay_ms"] <= 90:
                why.append(f"onset delay {x['onset_delay_ms']} ms")
            if why:
                flags.append([f"`{rid}`", f"`{c}`", "; ".join(why)])
        finish = ""
        if passes:
            last = passes[-1]
            finish = f"{last['at'][11:]} +{last.get('wall_s', 0):.0f}s"
        cr = bf.get("cloud_replay") or {}
        raw = cs.get("cloud-raw") or {}
        rr = raw.get("cloud_replay") or {}
        rows.append([f"`{rid}`", f"{len(recs)}/{len(REPLAY_IDS)}", finish, cr.get("retries", "—"),
                     "/".join(f"{d:g}" for d in drift) or "—",
                     f"{min(onset):+.0f}..{max(onset):+.0f}" if onset else "—",
                     f"{max(res):.0f}" if res else "—", f"{max(err):.1f}" if err else "—",
                     f"{min(lvc):+.0f}..{max(lvc):+.0f}" if lvc else "—",
                     f"{max(conf):.0f}" if conf else "—",
                     "0" if all(d in (None, 0, 0.0) for d in tld) else "NONZERO",
                     fmt(rr.get("level_db_vs_input"), "+.2f"), fmt(rr.get("env_corr"), ".4f"),
                     fmt(rr.get("wave_corr"), ".3f"), fmt(wer_of(cs.get("none")), ".3f"),
                     fmt(wer_of(raw), ".3f")])
    out = md(["run", "replayed", "finished", "retries", "control drift ms", "onset delay ms",
              "worst twin residual dB", "worst lead residual ms", "twin len vs control ms",
              "worst confirmed drift ms", "twin len diff", "cloud-raw level dB", "env corr",
              "wave corr", "none WER", "cloud-raw WER"], rows)
    out += "\n\nFlags (twin residual worse than -300 dB, non-zero twin length diff, onset outside -60..+90 ms):\n\n"
    out += md(["run", "candidate", "why"], flags) if flags else "none"
    return out


PROD_PARAMS = (0.6, 0.05, 0.3)
SWEEP_GATES = ("hush-aih", "hush", "gtcrn+hush", "dpdfnet2-8k+hush", "dpdfnet2-8k", "krisp-viva",
               "hush-atten12", "none", "lkaic-quail-l-bypass")
SWEEP_VADS = ("lk-inference", "silero-v4.0-8k", "firered-stream", "aic-vad-vf-2.0", "marblenet-v2",
              "lkaic-flag")
CONTINUITY = (0.5, 0.05, 0.4)


def _sweep_pool(runs: dict[str, dict], surf: str) -> dict[tuple, dict[tuple, dict]]:
    """{(gate, vad): {(act, ms, msil): pooled counts over the surface's marked runs}}."""
    pool: dict[tuple, dict[tuple, dict]] = {}
    for rid, m in runs.items():
        if surface(m) != surf or not has_truth(m):
            continue
        for key, c in ((m.get("vad_sweep_round2") or {}).get("cells") or {}).items():
            if c.get("turns") is None or c.get("rank_excluded"):
                continue
            g, v, *p = json.loads(key)
            d = pool.setdefault((g, v), {}).setdefault(tuple(p), {"runs": 0, "missed": 0, "phantom": 0,
                                                                  "merged": 0})
            d["runs"] += 1
            for k in ("missed", "phantom", "merged"):
                d[k] += c.get(k) or 0
    return pool


def _sweep_incident(runs: dict[str, dict], gate: str, vid: str, p: tuple) -> str:
    """Incident outcome at one parameter set: A click/clunk, B throat-clear, PR-42 catch."""
    inc = _incidents(runs)
    key = json.dumps([gate, vid, *p])
    out = []
    a = inc.get("PR-54 A")
    ca = ((a[1].get("vad_sweep_round2") or {}).get("cells") or {}).get(key) if a else None
    if ca:
        h = ca.get("nonspeech_hits") or {}
        out.append(f"A {ca.get('hit')}/{ca.get('turns')} click {'OPEN' if h.get('click-A') else 'no'}"
                   f" clunk {'OPEN' if h.get('hangup-clunk-A') else 'no'}")
    b = inc.get("PR-54 B")
    cb = ((b[1].get("vad_sweep_round2") or {}).get("cells") or {}).get(key) if b else None
    if cb:
        out.append(f"B throat {'OPEN' if (cb.get('nonspeech_hits') or {}).get('throat-clear-B') else 'no'}")
    c = inc.get("PR-42")
    cc = ((c[1].get("vad_sweep_round2") or {}).get("cells") or {}).get(key) if c else None
    if cc and cc.get("segments") is not None:
        snd = (c[1].get("incident") or {}).get("sounds") or []
        segs = cc["segments"]
        caught = lambda g: sum(any(_overlaps(x, (s["start"], s["end"])) for x in segs)  # noqa: E731
                               for s in snd if s["cls"] == g)
        out.append(f"PR-42 {caught('detected')}/21 +{caught('loud_missed')}/7 orphans {cc.get('phantom')}")
    return "; ".join(out) or "—"


@section("sweep")
def sec_sweep(runs: dict[str, dict]) -> str:
    """Parameter sweeps (meta["vad_sweep_round2"]): the best set per combo, how flat
    the optimum is, and the incidents at that set."""
    out = []
    for surf in ("phone", "web"):
        pool = _sweep_pool(runs, surf)
        if not pool:
            continue
        rows = []
        for (g, v), grid in pool.items():
            if g not in SWEEP_GATES or v not in SWEEP_VADS:
                continue   # e.g. aic-1to1's extra sources and VADs
            full = max(d["runs"] for d in grid.values())
            pts = {p: d for p, d in grid.items() if d["runs"] == full}
            sc = {p: (d["missed"] + d["phantom"], d["merged"]) for p, d in pts.items()}
            best = min(sc.values())
            # among the best, prefer the one nearest production's parameters
            bp = min((p for p in sc if sc[p] == best),
                     key=lambda p: (abs(p[0] - 0.6), abs(p[1] - 0.05), abs(p[2] - 0.3)))
            flat = sum(1 for x in sc.values() if x[0] <= best[0] + 1)
            prod, cont = pts.get(PROD_PARAMS), pts.get(CONTINUITY)
            rows.append((best, g, v, bp, pts[bp], flat, len(sc), prod, cont))
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        out.append(f"### {surf.title()} — best parameter set per (gate, VAD), pooled "
                   f"(act = deact; score = missed + phantom, then merged)\n")
        out.append(md(["gate → VAD", "best act / min speech / min silence", "missed", "phantom", "merged",
                       "grid points within +1", "at production 0.6/0.05/0.3", "at 08-04's 0.5/0.05/0.4",
                       "incidents at the best set"],
                      [[f"`{g}` → `{v}`", "/".join(f"{x:g}" for x in bp), d["missed"], d["phantom"],
                        d["merged"], f"{flat}/{n}",
                        f"{prod['missed']}/{prod['phantom']}/{prod['merged']}" if prod else "—",
                        f"{cont['missed']}/{cont['phantom']}/{cont['merged']}" if cont else "—",
                        _sweep_incident(runs, g, v, bp) if surf == "phone" else ""]
                       for best, g, v, bp, d, flat, n, prod, cont in rows]) + "\n")
    return "\n".join(out) or "(no sweep stored yet)"


PROD_COMBO = json.dumps([PROD_GATE, PROD_VAD, *PROD_PARAMS])


# The Cloud row §3.3 already ranked; every other combo on these two sources was run
# for aic-1to1 (2026-09-30) and is reported there, not in vad-wer.
CLOUD_ROW = json.dumps(["lkaic-quail-l-bypass", "lkaic-flag", 0.6, 0.25, 0.55])


def _aic_study_only(key: str) -> bool:
    return key != CLOUD_ROW and json.loads(key)[0] in ("lkaic-quail-l-bypass", "lkaic-vf22-l")


@section("vad-wer")
def sec_vad_wer(runs: dict[str, dict]) -> str:
    """VAD-cut WER (meta["vad_cut_wer_round2"]): one STT request per turn, per combo
    and read mode, as ΔWER against production's config on the same run (hush ->
    lk-inference at 0.6/0.05/0.3, STT reading raw)."""
    out = []
    for surf in ("phone", "web"):
        per: dict[tuple, list[float]] = {}
        absw: dict[tuple, list[float]] = {}
        dropped, excluded, valid = [], [], 0
        for rid, m in runs.items():
            if surface(m) != surf:
                continue
            w = m.get("vad_cut_wer_round2") or {}
            base = ((w.get(PROD_COMBO) or {}).get("raw") or {})
            if base.get("wer") is None:
                continue
            # as 08-04 excluded a run whose passthrough collapsed: a junk baseline
            # hands every combo a win on that run
            k = collapse(base.get("text", ""))
            if k >= 5 or base["wer"]["wer"] > 1.0:
                excluded.append((rid, m.get("note"), f"production baseline collapsed (x{k})" if k >= 5
                                 else f"production baseline WER {base['wer']['wer']:.2f} > 1.0"))
                continue
            valid += 1
            bw = base["wer"]["wer"]
            for key, res in w.items():
                if _aic_study_only(key):
                    continue
                for mode in ("same", "raw"):
                    r = res.get(mode) or {}
                    if r.get("wer") is None:
                        continue
                    ck = (tuple(json.loads(key)), mode)
                    # a loop is COUNTED here, not dropped as the whole-file tables do:
                    # in a VAD-cut transcript it comes from the fragment the VAD sent
                    # (a sentence split at a short pause), which is the VAD's cost
                    if collapse(r.get("text", "")) >= 5:
                        dropped.append(ck)
                    per.setdefault(ck, []).append(r["wer"]["wer"] - bw)
                    absw.setdefault(ck, []).append(r["wer"]["wer"])
        if not per:
            continue
        rows = []
        for (combo, mode), d in sorted(per.items(), key=lambda kv: (statistics.median(kv[1]),
                                                                    statistics.mean(kv[1]))):
            a = agg(d)
            g, v, act, ms, msil = combo
            rows.append([f"`{g}` → `{v}` @ {act:g}/{ms:g}/{msil:g}", mode, a["n"],
                         fmt(statistics.median(absw[(combo, mode)]), ".3f"), fmt(a["median"]),
                         fmt(a["mean"]), fmt(a["worst"]), a["W"], a["L"],
                         dropped.count((combo, mode))])
        out.append(f"### {surf.title()} — VAD-cut WER ({valid} runs; ΔWER vs production: `{PROD_GATE}` → "
                   f"`{PROD_VAD}` @ 0.6/0.05/0.3, STT reads raw)\n")
        out.append(md(["gate → VAD @ act/min speech/min silence", "STT reads", "n", "median WER",
                       "median ΔWER", "mean", "worst", "W", "L", "loops"], rows))
        if excluded:
            out.append("\nRuns excluded (their baseline is junk, so every combo would win them):\n")
            out.append(md(["run", "note", "why"], [[f"`{r}`", n, y] for r, n, y in excluded]))
        if dropped:
            out.append(f"\n\"loops\": transcripts with a 1-3-gram repeated >= 5x in a row. They are "
                       f"counted in the WER ({len(dropped)} on this surface).")
        out.append("")
    return "\n".join(out) or "(no VAD-cut WER stored yet)"


@section("minsil")
def sec_minsil(runs: dict[str, dict]) -> str:
    """Hush -> Silero v4.0 8 kHz at min_silence 0.3 / 0.4 / 0.55 (act 0.6, min speech
    0.05): phone VAD-cut WER (STT reads raw) against production, requests per run,
    added latency, and the incidents at each value."""
    out = []
    for msil in (0.3, 0.4, 0.55):
        key = json.dumps([PROD_GATE, "silero-v4.0-8k", 0.6, 0.05, msil])
        ds, loops, req, breq = [], 0, [], []
        for rid, m in runs.items():
            if surface(m) != "phone":
                continue
            w = m.get("vad_cut_wer_round2") or {}
            b, r = (w.get(PROD_COMBO) or {}).get("raw") or {}, (w.get(key) or {}).get("raw") or {}
            if not b.get("wer") or not r.get("wer"):
                continue
            if collapse(b.get("text", "")) >= 5 or b["wer"]["wer"] > 1.0:
                continue
            ds.append(r["wer"]["wer"] - b["wer"]["wer"])
            loops += collapse(r.get("text", "")) >= 5
            req.append(r.get("turns") or 0)
            breq.append(b.get("turns") or 0)
        if not ds:
            continue
        a = agg(ds)
        pool = (_sweep_pool(runs, "phone").get((PROD_GATE, "silero-v4.0-8k")) or {}).get((0.6, 0.05, msil)) or {}
        out.append([f"{msil:g}", a["n"], fmt(a["median"]), fmt(a["mean"]), f"{a['W']} / {a['L']}",
                    fmt(a["worst"]), loops, f"{statistics.mean(req):.1f} (prod {statistics.mean(breq):.1f})",
                    f"+{msil - PROD_PARAMS[2]:.2f} s", f"{pool.get('missed')}/{pool.get('phantom')}",
                    _sweep_incident(runs, "hush-aih", "silero-v4.0-8k", (0.6, 0.05, msil))])
    return md(["min silence", "runs", "median ΔWER", "mean", "W / L", "worst", "loops", "requests per run",
               "added latency vs production", "phone pool missed/phantom", "incidents on hush-aih"], out)


AIC_SOURCES = ("lkaic-quail-l-bypass", "lkaic-vf22-l")
AIC_VADS = ("lkaic-flag", "aic-vad-vf-2.0", "aic-vad-ms-2.1", "silero-v4.0-8k", PROD_VAD)
AIC_GRID = [(0.6, ms, msil) for ms in (0.05, 0.10, 0.15, 0.25) for msil in (0.3, 0.4, 0.55)]
SHORT_TURN_S = 0.4


def _cell(m: dict, g: str, v: str, p: tuple) -> dict | None:
    return ((m.get("vad_sweep_round2") or {}).get("cells") or {}).get(json.dumps([g, v, *p]))


def _wer_point(runs: dict[str, dict], key: str) -> dict | None:
    """Phone VAD-cut WER (STT reads raw) against production, the `minsil` rules."""
    ds, loops = [], 0
    for m in runs.values():
        if surface(m) != "phone":
            continue
        w = m.get("vad_cut_wer_round2") or {}
        b, r = (w.get(PROD_COMBO) or {}).get("raw") or {}, (w.get(key) or {}).get("raw") or {}
        if not b.get("wer") or not r.get("wer") or collapse(b.get("text", "")) >= 5 or b["wer"]["wer"] > 1.0:
            continue
        ds.append(r["wer"]["wer"] - b["wer"]["wer"])
        loops += collapse(r.get("text", "")) >= 5
    return {**agg(ds), "loops": loops} if ds else None


def aic_point(runs: dict[str, dict], g: str, v: str, p: tuple) -> dict:
    """One (source, VAD, parameter point): phone pool, incidents, WER, flag coverage."""
    out = {"runs": 0, "missed": 0, "phantom": 0, "merged": 0, "short_turns": 0, "short_missed": 0}
    used = [m for m in runs.values() if surface(m) == "phone" and has_truth(m)]
    for m in used:
        c = _cell(m, g, v, p)
        if not c or c.get("turns") is None or c.get("rank_excluded"):
            continue
        out["runs"] += 1
        for k in ("missed", "phantom", "merged"):
            out[k] += c.get(k) or 0
        out["short_missed"] += sum(1 for s, e in c.get("missed_spans") or [] if e - s < SHORT_TURN_S)
    inc = _incidents(runs)
    fails = 0
    a = inc.get("PR-54 A")
    ca = _cell(a[1], g, v, p) if a else None
    if ca:
        h = ca.get("nonspeech_hits") or {}
        known = [tuple(x) for x in (a[1].get("truth_spans") or []) + (a[1].get("ignore_spans") or [])] + [
            (w["start"], w["end"]) for w in a[1].get("known_nonspeech") or []]
        other = sum(1 for s in ca.get("segments") or [] if not any(_overlaps(s, w) for w in known))
        fails += (ca["turns"] - ca["hit"]) + int(bool(h.get("click-A"))) + int(bool(h.get("hangup-clunk-A"))) + other
        out["A"] = (f"{ca['hit']}/{ca['turns']} click {'OPEN' if h.get('click-A') else 'no'} clunk "
                    f"{'OPEN' if h.get('hangup-clunk-A') else 'no'}" + (f" +{other} phantom" if other else ""))
    b = inc.get("PR-54 B")
    cb, cp = (_cell(b[1], g, v, p), _cell(b[1], g, PROD_VAD, PROD_PARAMS)) if b else (None, None)
    if cb:
        # B discriminates only where production's VAD, on the same source, opens there
        disc = bool(cp and (cp.get("nonspeech_hits") or {}).get("throat-clear-B"))
        opened = bool((cb.get("nonspeech_hits") or {}).get("throat-clear-B"))
        fails += int(opened)
        out["B"] = "OPEN" if opened else ("rejects" if disc else "n/a")
    c = inc.get("PR-42")
    cc = _cell(c[1], g, v, p) if c else None
    if cc and cc.get("segments") is not None:
        snd = (c[1].get("incident") or {}).get("sounds") or []
        caught = lambda cls: sum(any(_overlaps(x, (s["start"], s["end"])) for x in cc["segments"])  # noqa: E731
                                 for s in snd if s["cls"] == cls)
        det, lm = caught("detected"), caught("loud_missed")
        nd = sum(1 for s in snd if s["cls"] == "detected")
        nl = sum(1 for s in snd if s["cls"] == "loud_missed")
        fails += (nd - det) + (nl - lm) + (cc.get("phantom") or 0)
        out["PR42"] = f"{det}/{nd} +{lm}/{nl} orphans {cc.get('phantom')}"
    out["fails"] = fails
    if v == "lkaic-flag":
        cov = []
        for m in used + [x[1] for x in inc.values()]:
            rec = cands(m).get(g) or {}
            info = rec.get("aicvad") or {}
            if info.get("frames"):
                cov.append(1 - (info.get("missing_frames") or 0) / info["frames"])
        out["coverage"] = min(cov) if cov else None
    out["wer"] = _wer_point(runs, json.dumps([g, v, *p]))
    return out


def aic_best(runs: dict[str, dict], g: str, v: str) -> tuple[tuple, dict] | None:
    """Best incident point: fewest incident failures, then pool missed + phantom, then
    merged, then nearest production's parameters."""
    pts = [(p, aic_point(runs, g, v, p)) for p in AIC_GRID]
    pts = [(p, d) for p, d in pts if d["runs"]]
    if not pts:
        return None
    return min(pts, key=lambda pd: (pd[1]["fails"], pd[1]["missed"] + pd[1]["phantom"], pd[1]["merged"],
                                    abs(pd[0][1] - 0.05) + abs(pd[0][2] - 0.3)))


def _wer_str(w: dict | None) -> str:
    if not w:
        return "—"
    return f"{w['median']:+.3f} / {w['mean']:+.3f} / {w['W']}-{w['L']} / {w['worst']:+.3f} / {w['loops']}"


@section("aic-1to1")
def sec_aic_1to1(runs: dict[str, dict]) -> str:
    """The LiveKit Cloud ai-coustics flag against the standalone ai-coustics VADs (and two
    references) on the same audio at the same settings: act = deact 0.6 (the flag is binary,
    so it's segmented at 0.5 anyway) x min speech x min silence, prefix pad 0.5."""
    head_rows, grid_rows = [], []
    for g in AIC_SOURCES:
        for v in AIC_VADS:
            best = aic_best(runs, g, v)
            if best is None:
                continue
            bp, d = best
            head_rows.append([f"`{g}`", f"`{v}`", f"{bp[1]:g} / {bp[2]:g}",
                              f"{d['missed']}/{d['phantom']}/{d['merged']}", d.get("A", "—"), d.get("B", "—"),
                              d.get("PR42", "—"), d["fails"], _wer_str(d["wer"]),
                              fmt(d.get("coverage"), ".2%") if v == "lkaic-flag" else ""])
            for p in AIC_GRID:
                e = aic_point(runs, g, v, p)
                if not e["runs"]:
                    continue
                grid_rows.append([f"`{g}`", f"`{v}`", f"{p[1]:g}", f"{p[2]:g}",
                                  f"{e['missed']}/{e['phantom']}/{e['merged']}", e["short_missed"],
                                  e.get("A", "—"), e.get("B", "—"), e.get("PR42", "—"), e["fails"],
                                  _wer_str(e["wer"])])
    ns = {aic_point(runs, g, v, p)["runs"] for g in AIC_SOURCES for v in AIC_VADS for p in AIC_GRID} - {0}
    phone = [m for m in runs.values() if surface(m) == "phone" and has_truth(m)]
    short_rows = []
    for g in AIC_SOURCES:
        for v in AIC_VADS:
            row = [f"`{g}`", f"`{v}`"]
            for msil in (0.3, 0.4, 0.55):
                n = sum(1 for m in phone for s, e in coalesce_truth(m["truth_spans"], msil,
                                                                    m.get("known_nonspeech")) if e - s < SHORT_TURN_S)
                lo, hi = (aic_point(runs, g, v, (0.6, ms, msil)) for ms in (0.05, 0.25))
                row.append(f"{lo['short_missed']} → {hi['short_missed']} of {n}" if hi["runs"] else "—")
            short_rows.append(row)
    off_rows = []
    for g in ("none", PROD_GATE):
        best = aic_best(runs, g, "aic-vad-vf-2.0")
        for p in dict.fromkeys([(0.6, 0.25, 0.55)] + ([best[0]] if best else [])):
            d = aic_point(runs, g, "aic-vad-vf-2.0", p)
            if d["runs"]:
                off_rows.append([f"`{g}`", "`aic-vad-vf-2.0`", f"{p[1]:g} / {p[2]:g}",
                                 f"{d['missed']}/{d['phantom']}/{d['merged']}", d.get("A", "—"), d.get("B", "—"),
                                 d.get("PR42", "—"), d["fails"]])
    out = ["### Best point per source and VAD\n",
           "\"Incident failures\" adds up every wrong call on the three incidents: A turns lost, A click / "
           "clunk / other phantoms opened, B's throat-clear opened, PR-42 detected utterances lost, "
           "loud-missed sounds not caught, and orphan triggers. The best point has the fewest, then the "
           "fewest pool missed + phantom, then the fewest merged, then the one nearest production's "
           "0.05 / 0.3. The phone pool is missed / phantom / merged over the "
           f"{'/'.join(map(str, sorted(ns)))} marked phone runs. VAD-cut WER is Δ vs production (median / "
           "mean / W-L / worst / loops, 9 phone runs, STT reads raw, one request per segment). Where "
           "production's own VAD, on the same source, doesn't open on B's throat-clear, B reads n/a.\n",
           md(["source", "VAD", "best min speech / min silence", "pool m/p/merged", "PR-54 A", "PR-54 B",
               "PR-42 det / missed caught / orphans", "incident failures", "VAD-cut WER", "flag coverage"],
              head_rows),
           "\n### Off the Cloud rail\n\nThe standalone VF 2.0 VAD on the stored input (`none`) and "
           "behind the bench's 16 kHz Hush, with no Cloud replay pass, at min speech 0.25 / min silence "
           "0.55, plus its best point where that differs:\n",
           md(["source", "VAD", "point", "pool m/p/merged", "PR-54 A", "PR-54 B", "PR-42", "fails"], off_rows),
           f"\n### Short-turn check\n\nMarked phone turns under {SHORT_TURN_S:g} s (after coalescing at "
           "that min silence) missed at min speech 0.05 → 0.25:\n",
           md(["source", "VAD", "min silence 0.3", "0.4", "0.55"], short_rows),
           "\n### Every point\n\nAct = deact 0.6, prefix pad 0.5. \"Short missed\" counts marked phone "
           f"turns under {SHORT_TURN_S:g} s that no segment touched. VAD-cut WER was run at min speech "
           "{0.05, 0.25} × every min silence on `lkaic-quail-l-bypass` (plus Silero v4.0's best point there), "
           "and at each VAD's best point on `lkaic-vf22-l`.\n",
           md(["source", "VAD", "min speech", "min silence", "pool m/p/merged", "short missed", "PR-54 A",
               "PR-54 B", "PR-42", "fails", "VAD-cut WER"], grid_rows)]
    return "\n".join(out)


@section("tail")
def sec_tail(runs: dict[str, dict]) -> str:
    """20260804-112511 alone, with and without its unmarked tail (23.8 s to the end,
    flagged for a listen): phantom segments per VAD, re-scored in memory."""
    m = runs.get(TAIL_RUN)
    if not m:
        return "(run missing)"
    rows = []
    for sid in ("none", PROD_GATE):
        for vid in (PROD_VAD, "silero-v4.0-8k", "firered-stream", "aic-vad-vf-2.0", "earshot", "ten"):
            r = vb_row(m, sid, vid)
            if not r:
                continue
            t = rescore(m, r["segments"], [[TAIL_FROM, 1e9]], r.get("params"))
            rows.append([f"`{sid}` → `{vid}`", ev(r, "phantom_n"), ev(t, "phantom_n"),
                         ev(t, "ignored_segments"), ev(r, "missed_n")])
    return md(["gate → VAD", "phantoms, tail as stored", "phantoms, tail ignored", "segments in the tail",
               "missed turns"], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--section", action="append", help="one or more of --list; default all")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, fn in SECTIONS.items():
            print(f"{k:12s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return
    runs = load_runs()
    for name in args.section or list(SECTIONS):
        print(f"\n<!-- report_round2.py --section {name} -->\n")
        print(SECTIONS[name](runs))


if __name__ == "__main__":
    main()
