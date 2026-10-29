"""Round-2 backfill: add NC candidates to stored runs, and re-run whole-file STT.

    .venv/bin/python scripts/backfill_round2.py process --runs all --cands aicsdk
    .venv/bin/python scripts/backfill_round2.py process --runs all --cands cloud
    .venv/bin/python scripts/backfill_round2.py process --runs 20260929-123329 --cands incident-offline
    .venv/bin/python scripts/backfill_round2.py stt --runs all
    .venv/bin/python scripts/backfill_round2.py status

`process` runs the server's own `_process_run`, the same merge path the UI's
"Re-run ticked candidates" uses: records are merged by candidate id, a record
not in this pass is left exactly as it was, and a failed attempt keeps the
earlier result under `last_attempt`. It never replaces meta["candidates"].
Runs go one at a time, one chain at a time (concurrency 1): the machine has
8 GB, and the cloud replay needs the CPU to itself.

Candidate sets (`--cands`, or a comma list of ids):
  aicsdk            every available aicsdk-* candidate
  cloud             cloud-raw, the three Krisp replay aliases, krisp-viva(-tel), every lkaic-*
  incident-offline  every available non-historical offline candidate (the incident
                    runs start with none)
  offline-all       the same list, on any run

**Replay aliases.** A stored run already holds `krisp-nc`, `krisp-bvc` and
`krisp-bvc-tel` from the LIVE rail. A cloud replay under the same id would
replace them, so the replayed ones get their own ids, `krisp-nc-replay` etc.,
on every run including the incidents. They exist only here, as an overlay on
the server's registry, so the UI can't tick them for a live session. Stored
cards render from the record's own label.

Resumable: a candidate whose record is already `done` with its output on disk
is skipped unless --force. Cloud candidates whose replay failed are retried up
to --retries times; a candidate still failing after that stops the backfill
(the replay gates are hard stops, not log-and-continue). Every pass is logged
in meta["backfill_round2"] and in data/backfill_round2.jsonl.

`stt` re-transcribes every candidate of every run that has a reference script
with the current STT setting (.env: boost_volume=false). The old transcript and
its WER move to `stt_v1` first, so every table reads one STT setting. Records
this backfill produced already carry that setting (`stt_setting`) and are
skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import config, scoring, store, stt  # noqa: E402

PASS = "round2-2026-09-29"
LOG = config.DATA_DIR / "backfill_round2.jsonl"

REPLAY_ALIASES = {
    "krisp-nc-replay": ("krisp-nc", "Krisp NC (LiveKit Cloud) (cloud replay)"),
    "krisp-bvc-replay": ("krisp-bvc", "Krisp BVC (LiveKit Cloud) (cloud replay)"),
    "krisp-bvc-tel-replay": ("krisp-bvc-tel", "Krisp BVCTelephony (LiveKit Cloud) (cloud replay)"),
}
CLOUD_SET = ["cloud-raw", *REPLAY_ALIASES, "krisp-viva", "krisp-viva-tel",
             "lkaic-quail-l", "lkaic-quail-l-bypass", "lkaic-vf22-l", "lkaic-vf22-s",
             "lkaic-rook-s", "lkaic-rook-s-bypass"]

# what the stored runs' transcripts were made with: .env.example and the code
# default both said true until 2026-09-29 (git 6dce540)
V1_SETTING = {"boost_volume": True, "source": "the .env.example / config default before 2026-09-29"}


def log(ev: dict) -> None:
    ev = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **ev}
    print(json.dumps(ev), flush=True)
    with LOG.open("a") as f:
        f.write(json.dumps(ev) + "\n")


def stt_setting() -> dict:
    return {"boost_volume": config.WHISPER_BOOST_VOLUME, "whole_file": True, "pass": PASS}


# -------------------------------------------------------------- registry

def _server():
    """The server module, with the replay aliases added to its registry."""
    from nc_bench import server

    base = server._candidates_by_id

    def with_aliases() -> dict[str, dict]:
        known = base()
        for alias, (cid, label) in REPLAY_ALIASES.items():
            if cid in known:
                known[alias] = {**known[cid], "id": alias, "label": label, "replay_alias_of": cid}
        return known

    server._candidates_by_id = with_aliases
    return server


def resolve_cands(spec: str, known: dict[str, dict]) -> list[str]:
    avail = lambda c: known[c]["available"] and not known[c].get("historical")  # noqa: E731
    if spec == "aicsdk":
        return [c for c in known if c.startswith("aicsdk-") and avail(c)]
    if spec == "cloud":
        return list(CLOUD_SET)
    if spec in ("incident-offline", "offline-all"):
        return [c for c, e in known.items() if e["rail"] == "offline" and avail(c)]
    ids = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in ids if c not in known]
    if unknown:
        raise SystemExit(f"unknown candidates: {unknown}")
    return ids


def resolve_runs(spec: str) -> list[str]:
    if spec == "all":
        return sorted(d.name for d in config.RUNS_DIR.iterdir()
                      if (d / "meta.json").is_file() and not d.name.startswith("_"))
    return [r.strip() for r in spec.split(",") if r.strip()]


_real_transcribe = stt.transcribe


async def _no_script_stt(_wav: Path) -> dict:
    raise RuntimeError("skipped: this run has no reference script (round-2 backfill)")


def _done(rec: dict | None, run_dir: Path, cloud: bool) -> bool:
    if not rec or rec.get("status") != "done" or not rec.get("output"):
        return False
    if not (run_dir / rec["output"]).is_file():
        return False
    return rec.get("rail") == "cloud_replay" if cloud else True


# --------------------------------------------------------------- process

def process(runs: list[str], spec: str, retries: int, force: bool, dry: bool,
            keep_going: bool = False) -> None:
    server = _server()
    known = server._candidates_by_id()
    cands = resolve_cands(spec, known)
    down = [c for c in cands if not known[c]["available"]]
    if down:
        raise SystemExit("unavailable: " + "; ".join(f"{c} ({known[c]['unavailable_reason']})"
                                                      for c in down))
    halted: list[str] = []
    for rid in runs:
        meta = store.load_meta(rid)
        run_dir = store.run_dir(rid)
        stored = {c.get("id"): c for c in meta.get("candidates") or []}
        todo = [c for c in cands if force or not _done(stored.get(c), run_dir,
                                                      known[c]["rail"] == "cloud")]
        if not todo:
            log({"run": rid, "cands": spec, "skip": "all done"})
            continue
        if dry:
            log({"run": rid, "cands": spec, "would_run": todo})
            continue
        attempts: list[dict] = []
        pending = list(todo)
        script = (meta.get("script") or "").strip()
        # With no reference script a whole-file transcript scores nothing, and on
        # a 7-minute incident call it only waits out the STT timeout per candidate
        server.stt.transcribe = _real_transcribe if script else _no_script_stt
        for attempt in range(retries + 1):
            t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
            w0 = time.time()
            asyncio.run(server._process_run(rid, pending, None, script, 1))
            with store.locked_meta(rid) as m:
                failed = {}
                for c in m.get("candidates") or []:
                    if c.get("id") not in pending:
                        continue
                    fresh = (c.get("processed_at") or "") >= t0
                    if c.get("status") == "done" and fresh and not c.get("last_attempt"):
                        c["stt_setting"] = stt_setting()
                        c["backfill"] = PASS
                    else:
                        la = c.get("last_attempt") or {}
                        failed[c["id"]] = la.get("error") or c.get("error") or "not processed"
                entry = {"at": t0, "attempt": attempt, "ids": pending,
                         "wall_s": round(time.time() - w0, 1), "failed": failed,
                         # the replay gates this pass ran under, when overridden
                         "env": {k: os.environ[k] for k in ("CLOUD_REPLAY_MAX_STREAMS",
                                                            "CLOUD_REPLAY_MAX_DRIFT_MS")
                                 if k in os.environ}}
                bf = m.setdefault("backfill_round2", {})
                bf.setdefault("passes", []).append({"cands": spec, **entry})
            attempts.append(entry)
            log({"run": rid, "cands": spec, **entry})
            if not failed:
                break
            # only the cloud replay is worth repeating: a failed offline chain
            # fails the same way twice
            pending = [c for c in failed if known[c]["rail"] == "cloud"]
            if not pending:
                break
        still = attempts[-1]["failed"]
        cloud_still = {c: e for c, e in still.items() if known[c]["rail"] == "cloud"}
        with store.locked_meta(rid) as m:
            bf = m.setdefault("backfill_round2", {})
            if any(known[c]["rail"] == "cloud" for c in todo):
                bf["cloud_replay"] = {"retries": len(attempts) - 1, "failed": cloud_still,
                                      "at": attempts[0]["at"], "env": attempts[-1]["env"]}
        if cloud_still:
            # a batch-level failure (control, starvation, drift) that survives the
            # retries means the rail itself is sick: that always halts. A
            # candidate's own failure with a clean control and twin is a result.
            # ...unless the batch that failed was a retry of candidates that had
            # already failed their own gates: that batch says nothing about the
            # rail (a 1-stream retry of a stream Krisp NC flattened, say)
            own_before = {c for a in attempts[:-1] for c, e in a["failed"].items()
                          if not e.startswith("cloud replay batch")}
            transport = {c: e for c, e in cloud_still.items()
                         if e.startswith("cloud replay batch") and c not in own_before}
            log({"run": rid, "halt" if (transport or not keep_going) else "candidate failures kept":
                 "cloud replay still failing after retries", "failed": cloud_still})
            if transport or not keep_going:
                raise SystemExit(f"{rid}: cloud replay failed {len(attempts)}x — stopping: {cloud_still}")
            halted.append(rid)
        # offline failures are never retried, so they sit in whichever pass had them
        offline_failed = {c: e for a in attempts for c, e in a["failed"].items()
                          if known[c]["rail"] != "cloud"}
        if offline_failed:
            log({"run": rid, "warn": "offline candidates failed", "failed": offline_failed})
    if halted:
        log({"keep_going": "candidate-level replay failures, recorded per run", "runs": halted})


# ------------------------------------------------------------------- stt

def run_stt(runs: list[str], force: bool, dry: bool) -> None:
    if config.WHISPER_BOOST_VOLUME:
        raise SystemExit("WHISPER_BOOST_VOLUME is true in .env; round 2 transcribes with false")
    for rid in runs:
        meta = store.load_meta(rid)
        run_dir = store.run_dir(rid)
        script = (meta.get("script") or "").strip()
        if not script:
            log({"run": rid, "stt": "skip: no reference script"})
            continue
        todo = [c for c in meta.get("candidates") or []
                if c.get("output") and (run_dir / c["output"]).is_file()
                and (force or (c.get("stt_setting") or {}).get("boost_volume") is not False)]
        if dry:
            log({"run": rid, "stt_would_run": [c["id"] for c in todo]})
            continue
        done = errors = 0
        for c in todo:
            try:
                got, err = asyncio.run(_real_transcribe(run_dir / c["output"])), None
            except Exception as e:  # noqa: BLE001 - recorded on the candidate
                got, err = None, str(e)
            with store.locked_meta(rid) as m:
                rec = next((x for x in m.get("candidates") or [] if x.get("id") == c["id"]), None)
                if rec is None or rec.get("output") != c["output"]:
                    continue   # re-processed meanwhile: this transcript is of other audio
                if "stt_v1" not in rec and (rec.get("stt_setting") or {}).get("boost_volume") is not False:
                    rec["stt_v1"] = {**(rec.get("stt") or {}),
                                     "wer": (rec.get("scores") or {}).get("wer"),
                                     "error": rec.get("stt_error"), "setting": V1_SETTING}
                if err is None:
                    rec["stt"] = got
                    rec.pop("stt_error", None)
                else:
                    rec.pop("stt", None)
                    rec["stt_error"] = err
                    errors += 1
                if err is None:
                    # an error stays retryable: only a transcript made with the
                    # new setting marks the record as done
                    rec["stt_setting"] = stt_setting()
                if isinstance(rec.get("scores"), dict):
                    rec["scores"]["wer"] = scoring.wer(script, (rec.get("stt") or {}).get("text", ""))
            done += 1
        log({"run": rid, "stt_done": done, "stt_errors": errors})


def status() -> None:
    for rid in resolve_runs("all"):
        m = store.load_meta(rid)
        cs = m.get("candidates") or []
        ids = {c["id"] for c in cs if c.get("status") == "done"}
        a = sum(c.startswith("aicsdk-") for c in ids)
        cl = sum(c in ids for c in CLOUD_SET)
        v2 = sum((c.get("stt_setting") or {}).get("boost_volume") is False for c in cs)
        print(f"{rid:20s} {m.get('source'):6s} cands {len(cs):3d} aicsdk {a:2d}/12 cloud {cl:2d}/12 "
              f"stt-v2 {v2:3d} backup {'yes' if (store.run_dir(rid) / 'meta.v1.json').is_file() else 'NO'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process")
    p.add_argument("--runs", required=True)
    p.add_argument("--cands", required=True)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-going", action="store_true",
                   help="a candidate whose own replay gate still fails after --retries (control and "
                        "twin clean) is recorded and the next run starts; a batch-level failure still halts")
    s = sub.add_parser("stt")
    s.add_argument("--runs", required=True)
    s.add_argument("--force", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "process":
        process(resolve_runs(args.runs), args.cands, args.retries, args.force, args.dry_run,
                args.keep_going)
    elif args.cmd == "stt":
        run_stt(resolve_runs(args.runs), args.force, args.dry_run)
    else:
        status()


if __name__ == "__main__":
    main()
