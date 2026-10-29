"""FastAPI app: session control, upload, candidate registry, run history,
and a websocket that streams recorder levels + processing progress to the UI."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from . import cloud_replay, config, export, lk_cloud, marks, scoring, store, stt, vad
from .pipeline import run_chain
from .processors import chain_available
from .recorder import (
    RECORD_RATE,
    Recorder,
    ensure_worker,
    shutdown_worker,
    worker_alive,
    worker_events,
)

lk_cloud.preload()  # cloud NC plugins must register on the main thread

# How many candidate chains process at once. Kept low by default: these are
# CPU-bound ONNX graphs sharing one machine with a local STT endpoint, and the
# per-block latency columns stop being comparable as soon as chains compete for
# the CPU (see meta["concurrency"], recorded per run).
CONCURRENCY_CHOICES = [1, 2, 3, 4, 6, 8]
DEFAULT_CONCURRENCY = 2


def _clamp_concurrency(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CONCURRENCY
    return max(1, min(max(CONCURRENCY_CHOICES), n))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Register with LiveKit here — after the port is bound, and before anyone can
    # press Start. After the bind, because a second copy of the bench that loses
    # the port must not register a second worker under the same agent name:
    # LiveKit would load-balance calls across both and the copy without an open
    # session would drop them. Before Start, because a SIP dispatch that finds no
    # worker means the room is never created and the call fails silently.
    ensure_worker()
    yield
    await asyncio.to_thread(shutdown_worker)


app = FastAPI(title="NC Bench", lifespan=_lifespan)

# ------------------------------------------------------------------ ws hub

_clients: set[WebSocket] = set()


async def broadcast(event: dict) -> None:
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive pings from the page; ignored
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


# ------------------------------------------------------------- candidates


def _load_candidates() -> list[dict]:
    entries = json.loads(config.CANDIDATES_FILE.read_text())
    for entry in entries:
        # rail: "offline" = a chain through pipeline.py; "cloud" = a LiveKit-hosted
        # filter, run on the live rail during a session and on the cloud replay
        # rail (cloud_replay.py) for uploads and reprocessing
        entry["rail"] = "cloud" if "lk_model" in entry else "offline"
        # nothing is live-only any more; kept because the page still reads it
        entry["live_only"] = False
        if entry.get("historical"):
            # kept so old runs keep their labels; the model behind it is gone
            ok, why = False, "historical — kept for old runs' labels, can't be re-run"
        elif entry["rail"] == "cloud":
            ok, why = lk_cloud.available(entry["lk_model"], entry.get("lk_options"))
        else:
            ok, why = chain_available(entry["chain"])
        entry["available"] = ok
        entry["unavailable_reason"] = why
    return entries


def _candidates_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in _load_candidates()}


def _check_runnable(candidate_ids: list[str], known: dict[str, dict]) -> None:
    """400 for candidates that cannot run on any rail.

    Historical entries only exist to label old runs. An unavailable cloud
    candidate is refused up front rather than failing inside a replay that costs
    a real-time pass per batch; unavailable offline chains still error per
    candidate, as they always have.
    """
    historical = [c for c in candidate_ids if known[c].get("historical")]
    if historical:
        raise HTTPException(400, f"historical candidates can't be re-run: {historical}")
    cloud_down = [f"{c} ({known[c]['unavailable_reason']})" for c in candidate_ids
                  if known[c]["rail"] == "cloud" and not known[c]["available"]]
    if cloud_down:
        raise HTTPException(400, f"cloud candidates unavailable: {cloud_down}")


@app.get("/api/candidates")
async def api_candidates():
    return _load_candidates()


@app.get("/api/config")
async def api_config():
    return {
        "livekit_url": config.LIVEKIT_URL,
        "whisper_url": config.WHISPER_URL,
        "hecttor_model_default": config.HECTTOR_MODEL,
        "agent_name": config.LK_AGENT_NAME,
        "worker_alive": worker_alive(),
        "concurrency_choices": CONCURRENCY_CHOICES,
        "concurrency_default": DEFAULT_CONCURRENCY,
        # the .env segmenter values seed the VAD bench's parameter boxes
        "vad_defaults": vad.defaults(),
        "vad_limits": vad.LIMITS,
    }


@app.get("/api/worker")
async def api_worker():
    """Job-level trail. Tells 'the call never arrived' apart from 'it arrived and
    was dropped', which is invisible from the run alone."""
    return {"agent_name": config.LK_AGENT_NAME, "alive": worker_alive(),
            "events": worker_events()}


# ---------------------------------------------------------------- session

_session: dict | None = None
_session_lock = asyncio.Lock()


@app.post("/api/session/start")
async def session_start(body: dict):
    global _session
    source = body.get("source")
    candidate_ids = body.get("candidates") or []
    if source not in ("phone", "web"):
        raise HTTPException(400, "source must be 'phone' or 'web'")
    if not candidate_ids:
        raise HTTPException(400, "tick at least one candidate")
    known = _candidates_by_id()
    unknown = [c for c in candidate_ids if c not in known]
    if unknown:
        raise HTTPException(400, f"unknown candidates: {unknown}")
    _check_runnable(candidate_ids, known)

    # a live session runs its cloud candidates on the live rail, as filters on
    # the track being recorded — no replay, so no second Opus pass
    live_cands = [
        {"id": cid, "lk_model": known[cid]["lk_model"],
         "lk_options": known[cid].get("lk_options") or {}}
        for cid in candidate_ids
        if known[cid]["rail"] == "cloud"
    ]
    async with _session_lock:
        if _session is not None:
            raise HTTPException(409, "a session is already active")
        run_id, run_dir = store.new_run(source, (body.get("note") or "").strip())
        recorder = Recorder(emit=broadcast, live_candidates=live_cands)
        _session = {
            "run_id": run_id,
            "run_dir": run_dir,
            "recorder": recorder,
            "candidates": candidate_ids,
            "script": (body.get("script") or "").strip(),
            "concurrency": _clamp_concurrency(body.get("concurrency")),
        }

    try:
        if source == "web":
            join = await recorder.start_web()
            await broadcast({"type": "session", "state": "web_ready", "room": join["room"]})
            return {"run_id": run_id, **join}
        dial = await recorder.start_phone()
        return {"run_id": run_id, **dial}
    except Exception:
        _session = None
        raise


@app.post("/api/session/stop")
async def session_stop():
    global _session
    async with _session_lock:
        if _session is None:
            raise HTTPException(409, "no active session")
        sess, _session = _session, None

    input_meta, live_results = await sess["recorder"].stop(sess["run_dir"] / "input.wav")
    detail = ""
    if input_meta["file"] is None or input_meta.get("silent"):
        detail = (
            # the SIP reason when there is one — "no call arrived" is useless next
            # to "the trunk rejected the number"
            (f"the outbound call never connected — {input_meta['dial_error']}"
             if input_meta.get("dial_error")
             else "no audio was recorded (no call arrived / no mic frames)")
            if input_meta["file"] is None
            else (
                f"recorded {input_meta['duration_s']}s at {input_meta.get('level_dbfs')} dBFS "
                f"from {(input_meta.get('diag') or {}).get('participant', 'the caller')} — the "
                "leg was up but carried no audio. One-way audio: check that the PBX is "
                "bridging the caller's media (and that the handset isn't muted)."
            )
        )
    # under the run's lock: a note typed while the call was up must survive
    with store.locked_meta(sess["run_id"]) as meta:
        meta["input"] = input_meta
        if detail:
            meta["status"] = "empty"
            meta["error"] = detail
        else:
            meta["status"] = "processing"
    if detail:
        await broadcast({"type": "session", "state": "idle"})
        return {"run_id": sess["run_id"], "status": "empty", "detail": detail}

    asyncio.create_task(
        _process_run(sess["run_id"], sess["candidates"], live_results,
                     sess["script"], sess["concurrency"])
    )
    return {"run_id": sess["run_id"], "status": "processing"}


@app.post("/api/runs/{run_id}/note")
async def set_note(run_id: str, body: dict):
    """Annotate a finished run.

    What made a run interesting is usually only clear after reading its results,
    and the runs already on disk were recorded before notes existed — so the note
    has to be editable, not just captured at Start.
    """
    try:
        with store.locked_meta(run_id) as meta:
            meta["note"] = (body.get("note") or "").strip()
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None
    return {"run_id": run_id, "note": meta["note"]}


@app.post("/api/runs/{run_id}/candidates/{cid}/stt")
async def rerun_stt(run_id: str, cid: str):
    """Re-transcribe one candidate from the audio already on disk.

    The NC chain is not re-run — the output wav is untouched — so only the
    transcript and the WER derived from it change. DNSMOS, band and gap-RMS are
    deliberately left alone: recomputing them here would be wasted work and would
    silently re-measure this one candidate under different conditions than the
    rest of its run.
    """
    try:
        meta = store.load_meta(run_id)
        run_dir = store.run_dir(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None

    entry = next((e for e in meta.get("candidates") or [] if e.get("id") == cid), None)
    if entry is None:
        raise HTTPException(404, f"{cid} is not in run {run_id}")
    out_wav = run_dir / (entry.get("output") or "")
    if not entry.get("output") or not out_wav.is_file():
        raise HTTPException(409, f"{cid} has no output audio to transcribe")

    try:
        got, err = await stt.transcribe(out_wav), None
    except Exception as e:
        got, err = None, str(e)

    # the STT ran unlocked; merge only this candidate's transcript into a fresh read
    with store.locked_meta(run_id) as meta:
        entry = next((e for e in meta.get("candidates") or [] if e.get("id") == cid), None)
        if entry is None:
            raise HTTPException(409, f"{cid} left run {run_id} while it was transcribed")
        if err is None:
            entry["stt"] = got
            entry.pop("stt_error", None)
        else:
            entry.pop("stt", None)
            entry["stt_error"] = err
        script = (meta.get("script") or "").strip()
        if script and isinstance(entry.get("scores"), dict):
            # WER is a function of the transcript, so it has to move with it
            entry["scores"]["wer"] = scoring.wer(
                script, (entry.get("stt") or {}).get("text", "")
            )
    await broadcast({"type": "progress", "run_id": run_id, "candidate": cid,
                     "stage": entry.get("status", "done"), "entry": entry})
    return entry


# ----------------------------------------------------------------- upload


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    candidates: str = Form(...),
    script: str = Form(""),
    note: str = Form(""),
    concurrency: int = Form(DEFAULT_CONCURRENCY),
):
    candidate_ids = json.loads(candidates)
    if not candidate_ids:
        raise HTTPException(400, "tick at least one candidate")
    known = _candidates_by_id()
    unknown = [c for c in candidate_ids if c not in known]
    if unknown:
        raise HTTPException(400, f"unknown candidates: {unknown}")
    _check_runnable(candidate_ids, known)
    if _session is not None:
        raise HTTPException(409, "a live session is active; stop it first")

    run_id, run_dir = store.new_run("upload", note.strip())
    suffix = Path(file.filename or "audio").suffix or ".bin"
    original = run_dir / f"original{suffix}"
    original.write_bytes(await file.read())

    # decode anything ffmpeg understands to mono wav at the source rate
    input_wav = run_dir / "input.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(original), "-ac", "1",
         "-acodec", "pcm_s16le", str(input_wav)],
        capture_output=True,
    )
    if proc.returncode != 0 or not input_wav.exists():
        with store.locked_meta(run_id) as meta:
            meta["status"] = "error"
            meta["error"] = "ffmpeg could not decode this file"
        raise HTTPException(400, f"could not decode audio: {proc.stderr.decode()[-300:]}")

    import soundfile as sf

    info = sf.info(input_wav)
    with store.locked_meta(run_id) as meta:
        meta["input"] = {
            "file": "input.wav",
            "duration_s": round(info.duration, 2),
            "sample_rate": info.samplerate,
            "original_name": file.filename,
        }
        meta["status"] = "processing"
    asyncio.create_task(_process_run(run_id, candidate_ids, None, script.strip(),
                                     _clamp_concurrency(concurrency)))
    return {"run_id": run_id, "status": "processing"}


# ------------------------------------------------------------- processing


def _referenced(rec: dict | None) -> set[str]:
    """The files in run_dir a candidate record points at."""
    if not rec:
        return set()
    return {f for f in (rec.get("output"), rec.get("output_seg"),
                        (rec.get("aicvad") or {}).get("file")) if f}


def _settle(run_dir: Path, old: dict | None, new: dict, staged: dict[str, Path]) -> dict:
    """The record to keep for one candidate after an attempt, with its files in place.

    - Success: the staged files are moved into run_dir (os.replace, same
      filesystem, so each move is atomic) and the new record replaces the old.
      Files derived from the previous audio that the new record does not point
      at — a VAD-cut wav, flags the new result does not have — are removed, so
      no file is left whose attempt is ambiguous.
    - Failure where a good record already exists: the old record and its files
      stay exactly as they were, and the failed attempt is attached under
      `last_attempt` — neither silently current nor wiped.
    - Failure with nothing good before: the error record is the record.

    Staged files of a failed attempt are dropped with the attempt directory.
    Synchronous on purpose: the caller saves meta right after, with no await
    between, so the files and the record that references them change together.
    """
    if new.get("status") == "done":
        for name, path in staged.items():
            os.replace(path, run_dir / name)
        # derived from this candidate's previous audio: whatever the new record
        # does not reference is stale, referenced by the old record or not
        derived = {f"{new['id']}.aicvad.npy", f"{new['id']}.vadcut.wav"}
        for name in (_referenced(old) | derived) - _referenced(new):
            if "/" not in name and name != "input.wav":
                (run_dir / name).unlink(missing_ok=True)
        return new
    if old and old.get("status") == "done" and old.get("output"):
        attempt = {"status": new.get("status"), "error": new.get("error"),
                   "at": new.get("processed_at")}
        for key in ("rail", "chain", "cloud_replay"):
            if new.get(key) is not None:
                attempt[key] = new[key]
        return {**old, "last_attempt": attempt}
    return new


async def _process_run(
    run_id: str,
    candidate_ids: list[str],
    live_results: dict | None = None,
    script: str = "",
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    # live_results is None for uploads and reprocessing: nothing was recorded
    # live, so the cloud candidates go through the cloud replay rail instead
    live = live_results is not None
    live_results = live_results or {}
    candidate_ids = list(dict.fromkeys(candidate_ids))
    run_dir = store.run_dir(run_id)
    input_wav = run_dir / "input.wav"
    known = _candidates_by_id()
    # A snapshot, for reading only. Every WRITE below goes through
    # store.locked_meta, which re-reads meta.json under the run's lock and saves
    # atomically, and writes only what this pass produced — so a VAD-bench CLI,
    # a note or a re-transcription saved meanwhile is not overwritten.
    meta = store.load_meta(run_id)
    # MERGE by candidate id, never replace the list: a reprocess of a few
    # candidates must leave every other stored record exactly as it was — the
    # stored runs are the only copy of those results. Stored order is kept and
    # new ids follow in ticked order, so the cards read the same before and
    # after. A selected record is swapped only once its new result is in hand
    # (_settle), so an interrupted pass loses nothing either.
    stored_ids = [e["id"] for e in meta.get("candidates") or [] if e.get("id")]
    order = stored_ids + [c for c in candidate_ids if c not in stored_ids]
    # Recorded because it invalidates the latency columns: block_ms_p95 measured
    # while N chains share the CPU is not the same number as measured alone. The
    # run-level value is the latest pass; each new record carries its own.
    run_level: dict = {"script": script, "concurrency": concurrency}

    # The input's own analysis is what every stored record was scored against
    # (noise reduction is relative to input_scores' gaps), so a reprocess reuses
    # it: new records are then measured against the same reference as the kept
    # ones. It is computed only when missing or failed.
    input_scores = meta.get("input_scores")
    if not isinstance(input_scores, dict) or not input_scores or "error" in input_scores:
        try:
            input_scores = await asyncio.to_thread(scoring.score_input, input_wav)
        except Exception as e:
            input_scores = {"error": str(e)}
        run_level["input_scores"] = input_scores
    # No VAD runs here any more: spans come from the VAD bench, on demand, for
    # whichever VADs and sources are asked for. Runs recorded before keep the
    # spans the old per-run Silero pass stored (`input_vad`, `candidates[].vad`),
    # which the bench reads as `legacy-silero`.
    with store.locked_meta(run_id) as m:
        m.update(run_level)

    # Every new output is written into this attempt directory first and moved
    # into run_dir only when its candidate succeeded, together with its record
    # (_settle). A failed attempt therefore leaves the old wav, flags and record
    # exactly as they were, and nothing half-written behind.
    attempts = run_dir / f".attempt-{uuid.uuid4().hex[:8]}"
    attempts.mkdir()
    try:
        await _process_candidates(
            run_id, run_dir, input_wav, known, order, candidate_ids,
            live, live_results, script, concurrency, input_scores, attempts,
        )
    finally:
        shutil.rmtree(attempts, ignore_errors=True)


def _put(cands: list[dict], rec: dict, order: list[str]) -> None:
    """Place one record in the stored list by id: in place if it is there,
    otherwise just before the next id of `order` already present (else at the
    end). Records this pass knows nothing about stay where they are."""
    ids = [c.get("id") for c in cands]
    if rec["id"] in ids:
        cands[ids.index(rec["id"])] = rec
        return
    after = order[order.index(rec["id"]) + 1:] if rec["id"] in order else []
    pos = next((ids.index(c) for c in after if c in ids), len(cands))
    cands.insert(pos, rec)


async def _process_candidates(
    run_id: str, run_dir: Path, input_wav: Path, known: dict,
    order: list[str], candidate_ids: list[str], live: bool, live_results: dict,
    script: str, concurrency: int, input_scores: dict, attempts: Path,
) -> None:
    """Run the selected candidates and merge each result into the stored meta
    (see _process_run). Outputs are staged under `attempts` and promoted by
    _settle, under the run's meta lock."""
    sem = asyncio.Semaphore(concurrency)

    # Cloud candidates on a stored wav share ONE replay (a real-time pass per
    # batch, whatever the candidate count), and it runs ALONE, before any chain
    # starts. Its subscriber has to keep up with real time: sharing the machine
    # with the chains and their all-core DNSMOS scoring put it 5.4 s behind, and
    # the receiver's jitter buffer then time-stretched the audio in ~50 ms steps
    # (control envelope corr 0.79, vs 1.00 run alone). Running it first also keeps
    # its two processes out of the chains' latency and CPU columns.
    replay_ids = [] if live else [c for c in candidate_ids if known[c]["rail"] == "cloud"]
    replay_out: dict[str, dict] = {}
    replay_dir = attempts / "replay"
    replay_dir.mkdir()
    if replay_ids:
        async def on_progress(ev: dict) -> None:
            detail = f"batch {ev['batch']}/{ev['batches']} · {ev['t']:.0f}/{ev['total']:.0f} s"
            for c in ev.get("ids") or replay_ids:
                await broadcast({"type": "progress", "run_id": run_id, "candidate": c,
                                 "stage": "cloud_replay", "detail": detail})

        for c in replay_ids:
            await broadcast({"type": "progress", "run_id": run_id, "candidate": c,
                             "stage": "cloud_replay", "detail": "starting"})
        try:
            # written into the attempt dir, promoted per candidate by _settle
            replay_out.update(await cloud_replay.replay(
                input_wav,
                [{"id": c, "lk_model": known[c]["lk_model"],
                  "lk_options": known[c].get("lk_options") or {}} for c in replay_ids],
                replay_dir, on_progress,
            ))
        except Exception as e:  # replay() handles batch errors; this is the rest
            for c in replay_ids:
                replay_out.setdefault(c, {"error": f"cloud replay: {e}"})

    async def _one(cid: str) -> tuple[dict, dict[str, Path]]:
        """The candidate's new record, plus the files it staged: final name in
        run_dir -> staged path. Nothing is written into run_dir here."""
        cand = known[cid]
        entry = {"id": cid, "label": cand["label"]}
        staged: dict[str, Path] = {}
        stage = attempts / cid
        stage.mkdir(exist_ok=True)
        await broadcast({"type": "progress", "run_id": run_id, "candidate": cid, "stage": "nc"})
        try:
            if cand["rail"] == "cloud" and live:
                entry["rail"] = "live"
                entry["chain"] = [f"livekit-cloud:{cand['lk_model']} (live rail)"]
                if cand.get("lk_options"):
                    entry["lk_options"] = cand["lk_options"]
                entry["plugins"] = lk_cloud.plugin_versions()
                result = live_results.get(cid)
                if result is None:
                    raise RuntimeError("the live rail recorded nothing for this candidate")
                if "error" in result:
                    raise RuntimeError(result["error"])
                out_wav = stage / f"{cid}.wav"
                audio = result["audio"].astype(np.float32) / 32768.0
                audio = soxr.resample(audio, RECORD_RATE, config.PIPELINE_RATE)
                sf.write(out_wav, (np.clip(audio, -1, 1) * 32767).astype(np.int16),
                         config.PIPELINE_RATE)
                entry["output"] = out_wav.name
                staged[out_wav.name] = out_wav
                flags = result.get("vad_flags")
                if flags is not None and len(flags):
                    # the live output is not lead-in trimmed, so frame 0 is t=0
                    # of <cid>.wav itself (see cloud_replay._aicvad for the format)
                    flags = np.asarray(flags, np.int8)  # -1 = frame had no flag
                    np.save(stage / f"{cid}.aicvad.npy", flags)
                    staged[f"{cid}.aicvad.npy"] = stage / f"{cid}.aicvad.npy"
                    entry["aicvad"] = {
                        "file": f"{cid}.aicvad.npy", "dtype": "int8",
                        "encoding": lk_cloud.AICVAD_ENCODING, "frame_s": result["frame_s"],
                        "t0_s": 0.0, "frames": int(len(flags)),
                        "speech_frames": int((flags == 1).sum()),
                        "missing_frames": int((flags == -1).sum()),
                        "source": f"{lk_cloud.AIC_VAD_KEY} per-frame enhancer flag (live rail)",
                    }
                    entry["aicvad_missing_frames"] = entry["aicvad"]["missing_frames"]
            elif cand["rail"] == "cloud":
                entry["rail"] = "cloud_replay"
                entry["chain"] = [f"livekit-cloud:{cand['lk_model']} (cloud replay)"]
                if cand.get("lk_options"):
                    entry["lk_options"] = cand["lk_options"]
                result = replay_out.get(cid) or {"error": "the cloud replay returned nothing"}
                if "error" in result:
                    if result.get("cloud_replay"):
                        entry["cloud_replay"] = result["cloud_replay"]
                    raise RuntimeError(result["error"])
                entry.update(result)
                out_wav = replay_dir / result["output"]
                staged[result["output"]] = out_wav
                if result.get("aicvad"):
                    staged[result["aicvad"]["file"]] = replay_dir / result["aicvad"]["file"]
            else:
                entry["chain_spec"] = cand["chain"]
                if not cand["available"]:
                    raise RuntimeError(cand["unavailable_reason"])
                out_wav = stage / f"{cid}.wav"
                # RSS is process-wide; at concurrency > 1 a neighbour's model
                # loading at the same moment lands in this chain's delta, so ask
                # for it only when this is the only chain being built
                timing = await asyncio.to_thread(
                    run_chain, input_wav, cand["chain"], out_wav, concurrency == 1
                )
                entry.update(timing)
                entry["output"] = out_wav.name
                staged[out_wav.name] = out_wav
            # validate before anything reads it: a rail that reports success
            # must have produced every file its record will point at
            missing = [name for name, path in staged.items() if not path.is_file()]
            if missing:
                raise RuntimeError(f"no output was written for {missing}")
            # STT and scores read the STAGED file; the record they end up in is
            # promoted together with it
            await broadcast({"type": "progress", "run_id": run_id, "candidate": cid, "stage": "stt"})
            try:
                entry["stt"] = await stt.transcribe(out_wav)
            except Exception as e:
                entry["stt_error"] = str(e)
            try:
                entry["scores"] = await asyncio.to_thread(
                    scoring.score_output, out_wav, input_scores,
                    script, (entry.get("stt") or {}).get("text", ""),
                )
            except Exception as e:
                entry["scores_error"] = str(e)
            entry["status"] = "done"
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
        entry["processed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry["concurrency"] = concurrency
        return entry, staged

    async def _slot(cid: str) -> None:
        async with sem:
            entry, staged = await _one(cid)
        # Under the run's lock: re-read meta, take the CURRENT stored record as
        # the old one (a re-transcription saved meanwhile counts), promote the
        # files, put the record in place, save. Milliseconds, no await inside —
        # the files and the record that points at them change together.
        with store.locked_meta(run_id) as m:
            cands = m.setdefault("candidates", [])
            old = next((c for c in cands if c.get("id") == cid), None)
            rec = _settle(run_dir, old, entry, staged)
            _put(cands, rec, order)
        await broadcast(
            {"type": "progress", "run_id": run_id, "candidate": cid,
             "stage": rec["status"], "entry": rec}
        )

    await asyncio.gather(*(_slot(cid) for cid in candidate_ids))

    # A re-processed candidate's VAD-bench rows go stale on their own (its audio
    # fingerprint changed); nothing else here reads spans. The records above were
    # scored against the reference as it was when this pass started; references
    # saved since are applied now.
    with store.locked_meta(run_id) as m:
        marks.sync_reference(m)
        m["status"] = "done"
    await broadcast({"type": "run_done", "run_id": run_id})


# ---------------------------------------------------------------- history


@app.get("/api/runs")
async def api_runs():
    return store.list_runs()


@app.get("/api/runs/{run_id}")
async def api_run(run_id: str):
    try:
        meta = store.load_meta(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, "run not found")
    # every mark with an id, also on runs marked before marks had one (derived
    # from the bounds, so the page and /truth name them the same way)
    meta["truth_marks"] = marks.ensure(meta)
    meta["reference_state"] = marks.reference_state(meta)
    return meta


_reprocessing: set[str] = set()


@app.post("/api/runs/{run_id}/reprocess")
async def reprocess_run(run_id: str, body: dict | None = None):
    """Re-run a finished run's stored input.wav through a different candidate set.

    Exists because the candidate ticks are chosen before the call and the
    interesting ones are obvious only after hearing it — forgetting to tick
    something used to mean the recording was wasted. Nothing in _process_run needs
    a live session; it works entirely off run_dir/input.wav. Cloud candidates get
    there through the cloud replay rail (cloud_replay.py), which republishes
    input.wav into a fresh Cloud room — a second Opus pass the live rail does not
    have, which is what the `cloud-raw` control measures.

    Keeps the run's id, source, note, hand marks and their references, which is
    the whole advantage over downloading input.wav and re-uploading it: a new
    upload run would start with none of them.
    """
    body = body or {}
    try:
        meta = store.load_meta(run_id)
        run_dir = store.run_dir(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None
    if not (run_dir / "input.wav").is_file():
        raise HTTPException(409, f"{run_id} has no input.wav to reprocess")
    if _session is not None:
        raise HTTPException(409, "a live session is active; stop it first")
    # Two overlapping _process_run calls on ONE run would interleave their writes
    # to the same meta.json — the upload path cannot collide because every upload
    # is a fresh id, but two clicks of this button can.
    if run_id in _reprocessing:
        raise HTTPException(409, f"{run_id} is already reprocessing")

    candidate_ids = body.get("candidates") or []
    if not candidate_ids:
        raise HTTPException(400, "tick at least one candidate")
    known = _candidates_by_id()
    unknown = [c for c in candidate_ids if c not in known]
    if unknown:
        raise HTTPException(400, f"unknown candidates: {unknown}")
    # cloud candidates replay input.wav through a fresh Cloud room (cloud_replay);
    # refuse the ones that cannot run here instead of erroring downstream
    _check_runnable(candidate_ids, known)

    # script defaults to the one already on the run — _process_run overwrites the
    # key, so not passing it through would silently drop every WER
    script = body.get("script")
    script = (meta.get("script") or "") if script is None else str(script).strip()

    async def _run() -> None:
        try:
            await _process_run(run_id, candidate_ids, None, script,
                               _clamp_concurrency(body.get("concurrency")))
        finally:
            _reprocessing.discard(run_id)

    _reprocessing.add(run_id)
    asyncio.create_task(_run())
    return {"run_id": run_id, "status": "processing", "candidates": candidate_ids}


def _merge_spans(spans, limit: float) -> list[list[float]]:
    """Sorted, clamped, non-overlapping.

    Hand-drawn regions arrive in paint order and overlap freely (a region can be
    redrawn over one that was slightly off). Storing them raw would leave any later
    interval arithmetic dependent on the order they were painted in.
    """
    clean: list[list[float]] = []
    for pair in spans or []:
        try:
            a, b = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        lo, hi = max(0.0, min(lo, limit)), max(0.0, min(hi, limit))
        if hi - lo >= 0.02:  # a 20 ms drag is a mis-click, not a region
            clean.append([lo, hi])
    clean.sort()
    out: list[list[float]] = []
    for lo, hi in clean:
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [[round(a, 3), round(b, 3)] for a, b in out]


@app.post("/api/runs/{run_id}/truth")
async def set_truth(run_id: str, body: dict):
    """Hand-marked speech regions for one run: the ground truth every candidate's
    VAD spans are scored against.

    Marked once on the raw input and reused for every candidate, deliberately —
    the question is "did this chain's audio still let the VAD find the speech",
    which needs a reference no chain can influence.

    Only the regions are stored. Scoring is interval arithmetic the page does
    against VAD spans already in meta, so re-marking re-scores every candidate
    instantly without re-reading a single wav.

    Two more label classes are optional and additive (a body without them
    leaves what the run already has):
      known_nonspeech  [{"name", "start", "end"}]  sounds known NOT to be speech
      ignore_spans     [[start, end], ...]         sounds whose label is uncertain;
                                                   scored as neither speech nor silence
    The three classes must not overlap; a body that would make them overlap
    (including new speech marks over an existing ignore span) is refused with
    400 and nothing is saved.

    Speech comes as `marks` [{id?, start, end}] (the page: an id keeps that
    mark and its reference, a region without one is new) or, from older
    callers, `spans` [[start, end], ...]. Optional `refs` {mark id: text, or
    null for expected empty} are saved first, so text typed before a re-mark
    survives it. Overlapping
    regions merge and keep every reference they swallowed (nc_bench/marks.py).
    The run's reference script and every WER that reads it follow.
    """
    from .vads.events import label_overlaps

    def _save() -> dict:
        with store.locked_meta(run_id) as meta:
            # clamp to the recording, so a drag past the right edge cannot invent speech
            limit = float((meta.get("input") or {}).get("duration_s") or 0) or 1e9
            if "marks" in body or "spans" in body:
                marks.apply_spans(meta, body.get("marks") if "marks" in body else body.get("spans"),
                                  limit, body.get("refs"))
            elif body.get("refs"):
                marks.apply_refs(meta, body["refs"])
            truth = meta.get("truth_spans") or []
            ignore = (_merge_spans(body.get("ignore_spans"), limit) if "ignore_spans" in body
                      else (meta.get("ignore_spans") or []))
            kns = (_clean_nonspeech(body.get("known_nonspeech"), limit) if "known_nonspeech" in body
                   else (meta.get("known_nonspeech") or []))
            clash = label_overlaps(truth, kns, ignore)
            if clash:
                # raised inside the lock: locked_meta saves nothing on an exception
                raise HTTPException(400, "label classes overlap: " + "; ".join(clash[:5])
                                    + (f" (+{len(clash) - 5} more)" if len(clash) > 5 else ""))
            if ignore or "ignore_spans" in meta:
                meta["ignore_spans"] = ignore
            if kns or "known_nonspeech" in meta:
                meta["known_nonspeech"] = kns
            # marks changed: the VAD bench's scores (and the old per-run spans'
            # truth scores) are re-scored against them, then the reference
            vad.score_run(meta)
            ref = marks.sync_reference(meta)
            return {
                "run_id": run_id,
                "marks": marks.ensure(meta),
                "spans": truth,
                "n": len(truth),
                "speech_s": round(sum(b - a for a, b in truth), 2),
                "ignore_spans": meta.get("ignore_spans") or [],
                "known_nonspeech": meta.get("known_nonspeech") or [],
                "reference": ref,
            }

    try:
        return await asyncio.to_thread(_save)
    except marks.UnknownMarks as e:
        raise HTTPException(409, f"marks {e.ids} are no longer on this run; reload it") from None
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None


@app.post("/api/runs/{run_id}/refs")
async def set_refs(run_id: str, body: dict):
    """What was actually said inside each mark: {refs: {mark id: text}}. null
    means expected empty (the STT should return nothing there), "" means no
    reference yet.

    The run's reference script is these joined in time order once every mark has
    one (marks.sync_reference), and every WER is re-scored from what is already
    stored: whole-file from each candidate's transcript, VAD-cut from the bench's
    per-cut texts. No STT call, so saving is instant whatever the run's length.
    """
    refs = body.get("refs")
    if not isinstance(refs, dict) or not refs:
        raise HTTPException(400, "refs must be {mark id: text, or null for expected empty}")

    def _save() -> dict:
        with store.locked_meta(run_id) as meta:
            marks.apply_refs(meta, refs)
            ref = marks.sync_reference(meta)
            return {"run_id": run_id, "marks": meta["truth_marks"], "reference": ref,
                    "script": meta.get("script") or ""}

    try:
        return await asyncio.to_thread(_save)
    except marks.UnknownMarks as e:
        raise HTTPException(409, f"marks {e.ids} are no longer on this run (re-marked "
                                 "meanwhile?); reload it, the text was not saved") from None
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None


def _clean_nonspeech(items, limit: float) -> list[dict]:
    """known_nonspeech entries, validated: a name and a non-empty window inside
    the recording. Anything malformed is refused, not silently dropped: these
    are the incident sounds a ranking is judged on."""
    out = []
    for i, w in enumerate(items or []):
        try:
            name = str(w["name"]).strip()
            a, b = sorted((float(w["start"]), float(w["end"])))
        except (TypeError, KeyError, ValueError):
            raise HTTPException(400, f"known_nonspeech[{i}] needs name, start and end") from None
        a, b = max(0.0, min(a, limit)), max(0.0, min(b, limit))
        if not name or b - a <= 0:
            raise HTTPException(400, f"known_nonspeech[{i}] ({name or 'no name'}) is empty")
        out.append({"name": name, "start": round(a, 3), "end": round(b, 3)})
    names = [w["name"] for w in out]
    if len(set(names)) != len(names):
        raise HTTPException(400, "known_nonspeech names must be unique")
    return sorted(out, key=lambda w: w["start"])


# -------------------------------------------------------------- VAD bench
# Many VAD models × this run's audio sources, through one shared segmenter.
# Everything is in nc_bench/vads/; these endpoints only validate, guard and
# broadcast. Results land in meta["vad_bench"] and nowhere else: the spans the
# old per-run Silero pass recorded (`vad`, `input_vad`) are only ever read, as
# the `legacy-silero` rows.

from .vads import bench as vad_bench  # noqa: E402
from .vads import registry as vad_registry  # noqa: E402


@app.get("/api/vads")
async def api_vads():
    """The VAD registry, expanded per rate, with availability — the VAD
    counterpart of /api/candidates."""
    return {"vads": vad_registry(), "params": vad_bench.PARAM_KEYS,
            "limits": vad.LIMITS, "stt_modes": list(vad_bench.STT_MODES),
            "stt_concurrency": vad_bench.STT_CONCURRENCY}


@app.get("/api/runs/{run_id}/vad_bench")
async def vad_bench_view(run_id: str):
    """meta["vad_bench"] checked against the audio on disk NOW: a row whose
    source wav (or flag file) changed since it was computed, or a transcript
    whose read audio or cut list changed, comes back `stale: true`. The page
    renders this rather than the stored copy, so a re-processed candidate can
    never keep ranking on old evidence. Read-only."""
    try:
        return await asyncio.to_thread(vad_bench.fresh_view, run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None


def _vad_bench_guard(run_id: str) -> dict:
    try:
        meta = store.load_meta(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None
    if _session is not None:
        raise HTTPException(409, "a live session is active; stop it first")
    # shared with reprocess: both rewrite this run's meta.json
    if run_id in _reprocessing:
        raise HTTPException(409, f"{run_id} is already busy")
    return meta


@app.post("/api/runs/{run_id}/vad_bench")
async def vad_bench_run(run_id: str, body: dict | None = None):
    """Run the ticked VADs over the chosen sources of one stored run.

    Body: {vads: [...] | "all", sources: [...] | "ticked" | "all",
    ticked: [NC candidate ids], params: {...}, thresholds: {vad: thr},
    force: bool, resegment_only: bool}. Probabilities already stored for a
    (source, VAD) are reused unless `force`; `resegment_only` never runs a
    model, so a parameter change costs milliseconds. Progress goes out on the
    websocket as {"type": "vad_bench", ...}.
    """
    body = body or {}
    meta = _vad_bench_guard(run_id)
    try:
        sids = vad_bench.resolve_sources(meta, body.get("sources") or "ticked", body.get("ticked"))
        entries = vad_bench.resolve_vads(body.get("vads") or "all")
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if not sids or not entries:
        raise HTTPException(400, "tick at least one VAD and one audio source")
    loop = asyncio.get_running_loop()

    def progress(ev: dict) -> None:   # called from the worker thread
        asyncio.run_coroutine_threadsafe(broadcast(ev), loop)

    async def _run() -> None:
        try:
            log = await asyncio.to_thread(
                vad_bench.run_bench, run_id, vads=[e["id"] for e in entries], sources=sids,
                params=body.get("params"), thresholds=body.get("thresholds"),
                force=bool(body.get("force")), resegment_only=bool(body.get("resegment_only")),
                progress=progress)
            await broadcast({"type": "vad_bench", "run_id": run_id, "finished": True, **log})
        except Exception as e:  # noqa: BLE001 - surfaced to the page, not swallowed
            await broadcast({"type": "vad_bench", "run_id": run_id, "finished": True,
                             "error": str(e)})
        finally:
            _reprocessing.discard(run_id)

    _reprocessing.add(run_id)
    asyncio.create_task(_run())
    return {"run_id": run_id, "status": "processing", "sources": sids,
            "vads": [e["id"] for e in entries], "pairs": len(sids) * len(entries)}


@app.post("/api/runs/{run_id}/vad_bench/stt")
async def vad_bench_stt(run_id: str, body: dict | None = None):
    """VAD-cut WER per (source, VAD): one STT request per turn, read from the
    same audio (`same`) or from raw (`raw`, production's split gate). A button,
    never automatic: it is one request per turn per pair per mode.

    Body: {vads, sources, modes: ["same", "raw"], force}; omitted = everything
    with stored segments. Cuts already transcribed with the same bounds on the
    same audio are reused (vad_bench.cut_cache) unless `force`.
    """
    body = body or {}
    meta = _vad_bench_guard(run_id)
    if not (meta.get("vad_bench") or {}).get("results"):
        raise HTTPException(409, "run the VAD bench on this run first")
    try:
        if body.get("sources"):
            vad_bench.resolve_sources(meta, body["sources"])
        if body.get("vads"):
            vad_bench.resolve_vads(body["vads"])
    except ValueError as e:
        raise HTTPException(400, str(e)) from None

    async def _run() -> None:
        try:
            log = await vad_bench.run_stt(run_id, vads=body.get("vads"),
                                          sources=body.get("sources"),
                                          modes=body.get("modes") or list(vad_bench.STT_MODES),
                                          progress=broadcast, force=bool(body.get("force")))
            await broadcast({"type": "vad_bench", "run_id": run_id, "finished": True, **log})
        except Exception as e:  # noqa: BLE001
            await broadcast({"type": "vad_bench", "run_id": run_id, "finished": True,
                             "error": str(e)})
        finally:
            _reprocessing.discard(run_id)

    _reprocessing.add(run_id)
    asyncio.create_task(_run())
    return {"run_id": run_id, "status": "processing"}


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Remove a run and every wav in it.

    Refuses the live one: the recorder still holds that directory and would go on
    writing input.wav into a path that no longer exists.
    """
    if _session is not None and _session["run_id"] == run_id:
        raise HTTPException(409, "that run is recording right now — stop the session first")
    try:
        d = store.run_dir(run_id)  # also rejects anything outside RUNS_DIR
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None
    await asyncio.to_thread(shutil.rmtree, d)
    return {"deleted": run_id}


@app.get("/api/export")
async def export_all(audio: str = "m4a"):
    """Every run as one zip: a standalone viewer HTML plus all the audio.

    Built to a temp file and handed to the browser's own download manager rather
    than streamed or held in memory — with wav audio this is hundreds of MB.
    Progress goes out on the existing websocket, since the build can take a
    couple of minutes and a bare spinner would look hung.
    """
    if audio not in export.AUDIO_FORMATS:
        raise HTTPException(400, f"audio must be one of {list(export.AUDIO_FORMATS)}")
    tmpdir = Path(tempfile.mkdtemp(prefix="nc-export-"))
    dest = tmpdir / f"nc-bench-{time.strftime('%Y%m%d-%H%M%S')}.zip"

    async def progress(done: int, total: int) -> None:
        await broadcast({"type": "export", "done": done, "total": total})

    try:
        stats = await export.build_zip(dest, audio, progress)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    await broadcast({"type": "export", "finished": True, **stats})
    return FileResponse(
        dest, filename=dest.name, media_type="application/zip",
        background=BackgroundTask(shutil.rmtree, tmpdir, ignore_errors=True),
    )


@app.get("/api/runs/{run_id}/files/{name}")
async def api_run_file(run_id: str, name: str):
    try:
        d = store.run_dir(run_id)
    except KeyError:
        raise HTTPException(404, "run not found")
    f = (d / name).resolve()
    if not f.is_relative_to(d) or not f.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(f)


# ------------------------------------------------------------------ page


@app.get("/")
async def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.exception_handler(Exception)
async def unhandled(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
