"""FastAPI app: session control, upload, candidate registry, run history,
and a websocket that streams recorder levels + processing progress to the UI."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from . import config, export, lk_cloud, scoring, store, stt, vad
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
        if "lk_model" in entry:  # cloud live-rail candidate (Krisp / ai-coustics)
            ok, why = lk_cloud.available(entry["lk_model"])
            entry["live_only"] = True
        else:
            ok, why = chain_available(entry["chain"])
            entry["live_only"] = False
        entry["available"] = ok
        entry["unavailable_reason"] = why
    return entries


def _candidates_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in _load_candidates()}


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
        "vad_enabled": config.VAD_ENABLED,
        # .env + the per-source rate seed the UI; the UI can then diverge
        "vad_defaults": vad.defaults(),
        "vad_limits": vad.LIMITS,
        "vad_rates": list(vad.SUPPORTED_RATES),
        "vad_source_rates": vad.SOURCE_RATES,
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

    live_cands = [
        {"id": cid, "lk_model": known[cid]["lk_model"]}
        for cid in candidate_ids
        if known[cid].get("live_only")
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
    meta = store.load_meta(sess["run_id"])
    meta["input"] = input_meta

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
        meta["status"] = "empty"
        meta["error"] = detail
        store.save_meta(sess["run_id"], meta)
        await broadcast({"type": "session", "state": "idle"})
        return {"run_id": sess["run_id"], "status": "empty", "detail": detail}

    meta["status"] = "processing"
    store.save_meta(sess["run_id"], meta)
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
        meta = store.load_meta(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None
    meta["note"] = (body.get("note") or "").strip()
    store.save_meta(run_id, meta)
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
        entry["stt"] = await stt.transcribe(out_wav)
        entry.pop("stt_error", None)
    except Exception as e:
        entry.pop("stt", None)
        entry["stt_error"] = str(e)

    script = (meta.get("script") or "").strip()
    if script and isinstance(entry.get("scores"), dict):
        # WER is a function of the transcript, so it has to move with it
        entry["scores"]["wer"] = scoring.wer(
            script, (entry.get("stt") or {}).get("text", "")
        )
    store.save_meta(run_id, meta)
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
        meta = store.load_meta(run_id)
        meta["status"] = "error"
        meta["error"] = "ffmpeg could not decode this file"
        store.save_meta(run_id, meta)
        raise HTTPException(400, f"could not decode audio: {proc.stderr.decode()[-300:]}")

    import soundfile as sf

    info = sf.info(input_wav)
    meta = store.load_meta(run_id)
    meta["input"] = {
        "file": "input.wav",
        "duration_s": round(info.duration, 2),
        "sample_rate": info.samplerate,
        "original_name": file.filename,
    }
    meta["status"] = "processing"
    store.save_meta(run_id, meta)
    asyncio.create_task(_process_run(run_id, candidate_ids, None, script.strip(),
                                     _clamp_concurrency(concurrency)))
    return {"run_id": run_id, "status": "processing"}


# ------------------------------------------------------------- processing


async def _process_run(
    run_id: str,
    candidate_ids: list[str],
    live_results: dict | None = None,
    script: str = "",
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    live_results = live_results or {}
    run_dir = store.run_dir(run_id)
    input_wav = run_dir / "input.wav"
    known = _candidates_by_id()
    meta = store.load_meta(run_id)
    meta["candidates"] = []
    meta["script"] = script
    # Recorded because it invalidates the latency columns: block_ms_p95 measured
    # while N chains share the CPU is not the same number as measured alone.
    meta["concurrency"] = concurrency

    try:
        input_scores = await asyncio.to_thread(scoring.score_input, input_wav)
    except Exception as e:
        input_scores = {"error": str(e)}
    meta["input_scores"] = input_scores
    # The raw leg's spans are the reference every candidate is read against: a
    # chain that drops a span dropped a turn a live deployment would transcribe.
    # Rate follows the source (a phone leg is narrowband whatever it arrives as),
    # but a reprocess keeps whatever the run was already measured at — resetting to
    # defaults would silently make the new candidates incomparable with the marks
    # and spans already stored on this run.
    vad_params = meta.get("vad_params") or vad.defaults(meta.get("source"))
    meta["vad_params"] = vad_params
    try:
        meta["input_vad"] = await vad.analyze(input_wav, vad_params)
    except Exception as e:
        meta["input_vad"] = {"error": str(e)}
    store.save_meta(run_id, meta)

    async def _one(cid: str) -> dict:
        cand = known[cid]
        entry = {"id": cid, "label": cand["label"]}
        await broadcast({"type": "progress", "run_id": run_id, "candidate": cid, "stage": "nc"})
        try:
            if cand.get("live_only"):
                entry["chain"] = [f"livekit-cloud:{cand['lk_model']} (live rail)"]
                result = live_results.get(cid)
                if result is None:
                    raise RuntimeError(
                        "live-rail candidate: only available on phone/web sessions, "
                        "not uploads or reprocessing"
                    )
                if "error" in result:
                    raise RuntimeError(result["error"])
                out_wav = run_dir / f"{cid}.wav"
                audio = result["audio"].astype(np.float32) / 32768.0
                audio = soxr.resample(audio, RECORD_RATE, config.PIPELINE_RATE)
                sf.write(out_wav, (np.clip(audio, -1, 1) * 32767).astype(np.int16),
                         config.PIPELINE_RATE)
                entry["output"] = out_wav.name
            else:
                entry["chain_spec"] = cand["chain"]
                if not cand["available"]:
                    raise RuntimeError(cand["unavailable_reason"])
                out_wav = run_dir / f"{cid}.wav"
                # RSS is process-wide; at concurrency > 1 a neighbour's model
                # loading at the same moment lands in this chain's delta, so ask
                # for it only when this is the only chain being built
                timing = await asyncio.to_thread(
                    run_chain, input_wav, cand["chain"], out_wav, concurrency == 1
                )
                entry.update(timing)
                entry["output"] = out_wav.name
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
            # Not in score_output: that runs in a worker thread and the VAD needs
            # the event loop (the plugin's stream is async).
            try:
                entry["vad"] = await vad.analyze(out_wav, vad_params)
            except Exception as e:
                entry["vad_error"] = str(e)
            entry["status"] = "done"
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
        return entry

    # Slot per candidate so meta stays in ticked order however completion
    # interleaves — otherwise the same candidate set lands in a different order
    # every run and two runs stop being readable side by side.
    entries: list[dict | None] = [None] * len(candidate_ids)
    sem = asyncio.Semaphore(concurrency)

    async def _slot(i: int, cid: str) -> None:
        async with sem:
            entry = await _one(cid)
        entries[i] = entry
        # No await between mutating and saving, so concurrent slots cannot
        # interleave a half-written meta.json.
        meta["candidates"] = [e for e in entries if e is not None]
        store.save_meta(run_id, meta)
        await broadcast(
            {"type": "progress", "run_id": run_id, "candidate": cid,
             "stage": entry["status"], "entry": entry}
        )

    await asyncio.gather(*(_slot(i, cid) for i, cid in enumerate(candidate_ids)))

    meta["candidates"] = [e for e in entries if e is not None]
    vad.score_run(meta)
    meta["status"] = "done"
    store.save_meta(run_id, meta)
    await broadcast({"type": "run_done", "run_id": run_id})


# ---------------------------------------------------------------- history


@app.get("/api/runs")
async def api_runs():
    return store.list_runs()


@app.get("/api/runs/{run_id}")
async def api_run(run_id: str):
    try:
        return store.load_meta(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, "run not found")


_reprocessing: set[str] = set()


@app.post("/api/runs/{run_id}/stt_segmented")
async def stt_segmented(run_id: str):
    """Re-transcribe every candidate the way a live deployment would hear it: cut the
    output audio at that candidate's own VAD spans, one STT request per turn,
    transcripts joined in time order, WER recomputed.

    The whole-file transcript the run was scored with is a bench artifact —
    production never posts two minutes of audio in one request, and long files
    push the autoregressive STT into repetition collapse. This mode makes the
    VAD's misses REAL: speech the VAD failed to find is simply never transcribed,
    exactly as a live deployment would, so the WER and missed-speech metric finally
    describe the same pipeline.

    Cuts use each candidate's OWN spans (production runs VAD on the NC-processed
    track, not on the raw leg), extended by the run's prefix padding — see
    stt.transcribe_turns. Candidates without measured spans are skipped, not
    silently transcribed whole-file: mixing modes inside a run would corrupt the
    within-run ranking.

    Stored ALONGSIDE the whole-file results, never over them: `stt_seg`,
    `scores.wer_seg`, and `output_seg` (the concatenated cut audio, so what the
    STT actually heard is listenable). The UI toggles between the two WERs.
    """
    try:
        meta = store.load_meta(run_id)
        run_dir = store.run_dir(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None
    if _session is not None:
        raise HTTPException(409, "a live session is active; stop it first")
    if run_id in _reprocessing:
        raise HTTPException(409, f"{run_id} is already busy")
    todo = [c for c in meta.get("candidates") or []
            if c.get("output") and (c.get("vad") or {}).get("segments") is not None]
    if not todo:
        raise HTTPException(409, "no candidate has VAD spans — press 're-run VAD "
                                 "on this run' first")
    pad = float((meta.get("vad_params") or {}).get("prefix_padding_duration", 0.5))
    script = (meta.get("script") or "").strip()

    async def _run() -> None:
        try:
            done = skipped = 0
            for c in meta.get("candidates") or []:
                if not c.get("output") or (c.get("vad") or {}).get("segments") is None:
                    skipped += 1
                    continue
                wav = run_dir / c["output"]
                cut = run_dir / f"{c['id']}.vadcut.wav"
                try:
                    c["stt_seg"] = await stt.transcribe_turns(
                        wav, c["vad"]["segments"], pad, cut_wav=cut)
                    c["output_seg"] = cut.name
                    c.pop("stt_seg_error", None)
                    if script and isinstance(c.get("scores"), dict):
                        c["scores"]["wer_seg"] = scoring.wer(script, c["stt_seg"]["text"])
                except Exception as e:
                    c["stt_seg_error"] = f"segmented stt: {e}"
                done += 1
                await broadcast({"type": "progress", "run_id": run_id,
                                 "candidate": c["id"], "stage": "done", "entry": c})
            store.save_meta(run_id, meta)
            await broadcast({"type": "run_done", "run_id": run_id})
        finally:
            _reprocessing.discard(run_id)

    _reprocessing.add(run_id)
    asyncio.create_task(_run())
    return {"run_id": run_id, "status": "processing", "candidates": len(todo)}


@app.post("/api/runs/{run_id}/reprocess")
async def reprocess_run(run_id: str, body: dict | None = None):
    """Re-run a finished run's stored input.wav through a different candidate set.

    Exists because the candidate ticks are chosen before the call and the
    interesting ones are obvious only after hearing it — forgetting to tick
    something used to mean the recording was wasted. Nothing in _process_run needs
    a live session; it works entirely off run_dir/input.wav.

    Keeps the run's id, source, script, note and hand marks, which is the whole
    advantage over downloading input.wav and re-uploading it: the same id keeps the
    VAD rate that the source implies, and a new upload run would be measured at
    the upload default instead.
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
    # live-rail candidates have no audio to replay: they only exist as a filter on
    # a live LiveKit track, so say so here instead of erroring 6 times downstream
    live_only = [c for c in candidate_ids if known[c].get("live_only")]
    if live_only:
        raise HTTPException(
            400,
            f"these only run on a live phone/web session, not a stored wav: {live_only}",
        )

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
    """
    try:
        meta = store.load_meta(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None
    # clamp to the recording, so a drag past the right edge cannot invent speech
    limit = float((meta.get("input") or {}).get("duration_s") or 0) or 1e9
    meta["truth_spans"] = _merge_spans(body.get("spans"), limit)
    vad.score_run(meta)   # marks changed: every candidate's score is now stale
    store.save_meta(run_id, meta)
    return {
        "run_id": run_id,
        "spans": meta["truth_spans"],
        "n": len(meta["truth_spans"]),
        "speech_s": round(sum(b - a for a, b in meta["truth_spans"]), 2),
    }


@app.post("/api/runs/{run_id}/vad")
async def rerun_vad(run_id: str, body: dict | None = None):
    """Re-analyze one run's input and every candidate output at the given
    thresholds.

    Only `vad` keys are written — no NC chain re-runs, no STT — so this is cheap
    and cannot move any other number in a stored run. Also the backfill path: a
    run recorded before VAD existed gets its spans the first time this is called.

    The thresholds are saved as meta["vad_params"] and shown above the run's
    results. That is what makes per-run re-runs safe: spans are only comparable
    across runs when the thresholds match, so each run has to carry the ones that
    produced it rather than leaving the history silently mixed.
    """
    try:
        meta = store.load_meta(run_id)
        run_dir = store.run_dir(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"unknown run: {run_id}") from None
    # unspecified keys fall back to this run's SOURCE default, not a global one
    p = vad.clean((body or {}).get("params"), meta.get("source"))

    names = []
    if (meta.get("input") or {}).get("file"):
        names.append(meta["input"]["file"])
    names += [c["output"] for c in meta.get("candidates") or [] if c.get("output")]
    names = list(dict.fromkeys(names))

    done = failed = 0
    for name in names:
        wav = run_dir / name
        result = None
        if wav.is_file():
            try:
                result = await vad.analyze(wav, p)
            except Exception as e:
                result = {"error": str(e)}
                failed += 1
        if (meta.get("input") or {}).get("file") == name:
            meta["input_vad"] = result
        for c in meta.get("candidates") or []:
            if c.get("output") == name:
                c["vad"] = result
                c.pop("vad_error", None)
        done += 1
        if done % 5 == 0 or done == len(names):
            await broadcast({"type": "vad", "run_id": run_id, "done": done,
                             "total": len(names)})

    meta["vad_params"] = p
    vad.score_run(meta)   # spans changed: rescore against the existing marks
    store.save_meta(run_id, meta)
    stats = {"run_id": run_id, "files": done, "failed": failed, "params": p}
    await broadcast({"type": "vad", "finished": True, **stats})
    return stats


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
