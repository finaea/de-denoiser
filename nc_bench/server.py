"""FastAPI app: session control, upload, candidate registry, run history,
and a websocket that streams recorder levels + processing progress to the UI."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import config, lk_cloud, scoring, store, stt
from .pipeline import run_chain
from .processors import chain_available
from .recorder import RECORD_RATE, Recorder

app = FastAPI(title="NC Bench")

lk_cloud.preload()  # cloud NC plugins must register on the main thread

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
    }


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
        run_id, run_dir = store.new_run(source)
        recorder = Recorder(emit=broadcast, live_candidates=live_cands)
        _session = {
            "run_id": run_id,
            "run_dir": run_dir,
            "recorder": recorder,
            "candidates": candidate_ids,
            "script": (body.get("script") or "").strip(),
        }

    try:
        if source == "web":
            join = await recorder.start_web()
            await broadcast({"type": "session", "state": "web_ready", "room": join["room"]})
            return {"run_id": run_id, **join}
        await recorder.start_phone()
        return {"run_id": run_id}
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

    if input_meta["file"] is None:
        meta["status"] = "empty"
        store.save_meta(sess["run_id"], meta)
        await broadcast({"type": "session", "state": "idle"})
        return {"run_id": sess["run_id"], "status": "empty",
                "detail": "no audio was recorded (no call arrived / no mic frames)"}

    meta["status"] = "processing"
    store.save_meta(sess["run_id"], meta)
    asyncio.create_task(
        _process_run(sess["run_id"], sess["candidates"], live_results, sess["script"])
    )
    return {"run_id": sess["run_id"], "status": "processing"}


# ----------------------------------------------------------------- upload


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...), candidates: str = Form(...), script: str = Form("")
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

    run_id, run_dir = store.new_run("upload")
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
    asyncio.create_task(_process_run(run_id, candidate_ids, None, script.strip()))
    return {"run_id": run_id, "status": "processing"}


# ------------------------------------------------------------- processing


async def _process_run(
    run_id: str,
    candidate_ids: list[str],
    live_results: dict | None = None,
    script: str = "",
) -> None:
    live_results = live_results or {}
    run_dir = store.run_dir(run_id)
    input_wav = run_dir / "input.wav"
    known = _candidates_by_id()
    meta = store.load_meta(run_id)
    meta["candidates"] = []
    meta["script"] = script

    try:
        input_scores = await asyncio.to_thread(scoring.score_input, input_wav)
    except Exception as e:
        input_scores = {"error": str(e)}
    meta["input_scores"] = input_scores
    store.save_meta(run_id, meta)

    for cid in candidate_ids:
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
                timing = await asyncio.to_thread(run_chain, input_wav, cand["chain"], out_wav)
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
            entry["status"] = "done"
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
        meta["candidates"].append(entry)
        store.save_meta(run_id, meta)
        await broadcast(
            {"type": "progress", "run_id": run_id, "candidate": cid,
             "stage": entry["status"], "entry": entry}
        )

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
