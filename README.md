# NC Bench UI

A dev web UI (styled after ai-handler's `/debug` UI) for A/B-testing
noise-cancellation candidates on real audio: record a **phone call**
(inbound, no number needed), a **web call** (browser mic), or **upload a
file** — then run the recording through every ticked NC candidate (singles
or chains) and compare **STT transcripts**, **waveforms**, and **playable /
downloadable output audio** side by side. Every run is stored and
revisitable from the history table.

## Run

```bash
uv sync --python 3.12      # once; installs vendor/hecttor_sdk wheel too
.venv/bin/python main.py   # http://localhost:8777
```

Configuration lives in `.env` (see `.env.example`): LiveKit project,
local STT URL (ai-handler's whisper `/recognize`), Hecttor key + default
model/weight/rate/chunk, DTLN model dir, port.

## Using it

1. Pick a **source**:
   - **Phone call** — press Start *first*, then dial the trunk pointed at the
     configured LiveKit project. The poller picks up the first room that gains
     a SIP participant (rooms existing before Start are ignored) and records
     the caller's track. No phone number entry.
   - **Web call** — Start joins a fresh room and publishes your mic with
     browser echo-cancellation / noise-suppression / AGC **disabled** (we want
     the noise to survive so the NC has something to do).
   - **Upload** — any format ffmpeg can read; goes straight to processing.
2. Tick **candidates** (at least one; "No NC" is the control). Chains run
   processors in order — edit `candidates.json` to add combos or Hecttor
   model/weight variants; restart to reload.
3. **Start** → record (single subscriber timeline shows live RMS) → **Stop**.
   Both buttons gray out while the candidates run (NC → STT per candidate);
   results stream in as each finishes.
4. **History**: every run keeps `input.wav` + one output wav per candidate +
   `meta.json` under `data/runs/<id>/`; click a row to reload it.

## Architecture

```
static/index.html        the whole UI (debug-ui styling, no build step)
nc_bench/server.py       FastAPI: session start/stop, upload, candidates,
                         runs history, /ws (levels + progress events)
nc_bench/recorder.py     LiveKit rtc subscriber: web room join + phone-room
                         poller; 48 kHz mono capture + 20 ms RMS events
nc_bench/pipeline.py     input wav → per-stage soxr resample → 20 ms blocks
                         through stateful processors → 16 kHz s16 output
nc_bench/processors/     registry: hecttor (SDK, same key/params as
                         ai-handler), dtln (ONNX, models/dtln/), passthrough
nc_bench/stt.py          ai-handler's /recognize protocol (16 kHz wav POST)
nc_bench/store.py        data/runs/<id>/ meta.json + wavs
scripts/selfcheck.py     assert-based check over all available candidates
```

## Cloud candidates (Krisp & ai-coustics via LiveKit)

Cloud NC models run on the **live rail only**: the recorder is an embedded
livekit-agents worker (`AGENT_NAME=nc-bench-recorder`, THREAD executor)
dispatched into the session's room, and each ticked cloud candidate gets its
own `rtc.AudioStream(noise_cancellation=...)` on the same track while
recording. They can't process uploads or old runs (the plugins authenticate
through the live Cloud room). Verify any of them headlessly with
`scripts/live_loopback_test.py <MODEL>` (e.g. `BVC`, `AIC:QUAIL_VF_S`).

- **ai-coustics (livekit-plugins-ai-coustics): WORKING** — verified
  2026-07-31 on the boostbank project (QUAIL_L and QUAIL_VF_S both actively
  filter; billed/metered by LiveKit Cloud, no separate key). Any
  `EnhancerModel` name works as a candidates.json `"lk_model": "AIC:<NAME>"`
  entry — the installed plugin also exposes `SPARROW_S` and `ROOK_S` beyond
  the documented QUAIL family. Both plugins are imported at server startup
  (`lk_cloud.preload()`) because rtc plugin registration must happen on the
  main thread.
- **Krisp NC / BVC / BVCTelephony: WORKING** — verified 2026-07-31 (NC and
  BVC both actively filter on the boostbank project). Two conditions, both
  handled by the code: the plugin must be imported on the **main thread**
  (`lk_cloud.preload()` — a job-thread import fails silently and the stream
  degrades to passthrough with `code=209`), and the recording participant
  must be a genuine agents-framework job (a plain rtc join is refused). If a
  Krisp candidate ever regresses to passthrough, the recorder detects
  output ≈ raw and reports it as a candidate error instead of presenting raw
  audio as an NC result.
- **ai-coustics standalone `aic-sdk`** (offline, runs on uploads too) is also
  wired, gated on a self-service trial key: set `AIC_LICENSE_KEY` +
  `AIC_MODEL_ID` in `.env`; the model downloads into `models/aic/` on first
  use. Untested until a key is present.

## Scoring

Every run is scored automatically (`nc_bench/scoring.py`); results live in
`meta.json` and on the cards.

| Metric | What | Reliability notes |
|---|---|---|
| **DNSMOS P.835** (SIG/BAK/OVRL) | Microsoft's reference-free MOS predictor (`models/dnsmos/sig_bak_ovr.onnx`), on the input and every output | The workhorse. Differences < ~0.1 MOS are noise. SIG = did the voice survive, BAK = did the background die |
| **Gap-RMS / noise reduction (dB)** | silero-VAD finds *confident* no-speech windows on the raw input (prob < 0.15 sustained ≥ 1 s, edges trimmed); every output is measured in those same windows; shown as dB vs input | Trustworthy when it speaks; abstains ("n/a") when VAD finds no confident gaps — notably under heavy background *speech*, where DNSMOS-BAK carries the comparison instead. Sanity check: the passthrough candidate should read ≈ 0 dB |
| **WER** | Word error rate vs the optional **reference script** field — read a fixed script during a test call (or provide the truth for an upload) | The only exact metric here, and the most decision-relevant (the NC feeds STT). Only as good as the script matching what was actually said |
| **Latency** | Per candidate: `init_ms` (chain construction), `block_ms_mean/p95` (processing per 20 ms block — must be ≪ 20 ms to be live-viable), `algo_delay_ms` (structural buffering the chain adds regardless of CPU) | Measured, not modeled — except `algo_delay_ms`, which is the sum of each stage's chunk/window size. Live-rail (cloud) candidates have no latency row: they process inside the rtc stack in real time |

Scoring never fails a run: errors land in `scores_error` on the affected
candidate and everything else proceeds.

## Notes / limits

- **Dev tool**: no auth on any endpoint; single session at a time.
- Phone mode records **only inbound (caller) audio** — the subscriber leg,
  before any NC. If ai-handler is also running against the same LiveKit
  project it will handle the call in parallel; we only subscribe.
- Hecttor init errors (e.g. the machine-bound installation ID going stale
  after an OS update — files under `~/Library/Application Support/Hecttor/`)
  surface as per-candidate errors, not crashes. Deleting those files forces
  a re-registration on next init (may consume an installation seat).
- Adding a processor = one file in `nc_bench/processors/` + a registry line
  (see `dtln.py` for the pattern) + optional `candidates.json` entries.
  Obvious next ones from the research doc: DPDFNet-8k via sherpa-onnx,
  FastEnhancer ONNX.
- STT compares *transcripts*, not WER — judging quality stays human for now
  (listen + read). Wire outputs into nc_bench for scored metrics.
