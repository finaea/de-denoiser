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
uv sync --python 3.12                        # once; installs vendor/hecttor_sdk wheel too
.venv/bin/python scripts/fetch_models.py     # once; open NC model files (~65 MB)
.venv/bin/python main.py                     # http://localhost:8777
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
nc_bench/processors/     registry: hecttor (SDK, same key/params as ai-handler),
                         specnc (DPDFNet/GTCRN/UL-UNAS), fastenhancer, dtln,
                         rnnoise (ffmpeg arnndn), aic, passthrough
scripts/fetch_models.py  downloads every open model file into models/
nc_bench/stt.py          ai-handler's /recognize protocol (16 kHz wav POST)
nc_bench/store.py        data/runs/<id>/ meta.json + wavs
scripts/selfcheck.py     assert-based check over all available candidates
```

## Offline candidates

All of these run on uploads, live recordings and re-runs alike, on CPU. Delay is
the *structural* live latency (framing + whatever the model holds back
internally, measured — see below); p95 is per 20 ms block on an M-series Mac.

| Candidate ids | What | Rate | Delay | Source |
|---|---|---|---|---|
| `hecttor-*` (coda-vi, coda, crest-1/2, mist, weight 0.75) | commercial SDK, incl. the only voice-isolation model we have offline | 16 k | 20 ms | wheel in `vendor/` |
| `dpdfnet2-8k`, `dpdfnet8-8k` | **native 8 kHz** — cleans in the PSTN band, before upsampling | 8 k | 40 ms | HF `Ceva-IP/DPDFNet` |
| `dpdfnet2`, `dpdfnet8`, `dpdfnet-baseline` | same family at 16 kHz, for the band-split comparison | 16 k | 40 ms | HF `Ceva-IP/DPDFNet` |
| `gtcrn` | 23.7 k params — the "how cheap can this get" point | 16 k | 16 ms | sherpa-onnx release |
| `ulunas` | UL-UNAS, ultra-light, different architecture family | 16 k | 16 ms | repo's streaming export |
| `fastenhancer-t/-s/-l` | best quality-per-FLOP claims; -T is the fastest thing here | 16 k | 16–26 ms | DNS-trained wav2wav release |
| `hush`, `hush-atten12` | **the only open background-*speaker* suppressor** — DFN3 retrained on competing voices, 16 kHz native (telephony's rate) | 16 k | 20 ms | pulp-vision/Hush prebuilt lib |
| `dtln` | the free fallback baseline | 16 k | 24 ms | `models/dtln/` (committed) |
| `rnnoise-sh`, `rnnoise-bd` | 2018 baseline row, via ffmpeg's `arnndn` | 48 k | 10 ms | GregorR/rnnoise-models |
| `aic-sdk` | ai-coustics standalone, needs a trial key in `.env` | — | — | PyPI `aic-sdk` |

DPDFNet, GTCRN and UL-UNAS all publish the *same* streaming ONNX shape (one STFT
frame + opaque caches in, enhanced frame + caches out), so one wrapper
(`spec_onnx.py`) drives all of them and reads the framing — n_fft, hop, window,
rate — out of each model's ONNX metadata. `fetch_models.py` is the only place
model URLs live.

**Two things the self-check pins down**, because both fail silently otherwise:

- *Reconstruction*: with the model bypassed, analysis → overlap-add must return
  the input bit-for-bit (max err 1.2e-07 for all seven). A wrong window or a
  missing window² normalisation still "cuts noise" — it just eats the speech.
- *Output shift*: every candidate's output is cross-correlated against its input.
  0 ms for FastEnhancer / GTCRN / UL-UNAS; **DPDFNet is 40 ms** (its deep-filter
  stage hands back audio 30 ms old on top of 10 ms of framing), DTLN 24 ms,
  arnndn 10 ms. Those numbers are what `algo_delay_ms` reports, so a live
  latency budget can be read straight off the cards.

Hush is the odd one out: its ONNX bundle is the three raw DeepFilterNet graphs,
which take ERB features and return gains rather than audio. But the project ships
a **prebuilt native library** with a frame-in/frame-out C API (all that DSP
compiled in), so `hush.py` drives it through ctypes. Its `atten_lim_db` runs
*backwards* from the name: **100 = unlimited** (their default), **0 = passthrough**
— set it to 0 expecting "no cap" and you benchmark a do-nothing chain.

Deliberately not wired yet: **LL-SDR** (no released checkpoint — training repo
only) and **faster-enhancer-py** (an int8 48 kHz C port of FastEnhancer; needs
`faster-enhancer.c` built locally, and only worth it if the float ONNX wins).
Combinations are next.

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
| **Measured band** (kHz) | The rolloff edge — highest frequency still carrying signal — on the input at its native rate and on every output at 16 kHz | The file's sample rate can't tell you this: LiveKit hands every track over at 48 kHz, so an 8 kHz phone call and a mic recording look identical in the header. ~4 kHz = PSTN, ~7 kHz = wideband trunk, 15 kHz+ = mic. Read it before trusting any comparison involving the 8 kHz models |
| **WER** | Word error rate vs the optional **reference script** field — read a fixed script during a test call (or provide the truth for an upload) | The only exact metric here, and the most decision-relevant (the NC feeds STT). Only as good as the script matching what was actually said |
| **Latency** | Per candidate: `init_ms` (chain construction), `block_ms_mean/p95` (processing per 20 ms block — must be ≪ 20 ms to be live-viable), `algo_delay_ms` (structural buffering the chain adds regardless of CPU) | Measured, not modeled — `algo_delay_ms` sums each stage's framing plus its measured internal lookahead (see *Output shift* above). Whole-file stages (`rnnoise-*`) report no per-block numbers rather than a flattering ~0 ms. Live-rail (cloud) candidates have no latency row: they process inside the rtc stack in real time |

Scoring never fails a run: errors land in `scores_error` on the affected
candidate and everything else proceeds.

**Rankings are per-run, never averaged across runs.** Each run is a different
recording, and each run only scores the candidates that were ticked for it — so a
mean across history would rank the *audio* (and the coverage) rather than the
models. The Rankings box therefore ranks within whichever run is loaded, and
shows each candidate's Δ against that run's own `none` control, which is the only
apples-to-apples number available.

## Notes / limits

- **Dev tool**: no auth on any endpoint; single session at a time.
- Phone mode records **only inbound (caller) audio** — the subscriber leg,
  before any NC. If ai-handler is also running against the same LiveKit
  project it will handle the call in parallel; we only subscribe.
- Hecttor init errors (e.g. the machine-bound installation ID going stale
  after an OS update — files under `~/Library/Application Support/Hecttor/`)
  surface as per-candidate errors, not crashes. Deleting those files forces
  a re-registration on next init (may consume an installation seat).
- Adding a processor = one file in `nc_bench/processors/` + a registry line +
  optional `candidates.json` entries. If it publishes a per-frame streaming ONNX,
  it's just a row in `spec_onnx.py`'s `_MODELS` plus a URL in `fetch_models.py`.
- **Gap-RMS needs noise in the gaps.** On a near-silent input (mic recording with
  the gate closed, gap RMS ≈ −65 dB) the "noise cut" column divides by silence
  and prints absurd numbers like 115 dB. Compare candidates on it only when the
  input's own `gap_rms_db` is well above the noise floor.
- **An 8 kHz model on wideband input scores well while destroying content.**
  Measured on a 15.6 kHz mic recording: `dpdfnet2-8k` came out on top on DNSMOS
  OVRL (3.14 vs 2.60 passthrough) *and* its output band read 3.8 kHz — it had
  thrown away everything above 4 kHz, and DNSMOS barely noticed. Check the band
  column before reading a narrowband model's win as a win. On a real phone call
  there is nothing above 4 kHz to lose, which is the whole point of those two
  candidates.
- `livekit-plugins-noise-cancellation` is pinned to **0.2.6** — the version the
  live Krisp path was verified on. Bump deliberately, then re-run
  `scripts/live_loopback_test.py NC`.
