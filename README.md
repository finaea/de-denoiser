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
   - **Phone call** — press Start *first*, **then** dial the trunk pointed at the
     configured LiveKit project. No phone number entry; the first inbound call
     wins. Start has to come first for two independent reasons: LiveKit
     dispatches the job the instant the call's room exists (a job arriving with
     no session open is rejected), and the fallback poller ignores any room that
     already existed at Start, so it can't grab a stale one.
     See *Phone mode & the agent name* below — it needs one `.env` value to match
     your SIP dispatch rule.
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

## Combination candidates

Chains run their stages in order, and the pipeline resamples between them — so
an 8 kHz stage feeding a 16 kHz stage is just two rows of JSON, no code. That is
the one combination shape with a structural argument behind it: **suppress in the
call's own band, then isolate at 16 kHz**, so each model sees input near its
training distribution. Everything else here exists to be falsified — stacking two
ML denoisers is artifact-on-artifact, and the point is to measure how badly.

| Candidate | Question |
|---|---|
| `dpdfnet2-8k+hecttor-coda-vi` | **C1** — the band-split: does in-band cleanup before isolation beat isolation alone? |
| `dpdfnet2-8k+hush` | **C5** — the same idea with zero licence cost |
| `dpdfnet2+hush` | **C5 control** — same pair, suppression at 16 kHz instead of 8 kHz. If it ties C5, the band-split isn't earning its 40 ms |
| `fastenhancer-t+hecttor-coda-vi` | **C4** — 16 kHz stacking with the cheapest suppressor (0.7 ms). Expected to lose; cheap to falsify |
| `gtcrn+hush` | the free floor: cheapest open suppressor + the open isolator |
| `dtln+hecttor-coda-vi` | the original combo, kept as a reference point |

Isolation-before-suppression is deliberately absent: the isolator is the fragile
model, and feeding it raw audio is the whole point of C1. Latency **adds** —
`algo_delay_ms` for C1 is 60 ms (40 + 20), which is a live-viability fact, not a
detail. Cloud candidates can't appear in chains at all: they process inside the
rtc stack on the live rail, not in this pipeline.

**First measurement, on a clean 11.8 kHz mic recording** (2026-08-01): every combo
landed at or below its own best single stage — `dpdfnet2+hush` worst overall at
2.80 OVRL vs 3.11 and 3.00 alone, `gtcrn+hush` 3.06 vs 3.20, `fastenhancer-t
+hecttor` 3.13 vs 3.21, `dpdfnet2-8k+hecttor` 3.18 ≈ the best single. So the
artifact-on-artifact rule showed up in the numbers, and the more aggressive the
pair the worse it got. Weak test though — nothing in that recording needed
removing. One mild signal worth chasing: the in-band variant beat its own control
(`dpdfnet2-8k+hush` 2.94 vs `dpdfnet2+hush` 2.80), which is the C1 hypothesis
pointing the right way on one clip.

## Phone mode & the agent name

A LiveKit SIP dispatch rule names the agent it hands each inbound call to, and
the room is created with that dispatch attached — so if nothing is registered
under that name, **the call gets no agent and drops**. The recorder therefore
registers under `LK_AGENT_NAME` (`.env`, default **`inbound-agent`**), matching
the usual rule, and lets LiveKit dispatch it directly. Verified 2026-08-01 by
simulating exactly what the rule does — create a `call-…` room, dispatch
`inbound-agent`, join as a `sip_…` participant — and the session recorded 5.5 s
with the live Krisp candidate attached.

Consequences worth knowing:

- **Never run ai-handler and the bench against the same LiveKit project at once.**
  LiveKit load-balances jobs across every worker sharing an agent name, so a real
  call could land on the bench — which only listens, so the caller gets silence.
  If you need both, give the bench its own name here *and* its own dispatch rule.
- The session is armed the moment you press Start, before any call exists, and a
  claim flag makes the first job the only recorder — so the fallback poller can't
  cause a second, duplicate job.
- `dispatch_rule_individual` with a room prefix (one fresh room per call) is the
  friendly case. A rule with a fixed room name would leave the room lingering
  between calls, which the poller path deliberately skips as stale.

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
