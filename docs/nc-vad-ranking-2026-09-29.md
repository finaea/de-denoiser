# NC × VAD ranking, round 2: which gate, VAD and parameters fix PR-54 and PR-42

**2026-09-29** · 20 stored runs plus the 3 incident calls (PR-54 A, PR-54 B, PR-42) in
`data/runs/`. Every table below is printed by `scripts/report_round2.py` from the stored
`meta.json` files alone. Each one names the `--section` that prints it, and some are cut to
their top rows. The method follows `docs/nc-ranking-2026-08-04.md`: every number is a delta
against the run's own control, aggregated as n, mean or median, worst, and W / L. W / L
counts strictly better / worse, so ties are in neither. A candidate in fewer than half of a
table's runs is left out. In the whole-file tables a collapsed transcript is excluded;
in VAD-cut WER a loop is counted, because the VAD's cut caused it.

## Bottom line

**On phone, keep the Hush@100 gate and swap production's VAD for Silero v4.0 at 8 kHz, with
activation = deactivation 0.6, min speech 0.05 s and min silence 0.55 s (production uses
0.3).** The scores below are for Hush on production's exact path (`hush-aih`, the production agent's
own Hush wrapper at 24 kHz):

- **PR-54 A:** the click and the hang-up clunk no longer open a turn, and all 3 real turns are kept.
- **PR-54 B:** the throat-clear no longer opens a turn. Production's own VAD opens on it on
  this same audio.
- **PR-42:** it catches 6 of the 7 loud missed sounds (production: 4/7) and keeps all 21 detected
  ones, at the cost of 1 extra orphan trigger and 4 on quiet/echo sounds (production: 0 and 2).
- **The 10 marked phone runs:** 0 missed turns and 0 phantom segments (production: 0 and 3).
- **VAD-cut WER:** better than production (median ΔWER −0.031, 5W / 3L, no repetition loops).

It needs no LiveKit Cloud and no licence. The catches:

- **Latency:** min silence 0.55 adds 0.25 s before every STT request, and so before every
  response. At production's 0.3 the incidents are fixed just the same, but VAD-cut WER gets
  worse (§8 has all three values).
- **Engineering:** the production agent has to run a Silero v4.0 ONNX (the older `h`/`c` state API) in place of
  `livekit.agents.inference.VAD`.

## 1. What changed since 08-04

- **New candidates.**
  - `aicsdk-*`: 12 ai-coustics SDK 3.2 models, offline, own key. They include the 8 kHz
    models and Rook Multi Speaker.
  - `lkaic-*`: 4 models from the official LiveKit ai-coustics plugin 0.3.2 (Voice Focus 2.2),
    plus 2 `bypass` variants that keep the audio raw and only stamp the VAD flag.
  - `krisp-viva` and `krisp-viva-tel`.
  - `cloud-raw`: the replay rail's unfiltered control.
  - The old `aic-quail-l`, `aic-vf-s` and `aic-vf-l` rows are **VF 2.1 / historical**. They
    came from plugin 0.3.0 and can't be reproduced now.
- **The cloud replay rail, and its Opus caveat.**
  - Cloud models now reach every stored run and upload. The replay publishes `input.wav` into a
    fresh LiveKit Cloud room at 256 kbps and records it back, which is a second Opus pass.
  - `cloud-raw` against `none` shows what that pass costs: level −0.16 to +0.09 dB, envelope
    corr 0.83–1.00. Its WER equals `none`'s on 13 of the 18 runs that have one. It differs on
    165058 (0.172 → 0.310), 171927, 172121, 173210 and the junk `noise test phone` (section
    `replay`). So cloud candidates are also compared against `cloud-raw`.
  - **Replay ids.** The replayed Krisp NC / BVC / BVC Telephony are new records
    (`krisp-nc-replay`, `krisp-bvc-replay`, `krisp-bvc-tel-replay`) with
    `rail: cloud_replay`. The live-rail `krisp-*` records stay as recorded. These ids aren't
    in `candidates.json`; stored cards render from the record's own label.
- **The Silero version trap.** Every VAD column in the 08-04 report came from
  livekit-plugins-silero 1.6.7, and that plugin's bundled ONNX is Silero **v6.0**. Production
  runs native **v6.2** through `livekit.agents.inference.VAD`. Every VAD number here comes from
  the VAD bench (`vad_bench`, one shared segmenter), never from the legacy `vad` /
  `truth_score` columns.
- **STT** is now `boost_volume=false`, what production's STT client sends. The old transcripts
  are kept under `stt_v1`. The change barely moved passthrough WER: it was the same on 15 of 17
  runs, +0.054 on one, and changed on `noise test phone`, which is junk either way (section
  `stt-v1`).
- **Labels.** Each run can now carry `known_nonspeech` (named non-speech sounds) and
  `ignore_spans` (uncertain audio) next to `truth_spans`. Event metrics score marks that are
  closer together than the row's own `min_silence` as one turn.
- **`20260804-131319` had lost 28 of its 35 records.** On 2026-08-10 it was re-run with 7
  candidates, and the old re-run replaced `meta["candidates"]` wholesale. The wavs survived.
  - **26 were re-attached** as `restored_orphan` records. Each passed a check on length, lag
    and mtime (lag against input time, or against the candidate's own usual offset on the
    other phone runs).
  - **2 stay orphaned:** historical `aic-vf-l` and `aic-vf-s`, whose lag was 60–65 ms off
    their own medians.
  - **Restored records have no latency, chain or cost numbers.**
  - This also explains 08-04's phone STT table: `hecttor-crest2` had n = 8 there, but today's
    meta gave n = 7.
- **Live-rail outputs aren't on input time.** The live rail's streams attach after the track
  starts. So the stored `aic-*` and `krisp-bvc-tel` outputs begin 20–290 ms into the call,
  and every event in them sits that much early against the hand marks. The 08-04 VAD tables
  therefore scored those rows up to 0.29 s early. They're moved out of the rankings, into a
  "live rail as recorded" table shown as stored and offset-corrected (§3.4).
- **Krisp NC's output couldn't be aligned on 4 of 23 runs.**
  - On 3 runs (113639, 172631, 173210) it was flattened below any envelope peak. On 113639 it
    was −27 dB with its first sound at 10.4 s.
  - On PR-42 its timeline tore: 558 ms of its own drift.
  - Its twin and control streams were clean every time, so this is the model, not the rail.
  - Where it did align (19 runs), DNSMOS OVRL was 2.59 against `cloud-raw`'s 2.29. Its
    whole-file WER against `cloud-raw` had a median of +0.041 and a worst of +0.562 (n = 14).
- **Two platforms.**
  - aicsdk-\* on the 20 stored runs, PR-42's other non-Hush offline candidates, and all VAD
    probabilities except TEN's were computed on Linux x86_64.
  - TEN, every Hush chain, and all earlier outputs are macOS arm64.
  - The cost numbers are single-thread Mac measurements (§3.2), never the parallel trainer runs.
- **Not available:**
  - `rnnoise-*`: no ffmpeg on this host.
  - Hecttor: no wheel. Stored outputs are still scored.
  - TEN on Linux: `libc++` missing, so it ran on the Mac.

## 2. Coverage

Records are candidates plus restored or imported sources. "VAD pairs" counts the fresh
(source, VAD) results. "unavail" is `lkaic-flag` on sources that carry no ai-coustics flags.
Every run has exactly sources × 22 VADs, plus `lkaic-flag` where flags exist (section `coverage`).

| run | surface | note | dur s | marks | records | aicsdk | replay | replay retries | control drift ms | VAD pairs | stale | unavail | errors | gaps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260802-203822` | phone | restaurant | 14.7 | 9 | 59 | 12/12 | 12/12 | 0 | 0 | 1304 | 0 | 53 | 0 | — |
| `20260802-204103` | web | restaurant | 12.9 | 2 | 59 | 12/12 | 12/12 | 0 | 0/46.1 | 1326 | 0 | 54 | 0 | — |
| `20260802-204218` | web | restaurant | 13.0 | 1 | 59 | 12/12 | 12/12 | 0 | 0/7.7 | 1326 | 0 | 54 | 0 | — |
| `20260803-164743` | phone | bg voices A low | 22.3 | 7 | 59 | 12/12 | 12/12 | 0 | 0/63.1 | 1304 | 0 | 53 | 0 | — |
| `20260803-165058` | phone | bg voices A | 20.6 | 8 | 59 | 12/12 | 12/12 | 1 | 44.7/46.8 | 1326 | 0 | 54 | 0 | — |
| `20260803-170156` | phone | bg voices B | 32.8 | 10 | 59 | 12/12 | 12/12 | 0 | 19/41.1 | 1304 | 0 | 53 | 0 | no script: no WER |
| `20260803-175858` | — | vendor sample (ignore refere | 15.0 | 0 | 53 | 12/12 | 12/12 | 0 | 0/33.8 | 1194 | 0 | 48 | 0 | no hand marks: event metrics null |
| `20260804-103947` | — | real call (no reference) | 119.5 | 0 | 53 | 12/12 | 6/12 | 2 | 0.1/114.1 | 1059 | 0 | 45 | 0 | replay 6 missing (krisp-bvc-replay, krisp-bvc-tel-replay, krisp-viva, lkaic-rook-s, lkaic-rook-s-bypass, lkaic-vf22-l failed); no hand marks: event metrics null; no script: no WER |
| `20260804-112511` | phone | bg voices C | 25.5 | 6 | 59 | 12/12 | 12/12 | 0 | 28.8/67.5 | 1282 | 0 | 52 | 0 | — |
| `20260804-113639` | phone | bg voices C saved (phone) | 23.3 | 6 | 53 | 12/12 | 11/12 | 2 | 2.5/20.9/42.8/62.7 | 1172 | 0 | 47 | 0 | replay 1 missing (krisp-nc-replay failed) |
| `20260804-131319` | phone | pantry | 20.6 | 7 | 57 | 12/12 | 12/12 | 0 | 0/0.1 | 1282 | 0 | 52 | 0 | 26 restored orphans |
| `20260804-171340` | web | office | 25.9 | 3 | 53 | 12/12 | 12/12 | 0 | 0/51.8 | 1194 | 0 | 48 | 0 | — |
| `20260804-171445` | web | bg voices D | 27.9 | 8 | 59 | 12/12 | 12/12 | 0 | 9.9/53.8 | 1326 | 0 | 54 | 0 | — |
| `20260804-171927` | web | bg voices E | 24.8 | 5 | 59 | 12/12 | 12/12 | 0 | 21.8/41.4 | 1326 | 0 | 54 | 0 | — |
| `20260804-172121` | web | bg voices E | 26.3 | 4 | 59 | 12/12 | 12/12 | 0 | 0/13.2 | 1326 | 0 | 54 | 0 | — |
| `20260804-172535` | phone | bg voices E | 21.7 | 10 | 53 | 12/12 | 12/12 | 1 | 0.2/37.2/46.7 | 1194 | 0 | 48 | 0 | — |
| `20260804-172631` | phone | bg voices E | 24.2 | 10 | 59 | 12/12 | 11/12 | 2 | 23.6/43.6/45.7 | 1304 | 0 | 53 | 0 | replay 1 missing (krisp-nc-replay failed) |
| `20260804-173210` | web | bg voices E | 22.6 | 6 | 59 | 12/12 | 11/12 | 2 | 0.1/25.9/65.4/125 | 1304 | 0 | 53 | 0 | replay 1 missing (krisp-nc-replay failed) |
| `20260804-174112` | web | noise test web | 33.9 | 4 | 59 | 12/12 | 12/12 | 0 | 23.9/26.2 | 1326 | 0 | 54 | 0 | — |
| `20260804-174342` | phone | noise test phone | 32.5 | 5 | 59 | 12/12 | 12/12 | 0 | 17/24.3 | 1304 | 0 | 53 | 0 | — |
| `20260929-123329` | incident | PR-54 A | 65.1 | 3 | 42 | 12/12 | 12/12 | 0 | 0.1/72.8 | 952 | 0 | 37 | 0 | no script: no WER |
| `20260929-123329-1` | incident | PR-54 B | 34.0 | 0 | 42 | 12/12 | 10/12 | 2 | 0/10.6/34.2/39.9 | 908 | 0 | 35 | 0 | replay 2 missing (krisp-bvc-replay, krisp-bvc-tel-replay failed); no hand marks: event metrics null; no script: no WER |
| `20260929-123329-2` | incident | PR-42 | 414.5 | 28 | 42 | 12/12 | 9/12 | 2 | 67.3/126.6/144 | 885 | 0 | 35 | 0 | replay 3 missing (krisp-bvc-tel-replay, krisp-nc-replay, lkaic-rook-s failed); no script: no WER |

The replay's missing cells, all recorded in each run's `meta["backfill_round2"]`:

- **103947:** 5 candidates failed the batch drift gate: the jitter-buffer spread was 95–114 ms
  against an 80 ms gate, on a 119 s file. `krisp-bvc-tel-replay` failed its own timeline gate:
  260 ms, a real candidate failure. The run has no marks and no script, so after 3 attempts it
  was stopped.
- **113639, 172631, 173210:** Krisp NC was unalignable (see §1).
- **PR-54 B:** Krisp BVC and BVC Telephony flattened the few sounds the file has.
- **PR-42:** 3 own-timeline tears: krisp-nc 558 ms, lkaic-rook-s 564 ms, krisp-bvc-tel 188 ms.
  PR-42 was replayed with a **150 ms** batch drift gate, since a 414 s file piles up jitter-buffer
  steps. Its kept records carry a control drift of 67, 127 and 144 ms, by batch.

Per-run replay diagnostics are all within the gates:

- twin residual −403 to −427 dB;
- lead residual ≤ 9.5 ms;
- twin length diff 0;
- onset delay −41 to +58 ms (section `replay`).

## 3. NC tables per surface

### 3.1 STT, whole-file

ΔWER vs `none`, round-2 STT, ranked by median. A run is excluded when its passthrough
transcript collapsed, **or** when its WER is above 1.0. `noise test phone`'s new passthrough
hallucinates instead of looping, WER 6.5 on an 11-word script, so it's excluded
(section `nc-stt`).

#### Phone — STT (8 valid runs · passthrough WER median 0.167, mean 0.183)

| # | candidate | n | median ΔWER | mean | worst | W | L |
|---|---|---|---|---|---|---|---|
| 1 | `aicsdk-vf22-s` | 8 | -0.032 | -0.063 | +0.000 | 7 | 0 |
| 2 | `hecttor-coda-vi-w075` | 8 | -0.029 | -0.031 | +0.035 | 5 | 1 |
| 3 | `aicsdk-vf22-l` | 8 | -0.029 | -0.037 | +0.029 | 5 | 2 |
| 4 | `hecttor-crest2` | 8 | -0.028 | -0.035 | +0.000 | 5 | 0 |
| 5 | `lkaic-vf22-l` | 8 | -0.028 | -0.026 | +0.054 | 5 | 2 |
| 6 | `dpdfnet2-8k` | 8 | -0.013 | -0.006 | +0.104 | 4 | 2 |
| 7 | `krisp-viva` | 8 | -0.013 | -0.019 | +0.031 | 4 | 1 |
| 8 | `lkaic-rook-s-bypass` | 8 | -0.013 | -0.023 | +0.000 | 4 | 0 |
| 9 | `rnnoise-sh` | 8 | -0.013 | -0.004 | +0.048 | 4 | 3 |
| 10 | `ulunas` | 8 | -0.013 | +0.027 | +0.218 | 4 | 3 |

**Phone, cloud replay rail against `cloud-raw`** (the second Opus pass cancelled out):

| candidate | n | median ΔWER vs cloud-raw | mean | worst | W | L |
|---|---|---|---|---|---|---|
| `krisp-viva` | 8 | -0.030 | -0.036 | +0.031 | 5 | 1 |
| `lkaic-rook-s-bypass` | 8 | -0.028 | -0.040 | +0.000 | 5 | 0 |
| `lkaic-vf22-l` | 8 | -0.028 | -0.043 | +0.054 | 5 | 2 |
| `lkaic-vf22-s` | 8 | -0.013 | -0.033 | +0.000 | 4 | 0 |
| `krisp-bvc-tel-replay` | 8 | +0.000 | -0.011 | +0.032 | 2 | 2 |
| `krisp-viva-tel` | 8 | +0.000 | -0.006 | +0.069 | 3 | 1 |
| `lkaic-quail-l` | 8 | +0.000 | +0.022 | +0.108 | 0 | 2 |
| `lkaic-quail-l-bypass` | 8 | +0.000 | +0.004 | +0.031 | 0 | 1 |
| `lkaic-rook-s` | 8 | +0.031 | +0.030 | +0.104 | 1 | 4 |
| `krisp-bvc-replay` | 8 | +0.069 | +0.552 | +3.376 | 1 | 4 |

#### Web — STT (8 valid runs · passthrough WER median 0.113, mean 0.103)

| # | candidate | n | median ΔWER | mean | worst | W | L |
|---|---|---|---|---|---|---|---|
| 1 | `aicsdk-quail-ms-l-16k` | 8 | +0.000 | +0.005 | +0.063 | 1 | 3 |
| 2 | `aicsdk-quail-ms-l-8k` | 8 | +0.000 | -0.010 | +0.032 | 2 | 2 |
| 3 | `aicsdk-quail-ms-s-16k` | 8 | +0.000 | +0.013 | +0.091 | 1 | 3 |
| 4 | `aicsdk-quail-ms-s-8k` | 8 | +0.000 | +0.001 | +0.091 | 1 | 1 |
| 5 | `cloud-raw` | 8 | +0.000 | +0.001 | +0.032 | 1 | 2 |
| 6 | `fastenhancer-l` | 8 | +0.000 | +0.022 | +0.094 | 1 | 3 |
| 7 | `hecttor-coda-vi-w075` | 8 | +0.000 | +0.005 | +0.063 | 1 | 2 |
| 8 | `hush-atten12` | 8 | +0.000 | +0.009 | +0.125 | 1 | 2 |
| 9 | `krisp-viva` | 8 | +0.000 | +0.016 | +0.125 | 1 | 2 |
| 10 | `krisp-viva-tel` | 8 | +0.000 | -0.003 | +0.032 | 1 | 2 |

**Web, cloud replay rail against `cloud-raw`** (the second Opus pass cancelled out):

| candidate | n | median ΔWER vs cloud-raw | mean | worst | W | L |
|---|---|---|---|---|---|---|
| `krisp-viva` | 8 | +0.000 | +0.016 | +0.125 | 1 | 2 |
| `krisp-viva-tel` | 8 | +0.000 | -0.003 | +0.000 | 1 | 0 |
| `lkaic-quail-l` | 8 | +0.000 | +0.020 | +0.157 | 0 | 1 |
| `lkaic-quail-l-bypass` | 8 | +0.000 | -0.007 | +0.000 | 2 | 0 |
| `lkaic-rook-s-bypass` | 8 | +0.000 | +0.008 | +0.063 | 0 | 1 |
| `lkaic-vf22-l` | 8 | +0.000 | +0.018 | +0.182 | 2 | 3 |
| `lkaic-rook-s` | 8 | +0.013 | +0.040 | +0.188 | 1 | 4 |
| `lkaic-vf22-s` | 8 | +0.015 | +0.024 | +0.063 | 0 | 4 |
| `krisp-bvc-tel-replay` | 8 | +0.029 | +0.078 | +0.250 | 1 | 5 |
| `krisp-nc-replay` | 7 | +0.046 | +0.040 | +0.094 | 1 | 5 |

Excluded runs:

| run | note | why |
|---|---|---|
| `20260803-170156` | bg voices B | no reference script |
| `20260803-175858` | vendor sample (ignore reference) | upload with no surface label |
| `20260804-103947` | real call (no reference) | upload with no surface label |
| `20260804-174342` | noise test phone | passthrough WER 6.54 > 1.0 (junk transcript) |

Candidate transcripts dropped for repetition collapse:

| run | note | candidate | repeat |
|---|---|---|---|
| `20260804-112511` | bg voices C | `dpdfnet-baseline` | x100 |

On phone the Voice Focus family (`aicsdk-vf22-*`, `lkaic-vf22-*`) and `hecttor-crest2` top
the table, at about one word better per call. On web nothing beats passthrough on the median,
as on 08-04.

### 3.2 VAD: missed speech, phantom segments and agreement, under production's VAD

Each gate's audio goes through `lk-inference` (production's Silero v6.2) at production's
parameters, scored as Δ vs `none` (section `nc-vad`). The stored phone runs almost never
lose a whole turn: `none` misses 0, and 44 of the 52 gates never add one. Misses show up in
PR-42 (§7).

#### Phone — phantom segments under `lk-inference` (10 runs · `none` 2.20 per run)

| # | gate | n | mean Δ | median | worst | W | L |
|---|---|---|---|---|---|---|---|
| 1 | `dpdfnet2+hush` | 10 | -2.20 | -1.50 | +0.00 | 9 | 0 |
| 2 | `hecttor-mist` | 10 | -2.10 | -1.50 | +0.00 | 9 | 0 |
| 3 | `gtcrn+hush` | 10 | -2.00 | -1.00 | +0.00 | 9 | 0 |
| 4 | `dpdfnet2-8k+hecttor-coda-vi` | 10 | -1.90 | -1.50 | +0.00 | 8 | 0 |
| 5 | `dpdfnet2-8k+hush` | 10 | -1.90 | -1.00 | +0.00 | 8 | 0 |
| 6 | `fastenhancer-t+hecttor-coda-vi` | 10 | -1.90 | -1.50 | +0.00 | 8 | 0 |
| 7 | `hush` | 10 | -1.90 | -1.00 | +0.00 | 8 | 0 |
| 8 | `krisp-bvc-replay` | 10 | -1.80 | -1.50 | +1.00 | 7 | 1 |
| 9 | `aicsdk-vf22-l` | 10 | -1.70 | -1.00 | +0.00 | 8 | 0 |
| 10 | `aicsdk-vf22-s` | 10 | -1.70 | -1.00 | +0.00 | 8 | 0 |

#### Phone — missed speech s under `lk-inference` (10 runs · `none` 1.20 per run)

| # | gate | n | mean Δ | median | worst | W | L |
|---|---|---|---|---|---|---|---|
| 1 | `dpdfnet8-8k` | 10 | -0.97 | +0.00 | +0.24 | 4 | 5 |
| 2 | `gtcrn` | 10 | -0.95 | -0.02 | +0.00 | 6 | 0 |
| 3 | `hecttor-coda` | 10 | -0.92 | +0.00 | +0.06 | 4 | 5 |
| 4 | `hush-atten12` | 10 | -0.90 | -0.02 | +0.21 | 5 | 4 |
| 5 | `dtln` | 10 | -0.89 | +0.02 | +0.17 | 4 | 6 |
| 6 | `fastenhancer-s` | 10 | -0.89 | -0.03 | +0.51 | 6 | 1 |
| 7 | `fastenhancer-t` | 10 | -0.88 | -0.01 | +0.13 | 5 | 2 |
| 8 | `lkaic-quail-l-bypass` | 10 | -0.86 | +0.00 | +0.25 | 4 | 1 |
| 9 | `aicsdk-quail-ms-l-16k` | 10 | -0.83 | +0.01 | +0.27 | 3 | 5 |
| 10 | `krisp-viva-tel` | 10 | -0.82 | +0.00 | +0.25 | 4 | 4 |

#### Phone — frame agreement under `lk-inference` (10 runs · `none` 77.8% per run)

| # | gate | n | mean Δ | median | worst | W | L |
|---|---|---|---|---|---|---|---|
| 1 | `gtcrn+hush` | 10 | +14.5% | +8.7% | +4.6% | 10 | 0 |
| 2 | `hush` | 10 | +13.7% | +8.5% | +1.0% | 10 | 0 |
| 3 | `krisp-bvc-tel-replay` | 10 | +13.5% | +8.2% | +3.0% | 10 | 0 |
| 4 | `hecttor-mist` | 10 | +13.0% | +8.0% | +3.2% | 10 | 0 |
| 5 | `hecttor-coda-vi-8k` | 10 | +12.6% | +7.3% | +2.8% | 10 | 0 |
| 6 | `fastenhancer-t+hecttor-coda-vi` | 10 | +12.5% | +7.7% | +2.1% | 10 | 0 |
| 7 | `hecttor-coda-vi` | 10 | +12.2% | +6.6% | +1.9% | 10 | 0 |
| 8 | `lkaic-vf22-l` | 10 | +12.2% | +7.6% | +1.8% | 10 | 0 |
| 9 | `hecttor-crest2` | 10 | +12.2% | +7.3% | -0.6% | 9 | 1 |
| 10 | `dpdfnet2-8k+hush` | 10 | +11.9% | +7.0% | +0.0% | 9 | 0 |

#### Web — missed speech s under `lk-inference` (8 runs · `none` 1.15 per run)

| # | gate | n | mean Δ | median | worst | W | L |
|---|---|---|---|---|---|---|---|
| 1 | `gtcrn` | 8 | -0.37 | -0.52 | +0.88 | 5 | 2 |
| 2 | `ulunas` | 8 | -0.28 | -0.35 | +0.67 | 4 | 3 |
| 3 | `lkaic-quail-l` | 8 | -0.23 | -0.48 | +1.48 | 4 | 3 |
| 4 | `lkaic-rook-s-bypass` | 8 | -0.13 | -0.03 | +0.04 | 5 | 1 |
| 5 | `dpdfnet8-8k` | 8 | -0.12 | -0.52 | +1.63 | 5 | 2 |
| 6 | `aicsdk-vf22-s` | 8 | -0.10 | -0.21 | +1.21 | 4 | 3 |
| 7 | `cloud-raw` | 8 | -0.10 | -0.02 | +0.08 | 4 | 2 |
| 8 | `lkaic-quail-l-bypass` | 8 | -0.10 | -0.02 | +0.08 | 4 | 2 |

### 3.3 VAD-cut WER (production-real: one STT request per turn)

Each VAD segment is **one request**, as production's gated STT sends it: 0.5 s of prefix pad +
the segment + the `min_silence` the VAD waited out before ending it. Neighbouring slices are
**not merged**, even when they overlap.

Rules:
- ΔWER is against **production's config on the same run** (`hush` → `lk-inference` at
  0.6/0.05/0.3, STT reading raw).
- A run is excluded when that baseline collapsed or has WER > 1.0. No phone run does.
- A repetition loop in a combo's transcript is **counted**, and it's shown as `loops`. In a
  VAD-cut transcript a loop comes from the fragment the VAD cut, which makes it that VAD's cost.

Section `vad-wer`:

#### Phone — VAD-cut WER (9 runs; ΔWER vs production: `hush` → `lk-inference` @ 0.6/0.05/0.3, STT reads raw)

| gate → VAD @ act/min speech/min silence | STT reads | n | median WER | median ΔWER | mean | worst | W | L | loops |
|---|---|---|---|---|---|---|---|---|---|
| `lkaic-quail-l-bypass` → `lkaic-flag` @ 0.6/0.25/0.55 | raw | 9 | 0.273 | -0.081 | -0.076 | +0.104 | 6 | 2 | 0 |
| `lkaic-quail-l-bypass` → `lkaic-flag` @ 0.6/0.25/0.55 | same | 9 | 0.273 | -0.081 | -0.062 | +0.207 | 6 | 2 | 0 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.05/0.55 | same | 9 | 0.265 | -0.062 | -0.035 | +0.182 | 5 | 4 | 0 |
| `hush` → `aic-vad-vf-2.0` @ 0.6/0.05/0.3 | same | 9 | 0.273 | -0.054 | -0.076 | +0.000 | 8 | 0 | 0 |
| `dpdfnet2-8k+hush` → `silero-v4.0-8k` @ 0.3/0.05/0.55 | raw | 9 | 0.297 | -0.054 | +0.016 | +0.719 | 6 | 3 | 1 |
| `gtcrn+hush` → `silero-v4.0-8k` @ 0.6/0.05/0.55 | same | 9 | 0.324 | -0.048 | -0.018 | +0.162 | 5 | 3 | 0 |
| `dpdfnet2-8k+hush` → `silero-v4.0-8k` @ 0.3/0.05/0.55 | same | 9 | 0.243 | -0.034 | -0.056 | +0.091 | 6 | 3 | 0 |
| `hush` → `lk-inference` @ 0.8/0.1/0.55 | same | 9 | 0.270 | -0.031 | -0.060 | +0.108 | 6 | 2 | 0 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.05/0.55 | raw | 9 | 0.294 | -0.031 | -0.018 | +0.345 | 5 | 3 | 0 |
| `gtcrn+hush` → `firered-stream` @ 0.6/0.05/0.3 | raw | 9 | 0.351 | -0.031 | +0.074 | +0.966 | 6 | 3 | 1 |
| `hush` → `lk-inference` @ 0.8/0.1/0.55 | raw | 9 | 0.324 | -0.029 | -0.031 | +0.345 | 5 | 2 | 0 |
| `gtcrn+hush` → `firered-stream` @ 0.6/0.05/0.3 | same | 9 | 0.276 | -0.029 | +0.024 | +0.454 | 5 | 3 | 0 |
| `gtcrn+hush` → `silero-v4.0-8k` @ 0.6/0.05/0.55 | raw | 9 | 0.294 | -0.027 | -0.012 | +0.311 | 5 | 2 | 0 |
| `hush` → `lk-inference` @ 0.6/0.1/0.3 | raw | 9 | 0.324 | +0.000 | -0.028 | +0.000 | 4 | 0 | 0 |
| `hush` → `lk-inference` @ 0.6/0.05/0.3 | raw | 9 | 0.324 | +0.000 | +0.000 | +0.000 | 0 | 0 | 0 |
| `gtcrn+hush` → `silero-v4.0-8k` @ 0.6/0.05/0.3 | raw | 9 | 0.438 | +0.000 | +0.045 | +0.207 | 2 | 4 | 0 |
| `hush` → `aic-vad-vf-2.0` @ 0.6/0.05/0.3 | raw | 9 | 0.379 | +0.000 | +0.094 | +0.862 | 4 | 4 | 1 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.05/0.3 | raw | 9 | 0.351 | +0.000 | +0.110 | +0.828 | 3 | 4 | 1 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.1/0.3 | raw | 9 | 0.351 | +0.027 | +0.103 | +0.828 | 3 | 5 | 1 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.05/0.4 | raw | 9 | 0.382 | +0.027 | +0.105 | +0.793 | 2 | 5 | 1 |
| `hush` → `lk-inference` @ 0.6/0.05/0.3 | same | 9 | 0.276 | +0.031 | -0.009 | +0.091 | 4 | 5 | 0 |
| `hush` → `lk-inference` @ 0.6/0.1/0.3 | same | 9 | 0.294 | +0.031 | -0.007 | +0.135 | 4 | 5 | 0 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.05/0.4 | same | 9 | 0.324 | +0.031 | +0.014 | +0.182 | 3 | 6 | 0 |
| `gtcrn+hush` → `silero-v4.0-8k` @ 0.6/0.05/0.3 | same | 9 | 0.364 | +0.031 | +0.124 | +0.966 | 2 | 5 | 1 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.1/0.3 | same | 9 | 0.364 | +0.063 | +0.152 | +0.966 | 1 | 8 | 1 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.05/0.3 | same | 9 | 0.364 | +0.091 | +0.156 | +0.966 | 1 | 7 | 1 |
| `none` → `lk-inference` @ 0.6/0.05/0.3 | same | 9 | 0.471 | +0.118 | +0.663 | +5.091 | 1 | 8 | 0 |
| `none` → `lk-inference` @ 0.6/0.05/0.3 | raw | 9 | 0.471 | +0.118 | +0.677 | +5.091 | 1 | 8 | 0 |

"loops": transcripts with a 1-3-gram repeated >= 5x in a row. They are counted in the WER (9 on this surface).

#### Web — VAD-cut WER (8 runs; ΔWER vs production: `hush` → `lk-inference` @ 0.6/0.05/0.3, STT reads raw)

| gate → VAD @ act/min speech/min silence | STT reads | n | median WER | median ΔWER | mean | worst | W | L | loops |
|---|---|---|---|---|---|---|---|---|---|
| `none` → `lk-inference` @ 0.6/0.05/0.3 | same | 8 | 0.212 | -0.256 | -0.261 | +0.108 | 6 | 1 | 0 |
| `none` → `lk-inference` @ 0.6/0.05/0.3 | raw | 8 | 0.212 | -0.256 | -0.258 | +0.108 | 6 | 1 | 0 |
| `lkaic-quail-l-bypass` → `lkaic-flag` @ 0.6/0.25/0.55 | raw | 8 | 0.220 | -0.187 | -0.190 | +0.108 | 7 | 1 | 0 |
| `lkaic-quail-l-bypass` → `lkaic-flag` @ 0.6/0.25/0.55 | same | 8 | 0.220 | -0.169 | -0.187 | +0.108 | 7 | 1 | 0 |
| `gtcrn+hush` → `firered-stream` @ 0.6/0.05/0.3 | raw | 8 | 0.386 | -0.045 | -0.063 | +0.135 | 4 | 2 | 0 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.05/0.55 | raw | 8 | 0.354 | -0.027 | -0.020 | +0.000 | 5 | 0 | 0 |
| `hush` → `silero-v4.0-8k` @ 0.6/0.05/0.4 | raw | 8 | 0.354 | -0.027 | -0.020 | +0.000 | 5 | 0 | 0 |
| `gtcrn+hush` → `silero-v4.0-8k` @ 0.6/0.05/0.55 | raw | 8 | 0.469 | -0.022 | -0.024 | +0.191 | 4 | 3 | 0 |

- **Phone, production's split read (`raw`):**
  - At production's own parameters, `hush → silero-v4.0-8k` **costs** WER: mean +0.110,
    3W / 4L. It splits sentences at short pauses (8.8 requests per run against production's
    8.1), and one fragment made Qwen loop (165058, +0.828).
  - At `min_silence` 0.55 it's better than production: median −0.031, 5W / 3L, 0 loops (§8).
  - `lkaic-quail-l-bypass → lkaic-flag` (LiveKit Cloud) has the best median, −0.081. It
    **doesn't fix PR-54**: at its own best set the ai-coustics flag still opens on A's click,
    and PR-54 measured it holding the click for 450 ms.
  - Production's VAD with min speech 0.10 is never worse than production (4W / 0L), but it
    doesn't fix either incident (§8).
  - Ungated (`none → lk-inference`) is the worst row, with a median of +0.118. Gating still
    pays, as on 08-05.
- **Web:** full-strength Hush costs speech (12 missed turns, §4). The web gate should stay
  capped, as 08-04 said.

### 3.4 Live rail as recorded (08-04), stored and offset-corrected

These rows don't compete in the rankings. The corrected rows move each segment onto input
time by the output's measured lag. That's what re-segmenting with `t0_s = −lag` gives. They
also ignore the head the output never had. They're re-scored in memory and never written
(section `live-rail`).

Measured offset of each live-rail output against its input (negative: the stream attached late and the output starts that far into the call):

| candidate | runs | lag ms min | median | max |
|---|---|---|---|---|
| `aic-quail-l` (VF 2.1 / historical) | 14 | -290 | -95 | -40 |
| `aic-vf-l` (VF 2.1 / historical) | 10 | -280 | -85 | -30 |
| `aic-vf-s` (VF 2.1 / historical) | 13 | -280 | -80 | -30 |
| `krisp-bvc` | 15 | -50 | -10 | +10 |
| `krisp-bvc-tel` | 15 | -190 | -70 | -20 |
| `krisp-nc` | 15 | +20 | +30 | +30 |

Under `lk-inference`, Δ vs `none` per run (corrected rows re-scored in memory, never written):

| surface | candidate | scored | runs | Δ missed turns | Δ phantoms | onset lag median |
|---|---|---|---|---|---|---|
| phone | `aic-quail-l` (VF 2.1 / historical) | stored | 7 | +0.00 | +0.29 | -0.292 |
| phone | `aic-quail-l` (VF 2.1 / historical) | corrected | 7 | +0.00 | +0.29 | -0.077 |
| phone | `aic-vf-l` (VF 2.1 / historical) | stored | 3 | +0.00 | -1.33 | -0.124 |
| phone | `aic-vf-l` (VF 2.1 / historical) | corrected | 3 | +0.00 | -1.33 | -0.024 |
| phone | `aic-vf-s` (VF 2.1 / historical) | stored | 6 | +0.17 | -1.83 | -0.114 |
| phone | `aic-vf-s` (VF 2.1 / historical) | corrected | 6 | +0.00 | -2.00 | -0.018 |
| phone | `krisp-bvc` | stored | 8 | +0.12 | -1.62 | -0.053 |
| phone | `krisp-bvc` | corrected | 8 | +0.12 | -1.62 | -0.033 |
| phone | `krisp-bvc-tel` | stored | 8 | +0.00 | -1.75 | -0.099 |
| phone | `krisp-bvc-tel` | corrected | 8 | +0.00 | -1.75 | -0.023 |
| phone | `krisp-nc` | stored | 8 | +0.25 | -0.12 | -0.013 |
| phone | `krisp-nc` | corrected | 8 | +0.25 | -0.12 | -0.043 |
| web | `aic-quail-l` (VF 2.1 / historical) | stored | 7 | +0.00 | -0.14 | -0.142 |
| web | `aic-quail-l` (VF 2.1 / historical) | corrected | 7 | +0.00 | -0.14 | -0.017 |
| web | `aic-vf-l` (VF 2.1 / historical) | stored | 7 | +0.00 | -0.14 | -0.069 |
| web | `aic-vf-l` (VF 2.1 / historical) | corrected | 7 | +0.00 | -0.14 | +0.013 |
| web | `aic-vf-s` (VF 2.1 / historical) | stored | 7 | +0.00 | -0.14 | -0.069 |
| web | `aic-vf-s` (VF 2.1 / historical) | corrected | 7 | +0.00 | -0.14 | +0.013 |
| web | `krisp-bvc` | stored | 7 | +0.14 | -0.14 | -0.022 |
| web | `krisp-bvc` | corrected | 7 | +0.14 | -0.14 | -0.012 |
| web | `krisp-bvc-tel` | stored | 7 | +0.29 | -0.14 | -0.090 |
| web | `krisp-bvc-tel` | corrected | 7 | +0.29 | -0.14 | +0.046 |
| web | `krisp-nc` | stored | 7 | +0.00 | +0.14 | -0.003 |
| web | `krisp-nc` | corrected | 7 | +0.00 | +0.14 | -0.032 |

The correction moves onsets by up to 0.2 s (`aic-quail-l` on phone: −0.292 → −0.077 s). It
barely moves missed turns or phantoms.

## 4. VAD tables per surface

Every VAD at production's parameters, on production's gate (`hush`) and on `none`, pooled
over each surface's marked runs (section `vad`). Onset lag is segment start minus the marked
start: positive means late.

#### Phone — VADs on `hush`

| VAD | runs | turn recall | missed turns | phantom segs | merged | overreach s | onset lag (median) | frame agree |
|---|---|---|---|---|---|---|---|---|
| `aic-vad-vf-2.0` | 10 | 100.0% | 0 | 0 | 21 | 15.5 | -0.030 | 89.9% |
| `silero-v4.0-8k` | 10 | 100.0% | 0 | 0 | 4 | 4.2 | +0.018 | 91.7% |
| `firered-stream` | 10 | 98.6% | 1 | 1 | 6 | 4.5 | +0.015 | 92.0% |
| `marblenet-v2` | 10 | 98.6% | 1 | 1 | 32 | 14.9 | -0.076 | 91.4% |
| `firered-offline` | 10 | 97.3% | 2 | 0 | 6 | 6.6 | -0.014 | 92.7% |
| `lk-inference` | 10 | 100.0% | 0 | 3 | 12 | 9.8 | -0.017 | 91.5% |
| `silero-v6.2-16k` | 10 | 100.0% | 0 | 3 | 12 | 9.8 | -0.017 | 91.5% |
| `silero-v6.2.3-16k` | 10 | 100.0% | 0 | 3 | 12 | 9.8 | -0.017 | 91.5% |
| `silero-v5.1.2-8k` | 10 | 98.6% | 1 | 2 | 6 | 7.2 | -0.004 | 91.5% |
| `silero-v6.2-8k` | 10 | 98.6% | 1 | 2 | 13 | 10.3 | +0.000 | 88.9% |
| `silero-v6.2.3-8k` | 10 | 98.6% | 1 | 2 | 13 | 10.3 | +0.000 | 88.9% |
| `silero-v6.0-16k` | 10 | 97.3% | 2 | 1 | 6 | 4.9 | +0.009 | 90.3% |
| `earshot` | 10 | 97.3% | 2 | 2 | 6 | 5.6 | +0.005 | 92.2% |
| `silero-v5.1.2-16k` | 10 | 97.3% | 2 | 2 | 10 | 8.2 | -0.004 | 91.4% |
| `pyannote-seg3` | 10 | 98.6% | 1 | 4 | 20 | 12.6 | -0.077 | 90.9% |
| `silero-v6.0-8k` | 10 | 98.6% | 1 | 4 | 6 | 4.3 | +0.019 | 88.8% |
| `silero-v4.0-16k` | 10 | 100.0% | 0 | 6 | 18 | 23.7 | -0.037 | 84.8% |
| `webrtc-3` | 10 | 100.0% | 0 | 6 | 8 | 9.0 | -0.023 | 92.5% |
| `ten` | 10 | 95.9% | 3 | 4 | 10 | 7.8 | -0.002 | 84.8% |
| `aic-vad-ms-2.1` | 10 | 100.0% | 0 | 10 | 29 | 24.3 | -0.066 | 87.1% |
| `fsmn-8k` | 10 | 100.0% | 0 | 19 | 17 | 14.7 | -0.046 | 87.6% |
| `fsmn-16k` | 10 | 100.0% | 0 | 22 | 40 | 35.8 | -0.129 | 79.2% |

#### Phone — VADs on `none`

| VAD | runs | turn recall | missed turns | phantom segs | merged | overreach s | onset lag (median) | frame agree |
|---|---|---|---|---|---|---|---|---|
| `aic-vad-vf-2.0` | 10 | 100.0% | 0 | 0 | 24 | 17.1 | -0.046 | 90.5% |
| `pyannote-seg3` | 10 | 100.0% | 0 | 14 | 37 | 29.0 | -0.188 | 81.4% |
| `marblenet-v2` | 10 | 97.3% | 2 | 12 | 52 | 46.3 | -1.236 | 72.3% |
| `aic-vad-ms-2.1` | 10 | 100.0% | 0 | 17 | 47 | 47.1 | -0.699 | 73.5% |
| `silero-v6.2-8k` | 10 | 100.0% | 0 | 18 | 29 | 28.4 | -0.068 | 80.4% |
| `silero-v6.2.3-8k` | 10 | 100.0% | 0 | 18 | 29 | 28.4 | -0.068 | 80.4% |
| `silero-v5.1.2-8k` | 10 | 100.0% | 0 | 19 | 17 | 22.3 | -0.052 | 82.3% |
| `ten` | 10 | 100.0% | 0 | 21 | 23 | 24.1 | -0.052 | 79.8% |
| `fsmn-8k` | 10 | 100.0% | 0 | 22 | 29 | 32.4 | -0.318 | 77.5% |
| `lk-inference` | 10 | 100.0% | 0 | 22 | 29 | 26.2 | -0.086 | 77.8% |
| `silero-v6.2-16k` | 10 | 100.0% | 0 | 22 | 29 | 26.2 | -0.086 | 77.8% |
| `silero-v6.2.3-16k` | 10 | 100.0% | 0 | 22 | 29 | 26.2 | -0.086 | 77.8% |
| `silero-v5.1.2-16k` | 10 | 100.0% | 0 | 23 | 27 | 24.6 | -0.089 | 80.9% |
| `firered-offline` | 10 | 100.0% | 0 | 25 | 19 | 23.6 | -0.079 | 82.6% |
| `firered-stream` | 10 | 100.0% | 0 | 25 | 22 | 25.9 | -0.104 | 81.1% |
| `fsmn-16k` | 10 | 100.0% | 0 | 25 | 44 | 44.5 | -0.634 | 73.0% |
| `silero-v4.0-8k` | 10 | 100.0% | 0 | 26 | 21 | 24.0 | -0.069 | 82.4% |
| `earshot` | 10 | 100.0% | 0 | 27 | 25 | 28.4 | -0.094 | 80.7% |
| `silero-v4.0-16k` | 10 | 100.0% | 0 | 27 | 18 | 26.2 | -0.101 | 81.3% |
| `silero-v6.0-8k` | 10 | 98.6% | 1 | 27 | 15 | 17.2 | -0.031 | 81.8% |
| `silero-v6.0-16k` | 10 | 97.3% | 2 | 27 | 15 | 16.1 | -0.031 | 83.0% |
| `webrtc-3` | 10 | 100.0% | 0 | 36 | 27 | 28.6 | -0.130 | 80.0% |

#### Web — VADs on `none`

| VAD | runs | turn recall | missed turns | phantom segs | merged | overreach s | onset lag (median) | frame agree |
|---|---|---|---|---|---|---|---|---|
| `lk-inference` | 8 | 100.0% | 0 | 1 | 2 | 6.1 | -0.028 | 90.7% |
| `silero-v6.2-16k` | 8 | 100.0% | 0 | 1 | 2 | 6.1 | -0.028 | 90.7% |
| `silero-v6.2.3-16k` | 8 | 100.0% | 0 | 1 | 2 | 6.1 | -0.028 | 90.7% |
| `aic-vad-vf-2.0` | 8 | 100.0% | 0 | 2 | 2 | 8.0 | -0.037 | 90.8% |
| `fsmn-8k` | 8 | 100.0% | 0 | 2 | 0 | 1.7 | +0.145 | 77.1% |
| `silero-v5.1.2-16k` | 8 | 100.0% | 0 | 2 | 4 | 12.2 | -0.016 | 87.5% |
| `silero-v4.0-16k` | 8 | 96.8% | 1 | 1 | 19 | 21.3 | +0.019 | 80.5% |
| `silero-v4.0-8k` | 8 | 93.5% | 2 | 0 | 7 | 11.5 | +0.040 | 78.8% |
| `silero-v6.0-16k` | 8 | 100.0% | 0 | 3 | 4 | 11.1 | -0.051 | 90.0% |
| `silero-v6.2-8k` | 8 | 96.8% | 1 | 2 | 0 | 5.0 | +0.028 | 84.4% |

Web on `hush` loses 8–12 of 31 marked turns with every VAD. Hush at full strength eats web
speech (section `vad`, "Web — VADs on `hush`").

**Cost:** single-thread, idle Mac, `scripts/vad_cost.py` (best of 3, one pass over
`20260804-112511` `none`):

| VAD | cpu ms / audio s (Darwin arm64, 20260804-112511 none, best of 3) |
|---|---|
| `lk-inference` | 2.5 |
| `silero-v6.2.3-8k` | 2.5 |
| `silero-v6.2.3-16k` | 3.3 |
| `silero-v6.2-8k` | 2.5 |
| `silero-v6.2-16k` | 3.3 |
| `silero-v6.0-8k` | 2.5 |
| `silero-v6.0-16k` | 3.3 |
| `silero-v5.1.2-8k` | 2.5 |
| `silero-v5.1.2-16k` | 3.3 |
| `silero-v4.0-8k` | 3.1 |
| `silero-v4.0-16k` | 3.1 |
| `firered-stream` | 2.5 |
| `firered-offline` | 3.0 |
| `ten` | 13.3 |
| `earshot` | 0.4 |
| `marblenet-v2` | 0.5 |
| `fsmn-16k` | 2.3 |
| `fsmn-8k` | 1.3 |
| `pyannote-seg3` | 5.6 |
| `webrtc-3` | 0.1 |
| `aic-vad-ms-2.1` | 5.7 |
| `aic-vad-vf-2.0` | 8.0 |

**`20260804-112511`'s unmarked tail (23.8 s to the end) — flagged for a listen.**
Section `tail` shows that run alone, with the tail as stored and with it ignored. The pooled
phone table with it ignored is the last block of section `vad`, and it changes no ranking.

| gate → VAD | phantoms, tail as stored | phantoms, tail ignored | segments in the tail | missed turns |
|---|---|---|---|---|
| `none` → `lk-inference` | 0 | 0 | 0 | 0 |
| `none` → `silero-v4.0-8k` | 1 | 0 | 1 | 0 |
| `none` → `firered-stream` | 1 | 0 | 1 | 0 |
| `none` → `aic-vad-vf-2.0` | 0 | 0 | 0 | 0 |
| `none` → `earshot` | 1 | 0 | 1 | 0 |
| `none` → `ten` | 1 | 0 | 1 | 0 |
| `hush` → `lk-inference` | 0 | 0 | 0 | 0 |
| `hush` → `silero-v4.0-8k` | 0 | 0 | 0 | 0 |
| `hush` → `firered-stream` | 0 | 0 | 0 | 0 |
| `hush` → `aic-vad-vf-2.0` | 0 | 0 | 0 | 0 |
| `hush` → `earshot` | 0 | 0 | 0 | 0 |
| `hush` → `ten` | 0 | 0 | 0 | 2 |

## 5. Gate × VAD matrix

Pooled per surface. This is the miss–phantom trade-off and its Pareto front (section `matrix`).

#### Phone — best 20 (gate, VAD) by missed + phantom (10 runs)

| gate → VAD | turns | missed | phantom | merged | overreach s |
|---|---|---|---|---|---|
| `hush` → `silero-v4.0-8k` | 74 | 0 | 0 | 4 | 4.2 |
| `gtcrn+hush` → `silero-v4.0-8k` | 74 | 0 | 0 | 4 | 4.4 |
| `gtcrn+hush` → `firered-stream` | 74 | 0 | 0 | 6 | 4.1 |
| `dpdfnet2-8k+hush` → `aic-vad-vf-2.0` | 74 | 0 | 0 | 19 | 16.0 |
| `hush` → `aic-vad-vf-2.0` | 74 | 0 | 0 | 21 | 15.5 |
| `fastenhancer-t+hecttor-coda-vi` → `aic-vad-vf-2.0` | 74 | 0 | 0 | 21 | 15.5 |
| `dtln+hecttor-coda-vi` → `aic-vad-vf-2.0` | 74 | 0 | 0 | 22 | 16.7 |
| `gtcrn+hush` → `aic-vad-vf-2.0` | 74 | 0 | 0 | 23 | 16.2 |
| `hecttor-coda-vi` → `aic-vad-vf-2.0` | 74 | 0 | 0 | 23 | 16.5 |
| `krisp-viva` → `aic-vad-vf-2.0` | 74 | 0 | 0 | 23 | 17.5 |
| **production: `hush` → `lk-inference`** | 74 | 0 | 3 | 12 | 9.8 |

Pareto front (20 combos): `input`→`aic-vad-vf-2.0` (0/0), `none`→`aic-vad-vf-2.0` (0/0), `hecttor-coda-vi`→`aic-vad-vf-2.0` (0/0), `hecttor-coda-vi-8k`→`aic-vad-vf-2.0` (0/0), `hecttor-crest2`→`aic-vad-vf-2.0` (0/0), `dpdfnet2-8k`→`aic-vad-vf-2.0` (0/0), `dpdfnet-baseline`→`aic-vad-vf-2.0` (0/0), `gtcrn`→`aic-vad-vf-2.0` (0/0), `hush`→`silero-v4.0-8k` (0/0), `hush`→`aic-vad-vf-2.0` (0/0), `hush-atten12`→`aic-vad-vf-2.0` (0/0), `dpdfnet2-8k+hush`→`aic-vad-vf-2.0` (0/0), `fastenhancer-t+hecttor-coda-vi`→`aic-vad-vf-2.0` (0/0), `gtcrn+hush`→`silero-v4.0-8k` (0/0), `gtcrn+hush`→`firered-stream` (0/0)

#### Web — best 20 (gate, VAD) by missed + phantom (8 runs)

| gate → VAD | turns | missed | phantom | merged | overreach s |
|---|---|---|---|---|---|
| `fastenhancer-l` → `silero-v6.0-16k` | 31 | 0 | 0 | 0 | 1.6 |
| `fastenhancer-l` → `lk-inference` | 31 | 0 | 0 | 0 | 2.4 |
| `fastenhancer-l` → `silero-v6.2.3-16k` | 31 | 0 | 0 | 0 | 2.4 |
| `fastenhancer-l` → `silero-v6.2-16k` | 31 | 0 | 0 | 0 | 2.4 |
| `krisp-bvc-replay` → `silero-v6.2.3-8k` | 31 | 0 | 0 | 0 | 3.2 |
| `krisp-bvc-replay` → `silero-v6.2-8k` | 31 | 0 | 0 | 0 | 3.2 |
| `dtln` → `silero-v5.1.2-16k` | 31 | 0 | 0 | 0 | 4.9 |
| `dtln` → `silero-v6.0-16k` | 31 | 0 | 0 | 0 | 5.0 |
| **production: `hush` → `lk-inference`** | 31 | 12 | 1 | 0 | 2.2 |

Pareto front (54 combos): `hecttor-coda-vi-w075`→`silero-v6.0-8k` (0/0), `hecttor-coda-vi-w075`→`silero-v6.0-16k` (0/0), `hecttor-coda-vi-w075`→`silero-v5.1.2-16k` (0/0), `hecttor-coda`→`silero-v6.0-16k` (0/0), `dpdfnet8-8k`→`aic-vad-vf-2.0` (0/0), `dpdfnet-baseline`→`pyannote-seg3` (0/0), `dpdfnet-baseline`→`aic-vad-ms-2.1` (0/0), `fastenhancer-l`→`lk-inference` (0/0), `fastenhancer-l`→`silero-v6.2.3-16k` (0/0), `fastenhancer-l`→`silero-v6.2-16k` (0/0), `fastenhancer-l`→`silero-v6.0-16k` (0/0), `fastenhancer-l`→`pyannote-seg3` (0/0), `dtln`→`silero-v6.0-16k` (0/0), `dtln`→`silero-v5.1.2-16k` (0/0), `cloud-raw`→`silero-v4.0-16k` (0/0)

On phone, `aic-vad-vf-2.0` reaches 0 / 0 behind almost any gate, but only by fusing
turns: 19–26 merged turns and 15–18 s of overreach. That's one long STT request per
exchange, and it opens on both A sounds (§7). Silero v4.0 at 8 kHz reaches 0 / 0 with 4 merged
turns.

## 6. Parameter sweeps

The stored probabilities are re-segmented in memory (`scripts/sweep_round2.py`), and the cells
are stored as `meta["vad_sweep_round2"]`. The sweep covers:

- **VADs (6):** lk-inference, silero-v4.0-8k, firered-stream, aic-vad-vf-2.0, marblenet-v2
  and lkaic-flag.
- **Gates (9):** hush-aih (incident runs only), hush, gtcrn+hush, dpdfnet2-8k+hush,
  dpdfnet2-8k, krisp-viva, hush-atten12, none and lkaic-quail-l-bypass.
- **Grid:** activation ∈ {0.3…0.8} (deactivation equal), min speech ∈ {0.05, 0.10, 0.15,
  0.25}, min silence ∈ {0.3, 0.4, 0.55}, plus 08-04's 0.5/0.05/0.4.

The sets were picked by one combined score: pooled phone missed + phantom + every incident
failure. "Within +1" counts the grid points whose score is at most one worse than the best.
That's how flat the optimum is (section `sweep`).

#### Phone — best parameter set per (gate, VAD), pooled (act = deact; score = missed + phantom, then merged)

| gate → VAD | best act / min speech / min silence | missed | phantom | merged | grid points within +1 | at production 0.6/0.05/0.3 | at 08-04's 0.5/0.05/0.4 | incidents at the best set |
|---|---|---|---|---|---|---|---|---|
| `dpdfnet2-8k+hush` → `silero-v4.0-8k` | 0.3/0.05/0.55 | 0 | 0 | 2 | 46/72 | 1/0/4 | 1/1/12 | A 3/3 click no clunk no; B throat no; PR-42 21/21 +4/7 orphans 0 |
| `gtcrn+hush` → `firered-stream` | 0.6/0.05/0.55 | 0 | 0 | 2 | 40/72 | 0/0/6 | 0/0/4 | A 3/3 click OPEN clunk no; B throat no; PR-42 21/21 +6/7 orphans 0 |
| `gtcrn+hush` → `silero-v4.0-8k` | 0.6/0.05/0.55 | 0 | 0 | 2 | 33/72 | 0/0/4 | 0/0/8 | A 3/3 click no clunk no; B throat no; PR-42 21/21 +7/7 orphans 0 |
| `hush` → `firered-stream` | 0.5/0.05/0.55 | 0 | 0 | 2 | 31/72 | 1/1/6 | 0/0/6 | A 3/3 click OPEN clunk no; B throat OPEN; PR-42 21/21 +4/7 orphans 0 |
| `hush` → `silero-v4.0-8k` | 0.6/0.05/0.55 | 0 | 0 | 2 | 43/72 | 0/0/4 | 0/0/10 | A 3/3 click no clunk no; B throat no; PR-42 21/21 +4/7 orphans 0 |
| `gtcrn+hush` → `lk-inference` | 0.8/0.15/0.3 | 0 | 0 | 6 | 40/72 | 0/2/10 | 0/3/13 | A 3/3 click no clunk no; B throat no; PR-42 14/21 +1/7 orphans 0 |
| `hush` → `lk-inference` | 0.8/0.1/0.55 | 0 | 0 | 6 | 32/72 | 0/3/12 | 0/4/13 | A 3/3 click no clunk no; B throat no; PR-42 20/21 +2/7 orphans 0 |
| `hush-atten12` → `aic-vad-vf-2.0` | 0.7/0.05/0.55 | 0 | 0 | 6 | 29/72 | 0/0/24 | 0/3/23 | A 3/3 click OPEN clunk OPEN; B throat OPEN; PR-42 21/21 +7/7 orphans 0 |
| `none` → `aic-vad-vf-2.0` | 0.7/0.05/0.55 | 0 | 0 | 6 | 26/72 | 0/0/24 | 0/5/21 | A 3/3 click OPEN clunk OPEN; B throat OPEN; PR-42 21/21 +6/7 orphans 1 |
| `dpdfnet2-8k` → `aic-vad-vf-2.0` | 0.7/0.05/0.55 | 0 | 0 | 8 | 39/72 | 0/0/23 | 0/4/27 | A 3/3 click OPEN clunk OPEN; B throat OPEN; PR-42 21/21 +5/7 orphans 0 |
| `dpdfnet2-8k+hush` → `lk-inference` | 0.7/0.15/0.55 | 0 | 0 | 8 | 40/72 | 0/3/11 | 0/3/13 | A 3/3 click no clunk no; B throat no; PR-42 17/21 +3/7 orphans 0 |
| `krisp-viva` → `aic-vad-vf-2.0` | 0.7/0.05/0.55 | 0 | 0 | 8 | 33/72 | 0/0/23 | 0/2/22 | A 3/3 click OPEN clunk OPEN; B throat no; PR-42 21/21 +7/7 orphans 0 |
| `dpdfnet2-8k+hush` → `aic-vad-vf-2.0` | 0.6/0.05/0.55 | 0 | 0 | 12 | 41/72 | 0/0/19 | 0/0/22 | A 3/3 click OPEN clunk OPEN; B throat no; PR-42 21/21 +3/7 orphans 0 |
| `gtcrn+hush` → `aic-vad-vf-2.0` | 0.6/0.05/0.55 | 0 | 0 | 15 | 43/72 | 0/0/23 | 0/0/24 | A 3/3 click OPEN clunk OPEN; B throat no; PR-42 21/21 +7/7 orphans 0 |
| `lkaic-quail-l-bypass` → `aic-vad-vf-2.0` | 0.6/0.25/0.55 | 0 | 0 | 15 | 17/72 | 0/2/24 | 0/6/22 | A 3/3 click no clunk no; B throat no; PR-42 21/21 +7/7 orphans 0 |
| `hush` → `aic-vad-vf-2.0` | 0.6/0.05/0.55 | 0 | 0 | 17 | 30/72 | 0/0/23 | 0/2/23 | A 3/3 click OPEN clunk OPEN; B throat OPEN; PR-42 21/21 +5/7 orphans 0 |

- **The optimum is flat for the Hush + Silero v4.0 8 kHz combos.** 33–46 of the 72 grid
  points are within one error of the best, and production's own 0.6/0.05/0.3 already scores
  0 / 0 behind `hush` and `gtcrn+hush`.
- **Production's VAD can't be tuned into it.** `hush → lk-inference`'s best set (0.8 / 0.10 /
  0.55) stops the A click, but still catches only 2/7 of PR-42's misses, and on the
  production Hush path B's throat-clear still opens.
- **min_silence 0.55 wins almost every row, partly as a scoring artifact.** Truth turns are
  coalesced at each row's own `min_silence`. Its real effect is fewer sentence splits, which
  helps VAD-cut WER (§3.3). Its cost is 0.25 s more before every STT request.
- **The "B throat" column above is scored on the bench's gate audio.** On `hush`, production's own
  VAD doesn't open on B either, so a "no" there doesn't discriminate. §7 has the
  production-path (`hush-aih`) rows.

## 7. Incidents

**How each span was derived.** A local import script holds the code (kept out of this repository: it reads the
private call recordings); each run's
`meta["incident"]` holds the provenance.

- **PR-54 A** (`20260929-123329`).
  - **Input:** the 24 kHz decode of the recorded caller track that PR-54 used. The three STT WAVs
    sit in it at ncc 0.94–0.997.
  - **Why not the VAD log:** production's VAD event times run about 0.4 s behind the audio (the
    click's audio is at 38.43 s, and prod logged 38.83 s). So every edge was read off a 20 ms
    level and pitch profile of the audio.
  - **Speech:** three short caller utterances.
  - **Known non-speech:** the click (38.42–39.07) and the hang-up clunk.
  - **Ignore:** the quiet background voice. Also a short voiced burst at 44.6 s, which is a
    **judgement call**: it has voiced frames in the caller's own F0 range, and Qwen reads words from it.
- **PR-54 B** (`-1`).
  - **Input:** the caller track with the production agent's own STT WAV of the phantom spliced in at 17.956 s
    (ncc 0.990). The track is 15.4 dB quieter than what the agent heard there, and without the splice
    production's VAD never opens.
  - **Labels:** the throat-clear is known non-speech. B has **no real caller speech**, so it has
    no truth spans. Its columns are read straight from the segments.
- **PR-42** (`-2`).
  - **Input:** the 16 kHz decode of the recorded caller track. It was copied from the PR-42
    investigation's working files into `data/incidents/pr42/src/`.
  - **Labels:** xref.py's heuristic with its paths fixed, which reproduces 40 sounds
    (21 / 7 / 12). The 21 detected + 7 loud-missed sounds are speech; the 12 quiet/echo
    sounds are ignore.
- **`hush-aih`.** The bench's `hush` resamples to 16 kHz. Production feeds its own Hush wrapper
  24 kHz 50 ms frames. `hush-aih` is PR-54's own replay of that path: the same library
  (sha 4bcd3863…) at atten 100, stored at +20 ms / +0 ms against the input.
  - **Sanity check:** production's `lk-inference` on `hush-aih` opens on A's click and B's
    throat-clear, as production did. On the bench's `hush` it doesn't open on B.
  - **B's alignment:** B's `hush-aih` failed only the prominence check (0.125 < 0.15), so it
    was attached as a judgement call (`import_check.override`).
- **The B column.** B's throat-clear is scored "rejects" only on sources where production's
  own VAD opens on it; elsewhere it reads "n/a".
- **"A phantoms excl. click/clunk"** counts phantom segments anywhere else in A.
- **PR-42's timeline.** Its replay sources carry up to 144 ms of control drift. A miss/hit call
  on a sound shorter than about 0.3 s is timeline-sensitive on those rows.

Selected combos at production's parameters (section `incidents`):

| gate → VAD | A turns kept | A click | A clunk | A phantoms excl. click/clunk | A on ignore | B throat-clear | B segs outside greeting | B segs | PR-42 detected kept | PR-42 missed caught | PR-42 on quiet/echo | PR-42 orphans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hush-aih` → `lk-inference` | 3/3 | OPEN | no | 0 | 0 | OPEN | 1 | 1 | 21/21 | 4/7 | 2 | 0 |
| `hush-aih` → `silero-v4.0-8k` | 3/3 | no | no | 0 | 0 | rejects | 0 | 0 | 21/21 | 6/7 | 4 | 1 |
| `hush-aih` → `firered-stream` | 3/3 | OPEN | OPEN | 0 | 0 | OPEN | 1 | 1 | 21/21 | 4/7 | 1 | 0 |
| `hush-aih` → `aic-vad-vf-2.0` | 3/3 | OPEN | OPEN | 0 | 1 | OPEN | 1 | 1 | 21/21 | 5/7 | 3 | 0 |
| `hush-aih` → `marblenet-v2` | 2/3 | no | no | 0 | 0 | rejects | 0 | 0 | 17/21 | 3/7 | 4 | 3 |
| `hush` → `lk-inference` | 3/3 | OPEN | no | 0 | 0 | n/a: prod VAD silent here | 0 | 0 | 20/21 | 2/7 | 1 | 0 |
| `hush` → `silero-v4.0-8k` | 3/3 | no | no | 0 | 0 | n/a: prod VAD silent here | 0 | 0 | 21/21 | 4/7 | 1 | 0 |
| `hush` → `firered-stream` | 3/3 | OPEN | no | 0 | 0 | n/a: prod VAD silent here | 0 | 0 | 21/21 | 4/7 | 3 | 0 |
| `hush` → `aic-vad-vf-2.0` | 3/3 | OPEN | OPEN | 0 | 1 | OPEN | 1 | 1 | 21/21 | 5/7 | 3 | 0 |
| `hush` → `marblenet-v2` | 3/3 | no | no | 0 | 0 | n/a: prod VAD silent here | 0 | 0 | 21/21 | 5/7 | 3 | 8 |
| `gtcrn+hush` → `lk-inference` | 3/3 | OPEN | no | 0 | 1 | n/a: prod VAD silent here | 0 | 0 | 17/21 | 2/7 | 1 | 0 |
| `gtcrn+hush` → `silero-v4.0-8k` | 3/3 | no | no | 0 | 0 | n/a: prod VAD silent here | 0 | 0 | 21/21 | 7/7 | 0 | 0 |
| `gtcrn+hush` → `firered-stream` | 3/3 | OPEN | no | 0 | 0 | n/a: prod VAD silent here | 0 | 0 | 21/21 | 6/7 | 1 | 0 |
| `gtcrn+hush` → `aic-vad-vf-2.0` | 3/3 | OPEN | OPEN | 0 | 1 | n/a: prod VAD silent here | 0 | 0 | 21/21 | 7/7 | 2 | 0 |
| `gtcrn+hush` → `marblenet-v2` | 3/3 | no | no | 0 | 0 | n/a: prod VAD silent here | 0 | 0 | 21/21 | 7/7 | 3 | 3 |
| `hush-atten12` → `lk-inference` | 3/3 | OPEN | no | 0 | 1 | OPEN | 1 | 1 | 18/21 | 2/7 | 1 | 0 |
| `hush-atten12` → `silero-v4.0-8k` | 3/3 | no | no | 0 | 0 | rejects | 0 | 0 | 21/21 | 7/7 | 1 | 1 |
| `hush-atten12` → `firered-stream` | 3/3 | OPEN | no | 0 | 0 | OPEN | 1 | 1 | 21/21 | 7/7 | 6 | 1 |
| `hush-atten12` → `aic-vad-vf-2.0` | 3/3 | OPEN | OPEN | 0 | 1 | OPEN | 2 | 2 | 21/21 | 7/7 | 3 | 0 |
| `hush-atten12` → `marblenet-v2` | 3/3 | no | no | 0 | 1 | OPEN | 1 | 1 | 21/21 | 7/7 | 4 | 6 |
| `none` → `lk-inference` | 3/3 | OPEN | no | 0 | 0 | OPEN | 1 | 1 | 14/21 | 1/7 | 0 | 0 |
| `none` → `silero-v4.0-8k` | 3/3 | no | no | 0 | 0 | OPEN | 1 | 1 | 21/21 | 7/7 | 6 | 1 |
| `none` → `firered-stream` | 3/3 | OPEN | no | 0 | 0 | OPEN | 1 | 1 | 21/21 | 7/7 | 12 | 4 |
| `none` → `aic-vad-vf-2.0` | 3/3 | OPEN | OPEN | 0 | 1 | OPEN | 2 | 3 | 21/21 | 7/7 | 5 | 5 |
| `none` → `marblenet-v2` | 3/3 | OPEN | no | 0 | 1 | rejects | 0 | 1 | 21/21 | 6/7 | 4 | 6 |
| `lkaic-quail-l-bypass` → `lkaic-flag` | 3/3 | OPEN | OPEN | 0 | 1 | rejects | 0 | 0 | 21/21 | 7/7 | 3 | 2 |
| `input` → `lk-inference` | 3/3 | OPEN | no | 0 | 0 | OPEN | 1 | 1 | 14/21 | 1/7 | 0 | 0 |

- **PR-54 A, the click:**
  - Production (`hush-aih → lk-inference`) opens on it.
  - So do FireRed-stream and aic-vad-vf-2.0.
  - **Silero v4.0 at 8 kHz rejects it on every source tested, and keeps all 3 real turns.**
- **PR-54 B, the throat-clear:**
  - Production opens on it on `hush-aih`.
  - **Silero v4.0 8 kHz rejects it there**, and so does MarbleNet, which loses a real A turn
    and 4 of PR-42's detected ones.
  - On raw audio production's VAD opens too. So do most others, except the Silero v5/v6.0
    and 8 kHz v6.2 variants, MarbleNet and FSMN-8k.
- **PR-42:**
  - Production on the production Hush path catches 4/7, and on the bench's `hush` 2/7.
  - `hush-aih → silero-v4.0-8k` catches 6/7 with 21/21 kept and 1 orphan.
  - Behind `gtcrn+hush`, v4.0 catches 7/7 with no extra trigger. That chain ran on the bench's
    16 kHz Hush path, where B doesn't discriminate.

## 8. Recommendation for PR-54 and PR-42

**The one best choice: phone gate Hush@100, as production already runs it, with VAD =
Silero v4.0 ONNX at 8 kHz, activation 0.6 / deactivation 0.6 / min speech 0.05 s / min
silence 0.55 s, and STT reading raw (unchanged).**

How min silence was picked: take the smallest value in {0.3, 0.4, 0.55} whose VAD-cut WER
doesn't lose to production (median ΔWER ≤ 0, W ≥ L, no loops), with every incident fixed. Only
0.55 qualifies (section `minsil`; phone, STT reads raw):

| min silence | runs | median ΔWER | mean | W / L | worst | loops | requests per run | added latency vs production | phone pool missed/phantom | incidents on hush-aih |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 9 | +0.000 | +0.110 | 3 / 4 | +0.828 | 1 | 8.8 (prod 8.1) | +0.00 s | 0/0 | A 3/3 click no clunk no; B throat no; PR-42 21/21 +6/7 orphans 1 |
| 0.4 | 9 | +0.027 | +0.105 | 2 / 5 | +0.793 | 1 | 7.8 (prod 8.1) | +0.10 s | 0/0 | A 3/3 click no clunk no; B throat no; PR-42 21/21 +6/7 orphans 1 |
| 0.55 | 9 | -0.031 | -0.018 | 5 / 3 | +0.345 | 0 | 6.0 (prod 8.1) | +0.25 s | 0/0 | A 3/3 click no clunk no; B throat no; PR-42 21/21 +6/7 orphans 1 |

- **Why:**
  - Of the 22 VADs on production's own Hush path (`hush-aih`), only Silero v4.0 rejects both
    PR-54 sounds while keeping every real A turn. The 8 kHz variant beats the 16 kHz one on
    PR-42 (6/7 vs 5/7) and on the phone pool (0 phantoms vs 6).
  - It lifts PR-42 from 4/7 to 6/7 missed sounds caught, with all 21 detected utterances kept.
  - On the 10 marked phone runs it has 0 missed turns and 0 phantoms, against production's 0 and 3.
  - Its incident outcomes and phone-pool counts are the same at all three min silence values.
  - At 0.55 it's better than production on VAD-cut WER (median −0.031, 5W / 3L, no loops).
- **No LiveKit Cloud and no licence:** Silero v4.0 is MIT, Hush is Apache-2.0, and it runs
  locally at 3.1 ms of CPU per audio second.
- **The catch:**
  - **+0.25 s per response.** The gated STT waits out min silence before it sends each turn.
    Production's 0.3 fixes the incidents just the same, but it costs VAD-cut WER (mean +0.110,
    3W / 4L, one loop). 0.4 is worse than both.
  - **Engineering.** The production agent has to load a Silero v4.0 ONNX with the older `h`/`c` state API at
    8 kHz, in place of `livekit.agents.inference.VAD`. livekit-plugins-silero 1.8.3 expects the
    v5 state API, so this is a small custom VAD, not a config change.
  - **It only works behind Hush.** It doesn't replace the gate: on raw audio it's slightly worse
    than production (phone pool: 26 phantoms vs 22 on `none`, section `vad`).
  - **PR-42 isn't free.** 1 orphan trigger and 4 hits on quiet/echo sounds, against
    production's 0 and 2. The seventh missed sound is still missed.
  - **Evidence and web.** v4.0 is a 2022 model, and every result here is 10 phone runs plus one
    click, one throat-clear and one PR-42 call. On web it doesn't help, because full-strength
    Hush is the web problem (§4). Keep the capped web gate.
- **Runner-up: GTCRN → Hush → Silero v4.0 8 kHz at 0.55.**
  - It catches **7/7** on PR-42 with no extra trigger, and has 0 / 0 on the phone pool.
  - VAD-cut WER at 0.55: median −0.027, 5W / 2L, no loops.
  - It adds a second NC model to the gate (GTCRN, MIT, 23.7 k parameters, 16 ms).
  - Its incident results come from the bench's 16 kHz Hush path, where B can't discriminate.
    Check it on the production path before choosing it over the first.
- **Doesn't fix either incident:**
  - Production's Silero v6.2 with min speech 0.10 is WER-safe (4W / 0L). But on the production
    Hush path it still opens on A's click and B's throat-clear, and catches only 2/7 on PR-42.
  - The LiveKit Cloud ai-coustics flag (best VAD-cut WER) opens on the click. That's the
    Quail-L flag on raw audio. The VF 2.2 flag is in §8.1.

### 8.1 The Cloud ai-coustics flag against the standalone ai-coustics VADs, 1:1

Same audio, same settings. Two Cloud sources:

- `lkaic-quail-l-bypass`: raw audio through Cloud, with Quail-L's flag.
- `lkaic-vf22-l`: the VF 2.2 enhancer's output, with its own flag.

On each source there are five VADs: the Cloud flag, the standalone VF 2.0 and MS 2.1 VADs, and
two references, Silero v4.0 8 kHz and production's VAD. They run at act = deact 0.6 × min speech
{0.05, 0.10, 0.15, 0.25} × min silence {0.3, 0.4, 0.55}, pad 0.5. The flag is binary, so the
activation doesn't change it (it's segmented at 0.5 either way). Flag coverage is 100% on both
sources. Everything was re-segmented and transcribed on the Linux box from the stored
probabilities, and the 288 cells that already existed came out identical (section `aic-1to1`).

**Which is better at equal settings: it depends on the source.**

- **On raw audio through Cloud, the standalone VF 2.0 VAD wins.**
  - At min speech 0.25 / min silence 0.55 it rejects A's click, the clunk and B's
    throat-clear. On PR-42 it keeps 21/21 and catches 7/7 with no orphan. The phone pool is
    0 / 0.
  - The Quail-L flag opens on the click at all 12 points.
  - Their VAD-cut WER medians tie at −0.081. The flag has mean −0.076 and 6W / 2L; VF 2.0 has
    mean −0.057 and 5W / 2L.
- **On VF 2.2's output, the Cloud flag wins on incidents, and VF 2.0 wins on WER.**
  - At 0.25 / 0.55 the flag is the only ai-coustics row with no incident failure and 0 / 0 on
    the pool. VF 2.0 adds one phantom on A and one PR-42 orphan.
  - VF 2.0's WER there is better: median −0.069 and 8W / 0L, against the flag's −0.031 and
    6W / 2L.
- **MS 2.1 loses on both sources.** B's throat-clear opens at every point.
- **Both need min speech 0.25.** At 0.05–0.15 the flag and VF 2.0 open on the click on both
  sources.
  - 0.25 lost none of the 3 marked phone turns under 0.4 s. Silero v4.0 lost 1–2 of them.
  - But 3 turns is a small sample.
  - At 0.25, START fires 0.2 s later than at 0.05.

**The recommendation doesn't move:**

- **VF 2.0's click rejection only holds on the Cloud rail.** At the same 0.25 / 0.55, on the
  stored input (`none`) and behind the bench's Hush, it opens on the click (table "Off the Cloud
  rail").
- **The VF 2.2 flag at 0.25 / 0.55 fixes every incident here**, and its WER is on par with the
  recommendation's: median −0.031 for both, mean −0.053 vs −0.018, 6W / 2L vs 5W / 3L. But:
  - it needs LiveKit Cloud and ai-coustics in the call path;
  - its incidents were measured after the replay's second Opus pass, not on the production
    path;
  - it needs min speech 0.25.
  
  If running on Cloud becomes an option, it's the one Cloud option worth a production-path test.

#### Best point per source and VAD

"Incident failures" adds up every wrong call on the three incidents: A turns lost, A click / clunk / other phantoms opened, B's throat-clear opened, PR-42 detected utterances lost, loud-missed sounds not caught, and orphan triggers. The best point has the fewest, then the fewest pool missed + phantom, then the fewest merged, then the one nearest production's 0.05 / 0.3. The phone pool is missed / phantom / merged over the 10 marked phone runs. VAD-cut WER is Δ vs production (median / mean / W-L / worst / loops, 9 phone runs, STT reads raw, one request per segment). Where production's own VAD, on the same source, doesn't open on B's throat-clear, B reads n/a.

| source | VAD | best min speech / min silence | pool m/p/merged | PR-54 A | PR-54 B | PR-42 det / missed caught / orphans | incident failures | VAD-cut WER | flag coverage |
|---|---|---|---|---|---|---|---|---|---|
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.25 / 0.55 | 0/1/24 | 3/3 click OPEN clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | -0.081 / -0.076 / 6-2 / +0.104 / 0 | 100.00% |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.25 / 0.55 | 0/0/15 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | -0.081 / -0.057 / 5-2 / +0.173 / 0 |  |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.25 / 0.55 | 0/8/42 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +5/7 orphans 0 | 4 | -0.054 / +0.387 / 7-1 / +4.272 / 0 |  |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.15 / 0.55 | 1/12/26 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 0 | 1 | +0.054 / +0.565 / 4-5 / +5.000 / 0 |  |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.05 / 0.55 | 0/13/27 | 3/3 click OPEN clunk no | OPEN | 15/21 +1/7 orphans 0 | 14 | +0.031 / +0.621 / 2-5 / +5.363 / 0 |  |
| `lkaic-vf22-l` | `lkaic-flag` | 0.25 / 0.55 | 0/0/15 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | -0.031 / -0.053 / 6-2 / +0.207 / 0 | 100.00% |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.25 / 0.55 | 0/1/13 | 3/3 click no clunk no +1 phantom | rejects | 21/21 +7/7 orphans 1 | 2 | -0.069 / -0.095 / 8-0 / +0.000 / 0 |  |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.25 / 0.55 | 1/2/24 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 3 | 5 | -0.027 / +0.025 / 5-3 / +0.727 / 0 |  |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.05 / 0.55 | 1/2/5 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | -0.034 / +0.043 / 5-3 / +0.727 / 0 |  |
| `lkaic-vf22-l` | `lk-inference` | 0.05 / 0.55 | 0/2/11 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | +0.000 / +0.071 / 4-4 / +0.727 / 0 |  |

#### Off the Cloud rail

The standalone VF 2.0 VAD on the stored input (`none`) and behind the bench's 16 kHz Hush, with no Cloud replay pass, at min speech 0.25 / min silence 0.55, plus its best point where that differs:

| source | VAD | point | pool m/p/merged | PR-54 A | PR-54 B | PR-42 | fails |
|---|---|---|---|---|---|---|---|
| `none` | `aic-vad-vf-2.0` | 0.25 / 0.55 | 0/0/13 | 3/3 click OPEN clunk no | rejects | 21/21 +7/7 orphans 0 | 1 |
| `hush` | `aic-vad-vf-2.0` | 0.25 / 0.55 | 0/0/17 | 3/3 click OPEN clunk no | OPEN | 21/21 +5/7 orphans 0 | 4 |

#### Short-turn check

Marked phone turns under 0.4 s (after coalescing at that min silence) missed at min speech 0.05 → 0.25:

| source | VAD | min silence 0.3 | 0.4 | 0.55 |
|---|---|---|---|---|
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0 → 0 of 3 | 0 → 0 of 3 | 0 → 0 of 1 |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0 → 0 of 3 | 0 → 0 of 3 | 0 → 0 of 1 |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0 → 0 of 3 | 0 → 0 of 3 | 0 → 0 of 1 |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0 → 1 of 3 | 0 → 1 of 3 | 0 → 0 of 1 |
| `lkaic-quail-l-bypass` | `lk-inference` | 0 → 0 of 3 | 0 → 0 of 3 | 0 → 0 of 1 |
| `lkaic-vf22-l` | `lkaic-flag` | 0 → 0 of 3 | 0 → 0 of 3 | 0 → 0 of 1 |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0 → 0 of 3 | 0 → 0 of 3 | 0 → 0 of 1 |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0 → 0 of 3 | 0 → 0 of 3 | 0 → 0 of 1 |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0 → 2 of 3 | 0 → 2 of 3 | 0 → 1 of 1 |
| `lkaic-vf22-l` | `lk-inference` | 0 → 0 of 3 | 0 → 0 of 3 | 0 → 0 of 1 |

<details><summary>Every point (section <code>aic-1to1</code>)</summary>

#### Every point

Act = deact 0.6, prefix pad 0.5. "Short missed" counts marked phone turns under 0.4 s that no segment touched. VAD-cut WER was run at min speech {0.05, 0.25} × every min silence on `lkaic-quail-l-bypass` (plus Silero v4.0's best point there), and at each VAD's best point on `lkaic-vf22-l`.

| source | VAD | min speech | min silence | pool m/p/merged | short missed | PR-54 A | PR-54 B | PR-42 | fails | VAD-cut WER |
|---|---|---|---|---|---|---|---|---|---|---|
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.05 | 0.3 | 1/5/23 | 0 | 3/3 click OPEN clunk OPEN | rejects | 21/21 +7/7 orphans 2 | 4 | +0.000 / +0.001 / 4-3 / +0.276 / 0 |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.05 | 0.4 | 1/5/24 | 0 | 3/3 click OPEN clunk OPEN | rejects | 21/21 +7/7 orphans 2 | 4 | -0.081 / -0.028 / 5-3 / +0.187 / 0 |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.05 | 0.55 | 0/2/24 | 0 | 3/3 click OPEN clunk OPEN | rejects | 21/21 +7/7 orphans 2 | 4 | -0.081 / -0.061 / 6-2 / +0.138 / 0 |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.1 | 0.3 | 1/4/23 | 0 | 3/3 click OPEN clunk OPEN | rejects | 21/21 +7/7 orphans 2 | 4 | — |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.1 | 0.4 | 1/4/24 | 0 | 3/3 click OPEN clunk OPEN | rejects | 21/21 +7/7 orphans 2 | 4 | — |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.1 | 0.55 | 0/2/24 | 0 | 3/3 click OPEN clunk OPEN | rejects | 21/21 +7/7 orphans 2 | 4 | — |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.15 | 0.3 | 1/3/23 | 0 | 3/3 click OPEN clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | — |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.15 | 0.4 | 1/3/24 | 0 | 3/3 click OPEN clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | — |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.15 | 0.55 | 0/2/24 | 0 | 3/3 click OPEN clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | — |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.25 | 0.3 | 1/2/23 | 0 | 3/3 click OPEN clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | -0.054 / -0.016 / 5-3 / +0.276 / 0 |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.25 | 0.4 | 1/2/24 | 0 | 3/3 click OPEN clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | -0.062 / -0.038 / 5-3 / +0.173 / 0 |
| `lkaic-quail-l-bypass` | `lkaic-flag` | 0.25 | 0.55 | 0/1/24 | 0 | 3/3 click OPEN clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | -0.081 / -0.076 / 6-2 / +0.104 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.05 | 0.3 | 0/2/24 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 4 | 7 | -0.054 / +0.059 / 5-2 / +0.828 / 1 |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.05 | 0.4 | 0/2/18 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 4 | 7 | -0.054 / -0.058 / 5-3 / +0.093 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.05 | 0.55 | 0/1/15 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 3 | 6 | -0.081 / -0.057 / 5-2 / +0.173 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.1 | 0.3 | 0/2/24 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +6/7 orphans 2 | 6 | — |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.1 | 0.4 | 0/2/18 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +6/7 orphans 2 | 6 | — |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.1 | 0.55 | 0/1/15 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 2 | 5 | — |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.15 | 0.3 | 0/2/24 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +6/7 orphans 1 | 5 | — |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.15 | 0.4 | 0/2/18 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +6/7 orphans 1 | 5 | — |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.15 | 0.55 | 0/1/15 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.25 | 0.3 | 0/1/24 | 0 | 3/3 click no clunk no | rejects | 21/21 +6/7 orphans 0 | 1 | -0.054 / +0.059 / 5-2 / +0.828 / 1 |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.25 | 0.4 | 0/1/18 | 0 | 3/3 click no clunk no | rejects | 21/21 +6/7 orphans 0 | 1 | -0.054 / -0.055 / 5-3 / +0.093 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-vf-2.0` | 0.25 | 0.55 | 0/0/15 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | -0.081 / -0.057 / 5-2 / +0.173 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.05 | 0.3 | 0/19/45 | 0 | 3/3 click no clunk OPEN +2 phantom | OPEN | 21/21 +6/7 orphans 4 | 9 | +0.000 / +0.520 / 4-3 / +5.000 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.05 | 0.4 | 0/13/47 | 0 | 3/3 click no clunk OPEN +2 phantom | OPEN | 21/21 +6/7 orphans 4 | 9 | -0.027 / +0.429 / 5-3 / +4.272 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.05 | 0.55 | 0/10/42 | 0 | 3/3 click no clunk OPEN +2 phantom | OPEN | 21/21 +6/7 orphans 4 | 9 | -0.059 / +0.787 / 8-1 / +7.909 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.1 | 0.3 | 0/18/45 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +6/7 orphans 2 | 5 | — |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.1 | 0.4 | 0/12/47 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +6/7 orphans 2 | 5 | — |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.1 | 0.55 | 0/10/42 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +6/7 orphans 2 | 5 | — |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.15 | 0.3 | 0/18/45 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +5/7 orphans 2 | 6 | — |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.15 | 0.4 | 0/12/47 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +5/7 orphans 2 | 6 | — |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.15 | 0.55 | 0/10/42 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +5/7 orphans 2 | 6 | — |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.25 | 0.3 | 0/15/45 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +5/7 orphans 0 | 4 | -0.048 / +0.520 / 6-2 / +5.182 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.25 | 0.4 | 0/10/47 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +5/7 orphans 0 | 4 | -0.031 / +0.425 / 6-2 / +4.454 / 0 |
| `lkaic-quail-l-bypass` | `aic-vad-ms-2.1` | 0.25 | 0.55 | 0/8/42 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +5/7 orphans 0 | 4 | -0.054 / +0.387 / 7-1 / +4.272 / 0 |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.05 | 0.3 | 0/27/21 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 1 | 2 | +0.118 / +0.530 / 2-7 / +4.182 / 0 |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.05 | 0.4 | 0/23/23 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 1 | 2 | +0.108 / +0.736 / 2-6 / +6.182 / 1 |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.05 | 0.55 | 0/15/26 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 1 | 2 | +0.054 / +0.545 / 4-5 / +4.636 / 0 |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.1 | 0.3 | 1/21/21 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 1 | 2 | — |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.1 | 0.4 | 1/19/23 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 1 | 2 | — |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.1 | 0.55 | 1/13/26 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 1 | 2 | — |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.15 | 0.3 | 1/17/21 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 0 | 1 | — |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.15 | 0.4 | 1/16/23 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 0 | 1 | — |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.15 | 0.55 | 1/12/26 | 0 | 3/3 click no clunk no | OPEN | 21/21 +7/7 orphans 0 | 1 | +0.054 / +0.565 / 4-5 / +5.000 / 0 |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.25 | 0.3 | 3/13/21 | 1 | 2/3 click no clunk no | OPEN | 21/21 +6/7 orphans 0 | 3 | +0.027 / +0.354 / 2-5 / +3.272 / 0 |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.25 | 0.4 | 2/12/23 | 1 | 2/3 click no clunk no | OPEN | 21/21 +6/7 orphans 0 | 3 | +0.000 / +0.649 / 2-4 / +5.727 / 1 |
| `lkaic-quail-l-bypass` | `silero-v4.0-8k` | 0.25 | 0.55 | 1/10/26 | 0 | 2/3 click no clunk no | OPEN | 21/21 +7/7 orphans 0 | 2 | -0.029 / +0.541 / 6-3 / +5.000 / 0 |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.05 | 0.3 | 0/25/29 | 0 | 3/3 click OPEN clunk no | OPEN | 15/21 +1/7 orphans 0 | 14 | +0.125 / +0.632 / 1-7 / +5.091 / 0 |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.05 | 0.4 | 0/19/27 | 0 | 3/3 click OPEN clunk no | OPEN | 15/21 +1/7 orphans 0 | 14 | +0.108 / +0.624 / 1-7 / +5.091 / 0 |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.05 | 0.55 | 0/13/27 | 0 | 3/3 click OPEN clunk no | OPEN | 15/21 +1/7 orphans 0 | 14 | +0.031 / +0.621 / 2-5 / +5.363 / 0 |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.1 | 0.3 | 0/23/29 | 0 | 3/3 click no clunk no | OPEN | 10/21 +1/7 orphans 0 | 18 | — |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.1 | 0.4 | 0/18/27 | 0 | 3/3 click no clunk no | OPEN | 10/21 +1/7 orphans 0 | 18 | — |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.1 | 0.55 | 0/13/27 | 0 | 3/3 click no clunk no | OPEN | 10/21 +1/7 orphans 0 | 18 | — |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.15 | 0.3 | 0/19/29 | 0 | 3/3 click no clunk no | OPEN | 8/21 +1/7 orphans 0 | 20 | — |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.15 | 0.4 | 0/15/27 | 0 | 3/3 click no clunk no | OPEN | 8/21 +1/7 orphans 0 | 20 | — |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.15 | 0.55 | 0/10/27 | 0 | 3/3 click no clunk no | OPEN | 8/21 +1/7 orphans 0 | 20 | — |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.25 | 0.3 | 0/14/29 | 0 | 3/3 click no clunk no | OPEN | 7/21 +0/7 orphans 0 | 22 | +0.000 / +0.520 / 4-4 / +4.909 / 0 |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.25 | 0.4 | 0/12/27 | 0 | 3/3 click no clunk no | OPEN | 7/21 +0/7 orphans 0 | 22 | -0.027 / +0.541 / 5-4 / +5.091 / 0 |
| `lkaic-quail-l-bypass` | `lk-inference` | 0.25 | 0.55 | 0/8/27 | 0 | 3/3 click no clunk no | OPEN | 7/21 +0/7 orphans 0 | 22 | -0.027 / +0.555 / 5-4 / +5.454 / 0 |
| `lkaic-vf22-l` | `lkaic-flag` | 0.05 | 0.3 | 0/2/15 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.05 | 0.4 | 0/1/19 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.05 | 0.55 | 0/1/15 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.1 | 0.3 | 0/2/15 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.1 | 0.4 | 0/1/19 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.1 | 0.55 | 0/1/15 | 0 | 3/3 click OPEN clunk OPEN | OPEN | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.15 | 0.3 | 0/2/15 | 0 | 3/3 click OPEN clunk no | OPEN | 21/21 +7/7 orphans 0 | 2 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.15 | 0.4 | 0/1/19 | 0 | 3/3 click OPEN clunk no | OPEN | 21/21 +7/7 orphans 0 | 2 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.15 | 0.55 | 0/1/15 | 0 | 3/3 click OPEN clunk no | OPEN | 21/21 +7/7 orphans 0 | 2 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.25 | 0.3 | 0/1/15 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.25 | 0.4 | 0/0/19 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | — |
| `lkaic-vf22-l` | `lkaic-flag` | 0.25 | 0.55 | 0/0/15 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | -0.031 / -0.053 / 6-2 / +0.207 / 0 |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.05 | 0.3 | 0/1/22 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | OPEN | 21/21 +6/7 orphans 3 | 8 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.05 | 0.4 | 0/1/18 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | OPEN | 21/21 +6/7 orphans 3 | 8 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.05 | 0.55 | 0/1/13 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | OPEN | 21/21 +7/7 orphans 3 | 7 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.1 | 0.3 | 0/1/22 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | rejects | 21/21 +6/7 orphans 1 | 5 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.1 | 0.4 | 0/1/18 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | rejects | 21/21 +6/7 orphans 1 | 5 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.1 | 0.55 | 0/1/13 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | rejects | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.15 | 0.3 | 0/1/22 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | rejects | 21/21 +6/7 orphans 1 | 5 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.15 | 0.4 | 0/1/18 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | rejects | 21/21 +6/7 orphans 1 | 5 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.15 | 0.55 | 0/1/13 | 0 | 3/3 click OPEN clunk OPEN +1 phantom | rejects | 21/21 +7/7 orphans 1 | 4 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.25 | 0.3 | 0/1/22 | 0 | 3/3 click no clunk no +1 phantom | rejects | 21/21 +6/7 orphans 1 | 3 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.25 | 0.4 | 0/1/18 | 0 | 3/3 click no clunk no +1 phantom | rejects | 21/21 +6/7 orphans 1 | 3 | — |
| `lkaic-vf22-l` | `aic-vad-vf-2.0` | 0.25 | 0.55 | 0/1/13 | 0 | 3/3 click no clunk no +1 phantom | rejects | 21/21 +7/7 orphans 1 | 2 | -0.069 / -0.095 / 8-0 / +0.000 / 0 |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.05 | 0.3 | 0/5/30 | 0 | 3/3 click OPEN clunk OPEN +2 phantom | OPEN | 21/21 +7/7 orphans 5 | 10 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.05 | 0.4 | 0/4/29 | 0 | 3/3 click OPEN clunk OPEN +2 phantom | OPEN | 21/21 +7/7 orphans 5 | 10 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.05 | 0.55 | 0/3/24 | 0 | 3/3 click OPEN clunk OPEN +2 phantom | OPEN | 21/21 +7/7 orphans 5 | 10 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.1 | 0.3 | 1/3/30 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 5 | 7 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.1 | 0.4 | 1/3/29 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 5 | 7 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.1 | 0.55 | 1/2/24 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 5 | 7 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.15 | 0.3 | 1/3/30 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 5 | 7 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.15 | 0.4 | 1/3/29 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 5 | 7 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.15 | 0.55 | 1/2/24 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 5 | 7 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.25 | 0.3 | 1/3/30 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 3 | 5 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.25 | 0.4 | 1/3/29 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 3 | 5 | — |
| `lkaic-vf22-l` | `aic-vad-ms-2.1` | 0.25 | 0.55 | 1/2/24 | 0 | 3/3 click no clunk no +1 phantom | OPEN | 21/21 +7/7 orphans 3 | 5 | -0.027 / +0.025 / 5-3 / +0.727 / 0 |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.05 | 0.3 | 1/3/4 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.05 | 0.4 | 1/2/11 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.05 | 0.55 | 1/2/5 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | -0.034 / +0.043 / 5-3 / +0.727 / 0 |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.1 | 0.3 | 1/3/4 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.1 | 0.4 | 1/2/11 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.1 | 0.55 | 1/2/5 | 0 | 3/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 0 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.15 | 0.3 | 1/3/4 | 0 | 2/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.15 | 0.4 | 1/2/11 | 0 | 2/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.15 | 0.55 | 1/2/5 | 0 | 2/3 click no clunk no | rejects | 21/21 +7/7 orphans 0 | 1 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.25 | 0.3 | 4/0/4 | 2 | 1/3 click no clunk no | rejects | 21/21 +5/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.25 | 0.4 | 3/0/11 | 2 | 1/3 click no clunk no | rejects | 21/21 +5/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `silero-v4.0-8k` | 0.25 | 0.55 | 2/0/5 | 1 | 1/3 click no clunk no | rejects | 21/21 +6/7 orphans 0 | 3 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.05 | 0.3 | 0/6/11 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.05 | 0.4 | 0/4/14 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.05 | 0.55 | 0/2/11 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | +0.000 / +0.071 / 4-4 / +0.727 / 0 |
| `lkaic-vf22-l` | `lk-inference` | 0.1 | 0.3 | 1/5/11 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.1 | 0.4 | 1/4/14 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.1 | 0.55 | 1/2/11 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.15 | 0.3 | 1/5/11 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.15 | 0.4 | 1/4/14 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.15 | 0.55 | 1/2/11 | 0 | 3/3 click no clunk no | OPEN | 21/21 +4/7 orphans 0 | 4 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.25 | 0.3 | 2/3/11 | 0 | 3/3 click no clunk no | rejects | 19/21 +4/7 orphans 0 | 5 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.25 | 0.4 | 1/2/14 | 0 | 3/3 click no clunk no | rejects | 19/21 +4/7 orphans 0 | 5 | — |
| `lkaic-vf22-l` | `lk-inference` | 0.25 | 0.55 | 1/1/11 | 0 | 3/3 click no clunk no | rejects | 19/21 +4/7 orphans 0 | 5 | — |

</details>

## 9. Caveats

- **Sample sizes.** 10 marked phone runs and 8 web runs. PR-54 is one click and one
  throat-clear. PR-42 is one call, with heuristic labels: the 7 "loud missed" sounds are
  inferred speech with no transcript, and the 12 quiet/echo sounds are ignored.
- **Hand marks** are drawn by eye on the raw waveform. The incident marks were read off
  level and pitch profiles.
- **The second Opus pass.** Every replay candidate has one, which `cloud-raw` measures. Also:
  - Outputs are not warped back, so each can be off input time by its batch's control drift
    (0–144 ms, section `replay`).
  - The replay gates can't catch two same-length corruptions: an onset loss that is
    compensated, and a mid-file tear that is compensated. Both are unlikely on real streams
    (see the README's cloud-replay caveats).
  - PR-42 used a 150 ms drift gate.
- **VAD-cut WER** is one request per VAD segment, as production sends it. The 08-05 report and
  `vad_bench`'s own `stt` merge overlapping cuts, so their numbers aren't comparable with these.
- **The STT stand-in.** Every WER comes from a self-hosted STT stand-in, which reproduced prod's text
  in PR-54 and PR-29. Whether it is prod's exact build is unverified.
- **Determinism.** The bench's own inference repeats bit for bit. livekit's resampling paths
  don't (README "Parity"). So a borderline sound that production's native path sees may flip
  between runs. PR-54 B's throat-clear is such a sound.
- **Platforms.** VAD probabilities come from Linux x86_64, except TEN's and every
  VAD's on `hush-aih`, which come from macOS. So do the Hush chains' NC outputs. On this Mac, `vad_bench`'s cache calls the Linux `lk-inference`, Earshot and WebRTC
  probabilities stale (their native-binary hash differs), so a parameter change made here
  re-infers them. The sweep reads them anyway when everything else in the fingerprint matches.
- **Restored records** on 131319 carry no latency, chain or cost numbers.
- **Kept on disk:** the orphan `aic-vf-l` / `aic-vf-s` wavs on 131319, and the PR-42
  investigation's own BVCTelephony replay. The latter wasn't attached: its stream lags its
  control by 190 ms.

## Reproduce

```bash
.venv/bin/python scripts/report_round2.py --list          # every table in this report
.venv/bin/python scripts/report_round2.py --section sweep
.venv/bin/python scripts/report_round2.py --section aic-1to1   # §8.1
.venv/bin/python scripts/vad_cost.py                       # the cost table (reads audio)
```

The backfill:

- the local incident import scripts (kept out of this repository);
- `scripts/backfill_round2.py` (`process`, `stt`);
- `scripts/restore_orphans.py`, `scripts/measure_live_offsets.py`;
- `scripts/vad_bench.py`;
- `scripts/sweep_round2.py` (`sweep`, `stt`).

Every run keeps its pre-round-2 `meta.json` as `meta.v1.json`.
