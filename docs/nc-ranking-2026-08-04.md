# NC candidate ranking — phone vs web, STT and VAD

**2026-08-04** · generated from the 20 runs in `data/runs/` by the nc_bench scoring
pipeline. Every number is reproducible from the stored `meta.json` files.

## What was measured

Each run records one real call (or an upload of one) processed through every ticked
NC candidate plus **`none` (passthrough — the untouched audio)**. Two independent
qualities are scored, because they fail independently:

1. **STT quality** — WER of the local Qwen3-ASR transcript against the reference
   script read during the call.
2. **VAD segmentation vs hand marks** — Silero VAD (the same
   livekit-agents plugin a live deployment runs, thresholds act/deact 0.50, min speech
   0.05 s, min silence 0.40 s, prefix pad 0.5 s; 8 kHz inference for phone,
   16 kHz for web) is run on each candidate's output, and its spans are compared
   against speech regions **hand-marked on the raw input waveform**. The marks are
   the only reference in this report produced by a human rather than another model.

Why VAD matters as much as WER: production never transcribes a whole call. The VAD
cuts it into turns and each turn is one STT request — speech the VAD misses is
never transcribed at all, and noise it mistakes for speech becomes a phantom turn.

### Surfaces

`source=phone` / `source=web` as recorded; uploads are classified by their note
(`yh bg saved (phone)` → phone). Phone runs: PSTN leg over the SIP trunk,
~3.5 kHz real bandwidth. Web runs: browser mic, wideband.

### Excluded from the STT ranking

The local STT is an autoregressive model and can collapse into repetition loops
("Hello? Hello? Hello? …"). A collapsed transcript measures the decoder's failure,
not the NC's, so it must not count. Detection: any 1–3-gram repeated ≥5×
consecutively.

| run | note | why excluded |
|---|---|---|
| `20260804-174342` | noise test phone | passthrough transcript collapsed (top n-gram repeated x42) |
| `20260804-103947` | Jasjeet Call (no reference) | no surface label / no passthrough |
| `20260803-175858` | Hecttoc Sample (ignore reference) | no surface label / no passthrough |
| `20260803-170156` | Daniel and hao bg | no reference script |

Plus 2 individual candidate transcripts dropped inside otherwise-valid runs:

| run | note | candidate | repetition |
|---|---|---|---|
| `20260804-171445` | shen | `gtcrn+hush` | ×73 |
| `20260804-112511` | yh bg | `dpdfnet-baseline` | ×96 |

VAD scoring is unaffected by transcript collapse (it never reads a transcript),
so those runs still count there.

## How to read the columns

Every table compares each candidate **against passthrough on the same recording**,
then aggregates those per-run deltas. Using deltas removes the run's own difficulty:
a noisy call hurts every candidate, but the *difference* vs untouched audio is the
NC's contribution.

- **n** — how many valid runs the candidate appeared in. Candidates in fewer than
  half the runs are dropped from the table (not enough evidence to rank).
- **mean** — the average delta across those runs: *"on a typical call, how much
  did this NC help or hurt vs doing nothing."*
- **worst** — the single most harmful delta among them: *"when it backfired, how
  badly."* This is downside exposure, not a typo for best. With n≈8–10 it tells
  that a failure of that size is possible, not how often it happens.
- **W / L** — runs strictly better / strictly worse than passthrough. Exact ties
  count in neither, so W+L rarely equals n.
- **abs agree** (agreement tables only) — the candidate's absolute agreement,
  for scale alongside the delta.

Worked example, phone STT: `hecttor-crest2 · mean −0.039 · worst +0.000 · 4W/0L`
means: averaged over 8 calls its transcript had 3.9 WER points fewer errors than
passthrough (~1 word on these ~30-word scripts), and in its single worst call it
exactly tied passthrough — it never made a transcript worse.

Units per metric:

- **ΔWER** — word error rate difference; negative = NC produced a better
  transcript. 0.035 ≈ one word on a 30-word script.
- **Δ missed speech (s)** — seconds of *hand-marked* speech the VAD failed to
  detect, minus passthrough's. Negative = the cleanup let the VAD find speech it
  was missing in the raw audio; positive = the NC dulled real speech into
  "silence" — words deleted before transcription ever happens.
- **Δ agree** — percentage-point change in frame-level agreement between the VAD
  and the marks (10 ms frames). Agreement counts both error directions, so it also
  rewards suppressing the noise that trips the VAD during silence (false alarms).

---

## Phone — STT (8 valid runs)

ΔWER vs passthrough; negative = NC helped the transcript.

| # | candidate | n | mean | worst | W | L |
|---|---|---|---|---|---|---|
| 1 | `hecttor-crest2` | 8 | -0.039 | +0.000 | 4 | 0 |
| 2 | `hecttor-coda-vi-w075` | 8 | -0.031 | +0.035 | 5 | 1 |
| 3 | `hecttor-coda` | 8 | -0.026 | +0.035 | 4 | 1 |
| 4 | `hecttor-coda-vi` | 8 | -0.019 | +0.035 | 4 | 2 |
| 5 | `dpdfnet2-8k` | 8 | -0.015 | +0.000 | 4 | 0 |
| 6 | `dpdfnet8-8k` | 8 | -0.011 | +0.035 | 2 | 2 |
| 7 | `rnnoise-sh` | 8 | -0.004 | +0.048 | 4 | 3 |
| 8 | `fastenhancer-t+hecttor-coda-vi` | 8 | -0.004 | +0.054 | 3 | 2 |
| 9 | `dpdfnet2-8k+hush` | 8 | -0.000 | +0.063 | 3 | 5 |
| 10 | `hush-atten12` | 8 | -0.000 | +0.048 | 2 | 3 |

Hecttor's ASR family holds the top 4. Only `hecttor-crest2` and `dpdfnet2-8k`
never hurt a transcript. From rank 9 down the mean is zero — most of the shelf
does nothing for phone STT.

## Phone — VAD, missed speech (10 runs · passthrough misses 0.44 s on average)

Δ seconds of hand-marked speech the VAD missed; negative = NC preserved more.

| # | candidate | n | mean | worst | W | L |
|---|---|---|---|---|---|---|
| 1 | `dtln` | 10 | -0.09 s | +0.36 s | 5 | 5 |
| 2 | `gtcrn` | 10 | -0.05 s | +0.43 s | 5 | 3 |
| 3 | `fastenhancer-s` | 10 | -0.01 s | +0.21 s | 4 | 3 |
| 4 | `ulunas` | 10 | -0.01 s | +0.43 s | 5 | 3 |
| 5 | `dpdfnet8-8k` | 10 | +0.07 s | +0.56 s | 3 | 7 |
| 6 | `hecttor-coda` | 10 | +0.08 s | +0.41 s | 2 | 8 |
| 7 | `rnnoise-bd` | 10 | +0.09 s | +0.69 s | 3 | 7 |
| 8 | `fastenhancer-t` | 10 | +0.12 s | +0.59 s | 3 | 5 |
| 9 | `hush-atten12` | 10 | +0.12 s | +0.91 s | 2 | 6 |
| 10 | `hecttor-coda-vi-w075` | 10 | +0.19 s | +0.87 s | 2 | 7 |

Near-inverse of the STT table: the tiny open suppressors (DTLN, GTCRN,
FastEnhancer-S, UL-UNAS) are the only ones that don't cost speech, while the
Hecttor family sits mid-table at +0.1–0.2 s. Note the small absolute scale —
passthrough already misses only 0.44 s, so phone wins here are marginal.

## Phone — VAD, agreement (10 runs · passthrough averages 82.4%)

Δ agreement with the marks; positive = better than passthrough.

| # | candidate | n | mean | worst | W | L | abs agree |
|---|---|---|---|---|---|---|---|
| 1 | `gtcrn+hush` | 10 | +9.5% | -5.0% | 8 | 2 | 91.9% |
| 2 | `krisp-bvc-tel` | 8 | +9.2% | -5.6% | 7 | 1 | 89.8% |
| 3 | `dpdfnet2-8k+hush` | 10 | +8.4% | -4.3% | 8 | 2 | 90.8% |
| 4 | `hush` | 10 | +8.4% | -9.7% | 8 | 2 | 90.8% |
| 5 | `hecttor-crest2` | 10 | +7.9% | -2.0% | 7 | 3 | 90.3% |
| 6 | `hecttor-coda-vi-8k` | 10 | +7.9% | -3.0% | 8 | 2 | 90.3% |
| 7 | `hecttor-mist` | 10 | +7.8% | -2.9% | 8 | 2 | 90.2% |
| 8 | `fastenhancer-t+hecttor-coda-vi` | 10 | +7.7% | -3.4% | 8 | 2 | 90.1% |
| 9 | `hecttor-coda-vi` | 10 | +7.4% | -1.5% | 8 | 2 | 89.8% |
| 10 | `dtln+hecttor-coda-vi` | 10 | +7.0% | -3.6% | 8 | 2 | 89.4% |

## Web — STT (8 valid runs)

| # | candidate | n | mean | worst | W | L |
|---|---|---|---|---|---|---|
| 1 | `aic-quail-l` | 7 | +0.009 | +0.063 | 0 | 1 |
| 2 | `hecttor-coda-vi-w075` | 8 | +0.015 | +0.063 | 0 | 3 |
| 3 | `hush-atten12` | 8 | +0.016 | +0.125 | 1 | 2 |
| 4 | `fastenhancer-l` | 8 | +0.023 | +0.094 | 0 | 2 |
| 5 | `rnnoise-sh` | 8 | +0.025 | +0.137 | 1 | 3 |
| 6 | `rnnoise-bd` | 8 | +0.028 | +0.156 | 1 | 2 |
| 7 | `aic-vf-s` | 7 | +0.029 | +0.108 | 1 | 3 |
| 8 | `gtcrn` | 8 | +0.036 | +0.091 | 0 | 6 |
| 9 | `hecttor-coda` | 8 | +0.040 | +0.125 | 0 | 5 |
| 10 | `dtln` | 8 | +0.046 | +0.125 | 1 | 4 |

**Read the signs: even rank 1 is a net loss.** Zero candidates beat passthrough on
average; 4 wins total across 78 candidate-runs. Web audio arrives clean enough
that NC has only downside for transcription.

## Web — VAD, missed speech (8 runs · passthrough misses 0.20 s on average)

| # | candidate | n | mean | worst | W | L |
|---|---|---|---|---|---|---|
| 1 | `gtcrn` | 8 | +0.02 s | +0.34 s | 3 | 2 |
| 2 | `ulunas` | 8 | +0.13 s | +0.61 s | 2 | 3 |
| 3 | `hecttor-coda-vi-w075` | 8 | +0.52 s | +1.62 s | 0 | 6 |
| 4 | `aic-quail-l` | 7 | +0.54 s | +1.55 s | 0 | 5 |
| 5 | `krisp-nc` | 7 | +0.63 s | +1.26 s | 0 | 7 |
| 6 | `hecttor-coda` | 8 | +0.76 s | +1.80 s | 0 | 6 |
| 7 | `hush-atten12` | 8 | +0.79 s | +1.85 s | 0 | 6 |
| 8 | `krisp-bvc` | 7 | +0.81 s | +1.64 s | 0 | 6 |
| 9 | `dpdfnet8-8k` | 8 | +0.83 s | +1.93 s | 0 | 7 |
| 10 | `rnnoise-bd` | 8 | +0.83 s | +3.59 s | 0 | 6 |

Cliff after rank 2: everything below costs **0.5–0.8 s of real speech per call**
with zero wins — including `krisp-nc` and `krisp-bvc`, a common default choice. `rnnoise-bd`'s worst case is 3.59 s of speech gone.

## Web — VAD, agreement (8 runs · passthrough averages 90.7%)

| # | candidate | n | mean | worst | W | L | abs agree |
|---|---|---|---|---|---|---|---|
| 1 | `hecttor-coda-vi-w075` | 8 | +3.8% | -3.8% | 6 | 2 | 94.5% |
| 2 | `krisp-bvc` | 7 | +3.5% | -0.4% | 6 | 1 | 93.1% |
| 3 | `hecttor-coda` | 8 | +2.8% | -5.8% | 7 | 1 | 93.4% |
| 4 | `hush-atten12` | 8 | +2.0% | -3.6% | 6 | 2 | 92.6% |
| 5 | `aic-vf-s` | 7 | +1.9% | -4.8% | 4 | 3 | 91.5% |
| 6 | `krisp-nc` | 7 | +1.5% | -3.6% | 5 | 2 | 91.1% |
| 7 | `aic-quail-l` | 7 | +1.3% | -6.9% | 5 | 2 | 91.0% |
| 8 | `rnnoise-sh` | 8 | +1.2% | -11.5% | 7 | 1 | 91.9% |
| 9 | `rnnoise-bd` | 8 | +1.1% | -7.4% | 5 | 3 | 91.7% |
| 10 | `dpdfnet8` | 8 | +1.0% | -8.7% | 5 | 3 | 91.7% |

## WER, VAD-cut — production-real (one STT request per turn)

*Added 2026-08-05, after backfilling every run with segmented STT.*

The whole-file WER above is a bench artifact: production never posts a whole call
to the STT. The VAD cuts the call into turns and each turn is one request — so
this table re-transcribes every candidate the way production hears it: audio cut
at that candidate's **own** VAD spans (start extended by the prefix padding,
overlaps merged), one request per turn, transcripts joined, WER recomputed. Both
measurements are kept on every candidate (`wer` whole-file, `wer_seg` VAD-cut);
the UI toggles between them.

The baseline moves first, and that is the finding. Passthrough WER, same audio:

| surface | whole-file | VAD-cut |
|---|---|---|
| phone | 0.183 | **0.330** |
| web | 0.096 | **0.245** |

Raw audio is far worse than the whole-file tables suggested, because every noise
stretch the VAD mistakes for speech becomes a turn that transcribes as junk
insertions. That is the cost the agreement tables measured in seconds, now in WER
— and suppressing those phantom turns is exactly what NC is for, so candidates
win here that lost whole-file.

**Ranked by median, not mean.** One re-admitted run (`noise test phone` — its
whole-file baseline was collapse-junk, its segmented baseline is honest at 4.545
on an 11-word script) hands good suppressors deltas near −4.0; a mean over nine
runs is dominated by it. The mean column stays for visibility — a large
median–mean gap on phone IS that run. In-turn repetition collapse still occurs
(10 candidate transcripts dropped by the same ≥5× filter).

### Phone (9 runs)

ΔWER vs the segmented passthrough; negative = NC helped.

| # | candidate | n | median | mean | worst | W | L |
|---|---|---|---|---|---|---|---|
| 1 | `aic-vf-s` | 6 | -0.116 | -0.671 | -0.035 | 6 | 0 |
| 2 | `fastenhancer-t` | 9 | -0.081 | -0.077 | +0.250 | 5 | 4 |
| 3 | `krisp-bvc` | 7 | -0.081 | -0.549 | +0.313 | 4 | 2 |
| 4 | `hecttor-mist` | 8 | -0.074 | -0.510 | +0.250 | 5 | 3 |
| 5 | `krisp-bvc-tel` | 7 | -0.059 | -0.622 | +0.034 | 4 | 2 |
| 6 | `dpdfnet2-8k+hush` | 9 | -0.059 | -0.528 | +0.094 | 5 | 3 |
| 7 | `gtcrn+hush` | 9 | -0.059 | -0.482 | +0.189 | 5 | 4 |
| 8 | `dtln` | 9 | -0.054 | -0.161 | +0.172 | 5 | 3 |
| 9 | `fastenhancer-t+hecttor-coda-vi` | 9 | -0.048 | -0.402 | +0.156 | 6 | 3 |
| 10 | `hecttor-coda-vi` | 9 | -0.035 | -0.493 | +0.031 | 6 | 1 |

### Web (8 runs)

| # | candidate | n | median | mean | worst | W | L |
|---|---|---|---|---|---|---|---|
| 1 | `aic-quail-l` | 7 | -0.063 | -0.122 | +0.091 | 5 | 1 |
| 2 | `fastenhancer-l` | 8 | -0.047 | -0.081 | +0.135 | 5 | 2 |
| 3 | `hecttor-coda-vi-w075` | 8 | -0.043 | -0.118 | +0.000 | 5 | 0 |
| 4 | `hush-atten12` | 8 | -0.032 | -0.088 | +0.091 | 5 | 2 |
| 5 | `dpdfnet-baseline` | 8 | -0.024 | -0.060 | +0.137 | 4 | 2 |
| 6 | `dpdfnet8` | 8 | -0.015 | -0.035 | +0.091 | 4 | 3 |
| 7 | `hecttor-coda-vi` | 8 | -0.000 | -0.066 | +0.091 | 4 | 4 |
| 8 | `hecttor-coda` | 8 | +0.000 | -0.036 | +0.187 | 2 | 3 |
| 9 | `dtln` | 8 | +0.000 | -0.045 | +0.162 | 3 | 3 |
| 10 | `krisp-nc` | 7 | +0.000 | +0.024 | +0.125 | 2 | 3 |

The suppressors that gut phantom turns — Hush and its combos, Krisp BVC, the
cloud rails — lead here, and `hecttor-crest2` goes 7W/1L. On web the sign finally
flips: `hecttor-coda-vi-w075` is 5W/0L with a worst case of exactly 0.000, where
whole-file it never won a single run.

### Split experiment: one model gates the VAD, another feeds the STT

*Added 2026-08-05 · `scripts/exp_split_vad_stt.py` — read-only over the stored
runs: turn spans come from the GATE candidate's stored VAD measurement, the audio
cut at those spans comes from the READ candidate's stored output, scored with the
same turn-by-turn STT as `wer_seg`, so pairs compare directly with every
self-gated row above. The gate's misses are inherited by construction: a turn the
gate never opens is never transcribed.*

ΔWER vs the segmented passthrough, median-ranked as above:

| pair (gate → read) | n | median | mean | worst | W/L |
|---|---|---|---|---|---|
| **phone** `hush → none (raw)` | 9 | **−0.063** | −0.498 | +0.069 | **7/2** |
| phone `gtcrn+hush → dpdfnet2-8k` | 9 | −0.054 | −0.465 | +0.103 | 6/3 |
| phone `hush → dpdfnet2-8k` | 9 | −0.030 | −0.459 | +0.103 | 5/3 |
| phone `gtcrn+hush → dtln` | 9 | −0.027 | −0.434 | +0.188 | 5/3 |
| **web** `hush-atten12 → none (raw)` | 8 | **−0.043** | −0.103 | +0.054 | **5/2** |
| web `hush-atten12 → dtln` | 8 | +0.000 | −0.049 | +0.219 | 3/3 |
| web `hush-atten12 → fastenhancer-l` | 8 | +0.000 | −0.071 | +0.135 | 3/2 |
| web `gtcrn → none (raw)` | 8 | −0.015 | +0.250 | +1.818 | 4/3 |

The winner on **both surfaces is the control pair: aggressive gate, RAW audio to
the STT.** `hush → none` posts the best median, the best W/L (7/2) and the
smallest worst-case of any free configuration measured in this report — better
than every self-gated candidate and better than the same gate feeding an
NC-cleaned reader. `hush-atten12 → none` does the same on web (5/2, worst-case
+0.054), where no self-gated free candidate managed a clean record.

Two implications:

1. **NC's measurable value in this stack is gating, not cleaning.** Suppression
   decides *which audio exists* (phantom turns die, that was worth up to −4.0
   WER on the noisiest run) — but for the audio that survives, the recogniser
   prefers it untouched. Every reader that "cleaned" the turns scored worse than
   raw.
2. **The gate must still fit the surface.** `gtcrn → none` on web carries a
   +1.818 catastrophe (its gate passed ten phantom turns on the noise-test web
   call), and full-strength `hush` remains phone-only — the web gate has to be
   the capped `hush-atten12`.

Caveats as elsewhere: n = 8–9, medians quoted because the re-admitted noise-test
run dominates every mean, and the split inherits the gate's missed speech
(`hush` +0.76 s / `hush-atten12` +0.79 s vs raw), which WER against a read script
underweights relative to a real conversation.

## The miss/agree inversion, and which to trust

The web agreement leaders (`hecttor-coda-vi-w075`, `krisp-bvc`, `hecttor-coda`,
`hush-atten12`) are the same chains at the bottom of the web missed-speech table.
Not a contradiction: they suppress hard in both directions. Agreement improves
because far more noise-time stops tripping the VAD (false alarms) than speech-time
is lost — but the speech-time *is* lost.

The two tables answer different production questions:

- **missed speech** → do the words still reach the STT? A miss is unrecoverable:
  the turn is never transcribed.
- **agreement** → will noise stop creating phantom turns? A false-alarm turn hands
  the STT junk; the agent may transcribe garbage or answer nobody.

A missed customer word is usually the costlier failure, which is why the summary
below weights the miss table over the agree table.

## Summary

**Phone: NC earns its keep, and `hecttor-coda-vi-w075` is the best all-rounder** —
top-3 STT (−0.031 mean, 5W/1L), agreement neutral (+0.4 %), acceptable +0.19 s miss.
`hecttor-crest2` has the best STT record (never worse than passthrough) and +7.9 %
agreement, but carries a known risk: on the one long (2 min) real call in the
archive it triggered the STT decoder's repetition collapse where 26 other
candidates did not. Until that's understood, prefer `coda-vi-w075` for anything
longer than a scripted test clip.

**Web: run nothing.** Three independent measurements agree — zero of ~30
candidates improved WER on average, everything except `gtcrn`/`ulunas` eats 0.5 s+
of real speech per call, and the agreement gains that do exist come from the same
over-suppression that eats the speech. This includes Krisp (0W/7L missed-speech
on web).

**The VAD-cut WER (added 2026-08-05) amends the web verdict.** Measured
production-real — one STT request per VAD turn — raw audio is much worse than the
whole-file tables suggested (passthrough 0.096 → 0.245 on web), because phantom
noise turns transcribe as junk. Under that measurement NC helps both surfaces:
`hecttor-coda-vi-w075` goes 5W/0L on web with worst-case ±0.000. "Run nothing on
web" holds only for whole-call transcription, which production does not do; the
production-shaped answer is that moderate suppression pays for itself by killing
phantom turns.

**Split gate/read (2026-08-05) is the strongest free configuration measured.**
Gating the VAD with Hush while the STT reads raw audio beat every single-model
setup on both surfaces (phone `hush → raw` 7W/2L, web `hush-atten12 → raw`
5W/2L). NC earns its keep deciding what gets transcribed, not polishing it.

**If one config must serve both surfaces:** `hecttor-coda-vi-w075` is the only
candidate in the top-10 of both STT tables and both missed-speech tables (its
phone agreement is a neutral +0.4 %, below that top-10). But the per-surface
answer is strictly better: Hecttor ASR on phone, passthrough on web.

### Caveats

- Sample sizes are 8–10 runs; a `worst` column is a single tail sample. Margins in
  the STT tables are ~1 word on ~30-word scripts.
- The `aisyah cheelam bg` condition contributes 5 of 16 valid runs (two of them
  the same phone script twice), so that room is over-weighted.
- Hand marks are drawn on the raw waveform by eye; sub-100 ms edges are subjective.
  Deltas vs passthrough cancel most of this; absolute miss/agree values inherit it.
- Phone VAD runs use 8 kHz Silero inference (matches the telephony band);
  a live deployment commonly pins 16 kHz everywhere. Deltas are internally
  consistent, but absolute spans differ from what a 16 kHz cut would give.
- All STT numbers are from the local Qwen3-ASR endpoint; a different recogniser
  can reorder the STT tables. The VAD tables are recogniser-independent.
- VAD spans carry inference jitter: onnxruntime is nondeterministic at the
  margin, and with min_speech 0.05 s a borderline span can flicker between
  identical analyses (measured: ±0.03–0.4 s of speech, ±1 span, on the noisiest
  runs). Table values round well above that, but exact re-runs of the pipeline
  may differ in the last decimal.
