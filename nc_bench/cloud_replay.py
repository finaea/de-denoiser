"""Cloud replay rail: run LiveKit-hosted NC models on a STORED recording.

The Krisp and ai-coustics plugins only run as filters on a track that lives in a
LiveKit Cloud room, so an upload or an old run cannot go through them offline.
This publishes the run's input.wav into a fresh room in real time and records it
back through one `rtc.AudioStream(noise_cancellation=...)` per candidate:

    publisher process  ── input.wav @48 kHz, Opus 256 kbps, RED off, DTX off ──▶
    LiveKit Cloud room ──▶ subscriber process: one AudioStream per candidate
                           (24 kHz mono, 50 ms frames — what a livekit-agents RoomIO delivers)
                           plus an unfiltered control stream

Decisions this rests on, measured in PR-42 and PR-54:

- **Two processes.** In one process the processors starved the publisher's
  pacing and it inserted 29 silence gaps (PR-42). The subscriber runs every
  FrameProcessor on its own event loop, which may fall behind real time; that
  only delays it, the audio is queued, not dropped.
- **256 kbps, RED off.** At LiveKit's default Opus bitrate a borderline
  throat-clear lost its VAD detection on the RAW stream (Silero peak 0.91 →
  0.35), so the second codec pass has to be as transparent as it can be made.
- **A `cloud-raw` control.** Every candidate here has been through a second Opus
  pass the offline candidates never saw; comparing against `cloud-raw` rather
  than only `none` separates the model from the transport.
- **Batches.** PR-42 ran 9 parallel streams in one subscriber without stalls.
  More candidates than CLOUD_REPLAY_MAX_STREAMS go in sequential batches, each
  in its own room with its own control stream.

Each output is aligned by cross-correlating it against the input (the lead-in is
the 3 s preroll plus join/transport delay), then checked: envelope correlation
and length against the input. Per batch, the publisher reports any time its send
queue ran dry (a stall: silence the recording never had), and the control
stream's local lag across the file is measured — the receiver's jitter buffer
time-stretches by tens of ms on its own, which is reported and tolerated, but
gross drift is not. A candidate that fails any of these is an error, not a result. Outputs are written like the offline
pipeline's: `<cid>.wav` at config.PIPELINE_RATE, plus `<cid>.aicvad.npy` for
ai-coustics streams (the enhancer's per-frame speech flag, bool, aligned to the
input; see `_aicvad`).

Roles, spawned by `replay()` (credentials come from .env, never argv):

    python -m nc_bench.cloud_replay pub <room> <input.wav>
    python -m nc_bench.cloud_replay sub <room> <workdir>     # reads workdir/streams.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import soundfile as sf
import soxr

from . import config, lk_cloud

PUB_RATE = 48_000
PUB_FRAME_MS = 10
SUB_RATE = 24_000
SUB_FRAME_MS = 50
BITRATE_KBPS = 256
# Digital silence before the audio. It has to cover the receiver's jitter buffer
# settling, not just the filters initialising: with 1 s, the control stream's lag
# sat at -27 ms for the first 2.6 s of audio and then stepped to 0 in every stream
# alike (NetEQ growing its target delay), which the drift gate rightly rejected.
# PR-54's 3 s pre-publish wait hid the same thing.
PREROLL_S = 3.0
TAIL_S = 1.0
# filtered streams per subscriber INCLUDING the control; PR-42 ran 9 without a
# stall. Each candidate's unfiltered twin comes on top: it does no processing
# (a 12-stream probe, half of them twins, ran with the subscriber ~0.4 s behind)
MAX_STREAMS = int(os.environ.get("CLOUD_REPLAY_MAX_STREAMS", "9"))

_PUB_ID, _SUB_ID = "nc-replay-pub", "nc-replay-sub"
_ALIGN_RATE = config.PIPELINE_RATE
_ENV_HOP = _ALIGN_RATE // 100  # 10 ms envelope
_MAX_LEAD_S = 30.0

# Acceptance gates for one aligned output. The envelope correlation alone cannot
# say "lines up": aggressive NC lowers it legitimately — Krisp NC/BVC read 0.50 /
# 0.56 at their BEST offset on 20260804-112511, live rail and replay alike. What
# misalignment does instead is flatten the correlation around the lag used, so
# the gate is the peak's prominence over offsets 150 ms-1 s away (real outputs
# measured 0.31-0.73; noise, and another run's recording, <= 0), plus a floor.
MIN_ENV_CORR = 0.3
MIN_PROMINENCE = 0.15
# The unfiltered control, as a backstop behind the drift gate: 0.88-1.00 with
# normal jitter-buffer steps, 0.79 on the saturated run the drift gate also fails.
CONTROL_MIN_ENV_CORR = 0.8
MAX_SHORT_S = 0.25        # output may end at most this much before the input
RAW_MAX_DB = -20.0        # "should be raw" (bypass / level 0) must match the control
# Publisher-side starvation, the stall itself: more than one 10 ms frame of
# silence the recording never had fails the batch.
MAX_STARVED_MS = 10.0
# Receiver-side lag steps are the jitter buffer (NetEQ) time-stretching: 17-33 ms
# steps mid-file, identical in every stream, waveform corr 1.00 on both sides,
# while the publisher never starved — inherent to WebRTC, and production audio
# goes through the same. Measured spreads (max - min local lag): 0-46 ms with the
# replay running alone, 102 ms with the machine saturated (subscriber 5.4 s
# behind). Reported always; a spread past this fails the batch. The steps pile up
# over long files (a 119 s upload read 95-114 ms with a 0.3 s backlog), so a
# long replay can raise it per run via the environment.
MAX_DRIFT_MS = float(os.environ.get("CLOUD_REPLAY_MAX_DRIFT_MS", "80"))
PASSTHROUGH_DB = -40.0    # residual vs the control below this = filter never ran

# Per-candidate timeline. The control's check says the TRANSPORT kept time; a
# gap or a duplicated stretch inside one candidate's own filter or receive path
# would not show there (a 120 ms zero gap in a Krisp VIVA stream passed every
# other gate). Measured per candidate on fine envelopes (2 ms hops), which
# survive heavy NC where the waveform does not, and compared with the control
# measured the same way: every stream of a batch shares its jitter-buffer steps,
# so only the difference is the candidate's own. On the 25-candidate smoke of
# 20260804-112511 that difference was 0-4 ms for every stream but Krisp NC
# (10 ms, the most distorted output, 3 of 8 windows voting), so 20 ms is twice
# the worst honest reading while a 120 ms insertion reads 120.
MAX_CAND_DRIFT_OVER_CONTROL_MS = 20.0
ENV_LAG_WIN_S = 1.5
ENV_LAG_SEARCH_S = 0.4
# Below 0.5, Krisp NC's windows voted for spurious lags at the search edge
# (-400, -304 ms) with correlations of 0.41-0.48; at 0.5 every real stream was clean.
ENV_LAG_MIN_CORR = 0.5
ENV_LAG_MIN_SHARE = 0.25  # of voiced windows; Krisp NC managed 3/8
ENV_LAG_CONFIRM_MS = 10.0  # a lag counts as confirmed when another window agrees
# Output length against the control's, after each one's lead-in: every stream
# ends at the same point of the track, so a candidate's tail can only differ
# by its filter's own delay (measured -57..0 ms) — or by audio it inserted or
# lost. The crop to the input length must not hide that, so it is a gate.
MAX_EXCESS_MS = 40.0      # longer than the control by more: audio was inserted
MIN_EXCESS_MS = -200.0    # shorter by more (~4x the largest filter delay): audio lost

# The onset. A gap soon after the audio starts is absorbed into the stream's
# own lead-in, and the length-vs-control check subtracts that lead-in, so it
# passed every check above (the reviewer's 120 ms gap 150 ms after onset). The
# control cannot be the reference there: streams start at different points of
# the track — a Krisp native filter initialises INSIDE AudioStream creation
# (57 and 120 ms in the probe), so everything opened after it starts later.
# Each candidate therefore gets an unfiltered twin opened right after it, which
# starts at the same point. Measured on a 12-stream probe (2026-09-29):
# - twin and candidate had the SAME sample count for Krisp NC (x2), BVC
#   Telephony, ai-coustics QUAIL_L and Krisp VIVA (364/364, 362/362, ...),
#   while the raw stream opened just BEFORE a native filter differed by 1-3
#   frames. Filters keep frame length, so any audio a candidate inserts or loses,
#   onset included, shows as a length difference, exactly.
# - candidate lead minus twin lead (the filter's own delay) was 26-30 ms for
#   all four; an absorbed onset gap adds its full length to it.
MAX_TWIN_LEN_DIFF_MS = 10.0   # a quarter of a 50 ms frame: legit reads 0
# The filter's delay measured 26-40 ms on every stream, except Krisp NC, whose
# heavily reshaped output gives the least certain lead-in: 26.3, 10.4 and -37.6
# ms on three runs (its twin anchored within 2 ms and its length matched exactly
# each time). The bounds sit outside that spread; an onset gap reads +120 + the
# delay, a 120 ms onset drop about -90.
MIN_ONSET_DELAY_MS = -60.0
MAX_ONSET_DELAY_MS = 90.0

# The twin itself is anchored to the control. A fault shared by a candidate
# and its twin (a pair-local stall: filters run on the subscriber's loop, the
# control was opened before them) passed the twin checks — the reviewer's
# shared 120 ms onset gap read diff 0 / onset 0. Both are raw streams of one
# decoded track, so the twin must be the control's own audio, and on the probe
# it was: BIT-IDENTICAL over the whole overlap (residual -416 dB, max |diff|
# 0) at an offset of exactly the creation skew (0 / 60 / 180 / 190 ms), each
# within 0-3 ms of what the streams' open times predict (10 ms chunks). Any gap
# or stall in the twin breaks the identity after it; a gap near the onset
# also moves the offset away from the prediction by the gap's length.
MAX_TWIN_RESIDUAL_DB = -60.0  # measured: exact identity; a gap reads ~0 dB
MAX_TWIN_SKEW_ERR_MS = 20.0   # measured 0-3 ms; two 10 ms chunks of margin
# The twin's length against the control's from the same offset. Both end at the
# same point of the track and each drops its own last partial 50 ms frame, so
# with r and r' those leftovers (0-50 ms) the difference is r - r': always
# strictly inside ±50 ms. (Fitted first to the probe's 0 / +10 / +30 ms, the
# range was too narrow: a smoke batch read -30 on five sound twins.) The
# residual above is the real anchor; this only bounds gross length errors.
TWIN_LEN_VS_CONTROL_MS = (-49.9, 49.9)

ProgressCb = Callable[[dict], Awaitable[None]]

log = logging.getLogger("nc_bench.cloud_replay")

# one replay at a time per server: two would share the machine's cores and the
# Cloud project, and the whole point of the separate processes is clean pacing
_lock = asyncio.Lock()


# ================================================================ orchestrator


def plan_batches(cands: list[dict], max_streams: int = MAX_STREAMS) -> list[list[dict]]:
    """Split candidates into batches of at most max_streams - 1 (the control rides
    along in every batch), as few and as even as possible.

    Variants of one model (e.g. `lkaic-quail-l` and its bypass) stay in one
    batch: each batch is a separate publish, so only streams of the same batch
    saw bit-identical input, and "does bypass change the flags" is only a fair
    question between them. Within a batch the given order is kept.
    """
    per = max(1, max_streams - 1)
    order = {id(c): i for i, c in enumerate(cands)}
    groups: dict[str, list[dict]] = {}
    for c in cands:
        groups.setdefault(c["lk_model"], []).append(c)
    batches: list[list[dict]] = [[] for _ in range(max(1, math.ceil(len(cands) / per)))]
    for group in sorted(groups.values(), key=len, reverse=True):
        for chunk in (group[i : i + per] for i in range(0, len(group), per)):
            room = [b for b in batches if len(b) + len(chunk) <= per]
            if not room:
                batches.append([])
                room = [batches[-1]]
            min(room, key=len).extend(chunk)
    return [sorted(b, key=lambda c: order[id(c)]) for b in batches if b]


async def replay(
    input_wav: Path,
    cands: list[dict],
    out_dir: Path,
    progress: ProgressCb | None = None,
) -> dict[str, dict]:
    """Replay input_wav through every cloud candidate; write their outputs to out_dir.

    cands: [{"id", "lk_model", "lk_options"?}]. The RAW control needs no entry of
    its own — every batch records one — but a `lk_model: "RAW"` candidate (the
    `cloud-raw` control) is written from the first batch's control stream.

    Returns candidate id -> the candidate record fields ({"output", "cloud_replay",
    "plugins", "aicvad"?, "duration_s"}) or {"error": str}. Never raises for a
    failed batch: its candidates carry the error and the other batches proceed.
    """
    raw_ids = [c["id"] for c in cands if c["lk_model"] == lk_cloud.RAW]
    filt = [c for c in cands if c["lk_model"] != lk_cloud.RAW]
    batches = plan_batches(filt) or [[]]
    results: dict[str, dict] = {}
    async with _lock:
        for k, batch in enumerate(batches):
            work = Path(tempfile.mkdtemp(prefix="nc-cloud-replay-"))
            # every candidate is followed by an unfiltered TWIN, opened right
            # after it: it starts at the same point of the track (see
            # _finish_batch), which is the reference its onset is checked against
            streams = [{"name": "0", "id": None, "lk_model": lk_cloud.RAW}]
            for i, c in enumerate(batch, 1):
                streams.append({"name": str(i), "id": c["id"], "lk_model": c["lk_model"],
                                "lk_options": c.get("lk_options") or {}})
                streams.append({"name": f"{i}t", "id": None, "lk_model": lk_cloud.RAW,
                                "twin_of": str(i)})
            ids = [c["id"] for c in batch] + (raw_ids if k == 0 else [])
            try:
                meta = await _run_batch(input_wav, streams, work, k, len(batches), progress, ids)
                done = await asyncio.to_thread(
                    _finish_batch, input_wav, streams, work, out_dir, meta,
                    raw_ids if k == 0 else [],
                )
                results.update(done)
            except Exception as e:  # noqa: BLE001 - a batch failure is per-candidate
                log.exception("cloud replay batch %d failed", k + 1)
                for cid in ids:
                    results[cid] = {"error": f"cloud replay batch {k + 1}/{len(batches)}: {e}"}
            finally:
                if os.environ.get("CLOUD_REPLAY_KEEP_WORK"):
                    log.warning("cloud replay batch %d: raw streams kept in %s", k + 1, work)
                else:
                    shutil.rmtree(work, ignore_errors=True)
    return results


def _http_url() -> str:
    return config.LIVEKIT_URL.replace("wss://", "https://").replace("ws://", "http://")


async def _read_event(stream: asyncio.StreamReader, want: str, timeout: float,
                      on_event: Callable[[dict], Awaitable[None]] | None = None) -> dict:
    """Read JSON-line events from a role process until `want` arrives."""
    async def _loop() -> dict:
        while True:
            line = await stream.readline()
            if not line:
                raise RuntimeError(f"process exited before '{want}'")
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "error":
                raise RuntimeError(ev.get("error", "unknown error"))
            if ev.get("event") == want:
                return ev
            if on_event is not None:
                await on_event(ev)
    return await asyncio.wait_for(_loop(), timeout)


def _tail(path: Path, n: int = 600) -> str:
    try:
        return path.read_text(errors="replace")[-n:].strip()
    except OSError:
        return ""


async def _run_batch(input_wav: Path, streams: list[dict], work: Path, k: int, n: int,
                     progress: ProgressCb | None, ids: list[str]) -> dict:
    from livekit import api

    duration = sf.info(input_wav).duration
    room = f"nc-replay-{uuid.uuid4().hex[:8]}"
    (work / "streams.json").write_text(json.dumps(streams))
    lk = api.LiveKitAPI(_http_url(), config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
    procs: list[asyncio.subprocess.Process] = []
    sub_log, pub_log = work / "sub.log", work / "pub.log"
    try:
        await lk.room.create_room(api.CreateRoomRequest(name=room, empty_timeout=60))
        spawn = dict(cwd=str(config.PROJECT_ROOT), stdout=asyncio.subprocess.PIPE)
        with sub_log.open("wb") as se, pub_log.open("wb") as pe:
            sub = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "nc_bench.cloud_replay", "sub", room, str(work),
                stderr=se, **spawn)
            procs.append(sub)
            await _read_event(sub.stdout, "connected", 45)
            pub = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "nc_bench.cloud_replay", "pub", room, str(input_wav),
                stdin=asyncio.subprocess.PIPE, stderr=pe, **spawn)
            procs.append(pub)
            await _read_event(pub.stdout, "published", 45)
            ready = await _read_event(sub.stdout, "ready", 45)
            if ready.get("streams") != len(streams):
                raise RuntimeError(f"subscriber attached {ready.get('streams')} of {len(streams)} streams")
            pub.stdin.write(b"go\n")
            await pub.stdin.drain()

            async def on_pub(ev: dict) -> None:
                if progress is not None and ev.get("event") == "progress":
                    await progress({"batch": k + 1, "batches": n, "t": ev["t"],
                                    "total": ev["total"], "room": room, "ids": ids})

            pub_done = await _read_event(pub.stdout, "done", duration + PREROLL_S + 90, on_pub)
            await asyncio.wait_for(pub.wait(), 30)
            # the subscriber drains whatever its filters still hold, which can
            # trail real time when many Python-side filters share its loop
            sub_done = await _read_event(sub.stdout, "done", 180 + duration)
            await asyncio.wait_for(sub.wait(), 60)
        if sub.returncode != 0:
            raise RuntimeError(f"subscriber exited {sub.returncode}: {_tail(sub_log)}")
        return {"room": room, "batch": k + 1, "batches": n, "pub": pub_done, "sub": sub_done}
    except Exception as e:
        detail = _tail(sub_log) or _tail(pub_log)
        raise RuntimeError(f"{e}" + (f" — {detail}" if detail else "")) from e
    finally:
        for p in procs:
            if p.returncode is None:
                p.kill()
                try:
                    await asyncio.wait_for(p.wait(), 10)
                except Exception:
                    pass
        try:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room))
        except Exception:
            pass
        await lk.aclose()


# ============================================================ alignment + checks


def _mono(path: Path, rate: int) -> np.ndarray:
    x, r = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x if r == rate else soxr.resample(x, r, rate).astype(np.float32)


def _envelope(x: np.ndarray, hop: int = _ENV_HOP) -> np.ndarray:
    n = len(x) // hop
    return np.sqrt((x[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(axis=1))


def _best_lag(ref: np.ndarray, sig: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag in [0, max_lag] maximising Pearson corr of ref against sig[lag:lag+len(ref)].

    FFT cross-correlation plus running sums for each window's own mean and
    variance, so a quiet stretch of sig cannot win on scale alone. sig is
    zero-padded when it runs out, which only lowers the score of lags that would
    need audio the output does not have.
    """
    ref = np.asarray(ref, np.float64)
    sig = np.asarray(sig, np.float64)
    n = len(ref)
    need = n + max_lag
    if len(sig) < need:
        sig = np.concatenate([sig, np.zeros(need - len(sig))])
    sig = sig[:need]
    # centred first: the running sums then stay small, so a window's variance is
    # not the difference of two large, nearly equal numbers
    sig = sig - sig.mean()
    r = (ref - ref.mean()) / (ref.std() + 1e-12)
    nfft = 1 << int(math.ceil(math.log2(len(sig) + n)))
    cross = np.fft.irfft(np.fft.rfft(sig, nfft) * np.fft.rfft(r[::-1], nfft), nfft)
    cross = cross[n - 1 : n + max_lag]
    cs = np.concatenate([[0.0], np.cumsum(sig)])
    cs2 = np.concatenate([[0.0], np.cumsum(sig * sig)])
    lags = np.arange(max_lag + 1)
    s, s2 = cs[lags + n] - cs[lags], cs2[lags + n] - cs2[lags]
    var = np.maximum(s2 / n - (s / n) ** 2, 0.0)
    # A window that is nearly all silent tail or zero padding has a variance at
    # rounding-error level, and dividing by it produced "correlations" far above
    # 1: on a 4 s input the lead-in came out 7.2 s instead of 3.2 s (true lead
    # corr 1.0000). The floor is that rounding level — a few thousand ulps of the
    # running sum per sample — not a loudness threshold, so a genuinely quiet
    # match (100 dB under its neighbours) still counts; and a "correlation"
    # above 1, which only rounding can produce, is discarded too.
    floor = 1e-12 * cs2[-1] / n
    ok = var > floor
    if not ok.any():
        return 0, 0.0
    corr = np.where(ok, cross / (n * np.sqrt(np.where(ok, var, 1.0))), -np.inf)
    corr = np.where(corr > 1.0 + 1e-6, -np.inf, corr)
    best = int(np.argmax(corr))
    return best, float(corr[best]) if np.isfinite(corr[best]) else 0.0


def lead_in(x: np.ndarray, y: np.ndarray) -> int:
    """Samples of y before x starts: coarse on 10 ms envelopes, then refined on
    the waveform within ±20 ms (kept coarse if the waveform peak is weak — a
    heavy voice-isolation output can lose the fine structure)."""
    ex, ey = _envelope(x), _envelope(y)
    max_hops = max(0, min(len(ey) - len(ex) // 2, int(_MAX_LEAD_S * 100)))
    coarse, _ = _best_lag(ex, ey, max_hops)
    guess = coarse * _ENV_HOP
    lo = max(0, guess - 2 * _ENV_HOP)
    fine, c = _best_lag(x, y[lo:], guess - lo + 2 * _ENV_HOP)
    return lo + fine if c > 0.2 else guess


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def prominence(env_x: np.ndarray, env_y: np.ndarray) -> float:
    """Envelope corr at the alignment used, minus the best corr 150 ms-1 s off it.

    A speech envelope decorrelates within ~150 ms, so an output that really
    lines up has a clear peak at 0 whatever its NC did to the level; a
    misaligned or unrelated one is flat. Envelopes at 10 ms hops, same length.
    """
    def at(d: int) -> float:
        a, b = (env_x[: len(env_x) - d], env_y[d:]) if d >= 0 else (env_x[-d:], env_y[: len(env_y) + d])
        return _corr(a, b)

    far = max(at(s * d) for d in range(15, 101, 5) for s in (1, -1))
    return round(at(0) - far, 4)


def _db(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return round(10 * math.log10(max(num, 1e-30) / den), 2)


def local_lags_ms(x: np.ndarray, y_aligned: np.ndarray, win_s: float = 2.0) -> list[float]:
    """Local lag, in ms, of an aligned output against the input, per window.

    Anything inserted or dropped mid-file shifts everything after it: the global
    alignment fits one side and the windows on the other side read the shift. On
    a real replay that is the receiver's jitter buffer (NetEQ) time-stretching —
    steps of 17-33 ms, once or twice per 25 s, identical in every stream of the
    batch — or, with the machine saturated, ~50 ms steps. A starved publisher
    would add whole 10 ms frames. Only windows with speech-level energy and a
    clear correlation peak vote. Each window searches ±150 ms, so a single step
    larger than the MAX_DRIFT_MS gate is still seen rather than lost off the edge.
    """
    w, pad = int(win_s * _ALIGN_RATE), int(0.15 * _ALIGN_RATE)
    starts = range(pad, len(x) - w - pad, w)
    energy = [float(np.sqrt((x[s : s + w] ** 2).mean())) for s in starts]
    if not energy:
        return []
    floor = float(np.percentile(energy, 40))
    lags = []
    for s, e in zip(starts, energy):
        if e < floor or e <= 0:
            continue
        lag, c = _best_lag(x[s : s + w], y_aligned[s - pad : s + w + pad], 2 * pad)
        if c >= 0.5:
            lags.append(round((lag - pad) / _ALIGN_RATE * 1000, 1))
    return lags


def lag_drift_ms(x: np.ndarray, y_aligned: np.ndarray, win_s: float = 2.0) -> tuple[float, int]:
    """Spread (max - min) of the local lags, in ms, and how many windows voted.

    The spread, not the largest lag, because the global alignment lands on one
    of the segments and the outputs are NOT warped back: splicing the stretches
    out would put discontinuities — clicks — into audio that a VAD bench then
    scores for phantom turns. So every output is off the input's timeline by
    up to this much somewhere in the file.
    """
    lags = local_lags_ms(x, y_aligned, win_s)
    if not lags:
        return 0.0, 0
    return round(max(lags) - min(lags), 1), len(lags)


_FINE_HOP, _FINE_WIN = 32, 160  # 2 ms hops of a 10 ms RMS, at 16 kHz


def _fine_envelope(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if len(x) < _FINE_WIN:
        return np.zeros(0)
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    i = np.arange((len(x) - _FINE_WIN) // _FINE_HOP + 1) * _FINE_HOP
    return np.sqrt(np.maximum(cs[i + _FINE_WIN] - cs[i], 0.0) / _FINE_WIN)


def env_timeline(x: np.ndarray, y_aligned: np.ndarray) -> dict:
    """A stream's own timeline against the input, from envelopes.

    The waveform check (local_lags_ms) needs the fine structure, which heavy NC
    destroys; the energy envelope survives it. Per voiced window (the louder
    half, by input energy), the lag within ±ENV_LAG_SEARCH_S that best matches
    the input's envelope, kept only when its correlation reaches
    ENV_LAG_MIN_CORR and it is not pinned to the search edge. Returns
    {drift_ms: spread of those lags, lags_ms, votes, voiced, established},
    where established means enough windows voted to say where the audio is.
    """
    ex, ey = _fine_envelope(x), _fine_envelope(y_aligned)
    ms = _FINE_HOP / _ALIGN_RATE * 1000
    w, pad = int(ENV_LAG_WIN_S * 1000 / ms), int(ENV_LAG_SEARCH_S * 1000 / ms)
    starts = list(range(pad, len(ex) - w - pad, w))
    energy = np.array([ex[s : s + w].mean() for s in starts])
    if not len(energy) or energy.max() <= 0:
        return {"drift_ms": None, "confirmed_drift_ms": None, "confirmed": 0, "lags_ms": [],
                "votes": 0, "voiced": 0, "established": False}
    floor = float(np.percentile(energy, 50))
    voiced = [s for s, e in zip(starts, energy) if e >= floor and e > 0]
    lags = []
    for s in voiced:
        lag, c = _best_lag(ex[s : s + w], ey[s - pad : s + w + pad], 2 * pad)
        if c >= ENV_LAG_MIN_CORR and 1 <= lag <= 2 * pad - 1:
            lags.append(round((lag - pad) * ms, 1))
    votes = len(lags)
    # A real tear moves EVERY window after it by the same amount, so its lags
    # confirm each other; a lag no other window agrees with is noise. Krisp NC,
    # whose output is reshaped hardest, voted [-6, +40, +14, -62] on one run
    # while its samples matched its raw twin exactly and every other stream of
    # the batch read the control's [-46, 0, 0, 0, 0, 0, 0, 0].
    confirmed = [a for i, a in enumerate(lags)
                 if any(abs(a - b) <= ENV_LAG_CONFIRM_MS for j, b in enumerate(lags) if j != i)]
    return {
        "drift_ms": round(max(lags) - min(lags), 1) if lags else None,
        "confirmed_drift_ms": (round(max(confirmed) - min(confirmed), 1)
                               if len(confirmed) >= 2 else None),
        "confirmed": len(confirmed),
        "lags_ms": lags,
        "votes": votes,
        "voiced": len(voiced),
        "established": votes >= 1 and votes >= ENV_LAG_MIN_SHARE * len(voiced),
    }


def twin_vs_control(ctl: np.ndarray, twin: np.ndarray, onset: int | None = None) -> dict:
    """Where a raw twin sits in the raw control (both int16 at SUB_RATE), and
    whether it IS the control's audio there.

    `offset_ms`: the twin's first sample is the control's sample at this point
    (the twin opened later). `residual_db`: the twin minus that stretch of the
    control, over the whole overlap; `onset_residual_db` the same over the
    first second of audio from `onset` (a twin sample index). Legit reads an
    exact match. `len_vs_control_ms`: the twin's length against the control's
    from that offset.
    """
    c, t = ctl.astype(np.float64), twin.astype(np.float64)
    if not len(c) or not len(t):
        return {"offset_ms": None, "residual_db": None, "onset_residual_db": None,
                "len_vs_control_ms": None}
    # twins open up to a few hundred ms after the control; 2 s is generous
    off, _ = _best_lag(t, c, min(2 * SUB_RATE, max(0, len(c) - 1)))
    n = max(0, min(len(t), len(c) - off))
    seg = c[off : off + n]

    def res(a: int, b: int) -> float | None:
        e = float((seg[a:b] ** 2).sum())
        return _db(float(((t[a:b] - seg[a:b]) ** 2).sum()), e) if e > 0 else None

    return {
        "offset_ms": round(off / SUB_RATE * 1000, 1),
        "residual_db": res(0, n),
        "onset_residual_db": (None if onset is None
                              else res(max(0, onset), min(n, max(0, onset) + SUB_RATE))),
        "len_vs_control_ms": round((len(t) - (len(c) - off)) / SUB_RATE * 1000, 1),
    }


def _aicvad(flags: np.ndarray, lead_s: float, duration_s: float) -> tuple[np.ndarray, dict]:
    """The enhancer's per-frame flags, cut to the input's own timeline.

    flags: int8 per received frame (1 speech, 0 not, -1 frame had no flag), and
    the result keeps that encoding (lk_cloud.AICVAD_ENCODING): a frame without
    a flag must stay distinguishable from a confident "not speech", or a
    missing flag over a phantom reads as the VAD rejecting it. Frame k covers
    [t0 + k*frame_s, t0 + (k+1)*frame_s) of the input (and of <cid>.wav), with
    t0 in (-frame_s, 0]: the lead-in removed from the audio is removed here
    too, to the nearest frame below. Frames past the end of the stream are -1.
    """
    frame_s = SUB_FRAME_MS / 1000
    start = int(math.floor(lead_s / frame_s + 1e-9))
    t0 = start * frame_s - lead_s
    n = int(math.ceil((duration_s - t0) / frame_s - 1e-9))
    cut = np.asarray(flags[start : start + n], np.int8)
    if len(cut) < n:
        cut = np.concatenate([cut, -np.ones(n - len(cut), np.int8)])
    return cut, {
        "dtype": "int8",
        "encoding": lk_cloud.AICVAD_ENCODING,
        "frame_s": frame_s,
        "t0_s": round(t0, 4),
        "frames": int(n),
        "speech_frames": int((cut == 1).sum()),
        "missing_frames": int((cut == -1).sum()),
    }


def _finish_batch(input_wav: Path, streams: list[dict], work: Path, out_dir: Path,
                  meta: dict, raw_ids: list[str]) -> dict[str, dict]:
    """Align every stream of one batch, gate it, and write the outputs."""
    x = _mono(input_wav, _ALIGN_RATE)
    duration = len(x) / _ALIGN_RATE
    env_x = _envelope(x)
    subsum = {s["name"]: s for s in meta["sub"].get("streams", [])}
    versions = lk_cloud.plugin_versions()

    def load(name: str) -> tuple[np.ndarray, np.ndarray]:
        pcm = np.load(work / f"{name}.pcm.npy")
        flags = np.load(work / f"{name}.flags.npy")
        if not len(pcm):
            return np.zeros(0, np.float32), flags
        y = pcm.astype(np.float32) / 32768.0
        return soxr.resample(y, SUB_RATE, _ALIGN_RATE).astype(np.float32), flags

    def align(y: np.ndarray) -> tuple[int, np.ndarray, float]:
        lead = lead_in(x, y)
        seg = y[lead : lead + len(x)]
        short = (len(x) - len(seg)) / _ALIGN_RATE
        if len(seg) < len(x):
            seg = np.concatenate([seg, np.zeros(len(x) - len(seg), np.float32)])
        return lead, seg, short

    raw_y, _ = load("0")
    if not len(raw_y):
        raise RuntimeError("the control stream received no audio")
    raw_lead, raw_al, raw_short = align(raw_y)
    drift, drift_windows = lag_drift_ms(x, raw_al)
    # the same envelope measure every candidate gets, so the two compare like
    # for like (the waveform drift above is the finer transport check)
    ctl_tl = env_timeline(x, raw_al)
    raw_tail = len(raw_y) - raw_lead - len(x)

    batch_info = {
        "rail": "cloud_replay",
        "room": meta["room"],
        "batch": meta["batch"],
        "batches": meta["batches"],
        "streams_in_batch": len(streams),
        "bitrate_kbps": BITRATE_KBPS,
        "red": False,
        "dtx": False,
        "publish_rate": PUB_RATE,
        "subscribe_rate": SUB_RATE,
        "frame_ms": SUB_FRAME_MS,
        "preroll_s": PREROLL_S,
        # receive-side time-stretching: the spread of the control's local lag
        # against the input across the file — every stream of the batch shares
        # it, and is off the input's timeline by up to this much somewhere
        "control_lag_drift_ms": drift,
        "control_drift_windows": drift_windows,
        # the batch's unfiltered control against the input, on every record, so
        # a batch whose cloud-raw is not stored can still be judged
        "control_env_corr": round(_corr(env_x, _envelope(raw_al)), 4),
        "control_env_lag_drift_ms": ctl_tl["drift_ms"],
        "control_env_lag_confirmed_drift_ms": ctl_tl["confirmed_drift_ms"],
        "publisher_wall_s": meta["pub"].get("wall_s"),
        "publisher_audio_s": meta["pub"].get("audio_s"),
        "publisher_starved_ms": meta["pub"].get("starved_ms"),
        "subscriber_max_backlog_s": meta["sub"].get("max_backlog_s"),
    }

    def record(name: str, y: np.ndarray, lead: int, al: np.ndarray, short: float) -> dict:
        s = subsum.get(name) or {}
        rms_x = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
        rms_y = float(np.sqrt((al.astype(np.float64) ** 2).mean()))
        env_y = _envelope(al)
        tl = ctl_tl if name == "0" else env_timeline(x, al)
        return {
            **batch_info,
            "lead_in_s": round(lead / _ALIGN_RATE, 4),
            # lead-in beyond the control's. NOT a clean filter delay: each stream
            # starts at whatever audio follows its creation, so this also holds
            # the start skew between streams (the ai-coustics bypass, measured at
            # ~30 ms by PR-54, read -10 ms here)
            "lead_vs_control_ms": round((lead - raw_lead) / _ALIGN_RATE * 1000, 1),
            "env_corr": round(_corr(env_x, env_y), 4),
            "env_prominence": prominence(env_x, env_y),
            # sample-level agreement; also drops with any jitter-buffer step
            # (control_lag_drift_ms), so read it alongside that
            "wave_corr": round(_corr(x, al), 4),
            "level_db_vs_input": _db(rms_y ** 2, rms_x ** 2),
            "input_s": round(duration, 3),
            "output_s": round(len(y) / _ALIGN_RATE, 3),
            "tail_s": round((len(y) - lead - len(x)) / _ALIGN_RATE, 3),
            "short_s": round(max(0.0, short), 3),
            # length against the control's, both after their own lead-in: all
            # streams end at the same point of the track, so beyond the filter's
            # delay this is audio the stream inserted (+) or lost (-)
            "excess_ms": round((len(y) - lead - len(x) - raw_tail) / _ALIGN_RATE * 1000, 1),
            # this stream's own timeline: spread of its local envelope lags
            "lag_drift_ms": tl["drift_ms"],
            "lag_confirmed_drift_ms": tl["confirmed_drift_ms"],
            "lag_confirmed": tl["confirmed"],
            "lag_votes": tl["votes"],
            "lag_voiced": tl["voiced"],
            "lag_local_ms": tl["lags_ms"],
            "cpu_per_audio_s": s.get("cpu_per_audio_s"),
            "frames": s.get("frames"),
        }

    def gate(rec: dict, control: bool = False) -> str:
        if not 0 <= rec["lead_in_s"] <= _MAX_LEAD_S:
            return f"lead-in {rec['lead_in_s']} s is outside 0–{_MAX_LEAD_S:g} s"
        if rec["env_prominence"] < MIN_PROMINENCE:
            return (f"output does not line up with the input (no clear envelope "
                    f"correlation peak: prominence {rec['env_prominence']} < {MIN_PROMINENCE})")
        if rec["env_corr"] < MIN_ENV_CORR:
            return (f"output barely follows the input (envelope corr {rec['env_corr']} "
                    f"< {MIN_ENV_CORR}) — silence or garbage")
        if rec["short_s"] > MAX_SHORT_S:
            return f"output ends {rec['short_s']} s before the input does (truncated)"
        if control:
            return ""
        # the crop to the input length would hide an insertion, so it is judged
        # here, on the uncropped stream, before anything is written
        anchor = rec.get("twin_vs_control")
        if anchor is not None:
            # first the reference itself: the twin must be the control's audio
            res = anchor.get("residual_db")
            if res is None or res > MAX_TWIN_RESIDUAL_DB:
                return (f"its raw twin is not the control's audio (residual {res} dB; legit "
                        "is an exact match): a gap or stall in this pair's receive path")
            err = anchor.get("offset_err_ms")
            if err is not None and abs(err) > MAX_TWIN_SKEW_ERR_MS:
                return (f"its raw twin starts {err:+g} ms from where its opening time puts it "
                        f"(allowed ±{MAX_TWIN_SKEW_ERR_MS:g}): audio inserted or lost at the "
                        "onset of this pair")
            lo, hi = TWIN_LEN_VS_CONTROL_MS
            lvc = anchor.get("len_vs_control_ms")
            if lvc is None or not lo <= lvc <= hi:
                return (f"its raw twin's length is {lvc} ms against the control's "
                        f"(allowed {lo:g}..{hi:g}): this pair inserted or lost audio")
        if rec.get("twin_len_diff_ms") is not None:
            # exact: same start, same end, and filters keep frame length
            if abs(rec["twin_len_diff_ms"]) > MAX_TWIN_LEN_DIFF_MS:
                return (f"output is {rec['twin_len_diff_ms']:+g} ms against the raw stream "
                        f"opened with it (legit: 0): this stream "
                        f"{'inserted' if rec['twin_len_diff_ms'] > 0 else 'lost'} audio")
            if not MIN_ONSET_DELAY_MS <= rec["onset_delay_ms"] <= MAX_ONSET_DELAY_MS:
                return (f"its audio starts {rec['onset_delay_ms']:+g} ms against the raw stream "
                        f"opened with it (a filter's own delay: 26-40 ms measured; allowed "
                        f"{MIN_ONSET_DELAY_MS:g}..{MAX_ONSET_DELAY_MS:g}): the opening was "
                        "shifted — audio inserted or lost near the onset")
        else:
            # no twin (a batch planned before twins existed): the control is
            # the only length reference, with its start skew cancelled by the lead
            if rec["excess_ms"] > MAX_EXCESS_MS:
                return (f"output runs {rec['excess_ms']} ms longer than the control's "
                        f"(> {MAX_EXCESS_MS:g}): this stream inserted audio")
            if rec["excess_ms"] < MIN_EXCESS_MS:
                return (f"output runs {-rec['excess_ms']} ms shorter than the control's "
                        f"(> {-MIN_EXCESS_MS:g}): this stream lost audio")
        if ctl_tl["established"] and rec.get("twin_len_diff_ms") is not None:
            # With a twin, net inserted or lost audio anywhere is already
            # caught exactly (twin_len_diff_ms) and so is the onset; the
            # envelope drift remains for a shift that keeps the length, and is
            # judged on confirmed lags only, so one distorted window cannot fail
            # a stream whose samples are proven intact. It abstains (recorded)
            # when fewer than two lags confirm each other.
            cand_d, ctl_d = rec.get("lag_confirmed_drift_ms"), ctl_tl["confirmed_drift_ms"]
            if cand_d is not None:
                over = cand_d - (ctl_d or 0.0)
                if over > MAX_CAND_DRIFT_OVER_CONTROL_MS:
                    return (f"its own timeline drifts {cand_d} ms against the input (confirmed "
                            f"windows), {round(over, 1)} ms more than its control's "
                            f"({ctl_d} ms): audio was shifted inside this stream")
        elif ctl_tl["established"]:
            # no twin: the envelope timeline is the only mid-file evidence, so
            # it is strict — every voting window counts
            if not (rec["lag_votes"] >= 1
                    and rec["lag_votes"] >= ENV_LAG_MIN_SHARE * rec["lag_voiced"]):
                return (f"its timeline can't be established: {rec['lag_votes']} of "
                        f"{rec['lag_voiced']} voiced windows matched the input's envelope")
            over = rec["lag_drift_ms"] - (ctl_tl["drift_ms"] or 0.0)
            if over > MAX_CAND_DRIFT_OVER_CONTROL_MS:
                return (f"its own timeline drifts {rec['lag_drift_ms']} ms against the "
                        f"input, {round(over, 1)} ms more than its control's "
                        f"({ctl_tl['drift_ms']} ms): audio was inserted or dropped "
                        "inside this stream")
        return ""

    def write(cid: str, al: np.ndarray) -> str:
        out = out_dir / f"{cid}.wav"
        y16 = al if _ALIGN_RATE == config.PIPELINE_RATE else soxr.resample(
            al, _ALIGN_RATE, config.PIPELINE_RATE)
        sf.write(out, (np.clip(y16, -1.0, 1.0) * 32767).astype(np.int16), config.PIPELINE_RATE)
        return out.name

    results: dict[str, dict] = {}
    raw_rec = record("0", raw_y, raw_lead, raw_al, raw_short)
    # batch-level gates: the transport is shared, so these fail every stream —
    # each still carries the control's record, so the failure can be read
    starved = float(meta["pub"].get("starved_ms") or 0.0)
    batch_error = ""
    if starved > MAX_STARVED_MS:
        batch_error = (
            f"the publisher stalled: its send queue ran dry for {starved} ms over "
            f"{meta['pub'].get('starved_events')} gaps, so every stream of this batch "
            "carries silence the recording never had")
    elif drift > MAX_DRIFT_MS:
        batch_error = (
            f"the receiver's jitter buffer stretched the timeline by {drift} ms across "
            f"the file (> {MAX_DRIFT_MS:g} ms; subscriber up to "
            f"{batch_info['subscriber_max_backlog_s']} s behind real time — was the "
            "machine busy?)")
    elif raw_rec["env_corr"] < CONTROL_MIN_ENV_CORR:
        # seen once: the machine was saturated, the subscriber fell 5.4 s behind
        # and the jitter buffer stretched the audio in ~50 ms steps
        batch_error = (
            f"the unfiltered control only correlates {raw_rec['env_corr']} with the input "
            f"(< {CONTROL_MIN_ENV_CORR}): the transport time-stretched or mangled the audio "
            f"(subscriber was up to {batch_info['subscriber_max_backlog_s']} s behind real "
            "time — was the machine busy?)")
    if batch_error:
        return {cid: {"error": f"cloud replay batch {meta['batch']}/{meta['batches']}: "
                               f"{batch_error}", "cloud_replay": raw_rec}
                for cid in [s["id"] for s in streams[1:] if s.get("id")] + raw_ids}
    for cid in raw_ids:
        why = gate(raw_rec, control=True)
        results[cid] = {"error": f"cloud-raw: {why}", "cloud_replay": raw_rec} if why else {
            "output": write(cid, raw_al), "duration_s": round(duration, 2),
            "cloud_replay": raw_rec, "plugins": versions,
        }

    raw_energy = float((raw_al.astype(np.float64) ** 2).sum())
    twins = {s["twin_of"]: s["name"] for s in streams if s.get("twin_of")}
    ctl_pcm = np.load(work / "0.pcm.npy")
    ctl_open = (subsum.get("0") or {}).get("open_ms")
    for st in streams[1:]:
        if not st.get("id"):
            continue  # a twin: read only as its candidate's reference
        cid, model, opts = st["id"], st["lk_model"], st.get("lk_options") or {}
        # a missing or torn artifact fails its own candidate, not the batch
        try:
            y, flags = load(st["name"])
            pcm_cand = np.load(work / f"{st['name']}.pcm.npy", mmap_mode="r")
        except (OSError, ValueError, EOFError) as e:
            results[cid] = {"error": f"this stream's recording is missing or unreadable: {e}"}
            continue
        if not len(y):
            results[cid] = {"error": "no audio received on this stream"}
            continue
        lead, al, short = align(y)
        rec = record(st["name"], y, lead, al, short)
        rec["residual_vs_control_db"] = _db(
            float(((al.astype(np.float64) - raw_al) ** 2).sum()), raw_energy)
        twin = twins.get(st["name"])
        if twin is not None:
            try:
                pcm_twin = np.load(work / f"{twin}.pcm.npy")
                y_twin, _ = load(twin)
            except (OSError, ValueError, EOFError) as e:
                results[cid] = {"error": f"its raw twin's recording is missing or unreadable "
                                         f"({e}), so its onset can't be checked",
                                "cloud_replay": rec}
                continue
            lead_twin = lead_in(x, y_twin) if len(y_twin) else None
            if lead_twin is None:
                results[cid] = {"error": "its raw twin received no audio, so its onset "
                                         "can't be checked", "cloud_replay": rec}
                continue
            rec["twin_len_diff_ms"] = round((len(pcm_cand) - len(pcm_twin)) / SUB_RATE * 1000, 1)
            # the filter's own delay, measured against a stream that started at
            # the same point; an onset gap the lead-in absorbed adds to it
            rec["onset_delay_ms"] = round((lead - lead_twin) / _ALIGN_RATE * 1000, 1)
            # ...and the twin anchored to the control, so a fault the pair shares
            # cannot hide: see MAX_TWIN_RESIDUAL_DB
            anchor = twin_vs_control(ctl_pcm, pcm_twin,
                                     onset=int(lead_twin * SUB_RATE / _ALIGN_RATE))
            rec["twin_vs_control"] = anchor
            twin_open = (subsum.get(twin) or {}).get("open_ms")
            predicted = (twin_open[1] - ctl_open[1]
                         if twin_open and ctl_open and None not in (twin_open[1], ctl_open[1])
                         else None)
            anchor["predicted_offset_ms"] = None if predicted is None else round(predicted, 1)
            anchor["offset_err_ms"] = (None if predicted is None or anchor["offset_ms"] is None
                                       else round(anchor["offset_ms"] - predicted, 1))
        why = gate(rec)
        if not why and lk_cloud.changes_audio(model, opts) and (
                rec["residual_vs_control_db"] is not None
                and rec["residual_vs_control_db"] < PASSTHROUGH_DB):
            why = (f"output is identical to the unfiltered control (residual "
                   f"{rec['residual_vs_control_db']} dB) — the filter never ran; check "
                   "the project's Cloud noise-cancellation enablement")
        if not why and lk_cloud.expects_raw(model, opts) and (
                rec["residual_vs_control_db"] is None
                or rec["residual_vs_control_db"] > RAW_MAX_DB):
            # plugin 0.3.2 ignores bypass=1.0 on the Voice Focus models: the
            # "raw" variant would silently be the normal enhancer
            why = (f"{opts} was not honoured: the output differs from the unfiltered "
                   f"control by {rec['residual_vs_control_db']} dB (> {RAW_MAX_DB:g}), so "
                   "this is enhanced audio, not raw")
        aicvad = None
        if not why and lk_cloud.is_aic(model):
            if not (flags >= 0).any():
                why = f"no {lk_cloud.AIC_VAD_KEY} flag on any frame — the enhancer never ran"
            else:
                cut, aicvad = _aicvad(flags, lead / _ALIGN_RATE, duration)
                np.save(out_dir / f"{cid}.aicvad.npy", cut)
                aicvad = {"file": f"{cid}.aicvad.npy",
                          "source": f"{lk_cloud.AIC_VAD_KEY} per-frame enhancer flag "
                                    f"(livekit-plugins-ai-coustics "
                                    f"{versions.get('livekit-plugins-ai-coustics', '?')})",
                          **aicvad}
        if why:
            results[cid] = {"error": why, "cloud_replay": rec}
            continue
        results[cid] = {"output": write(cid, al), "duration_s": round(duration, 2),
                        "cloud_replay": rec, "plugins": versions}
        if aicvad is not None:
            results[cid]["aicvad"] = aicvad
            # top level too, so a missing-flag count is visible without
            # opening the flags record
            results[cid]["aicvad_missing_frames"] = aicvad["missing_frames"]
    return results


# ===================================================================== roles


def _emit(event: str, **kw) -> None:
    sys.stdout.write(json.dumps({"event": event, **kw}) + "\n")
    sys.stdout.flush()


def _token(identity: str, room: str, publish: bool) -> str:
    from livekit import api

    return (
        api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=publish,
                                     can_subscribe=not publish))
        .to_jwt()
    )


async def _publish(room_name: str, input_wav: Path) -> None:
    from livekit import rtc
    from livekit.rtc._proto import room_pb2

    x = _mono(input_wav, PUB_RATE)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    n = PUB_RATE * PUB_FRAME_MS // 1000
    pcm = np.concatenate([np.zeros(int(PREROLL_S * PUB_RATE), np.int16), pcm,
                          np.zeros(int(TAIL_S * PUB_RATE) + (-len(pcm)) % n, np.int16)])
    total = len(pcm) // n

    room = rtc.Room()
    await room.connect(config.LIVEKIT_URL, _token(_PUB_ID, room_name, True))
    source = rtc.AudioSource(PUB_RATE, 1, queue_size_ms=200)
    track = rtc.LocalAudioTrack.create_audio_track("replay", source)
    await room.local_participant.publish_track(track, rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE, dtx=False, red=False,
        audio_encoding=room_pb2.AudioEncoding(max_bitrate=BITRATE_KBPS * 1000)))
    _emit("published")
    # wait until the subscriber has every stream attached, so no filter misses
    # the start (the parent relays the subscriber's "ready" as this line)
    line = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(
        None, sys.stdin.readline), 120)
    if line.strip() != "go":
        raise RuntimeError("no go signal from the parent")

    t0 = time.monotonic()
    step = int(5000 / PUB_FRAME_MS)
    # A stall is the send queue running dry: the encoder then sends silence the
    # recording never had (PR-42's 29 gaps). queued_duration says when the queue
    # will empty, so any capture that arrives later than that was starved for the
    # difference. This, not the receiver's timeline, is the stall proof — the
    # receiver's jitter buffer time-stretches on its own (see lag_drift_ms).
    empty_at, starved_s, starved_n = None, 0.0, 0
    for i in range(total):
        now = time.monotonic()
        if empty_at is not None and now > empty_at + 0.001:
            starved_s += now - empty_at
            starved_n += 1
        await source.capture_frame(rtc.AudioFrame(pcm[i * n : (i + 1) * n].tobytes(), PUB_RATE, 1, n))
        empty_at = time.monotonic() + source.queued_duration
        if i % step == 0:
            _emit("progress", t=round(i * PUB_FRAME_MS / 1000, 1),
                  total=round(total * PUB_FRAME_MS / 1000, 1))
    await source.wait_for_playout()
    wall = time.monotonic() - t0
    await asyncio.sleep(1.5)
    await room.disconnect()
    _emit("done", wall_s=round(wall, 2), audio_s=round(total * PUB_FRAME_MS / 1000, 2),
          starved_ms=round(starved_s * 1000, 1), starved_events=starved_n)


def _timed(inner):
    """Wrap a FrameProcessor to bill the CPU it spends per frame.

    The filters run on the subscriber's event-loop thread (AudioStream._run calls
    _process inline), so thread_time around _process is that filter's own cost —
    the Cloud only authenticates, the model runs here.
    """
    from livekit import rtc

    class Timed(rtc.FrameProcessor[rtc.AudioFrame]):
        cpu_s = 0.0

        @property
        def enabled(self) -> bool:
            return inner.enabled

        @enabled.setter
        def enabled(self, value: bool) -> None:
            inner.enabled = value

        def _process(self, frame):
            t = time.thread_time()
            try:
                return inner._process(frame)
            finally:
                self.cpu_s += time.thread_time() - t

        def _on_stream_info_updated(self, **kw) -> None:
            inner._on_stream_info_updated(**kw)

        def _on_stream_info_cleared(self) -> None:
            getattr(inner, "_on_stream_info_cleared", lambda: None)()

        def _on_credentials_updated(self, **kw) -> None:
            inner._on_credentials_updated(**kw)

        def _on_credentials_cleared(self) -> None:
            getattr(inner, "_on_credentials_cleared", lambda: None)()

        def _close(self) -> None:
            inner._close()

    return Timed()


async def _subscribe(room_name: str, work: Path) -> None:
    from livekit import rtc

    streams = json.loads((work / "streams.json").read_text())
    state = {s["name"]: {"pcm": [], "flags": [], "first": None, "last": None, "late": 0.0,
                         "proc": None, "done": False} for s in streams}
    tasks: list[asyncio.Task] = []
    handles: list = []
    gone = asyncio.Event()
    frame_s = SUB_FRAME_MS / 1000

    async def record(name: str, stream) -> None:
        st = state[name]
        try:
            async for ev in stream:
                f = ev.frame
                now = time.monotonic()
                if st["first"] is None:
                    st["first"] = now
                # how far behind real time this stream is being delivered: the
                # filters share one loop, so a slow one delays all of them
                st["late"] = max(st["late"], now - (st["first"] + len(st["pcm"]) * frame_s))
                st["pcm"].append(np.frombuffer(f.data, dtype=np.int16).copy())
                v = (f.userdata or {}).get(lk_cloud.AIC_VAD_KEY)
                st["flags"].append(-1 if v is None else int(bool(v)))
                st["last"] = now
        finally:
            st["done"] = True

    def attach(track) -> None:
        # Build every filter FIRST, then open the streams back to back: a stream
        # starts at whatever audio arrives after it is created, and building a
        # Krisp VIVA or ai-coustics filter takes tens of ms — built in between,
        # the later streams started ~60 ms after the control, which made each
        # one's lead-in against the control (its proc_delay_ms) meaningless.
        filters = []
        for s in streams:
            nc = lk_cloud.build(s["lk_model"], s.get("lk_options"))
            if nc is not None and isinstance(nc, rtc.FrameProcessor):
                nc = _timed(nc)
                state[s["name"]]["proc"] = nc
            filters.append(nc)
        t_attach = time.monotonic()
        for s, nc in zip(streams, filters):
            state[s["name"]]["created"] = [time.monotonic() - t_attach, None]
            stream = rtc.AudioStream.from_track(
                track=track, sample_rate=SUB_RATE, num_channels=1,
                frame_size_ms=SUB_FRAME_MS, noise_cancellation=nc)
            # how long opening this stream took: a Krisp native filter
            # initialises inside the call, which delays every stream after it
            state[s["name"]]["created"][1] = time.monotonic() - t_attach
            handles.append(stream)
            tasks.append(asyncio.create_task(record(s["name"], stream)))
        _emit("ready", streams=len(streams))

    room = rtc.Room()

    @room.on("track_subscribed")
    def _on_track(track, _pub, participant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity == _PUB_ID and not tasks:
            try:
                attach(track)
            except Exception as e:  # noqa: BLE001 - reported to the parent
                _emit("error", error=f"could not attach streams: {e}")
                gone.set()

    @room.on("participant_disconnected")
    def _on_left(participant) -> None:
        if participant.identity == _PUB_ID:
            gone.set()

    await room.connect(config.LIVEKIT_URL, _token(_SUB_ID, room_name, False))
    _emit("connected")
    await gone.wait()
    # drain: every stream either ended or has delivered nothing for 2 s
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        now = time.monotonic()
        if all(st["done"] or (st["last"] is not None and now - st["last"] > 2.0)
               or st["first"] is None for st in state.values()):
            break
        await asyncio.sleep(0.2)
    for s in handles:
        try:
            await asyncio.wait_for(s.aclose(), 5)
        except Exception:
            pass
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await room.disconnect()

    summary = []
    for s in streams:
        st = state[s["name"]]
        pcm = np.concatenate(st["pcm"]) if st["pcm"] else np.zeros(0, np.int16)
        np.save(work / f"{s['name']}.pcm.npy", pcm)
        np.save(work / f"{s['name']}.flags.npy", np.array(st["flags"], np.int8))
        audio_s = len(pcm) / SUB_RATE
        proc = st["proc"]
        created = st.get("created") or [None, None]
        summary.append({
            "name": s["name"], "frames": len(st["flags"]), "audio_s": round(audio_s, 3),
            "late_s": round(st["late"], 3),
            # stream open, begin/end, ms after the first one opened
            "open_ms": [None if c is None else round(c * 1000, 1) for c in created],
            "cpu_per_audio_s": (round(proc.cpu_s / audio_s, 4)
                                if proc is not None and audio_s > 0 else None),
        })
    _emit("done", streams=summary,
          max_backlog_s=max((x["late_s"] for x in summary), default=0.0))


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    role, room = argv[0], argv[1]
    if role == "sub":
        lk_cloud.preload()  # plugins register on the main thread, before the loop
    try:
        if role == "pub":
            asyncio.run(_publish(room, Path(argv[2])))
        elif role == "sub":
            asyncio.run(_subscribe(room, Path(argv[2])))
            # Everything is saved and "done" is sent. Skip interpreter teardown:
            # the Krisp / ai-coustics native filters SIGABRT in their destructors
            # about half the time on Linux ("panic in a function that cannot
            # unwind", Nc::~Nc), and the exit code alone fails the whole batch.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        else:
            raise SystemExit(f"unknown role {role!r}")
    except Exception as e:
        _emit("error", error=f"{role}: {type(e).__name__}: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main(sys.argv[1:])
