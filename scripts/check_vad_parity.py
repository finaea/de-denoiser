"""Does the VAD bench reproduce what it replaces? Three checks, all read-only.

    .venv/bin/python scripts/check_vad_parity.py                 # all three
    .venv/bin/python scripts/check_vad_parity.py --only loop      # just A
    .venv/bin/python scripts/check_vad_parity.py --runs 20260804-112511,... --only runs

A. loop: the shared segmenter against livekit's OWN stream loops. Scripted
   probabilities are fed through livekit-plugins-silero's VADStream (with its
   ExpFilter) and livekit.agents.inference's _VADStream (unsmoothed) by
   swapping the model for a fake. Spans must be identical, since the state
   machine is the only thing under test.

B. runs: on stored runs, span sets from the new path vs the plugin path, per file
   (input + every candidate output):
   B1  vad.analyze() today (livekit-plugins-silero 1.8.3 = Silero v6.2 ONNX,
       ExpFilter) vs silero-v6.2-<rate> + smoothing, both given the same
       resampled audio (so neither side resamples);
   B1n the same, but the plugin fed the native-rate file and resampling in its
       stream (rtc QUICK, which jitters): informational, noise-aware;
   B2  the spans STORED in meta (livekit-plugins-silero 1.6.7 = Silero v6.0
       ONNX, ExpFilter) vs silero-v6.0-<rate> + smoothing;
   B3  livekit.agents.inference.VAD's own START/END events vs lk-inference
       probabilities through the segmenter (unsmoothed).
   Reported as span-count agreement and the worst edge difference, in 32 ms
   frames.

C. bakeoff: probabilities on $VAD_BAKEOFF_DIR/pr42_caller.nc.wav from the
   new loaders vs the PR-42 bake-off's own code (vad_bench.py imported as-is;
   probe_loaders.py's functions copied below verbatim, since that module runs
   an sklearn/scipy probe at import). Same resampled input to both, so the
   correlation isolates the loader. Also against the bake-off's STORED output:
   native_nc.npy (the production engine's probabilities) and the peak table
   in bench_results*.json.

D. firered: firered-stream's whole-sequence call vs live chunked streaming with
   carried caches (the publisher's _with_cache graph): 20/50 ms waveform chunks
   into one OnlineFbank, 10/20/50 ms of model input per call, on 3 stored files.

Writes nothing under data/runs/.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nc_bench import store, vad, vads  # noqa: E402
from nc_bench.vads.loaders import _i16  # noqa: E402
from nc_bench.vads.segmenter import segment  # noqa: E402

# the bake-off's folder is private; C is skipped when it isn't set or isn't there
BAKEOFF = Path(os.environ.get("VAD_BAKEOFF_DIR", "/nonexistent"))
FRAME = 0.032


# ------------------------------------------------------------ A: the loop


def _events_to_spans(events, duration_s):
    """nc_bench/vad.py's START/END -> span conversion, verbatim."""
    from livekit.agents import vad as agents_vad

    segments, open_start = [], None
    for ev in events:
        if ev.type is agents_vad.VADEventType.START_OF_SPEECH:
            open_start = max(0.0, float(ev.timestamp) - float(ev.speech_duration))
        elif ev.type is agents_vad.VADEventType.END_OF_SPEECH:
            end = float(ev.timestamp) - float(ev.silence_duration)
            start = max(0.0, end - float(ev.speech_duration))
            segments.append([round(start, 3), round(min(end, duration_s or end), 3)])
            open_start = None
    if open_start is not None and duration_s > open_start:
        segments.append([round(open_start, 3), round(duration_s, 3)])
    return segments


async def _drive(stream, n_samples, rate=16000, push=1600):
    from livekit import rtc

    zeros = np.zeros(push, dtype=np.int16)
    for i in range(0, n_samples, push):
        k = min(push, n_samples - i)
        stream.push_frame(rtc.AudioFrame(data=zeros[:k].tobytes(), sample_rate=rate,
                                         num_channels=1, samples_per_channel=k))
    stream.end_input()
    evs = [ev async for ev in stream]
    await stream.aclose()
    return evs


class _Scripted:
    def __init__(self, probs):
        self.probs, self.i = list(probs), 0
        self.window_size_samples, self.sample_rate = 512, 16000

    def __call__(self, _x):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return float(p)

    predict = __call__

    def reset(self):
        pass


async def _plugin_spans(probs, p):
    from livekit.plugins.silero import vad as sv

    opts = sv._VADOptions(min_speech_duration=p["min_speech_duration"],
                          min_silence_duration=p["min_silence_duration"],
                          prefix_padding_duration=0.5, max_buffered_speech=60.0,
                          activation_threshold=p["activation_threshold"],
                          deactivation_threshold=p["deactivation_threshold"], sample_rate=16000)
    parent = sv.VAD(session=None, opts=opts)
    stream = sv.VADStream(parent, opts, _Scripted(probs))
    n = len(probs) * 512
    return _events_to_spans(await _drive(stream, n), n / 16000)


async def _inference_spans(probs, p):
    from livekit.agents import inference

    stream = inference.VAD(min_speech_duration=p["min_speech_duration"],
                           min_silence_duration=p["min_silence_duration"],
                           activation_threshold=p["activation_threshold"],
                           deactivation_threshold=p["deactivation_threshold"]).stream()
    stream._native_vad = _Scripted(probs)   # before the task's first await
    n = len(probs) * 512
    return _events_to_spans(await _drive(stream, n), n / 16000)


def check_loop() -> bool:
    rng = np.random.default_rng(7)
    sets = [
        dict(activation_threshold=0.5, deactivation_threshold=0.5, min_speech_duration=0.05,
             min_silence_duration=0.4),     # nc_bench defaults
        dict(activation_threshold=0.6, deactivation_threshold=0.6, min_speech_duration=0.05,
             min_silence_duration=0.3),     # production
        dict(activation_threshold=0.6, deactivation_threshold=0.35, min_speech_duration=0.12,
             min_silence_duration=0.55),    # livekit defaults, wide hysteresis
        dict(activation_threshold=0.5, deactivation_threshold=0.5, min_speech_duration=0.0,
             min_silence_duration=0.05),    # edge: every speech window opens
    ]
    ok = True
    for trial in range(12):
        # bursts of speech-like runs over noise, so turns open, bridge and close
        n = int(rng.integers(200, 900))
        probs = np.clip(rng.normal(0.2, 0.15, n), 0, 1)
        for _ in range(int(rng.integers(2, 9))):
            a = int(rng.integers(0, n - 5))
            probs[a:a + int(rng.integers(1, 40))] = np.clip(rng.normal(0.8, 0.12), 0, 1)
        for p in sets:
            want_p = asyncio.run(_plugin_spans(probs, p))
            got_p = segment(probs, FRAME, duration_s=n * 512 / 16000, smoothing=0.35, **p)
            want_i = asyncio.run(_inference_spans(probs, p))
            got_i = segment(probs, FRAME, duration_s=n * 512 / 16000, smoothing=None, **p)
            if want_p != got_p or want_i != got_i:
                ok = False
                print(f"  MISMATCH trial {trial} {p}\n    plugin    {want_p}\n    segmenter {got_p}"
                      f"\n    inference {want_i}\n    segmenter {got_i}")
    print(f"A. loop: 12 random probability tracks × 4 parameter sets through the real livekit "
          f"loops — {'IDENTICAL spans' if ok else 'MISMATCHES above'}")
    return ok


# ------------------------------------------------------------- B: stored runs


def _edge_diff(a, b):
    """(same count?, worst |edge delta| in frames over paired spans)."""
    if len(a) != len(b):
        return False, None
    if not a:
        return True, 0.0
    d = max(max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(a, b))
    return True, d / FRAME


async def _inference_file_spans(wav, p):
    from livekit import rtc
    from livekit.agents import inference

    data, rate = sf.read(wav, dtype="int16", always_2d=True)
    x = np.ascontiguousarray(data[:, 0])
    stream = inference.VAD(min_speech_duration=p["min_speech_duration"],
                           min_silence_duration=p["min_silence_duration"],
                           activation_threshold=p["activation_threshold"],
                           deactivation_threshold=p["deactivation_threshold"]).stream()
    step = rate // 10
    for i in range(0, len(x), step):
        c = x[i:i + step]
        stream.push_frame(rtc.AudioFrame(data=c.tobytes(), sample_rate=rate, num_channels=1,
                                         samples_per_channel=len(c)))
    stream.end_input()
    evs = [ev async for ev in stream]
    await stream.aclose()
    return _events_to_spans(evs, len(x) / rate)


def _plugin_on_resampled(wav, p):
    """vad.analyze() on the file already resampled to the plugin's inference
    rate with the bench's resampler, so the plugin does no resampling itself."""
    data, rate = sf.read(wav, dtype="int16", always_2d=True)
    x = _i16(np.ascontiguousarray(data[:, 0]), rate, p["sample_rate"])
    with tempfile.TemporaryDirectory(prefix="vadparity-") as tmp:
        f = Path(tmp) / "at-rate.wav"
        sf.write(f, x, p["sample_rate"], subtype="PCM_16")
        return asyncio.run(vad.analyze(f, p))["segments"]


def _new_spans(vad_id, wav, p, smoothing):
    data, rate = sf.read(wav, dtype="int16", always_2d=True)
    x = np.ascontiguousarray(data[:, 0])
    r = vads.infer(vad_id, x, rate)
    return segment(r["probs"], r["hop_s"], t0_s=r["t0_s"], duration_s=len(x) / rate,
                   smoothing=smoothing, **{k: p[k] for k in (
                       "activation_threshold", "deactivation_threshold",
                       "min_speech_duration", "min_silence_duration")})


def _params(saved, meta):
    """A run's saved VAD params. One saved before the rate was selectable was
    measured at the plugin default, 16 kHz; vad.clean() would fill in the
    source default (8 kHz for phone) instead and compare the wrong thing."""
    saved = dict(saved or {})
    saved.setdefault("sample_rate", 16000)
    return vad.clean(saved, meta.get("source"))


def check_runs(run_ids) -> bool:
    ok = True
    for rid in run_ids:
        meta = store.load_meta(rid)
        d = store.run_dir(rid)
        p = _params(meta.get("vad_params") or vad.defaults(meta.get("source")), meta)
        k = f"{p['sample_rate'] // 1000}k"
        files = [("input", meta["input"]["file"], meta.get("input_vad"))] + [
            (c["id"], c["output"], c.get("vad")) for c in meta.get("candidates") or []
            if c.get("output")]
        rows = {"B1": [], "B1n": [], "B2": [], "B3": []}
        flips = []   # files where the plugin path disagrees with ITSELF
        noise = set()  # ...and where the new path lands inside that disagreement
        for sid, name, stored in files:
            wav = d / name
            new = _new_spans(f"silero-v6.2-{k}", wav, p, 0.35)
            # B1: the plugin given the SAME resampled audio the bench uses, so
            # neither side resamples and only the model + state machine differ
            rows["B1"].append((sid, *_edge_diff(_plugin_on_resampled(wav, p), new)))
            # The plugin path resamples inside its stream, and that path is not
            # repeatable at 8 kHz: the same file can give different spans on
            # different runs. So when it disagrees with the new path, ask it
            # again, and count a match against ANY of its own outcomes.
            outcomes = [asyncio.run(vad.analyze(wav, p))["segments"]]
            res = _edge_diff(outcomes[0], new)
            while not (res[0] and res[1] <= TOL) and len(outcomes) < PLUGIN_TRIES:
                outcomes.append(asyncio.run(vad.analyze(wav, p))["segments"])
                res = min((_edge_diff(o, new) for o in outcomes),
                          key=lambda r: (not r[0], r[1] if r[1] is not None else 1e9))
            if len({str(o) for o in outcomes}) > 1:
                flips.append(f"{sid} ({len({str(o) for o in outcomes})} span sets in {len(outcomes)} runs)")
                # The plugin cannot agree with itself here, so no single outcome
                # is "the" plugin answer. The deterministic new path passes if it
                # lands inside the plugin's own spread: a span count the plugin
                # itself produced on one of its runs.
                if not (res[0] and res[1] <= TOL) and len(new) in {len(o) for o in outcomes}:
                    noise.add(sid)
            rows["B1n"].append((sid, *res))
            if stored and stored.get("segments") is not None:
                sp = _params(stored.get("params"), meta)
                sk = f"{sp['sample_rate'] // 1000}k"
                rows["B2"].append((sid, *_edge_diff(stored["segments"],
                                                    _new_spans(f"silero-v6.0-{sk}", wav, sp, 0.35))))
            lk = asyncio.run(_inference_file_spans(wav, p))
            rows["B3"].append((sid, *_edge_diff(lk, _new_spans("lk-inference", wav, p, None))))
        print(f"\nB. run {rid} ({meta.get('source')}, {len(files)} files, "
              f"params {p['activation_threshold']}/{p['deactivation_threshold']}/"
              f"{p['min_speech_duration']}/{p['min_silence_duration']} @ {k})")
        names = {"B1": "plugin 1.8.3 (v6.2) on the bench's resampled audio vs silero-v6.2 + smoothing",
                 "B1n": "plugin resampling in-stream (native-rate file) vs the same [informational]",
                 "B2": "stored spans (1.6.7, v6.0) vs silero-v6.0 + smoothing [informational]",
                 "B3": "inference.VAD events vs lk-inference + segmenter"}
        for key, rs in rows.items():
            same = [r for r in rs if r[1]]
            worst = max((r[2] for r in same), default=0.0)
            within = sum(1 for r in same if r[2] <= TOL)
            bad = [r[0] for r in rs if not r[1]]
            in_noise = [r[0] for r in rs if key == "B1n" and r[0] in noise]
            print(f"  {key} {names[key]}: {len(same)}/{len(rs)} files same span count, "
                  f"{within}/{len(rs)} within 2 frames, worst {worst:.2f} frames"
                  + (f"; count differs on {bad}" if bad else "")
                  + (f"; {len(in_noise)} more inside the plugin's own run-to-run spread: {in_noise}"
                     if in_noise else ""))
            # B2 cannot re-run livekit 1.6.7, so it cannot separate "the new path is
            # off" from "the stored run landed on the other side of a flip"; B1n
            # measures livekit's own resampler jitter (in-stream QUICK), which
            # scripts/vad_repeat.py --path plugin quantifies per file
            if within + len(in_noise) < len(rs):
                if key not in ("B2", "B1n"):
                    ok = False
                for sid, same_n, dfr in rs:
                    if (not same_n or dfr > TOL) and sid not in in_noise:
                        print(f"     {sid}: {'count differs' if not same_n else f'{dfr:.2f} frames'}")
        if flips:
            print(f"  the plugin path disagreed with itself on {len(flips)} file(s): {', '.join(flips)}")
    return ok


PLUGIN_TRIES = 5
TOL = 2.0 + 1e-6   # frames; "within a frame or two", robust to float edges


# ------------------------------------------------------------- C: bake-off
# probe_loaders.py (PR-42 bake-off), copied verbatim apart from paths.


def _pl_firered(x16, onnx, fr):
    import kaldi_native_fbank as knf
    import kaldiio
    import onnxruntime as ort

    _cm = kaldiio.load_mat(f"{fr}/cmvn.ark"); _dim = _cm.shape[1] - 1; _n = _cm[0, _dim]
    _mean = _cm[0, :_dim] / _n; _istd = 1 / np.sqrt(np.maximum(_cm[1, :_dim] / _n - _mean ** 2, 1e-20))
    o = knf.FbankOptions(); o.frame_opts.samp_freq = 16000; o.frame_opts.dither = 0
    o.frame_opts.snip_edges = True; o.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(o); fb.accept_waveform(16000, (np.clip(x16, -1, 1) * 32767).tolist())
    f = np.vstack([fb.get_frame(i) for i in range(fb.num_frames_ready)])
    f = ((f - _mean) * _istd).astype(np.float32)[None]
    s = ort.InferenceSession(f"{fr}/{onnx}", providers=["CPUExecutionProvider"])
    p = s.run(None, {s.get_inputs()[0].name: f})[0].reshape(-1)
    return p, 0.010


def _pl_earshot(x16):
    import earshot
    d = earshot.Detector()
    pcm = (np.clip(x16, -1, 1) * 32767).astype(np.int16)
    return np.array([d.predict_i16(pcm[i * 256:(i + 1) * 256]) for i in range(len(pcm) // 256)]), 0.016


def _pl_vadonnx(name, x, sr):
    from vadonnx import load_vad
    v = load_vad(name)
    p = np.asarray(v.probabilities(x.astype(np.float32), sample_rate=sr)).reshape(-1)
    return p, len(x) / sr / max(len(p), 1)


def _corr(a, b):
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    if n < 2 or a.std() == 0 or b.std() == 0:
        return float("nan"), n
    return float(np.corrcoef(a, b)[0, 1]), n


def check_bakeoff() -> bool:
    scripts = BAKEOFF / "scripts"
    wav = BAKEOFF / "pr42_caller.nc.wav"
    if not wav.is_file():
        print(f"C. bake-off: {wav} not found, skipped")
        return True
    sys.path.insert(0, str(scripts))
    import vad_bench as vb  # the bake-off's own module; import-safe (main-guarded)

    data, rate = sf.read(wav, dtype="int16", always_2d=True)
    pcm = np.ascontiguousarray(data[:, 0])
    # ONE input for both codes: the bench's own QUICK resampling to int16, and the
    # old code gets exactly those samples as floats. Feeding the old code soxr
    # floats instead measured the INPUT, not the loader: kaldi-fbank frontends
    # (FireRed, FSMN) take int16-valued audio, and 1-LSB differences in near-
    # digital-silence frames move their log-mel features, and so their output.
    i16 = _i16(pcm, rate, 16000)
    i8 = _i16(pcm, rate, 8000)
    x16 = i16.astype(np.float32) / 32768.0
    x8 = i8.astype(np.float32) / 32768.0
    M = vads.MODELS_DIR / "vad"
    fr = str(M / "firered")
    # the bake-off's vadonnx loaders download their ONNX into a cache; keep that
    # out of the home directory
    os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp(prefix="vadparity-")

    old = {
        "silero-v6.2-16k": lambda: vb.silero(str(M / "silero/v6.2/silero_vad.onnx"), 16000)(x16),
        "silero-v6.2-8k": lambda: vb.silero(str(M / "silero/v6.2/silero_vad.onnx"), 8000)(x8),
        "silero-v6.0-16k": lambda: vb.silero(str(M / "silero/v6.0/silero_vad.onnx"), 16000)(x16),
        "silero-v6.0-8k": lambda: vb.silero(str(M / "silero/v6.0/silero_vad.onnx"), 8000)(x8),
        "silero-v5.1.2-16k": lambda: vb.silero(str(M / "silero/v5.1.2/silero_vad.onnx"), 16000)(x16),
        "silero-v5.1.2-8k": lambda: vb.silero(str(M / "silero/v5.1.2/silero_vad.onnx"), 8000)(x8),
        "silero-v4.0-16k": lambda: vb.silero(str(M / "silero/v4.0/silero_vad.onnx"), 16000)(x16),
        "silero-v4.0-8k": lambda: vb.silero(str(M / "silero/v4.0/silero_vad.onnx"), 8000)(x8),
        "ten": lambda: vb.ten_vad(16000)(x16),
        "webrtc-3": lambda: vb.webrtc(3, 16000)(x16),
        "firered-stream": lambda: _pl_firered(x16, "fireredvad_stream_vad.onnx", fr)[::-1],
        "firered-offline": lambda: _pl_firered(x16, "fireredvad_vad.onnx", fr)[::-1],
        "earshot": lambda: _pl_earshot(x16)[::-1],
        "marblenet-v2": lambda: _pl_vadonnx("marblenet", x16, 16000)[::-1],
        "fsmn-16k": lambda: _pl_vadonnx("fsmn", x16, 16000)[::-1],
        "pyannote-seg3": lambda: _pl_vadonnx("pyannote", x16, 16000)[::-1],
    }
    print(f"\nC. bake-off parity on {wav.name} ({len(pcm) / rate:.0f} s): new loader vs the "
          "bake-off's code, same QUICK-resampled int16 input")
    ok = True
    reg = vads.by_id()
    for vid, fn in old.items():
        try:
            a = fn()
            op, ohop = (a[0], a[1]) if not isinstance(a[0], float) else (a[1], a[0])
        except Exception as ex:  # noqa: BLE001
            print(f"  {vid:20} old code failed: {type(ex).__name__}: {str(ex)[:120]}")
            continue
        src = i8 if reg[vid].get("rate") == 8000 else i16
        new = vads.infer(vid, src, reg[vid].get("rate") or 16000)
        r, n = _corr(op, new["probs"])[:2]
        mad = float(np.mean(np.abs(np.asarray(op[:n], float) - new["probs"][:n])))
        flag = "ok" if r >= 0.99 else "LOW"
        ok &= r >= 0.99
        print(f"  {vid:20} r={r:.5f}  mean|Δ|={mad:.4f}  frames old/new {len(op)}/{len(new['probs'])}"
              f"  hop {ohop:.4f}/{new['hop_s']:.4f}  {flag}")

    # the bake-off's STORED outputs
    nat = np.load(scripts / "native_nc.npy")[:, 1]
    lk = vads.infer("lk-inference", pcm, rate)["probs"]
    s62 = vads.infer("silero-v6.2-16k", i16, 16000)["probs"]
    xs = soxr.resample(pcm.astype(np.float32) / 32768.0, rate, 16000)
    s62x = vads.infer("silero-v6.2-16k", (np.clip(xs, -1, 32767 / 32768) * 32768).astype(np.int16),
                      16000)["probs"]
    print(f"  stored native_nc.npy (prod engine, livekit-agents 1.6.3 / local-inference 0.2.6):"
          f"\n    vs lk-inference (1.8.3 / 0.2.7)  r={_corr(nat, lk)[0]:.5f}"
          f"  mean|Δ|={np.mean(np.abs(nat[:len(lk)] - lk[:len(nat)])):.4f}"
          f"\n    vs silero-v6.2-16k ONNX          r={_corr(nat, s62)[0]:.5f}"
          f"  (bake-off reported 0.9995)"
          f"\n    vs silero-v6.2-16k ONNX, soxr    r={_corr(nat, s62x)[0]:.5f}"
          f"  (the resampler alone; why every loader uses livekit's QUICK)")
    ok &= _corr(nat, lk)[0] >= 0.99

    stored = []
    for fn in ("bench_results.json", "bench_results_extra.json", "bench_results_extra_firered.json"):
        p = scripts / fn
        if p.is_file():
            stored += json.loads(p.read_text())
    name_map = {"livekit-native (prod)": "lk-inference", "silero-v6.2": "silero-v6.2",
                "silero-v6.0-onnx": "silero-v6.0", "silero-v5.1.2": "silero-v5.1.2",
                "silero-v4.0": "silero-v4.0", "ten-vad": "ten", "firered-stream": "firered-stream",
                "firered-vad (offline)": "firered-offline", "earshot": "earshot",
                "marblenet-v2.0": "marblenet-v2", "fsmn-16k": "fsmn-16k",
                "pyannote-seg3 (offline)": "pyannote-seg3"}
    seen = set()
    print("  stored peak table (7 missed sounds + detected median/min) vs the same peaks from new probs:")
    for row in stored:
        if row.get("audio") != "nc" or row["model"] not in name_map:
            continue
        fam = name_map[row["model"]]
        sr = row.get("sr", 16000)
        vid = fam if fam in reg else f"{fam}-{sr // 1000}k"
        if vid in seen or vid not in reg:
            continue
        seen.add(vid)
        # the bench's real path: native-rate audio in, each loader resamples
        r = vads.infer(vid, pcm, rate)
        pk, med, mn = vb.peaks(r["hop_s"], np.clip(r["probs"], 0, 1))
        old_pk = row["miss_peaks"]
        diffs = [abs(pk[k] - old_pk[k]) for k in old_pk] + [abs(med - row["det_median"]),
                                                             abs(mn - row["det_min"])]
        print(f"    {vid:20} max|Δpeak|={max(diffs):.2f}  new {list(pk.values())} "
              f"med/min {med}/{mn}  stored {list(old_pk.values())} {row['det_median']}/{row['det_min']}")
    return ok


# ------------------------------------------------ D: FireRed live streaming


FIRERED_FILES = ("20260804-112511/none.wav", "20260804-112511/hush.wav",
                 "20260804-171340/none.wav")


def check_firered() -> bool:
    """firered-stream runs as ONE whole-sequence call. Live would stream it: the
    waveform arrives in 20-50 ms chunks into one OnlineFbank, and the model runs
    chunk by chunk carrying its caches (the publisher's _with_cache graph,
    `caches_in` [8, 1, 128, 19]). Equal, or does the loader have to chunk?

    The control resets the caches every chunk; it has to differ, or the test
    would pass just as well with caches that do nothing.
    """
    import kaldi_native_fbank as knf

    from nc_bench.vads.loaders import _fbank, _kaldi_cmvn, _ort_session

    fr = vads.MODELS_DIR / "vad" / "firered"
    cache_onnx = fr / "fireredvad_stream_vad_with_cache.onnx"
    if not cache_onnx.is_file():
        print(f"\nD. firered: {cache_onnx.name} missing (scripts/fetch_vads.py firered-stream), skipped")
        return True
    mean, istd = _kaldi_cmvn(fr / "cmvn.ark")
    whole, cached = _ort_session(fr / "fireredvad_stream_vad.onnx"), _ort_session(cache_onnx)

    def stream_feats(x, chunk_ms):
        o = knf.FbankOptions()
        o.frame_opts.samp_freq = 16000
        o.frame_opts.frame_length_ms, o.frame_opts.frame_shift_ms = 25, 10
        o.frame_opts.dither, o.frame_opts.snip_edges = 0.0, True
        o.mel_opts.num_bins = 80
        fb, got, rows = knf.OnlineFbank(o), 0, []
        for i in range(0, len(x), 16 * chunk_ms):
            fb.accept_waveform(16000, x[i:i + 16 * chunk_ms].astype(np.float32).tolist())
            while got < fb.num_frames_ready:
                rows.append(fb.get_frame(got))
                got += 1
        return ((np.vstack(rows) - mean) * istd).astype(np.float32)

    def chunked(f, k, carry=True):
        caches, out = np.zeros((8, 1, 128, 19), np.float32), []
        for i in range(0, len(f), k):
            p, nxt = cached.run(None, {"feat": f[i:i + k][None], "caches_in": caches})
            caches = nxt if carry else np.zeros_like(caches)
            out.append(p.reshape(-1))
        return np.concatenate(out)

    print("\nD. firered-stream: live chunked streaming (carried caches) vs the bench's whole call")
    ok, P = True, dict(deactivation_threshold=0.5, min_speech_duration=0.05, min_silence_duration=0.4)
    for rel in FIRERED_FILES:
        data, rate = sf.read(store.config.RUNS_DIR / rel, dtype="int16", always_2d=True)
        x = _i16(np.ascontiguousarray(data[:, 0]), rate, 16000)
        batch = ((_fbank(x, 16000) - mean) * istd).astype(np.float32)
        ref = vads.infer("firered-stream", x, 16000)["probs"]
        worst, spans_same = 0.0, True
        for chunk_ms in (20, 50):
            f = stream_feats(x, chunk_ms)
            assert np.array_equal(f, batch), f"{rel}: streamed fbank != batch fbank"
            for k in (1, 2, 5):            # 10, 20, 50 ms of model input per call
                p = chunked(f, k)
                worst = max(worst, float(np.abs(p - ref).max()))
                for thr in (0.5, 0.7):
                    kw = dict(P, activation_threshold=thr, deactivation_threshold=thr)
                    spans_same &= (segment(p, 0.01, duration_s=len(x) / 16000, **kw)
                                   == segment(ref, 0.01, duration_s=len(x) / 16000, **kw))
        ctrl = float(np.abs(chunked(batch, 2, carry=False) - ref).max())
        good = worst < 1e-5 and spans_same and ctrl > 0.1
        ok &= good
        print(f"  {rel}: max|Δp| {worst:.1e} over fbank 20/50 ms × model 10/20/50 ms, "
              f"spans identical at 0.5 and 0.7: {spans_same}; control (caches reset every "
              f"20 ms) max|Δp| {ctrl:.3f}  {'ok' if good else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["loop", "runs", "bakeoff", "firered"])
    ap.add_argument("--runs", default="20260804-112511,20260804-171340,20260803-164743")
    a = ap.parse_args()
    ok = True
    if a.only in (None, "loop"):
        ok &= check_loop()
    if a.only in (None, "runs"):
        ok &= check_runs([r for r in a.runs.split(",") if r])
    if a.only in (None, "bakeoff"):
        ok &= check_bakeoff()
    if a.only in (None, "firered"):
        ok &= check_firered()
    print("\nPARITY OK" if ok else "\nPARITY: see mismatches above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
