"""How repeatable is one (source, VAD)? Run inference N times, report the spread.

    .venv/bin/python scripts/vad_repeat.py --run 20260804-112511 --source hush --vad silero-v6.2-8k
    .venv/bin/python scripts/vad_repeat.py --run 20260804-112511 --source input --vad lk-inference \\
        --path lk-native-rate -n 8
    .venv/bin/python scripts/vad_repeat.py --run 20260803-164743 --source dpdfnet2 --path plugin \\
        --rate 8000 -n 6 --json

The noise floor for borderline results: a turn that one run opens and the next
does not is not evidence for or against a model. Three paths:

--path bench           the VAD bench's own loader (what the stored .vadp.npy came
                       from). Expected to repeat exactly; --fresh rebuilds the
                       model every time to rule out state leaking between calls.
--path plugin          the legacy livekit-plugins-silero path behind the cyan
                       spans (vad.analyze): file-rate frames in, the plugin
                       resamples in-stream and applies its ExpFilter. Silero only
                       (whatever ONNX the installed plugin bundles); --rate picks
                       8000/16000. NOT repeatable at 8 kHz (measured 2026-09-29).
--path lk-native-rate  livekit.agents.inference.VAD fed file-rate frames, as a
                       production worker feeds it. NOT repeatable on 48 kHz frames
                       (measured 2026-09-29); the bench feeds it 16 kHz instead.

Segmentation uses the run's VAD-bench parameters (else its VAD panel values),
overridable with --params k=v. Read-only: nothing under data/runs/ is written.

Reported: per-window probability spread across runs (max, mean, and how many
windows moved more than 0.01 / 0.05), then the span sets: how many distinct
ones, the range of span counts and speech seconds, and the worst edge
difference against run 1 where the counts match.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import store, vad, vads  # noqa: E402
from nc_bench.vads import bench  # noqa: E402
from nc_bench.vads.segmenter import segment  # noqa: E402


def _events_to_spans(events, duration_s):
    """nc_bench/vad.py's START/END -> span conversion."""
    from livekit.agents import vad as agents_vad

    segs, open_start = [], None
    for ev in events:
        if ev.type is agents_vad.VADEventType.START_OF_SPEECH:
            open_start = max(0.0, float(ev.timestamp) - float(ev.speech_duration))
        elif ev.type is agents_vad.VADEventType.END_OF_SPEECH:
            end = float(ev.timestamp) - float(ev.silence_duration)
            segs.append([round(max(0.0, end - float(ev.speech_duration)), 3),
                         round(min(end, duration_s), 3)])
            open_start = None
    if open_start is not None and duration_s > open_start:
        segs.append([round(open_start, 3), round(duration_s, 3)])
    return segs


async def _stream(make_stream, x, rate):
    """Push file-rate frames (100 ms, as vad.analyze does); probabilities + spans.
    The stream is built inside the loop: livekit's VADStream starts a task in
    its constructor, so it needs a running event loop."""
    from livekit import rtc
    from livekit.agents import vad as agents_vad

    stream = make_stream()
    step = max(1, rate // 10)
    for i in range(0, len(x), step):
        c = x[i:i + step]
        stream.push_frame(rtc.AudioFrame(data=c.tobytes(), sample_rate=rate, num_channels=1,
                                         samples_per_channel=len(c)))
    stream.end_input()
    evs = [ev async for ev in stream]
    await stream.aclose()
    probs = np.array([float(e.probability) for e in evs
                      if e.type is agents_vad.VADEventType.INFERENCE_DONE], dtype=np.float32)
    return probs, _events_to_spans(evs, len(x) / rate)


def _once(path, entry, x, rate, p, th, fresh, plugin_rate):
    dur = len(x) / rate
    if path == "bench":
        if fresh:
            vads._models.pop(entry["id"], None)
        r = vads.infer(entry, x, rate)
        ep = bench.effective_params(p, entry, th)
        spans = segment(r["probs"], r["hop_s"], t0_s=r["t0_s"], duration_s=dur,
                        smoothing=ep["smoothing"], **{k: ep[k] for k in (
                            "activation_threshold", "deactivation_threshold",
                            "min_speech_duration", "min_silence_duration")})
        return r["probs"], spans, r["hop_s"]
    kw = {k: p[k] for k in ("activation_threshold", "deactivation_threshold",
                            "min_speech_duration", "min_silence_duration")}
    if path == "plugin":
        from livekit.plugins import silero

        model = silero.VAD.load(sample_rate=plugin_rate,
                                prefix_padding_duration=p["prefix_padding_duration"], **kw)
    else:  # lk-native-rate
        from livekit.agents import inference

        model = inference.VAD(prefix_padding_duration=p["prefix_padding_duration"], **kw)
    probs, spans = asyncio.run(_stream(model.stream, x, rate))
    return probs, spans, 0.032


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True, help="input or a candidate id")
    ap.add_argument("--vad", default="lk-inference", help="VAD id (bench path)")
    ap.add_argument("--path", choices=["bench", "plugin", "lk-native-rate"], default="bench")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--rate", type=int, choices=[8000, 16000], default=None,
                    help="plugin path only; default: the run's VAD panel rate")
    ap.add_argument("--params", nargs="*", default=[])
    ap.add_argument("--fresh", action="store_true", help="bench path: rebuild the model each run")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    meta = store.load_meta(a.run)
    files = bench.source_files(meta)
    if a.source not in files:
        sys.exit(f"no audio for source {a.source!r}; have {list(files)}")
    p = bench.clean_params(dict(kv.split("=", 1) for kv in a.params) or None, meta)
    th = (meta.get("vad_bench") or {}).get("thresholds") or {}
    entry = vads.by_id().get(a.vad) if a.path == "bench" else None
    if a.path == "bench" and (entry is None or not entry["available"] or entry["loader"] == "lkaic_flag"):
        sys.exit(f"{a.vad}: not an available inference VAD")
    plugin_rate = a.rate or vad.clean(meta.get("vad_params"), meta.get("source"))["sample_rate"]

    data, rate = sf.read(store.run_dir(a.run) / files[a.source], dtype="int16", always_2d=True)
    x = np.ascontiguousarray(data[:, 0])
    runs = [_once(a.path, entry, x, rate, p, th, a.fresh, plugin_rate) for _ in range(a.n)]

    n = min(len(r[0]) for r in runs)
    P = np.stack([r[0][:n] for r in runs])
    spread = P.max(axis=0) - P.min(axis=0) if n else np.zeros(0)
    hop = runs[0][2]
    sets = [r[1] for r in runs]
    distinct = {json.dumps(s) for s in sets}
    edge = []
    for s in sets[1:]:
        if len(s) == len(sets[0]) and s:
            edge.append(max(max(abs(u[0] - v[0]), abs(u[1] - v[1])) for u, v in zip(s, sets[0])) / hop)
    speech = [round(sum(e - b for b, e in s), 2) for s in sets]
    what = {"bench": a.vad, "plugin": f"livekit-plugins-silero {vad.model_info()['silero_plugin_version']}"
            f" ({vad.model_info()['model']}) @ {plugin_rate // 1000} kHz",
            "lk-native-rate": f"inference.VAD fed {rate // 1000} kHz frames"}[a.path]
    out = {
        "run": a.run, "source": a.source, "path": a.path, "vad": what, "runs": a.n,
        "params": {k: p[k] for k in bench.PARAM_KEYS},
        "windows": int(n), "frame_counts": sorted({len(r[0]) for r in runs}),
        "prob_spread_max": round(float(spread.max()), 4) if n else None,
        "prob_spread_mean": round(float(spread.mean()), 5) if n else None,
        "windows_spread_gt_0.01": int((spread > 0.01).sum()),
        "windows_spread_gt_0.05": int((spread > 0.05).sum()),
        "span_sets_distinct": len(distinct),
        "span_counts": sorted({len(s) for s in sets}),
        "speech_s_range": [min(speech), max(speech)],
        "worst_edge_frames_vs_run1": round(max(edge), 2) if edge else (0.0 if len(distinct) == 1 else None),
        "repeatable": len(distinct) == 1 and (not n or float(spread.max()) == 0.0),
    }
    if a.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{a.run} · {a.source} · {what} · {a.n} runs")
    print(f"  probabilities: {n} windows; spread max {out['prob_spread_max']}, mean "
          f"{out['prob_spread_mean']}; {out['windows_spread_gt_0.01']} windows moved > 0.01, "
          f"{out['windows_spread_gt_0.05']} > 0.05")
    print(f"  spans: {out['span_sets_distinct']} distinct set(s) in {a.n} runs; counts "
          f"{out['span_counts']}; speech {out['speech_s_range'][0]}–{out['speech_s_range'][1]} s; "
          f"worst edge vs run 1 {out['worst_edge_frames_vs_run1']} frames")
    print("  REPEATABLE" if out["repeatable"] else "  NOT repeatable: treat differences this size as noise")


if __name__ == "__main__":
    main()
