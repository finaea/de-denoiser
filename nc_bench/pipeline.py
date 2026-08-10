"""Run one recorded input through a candidate chain, offline but streaming.

Audio flows: input wav (any rate) → per-stage soxr resample → 20 ms blocks
through each stateful processor in order → 16 kHz s16 wav output. Blocks are
fed sequentially so LSTM/cache state behaves exactly as it would live.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import config
from .processors import build_chain

_BLOCK_MS = 20


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), rate


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    return soxr.resample(x, src, dst).astype(np.float32)


# Measuring a model's footprint by RSS delta INSIDE this process does not work:
# the server is long-lived, and ONNX Runtime hands a closed session's arena back
# to its own pool rather than to the OS, so the next model loads into memory that
# is already resident. Measured that way gtcrn came out at -3.2 MB. A throwaway
# process is the only place the delta means what it says.
_RSS_PROBE = """
import json, sys
import psutil
sys.path.insert(0, sys.argv[1])
from nc_bench.processors import build_chain
p = psutil.Process()
before = p.memory_info().rss
chain = build_chain(json.loads(sys.argv[2]))
after = p.memory_info().rss
for c in chain:
    c.close()
print((after - before) / 1e6)
"""


def _model_rss_mb(chain_spec: list[dict]) -> float | None:
    """RSS the chain's models add, measured in a clean interpreter.

    Costs one process spawn plus one model load, so it is gated on the caller
    (concurrency 1 only). Never raises: a diagnostic must not fail a run.
    """
    root = str(Path(__file__).resolve().parent.parent)
    try:
        r = subprocess.run(
            [sys.executable, "-c", _RSS_PROBE, root, json.dumps(chain_spec)],
            capture_output=True, text=True, timeout=180,
        )
        return round(float(r.stdout.strip().splitlines()[-1]), 1) if r.returncode == 0 else None
    except Exception:
        return None


def run_chain(
    input_wav: Path, chain_spec: list[dict], output_wav: Path, measure_rss: bool = False
) -> dict:
    """Process input through the chain; returns timing/metadata.

    `measure_rss` spends a subprocess and a second model load to size the
    models (see _model_rss_mb), so the server only asks for it at concurrency 1
    — where the run is already the slow, measure-everything-properly setting.
    """
    audio, rate = load_mono(input_wav)
    duration_s = len(audio) / rate

    model_rss = _model_rss_mb(chain_spec) if measure_rss else None
    init_started = time.perf_counter()
    procs = build_chain(chain_spec)
    init_s = time.perf_counter() - init_started

    block_times: list[float] = []  # seconds per 20 ms block, across all stages
    started = time.perf_counter()
    # thread_time, not process_time: the server runs candidates concurrently in
    # threads of one process, so a process-wide clock would bill this chain for
    # its neighbours. It reads the FULL cost only because every ORT session here
    # is pinned to one thread (processors/base.py ort_options) — unpinned, ORT
    # does ~80% of the work on its own workers and this would undercount ~5x.
    # scripts/selfcheck.py check_single_threaded() holds that invariant.
    cpu_started = time.thread_time()
    try:
        for proc in procs:
            audio = _resample(audio, rate, proc.rate)
            rate = proc.rate
            block = max(1, int(rate * _BLOCK_MS / 1000))
            outs = []
            for i in range(0, len(audio), block):
                t0 = time.perf_counter()
                outs.append(proc.process_block(audio[i : i + block]))
                block_times.append(time.perf_counter() - t0)
            outs.append(proc.flush())
            audio = np.concatenate([o for o in outs if len(o)] or [np.zeros(0, np.float32)])
        proc_s = time.perf_counter() - started
        cpu_s = time.thread_time() - cpu_started
    finally:
        for proc in procs:
            proc.close()

    audio = _resample(audio, rate, config.PIPELINE_RATE)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(output_wav, (audio * 32767).astype(np.int16), config.PIPELINE_RATE)

    # whole-file stages (e.g. ffmpeg arnndn) do their work in flush(), so
    # per-block numbers would flatter them — report none for the whole chain
    bt = np.array([] if any(p.whole_file for p in procs) else block_times) * 1000
    return {
        "proc_ms": round(proc_s * 1000, 1),
        "rtf": round(proc_s / duration_s, 4) if duration_s > 0 else None,
        "duration_s": round(duration_s, 2),
        "chain": [p.name for p in procs],
        "latency": {
            "init_ms": round(init_s * 1000, 1),
            "block_ms_mean": round(float(bt.mean()), 3) if len(bt) else None,
            "block_ms_p95": round(float(np.percentile(bt, 95)), 3) if len(bt) else None,
            # sum of each stage's structural buffering (block/window sizes) —
            # what the chain would add to live audio even on an infinite CPU
            "algo_delay_ms": round(sum(p.algo_delay_ms for p in procs), 1),
        },
        # What it costs to RUN, as opposed to how fast it finishes. rtf is wall
        # clock and flatters anything that spreads over cores; cpu_per_audio_s
        # is core-seconds per second of audio, so 1/it is how many concurrent
        # calls one core carries. That is the number that sizes a box.
        "cost": {
            "cpu_ms": round(cpu_s * 1000, 1),
            "cpu_per_audio_s": round(cpu_s / duration_s, 4) if duration_s > 0 else None,
            # RSS the models added at load: a per-session cost, paid once
            "model_rss_mb": model_rss,
        },
    }
