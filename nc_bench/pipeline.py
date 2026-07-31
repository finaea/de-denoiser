"""Run one recorded input through a candidate chain, offline but streaming.

Audio flows: input wav (any rate) → per-stage soxr resample → 20 ms blocks
through each stateful processor in order → 16 kHz s16 wav output. Blocks are
fed sequentially so LSTM/cache state behaves exactly as it would live.
"""

from __future__ import annotations

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


def run_chain(input_wav: Path, chain_spec: list[dict], output_wav: Path) -> dict:
    """Process input through the chain; returns timing/metadata."""
    audio, rate = load_mono(input_wav)
    duration_s = len(audio) / rate

    init_started = time.perf_counter()
    procs = build_chain(chain_spec)
    init_s = time.perf_counter() - init_started

    block_times: list[float] = []  # seconds per 20 ms block, across all stages
    started = time.perf_counter()
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
    finally:
        for proc in procs:
            proc.close()

    audio = _resample(audio, rate, config.PIPELINE_RATE)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(output_wav, (audio * 32767).astype(np.int16), config.PIPELINE_RATE)

    bt = np.array(block_times) * 1000
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
    }
