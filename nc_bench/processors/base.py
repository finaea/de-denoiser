"""Streaming processor contract.

A processor operates on float32 mono audio at its own fixed sample rate.
Feed arbitrary-length blocks in order; the processor buffers internally and
returns whatever enhanced audio is ready. Call flush() at end-of-stream to
drain (pads the tail with silence if the model needs whole chunks). Each
instance is stateful — build one per run.
"""

from __future__ import annotations

import numpy as np


def ort_options():
    """Session options for a STREAMING processor: one thread, sequential.

    ONNX Runtime defaults to one intra-op thread per physical core, which on
    20 ms frames costs more in thread hand-off than the parallelism buys.
    Measured over a 119 s file, pinning to one thread cut CPU by ~5.5x *and*
    cut wall-clock at the same time:

        dpdfnet2        0.435 -> 0.079 CPU-s per audio-s,  RTF 0.088 -> 0.079
        gtcrn           0.187 -> 0.030 CPU-s per audio-s,  RTF 0.038 -> 0.030
        fastenhancer-t  0.065 -> 0.009 CPU-s per audio-s,  RTF 0.013 -> 0.009

    It also stops one candidate from taking every core while the bench runs
    others beside it, and it is what makes pipeline.py's thread_time() reading
    the whole cost rather than a fifth of it — check_single_threaded() in
    scripts/selfcheck.py pins that invariant.

    Whole-file scoring (DNSMOS, Silero) deliberately keeps the ORT default:
    one big tensor is where intra-op parallelism actually pays.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


class Processor:
    name: str = "base"
    rate: int = 16_000
    # structural buffering the processor imposes on a live stream (ms):
    # its chunk/window size — independent of CPU speed
    algo_delay_ms: float = 0.0
    # True for processors that do all their work in flush() (whole-file tools):
    # their per-block timings are meaningless, so the pipeline suppresses them
    whole_file: bool = False

    def process_block(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def flush(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        pass


class Passthrough(Processor):
    name = "none"

    def __init__(self, rate: int = 16_000):
        self.rate = rate

    def process_block(self, x: np.ndarray) -> np.ndarray:
        return x
