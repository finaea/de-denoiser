"""Streaming processor contract.

A processor operates on float32 mono audio at its own fixed sample rate.
Feed arbitrary-length blocks in order; the processor buffers internally and
returns whatever enhanced audio is ready. Call flush() at end-of-stream to
drain (pads the tail with silence if the model needs whole chunks). Each
instance is stateful — build one per run.
"""

from __future__ import annotations

import numpy as np


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
