"""Does the VAD-turn cut list send exactly the audio a live deployment would — once?

Worth a script because every failure mode here corrupts WER invisibly: an
unmerged overlap transcribes the same words twice (scored as insertions), a
missing pad shaves onsets (deletions), and both produce transcripts that read
plausibly. merge_turns is pure precisely so this can run without an STT server.

    .venv/bin/python scripts/check_stt_turns.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench.stt import merge_turns  # noqa: E402

DUR = 30.0
PAD = 0.5

# plain spans, far apart: each padded by 0.5 s at the start, ends untouched
cuts = merge_turns([[5.0, 8.0], [15.0, 18.0]], PAD, DUR)
assert cuts == [[4.5, 8.0], [14.5, 18.0]], cuts
print("OK  pad extends each start by prefix padding, ends untouched")

# padding bridges a short gap: 0.4 s apart -> one request, no duplicated audio
cuts = merge_turns([[5.0, 8.0], [8.4, 10.0]], PAD, DUR)
assert cuts == [[4.5, 10.0]], cuts
print("OK  turns closer than the pad merge into one request")

# already-overlapping spans merge too
cuts = merge_turns([[5.0, 9.0], [8.0, 12.0]], PAD, DUR)
assert cuts == [[4.5, 12.0]], cuts
total = sum(e - s for s, e in cuts)
assert abs(total - 7.5) < 1e-9, total
print("OK  overlapping spans send their audio exactly once")

# clamped at both ends: pad cannot reach before 0, span cannot outrun the file
cuts = merge_turns([[0.2, 3.0], [28.0, 99.0]], PAD, DUR)
assert cuts == [[0.0, 3.0], [27.5, 30.0]], cuts
print("OK  clamped to [0, duration]")

# slivers are dropped, empty input sends nothing
assert merge_turns([[5.0, 5.005]], 0.0, DUR) == []
assert merge_turns([], PAD, DUR) == []
assert merge_turns(None, PAD, DUR) == []
print("OK  sub-20 ms slivers and empty span sets send nothing")

# monotone, non-overlapping output whatever comes in
cuts = merge_turns([[1, 2], [2.1, 3], [3.05, 4], [10, 11]], PAD, DUR)
for a, b in zip(cuts, cuts[1:]):
    assert a[1] < b[0], cuts
print("OK  output is ordered and disjoint")

print("\ncheck_stt_turns passed")
