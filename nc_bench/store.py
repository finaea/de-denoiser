"""Run history on disk: data/runs/<id>/ holds meta.json, input.wav and one
<candidate-id>.wav per candidate. meta.json is the single source of truth."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from . import config


def new_run(source: str, note: str = "") -> tuple[str, Path]:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = config.RUNS_DIR / run_id
    # avoid collision if two runs start within a second
    n = 1
    while run_dir.exists():
        run_dir = config.RUNS_DIR / f"{run_id}-{n}"
        n += 1
    run_dir.mkdir(parents=True)
    meta = {
        "id": run_dir.name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "status": "recording",
        # free-text conditions for this run ("someone talking nearby",
        # "speakerphone at arm's length"). Deliberately NOT part of `script`,
        # which is the WER reference — describing the room in there would be
        # scored as words the speaker failed to say.
        "note": note,
        "input": None,
        "candidates": [],
    }
    save_meta(run_dir.name, meta)
    return run_dir.name, run_dir


def run_dir(run_id: str) -> Path:
    d = (config.RUNS_DIR / run_id).resolve()
    if not d.is_relative_to(config.RUNS_DIR) or not d.is_dir():
        raise KeyError(run_id)
    return d


def save_meta(run_id: str, meta: dict) -> None:
    """Atomic: written to a temp file in the same directory, then os.replace()d.

    A reader (or a crash) mid-write can therefore never see half a meta.json.
    Atomic is not the same as safe from lost updates: a writer that read meta
    a while ago and saves it now still replaces whatever changed in between.
    Anything that reads, merges and writes must do it inside locked_meta().
    """
    d = config.RUNS_DIR / run_id
    fd, tmp = tempfile.mkstemp(prefix=".meta.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(meta, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / "meta.json")
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def load_meta(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "meta.json").read_text())


# One lock per run, shared by every writer: the server's threads (a VAD-bench
# job runs in a worker thread while the event loop handles marks, notes and
# candidate results) and other processes (scripts/vad_bench.py, ticket 3's
# backfill). flock() covers other processes; the RLock covers other threads in
# this one, and makes the lock re-entrant: flock alone is per open file, so a
# second open() of .meta.lock in the same thread would wait on itself forever.
_run_locks: dict[str, threading.RLock] = {}
_run_locks_guard = threading.Lock()
_held = threading.local()   # run id -> [depth, meta] for this thread


@contextlib.contextmanager
def locked_meta(run_id: str, *, save: bool = True):
    """Exclusive read-merge-write transaction on one run's meta.json.

        with store.locked_meta(run_id) as meta:
            meta["vad_bench"] = merged

    Enter: takes this run's lock (per-run threading.RLock, then an exclusive
    fcntl.flock on <run>/.meta.lock) and yields meta freshly read from disk.
    Normal exit: saves it atomically (save_meta) and releases. An exception
    inside the block: nothing is saved. `save=False` makes it a consistent read.

    Re-entrant within a thread: a nested locked_meta() on the same run yields
    the SAME dict and only the outermost block saves. Blocking, so hold it only
    across read -> merge -> write (milliseconds), never across inference or STT;
    compute outside, then merge the result into the fresh meta inside.
    """
    held = getattr(_held, "runs", None)
    if held is None:
        held = _held.runs = {}
    if run_id in held:                       # re-entry from this same thread
        held[run_id][0] += 1
        try:
            yield held[run_id][1]
        finally:
            held[run_id][0] -= 1
        return
    d = run_dir(run_id)
    with _run_locks_guard:
        lock = _run_locks.setdefault(run_id, threading.RLock())
    with lock:
        with open(d / ".meta.lock", "a+") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                meta = load_meta(run_id)
                held[run_id] = [1, meta]
                try:
                    yield meta
                    if save:
                        save_meta(run_id, meta)
                finally:
                    del held[run_id]
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def list_runs() -> list[dict]:
    runs = []
    for d in sorted(config.RUNS_DIR.iterdir(), reverse=True):
        meta_file = d / "meta.json"
        if meta_file.is_file():
            try:
                runs.append(json.loads(meta_file.read_text()))
            except json.JSONDecodeError:
                continue
    return runs
