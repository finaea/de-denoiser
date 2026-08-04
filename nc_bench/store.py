"""Run history on disk: data/runs/<id>/ holds meta.json, input.wav and one
<candidate-id>.wav per candidate. meta.json is the single source of truth."""

from __future__ import annotations

import json
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
        # free-text conditions for this run ("someone talking beside me",
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
    (config.RUNS_DIR / run_id / "meta.json").write_text(json.dumps(meta, indent=2))


def load_meta(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "meta.json").read_text())


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
