"""One zip holding every run: a viewer HTML that opens with a double click, and
all the audio next to it. No clone, no Python, no server.

The viewer IS static/index.html with a data blob injected in place of a marker
comment — deliberately, because a second renderer would drift from the app it
exists to mirror. The blob carries what the page normally fetches (run metas)
plus the one thing it normally computes itself: waveform peaks. A file:// page
cannot fetch() a local wav to decode it, so peaks either arrive precomputed or
the export has no waveforms at all.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config, store

PEAK_COLS = 1600  # canvas.wave width in index.html — peaks are one per pixel

# AAC in m4a rather than Opus in Ogg: every browser plays it including Safari,
# which is the entire point of a file someone else opens. The waveforms and every
# metric come from the ORIGINAL pcm regardless of what is stored here, so a lossy
# copy costs listening fidelity only — never a number.
AUDIO_FORMATS = {
    "m4a": {
        "ext": "m4a",
        "args": ["-c:a", "aac", "-b:a", "48k"],
        "store": True,  # already compressed; deflating it again just burns CPU
        "note": "audio re-encoded to 48 kbps mono AAC to keep this zip shareable "
                "— waveforms and all metrics come from the original 16-bit PCM",
    },
    "wav": {
        "ext": "wav",
        "args": None,
        "store": False,
        "note": "audio is the original 16-bit PCM, bit-for-bit",
    },
}

MARKER = "<!-- NC_EXPORT_DATA -->"

README = """NC Bench export
===============

Open index.html in any browser (double click it). Everything works from disk —
no server, no install.

  index.html   the viewer: history, per-candidate metrics, transcripts,
               rankings, waveforms, and playback
  runs/<id>/   the audio for each run

Waveform scale/height controls work exactly as they do in the live app, and
clicking a waveform plays from that point.
"""


def peaks_b64(wav: Path, cols: int = PEAK_COLS) -> str | None:
    """min/max per pixel column as int16, base64.

    Mirrors what the browser computes in drawWave() from decodeAudioData: channel
    0, `ceil(n/cols)` samples per column. int16 and not int8 because the 64x and
    dB views exist to show residual noise near -60 dBFS, which an int8 quantum
    (1/127, i.e. -42 dBFS) would flatten to silence.
    """
    try:
        data, _ = sf.read(wav, dtype="int16", always_2d=True)
    except Exception:
        return None
    mono = np.ascontiguousarray(data[:, 0])
    n = mono.size
    if n == 0:
        return None
    step = -(-n // cols)
    pad = cols * step - n
    grid = np.concatenate([mono, np.zeros(pad, "int16")]) if pad else mono
    grid = grid.reshape(cols, step)
    mins, maxs = grid.min(axis=1), grid.max(axis=1)
    # the last column was zero-padded above; redo it from the real tail so a clip
    # that ends mid-word does not draw a false return to zero
    tail = mono[(cols - 1) * step:]
    if tail.size:
        mins[-1], maxs[-1] = tail.min(), tail.max()
    return base64.b64encode(
        np.concatenate([mins, maxs]).astype("<i2").tobytes()
    ).decode()


def audio_bytes(src: Path, fmt: str) -> bytes | None:
    """The bytes to store for one wav. None means "could not encode" — the caller
    keeps the metrics and the waveform and drops only the player, which is far
    better than shipping a file the browser refuses to open."""
    spec = AUDIO_FORMATS[fmt]
    if spec["args"] is None:
        return src.read_bytes()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"a.{spec['ext']}"
        p = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1",
             *spec["args"], str(out)],
            capture_output=True,
        )
        if p.returncode != 0 or not out.is_file():
            return None
        return out.read_bytes()


def viewer_html(payload: dict) -> str:
    tpl = (config.STATIC_DIR / "index.html").read_text()
    if MARKER not in tpl:
        raise RuntimeError(f"{MARKER} is missing from index.html — nowhere to inject")
    # "<" is escaped so a transcript or note containing "</script>" cannot close
    # the tag it is embedded in
    blob = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return tpl.replace(MARKER, f"<script>window.NC_EXPORT={blob};</script>")


def _plan() -> tuple[list[dict], list[tuple[str, str, Path]]]:
    """Every run's meta, and every audio file the viewer will reference.

    `original.*` (the raw upload) is left out on purpose: nothing in the UI links
    to it and on one run alone it is 118 MB.
    """
    runs = store.list_runs()
    items: list[tuple[str, str, Path]] = []
    for meta in runs:
        rid = meta.get("id")
        if not rid:
            continue
        try:
            rd = store.run_dir(rid)
        except (KeyError, FileNotFoundError):
            continue
        names = []
        if (meta.get("input") or {}).get("file"):
            names.append(meta["input"]["file"])
        for c in meta.get("candidates") or []:
            names += [c[k] for k in ("output", "output_seg") if c.get(k)]
        for name in dict.fromkeys(names):
            p = rd / name
            if p.is_file():
                items.append((rid, name, p))
    return runs, items


async def build_zip(dest: Path, fmt: str, progress=None) -> dict:
    """Write the export to `dest`. `progress(done, total)` is awaited as it goes.

    Each file hops to a thread rather than the whole build: ffmpeg and the peak
    scan are blocking, and the event loop has to stay free to push progress and
    keep the websocket alive across what can be a couple of minutes.
    """
    runs, items = _plan()
    spec = AUDIO_FORMATS[fmt]
    peaks: dict[str, str] = {}
    skipped: list[str] = []

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, (rid, name, p) in enumerate(items):
            pk = await asyncio.to_thread(peaks_b64, p)
            if pk:
                peaks[f"{rid}/{name}"] = pk
            data = await asyncio.to_thread(audio_bytes, p, fmt)
            if data is None:
                skipped.append(f"{rid}/{name}")
            else:
                info = zipfile.ZipInfo(f"runs/{rid}/{Path(name).stem}.{spec['ext']}")
                info.compress_type = (
                    zipfile.ZIP_STORED if spec["store"] else zipfile.ZIP_DEFLATED
                )
                # a hand-built ZipInfo defaults to external_attr 0, which some
                # unzip tools honour literally and extract as mode 000
                info.external_attr = 0o644 << 16
                z.writestr(info, data)
            if progress and (i % 4 == 0 or i == len(items) - 1):
                await progress(i + 1, len(items))

        z.writestr("index.html", viewer_html({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_ext": spec["ext"],
            "audio_note": spec["note"],
            "peak_cols": PEAK_COLS,
            "skipped": skipped,
            "runs": runs,
            "peaks": peaks,
        }))
        z.writestr("README.txt", README)

    return {
        "runs": len(runs),
        "files": len(items),
        "skipped": len(skipped),
        "bytes": dest.stat().st_size,
    }
