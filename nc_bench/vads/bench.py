"""One VAD-bench run over one stored run: sources × VADs → stored
probabilities → shared segmenter → frame + event scores, and on demand the
VAD-cut WER.

Storage, per (audio source, VAD), next to the audio in data/runs/<id>/:

    <source>.<vad>.vadp.npy    float16 probabilities, one per window
    <source>.<vad>.vadp.json   hop_s, t0_s, rate, cpu cost, and what they were
                               computed from: the source fingerprint (size,
                               mtime, sha256) and the inference fingerprint
                               (model file hashes, package versions, embedded
                               weights, loader config, resampler), so a
                               re-processed candidate, a re-fetched model or a
                               package upgrade invalidates them

`<source>` is `input` (the raw input.wav) or a candidate id. Probabilities are
stored raw, before any smoothing, so every segmentation choice stays a cheap
re-segmentation rather than a re-inference; ticket 3's sweeps over 20+ runs
depend on that.

meta.json gets one additive key, `vad_bench`:

    {"params": {...}, "thresholds": {...}, "runs": [...],
     "results": {source: {vad: {...}}}}

Existing keys (`vad`, `input_vad`, `truth_score`, `stt_seg`, ...) are never
touched, so every older view still renders. Every write is a merge inside
store.locked_meta (a per-run flock plus an atomic replace): results are
computed outside the lock, then merged pair by pair into a fresh read, so
concurrent writers (the UI, the CLI, a backfill, new hand marks) never lose
each other's work. The first write to a run leaves a one-time backup,
meta.pre-vadbench.json.

Every result carries the fingerprint of the audio it was computed from. A row
whose audio has changed since is marked stale (mark_stale) and left out of the
rankings, and so is a transcript whose cut list or read audio has changed.

`legacy-silero` is not a model: it reads the spans the old per-run Silero pass
stored with the run (`input_vad`, `candidates[].vad`) as they are, under the
parameters they were recorded with. Its rows are `fixed`: never re-segmented,
never ranked, there so an old run keeps the exact cyan it was judged by.

A VAD-cut transcript stores every cut's own text, so its WER is a pure function
of the run's reference and ignore spans, re-derived (rescore_wer) whenever
either changes instead of going stale.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import config, scoring, store, stt, vad
from . import MODELS_DIR, infer, registry
from .events import coalesce_truth, score_events, spans_of
from .segmenter import PLUGIN_ALPHA, segment, summary

PARAM_KEYS = ("activation_threshold", "deactivation_threshold", "min_speech_duration",
              "min_silence_duration", "prefix_padding_duration")
STT_MODES = ("same", "raw")
# (source, VAD) pairs transcribed at once. Each pair still sends its turns one
# after another; this only overlaps pairs, which is where a long call's minutes go.
STT_CONCURRENCY = 4


# ------------------------------------------------------------------ params


def default_params(meta: dict) -> dict:
    """The run's own VAD panel values (meta["vad_params"]), else the source
    default: one parameter set per bench run, reusing today's panel fields.
    The panel's sample_rate is not used; each VAD runs at its own rate."""
    base = meta.get("vad_params") or vad.defaults(meta.get("source"))
    p = {k: float(base.get(k, vad.defaults()[k])) for k in PARAM_KEYS}
    p["smoothing"] = False
    return p


def clean_params(p: dict | None, meta: dict) -> dict:
    """Defaults overlaid with what the caller sent, clamped to vad.LIMITS."""
    out = dict((meta.get("vad_bench") or {}).get("params") or default_params(meta))
    out.setdefault("smoothing", False)
    for k in PARAM_KEYS:
        if p and k in p:
            lo, hi = vad.LIMITS[k]
            try:
                out[k] = max(lo, min(hi, float(p[k])))
            except (TypeError, ValueError):
                pass
    if p and "smoothing" in p:
        out["smoothing"] = bool(p["smoothing"]) and str(p["smoothing"]).lower() not in ("0", "false", "no")
    return out


def clean_thresholds(t: dict | None) -> dict:
    """{vad id or family id: activation} or {id: {"activation": a, "deactivation": d}}."""
    out: dict = {}
    lo, hi = vad.LIMITS["activation_threshold"]
    for k, v in (t or {}).items():
        if v is None or v == "":
            continue
        try:
            if isinstance(v, dict):
                out[k] = {kk: max(lo, min(hi, float(vv))) for kk, vv in v.items()
                          if kk in ("activation", "deactivation")}
            else:
                out[k] = max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            continue
    return out


def effective_params(params: dict, entry: dict, thresholds: dict | None) -> dict:
    """What the segmenter actually runs for this VAD.

    A threshold override replaces the activation and keeps the run's
    activation-deactivation gap, so with the default equal pair (0.5/0.5) an
    override of 0.7 means 0.7/0.7. An override may be keyed by the rate
    variant (`silero-v4.0-8k`) or the family (`silero-v4.0`, both rates). A
    binary detector has no probability to threshold: its flags are segmented
    at 0.5 with no smoothing, whatever the run asked for.
    """
    p = {k: params[k] for k in PARAM_KEYS}
    if entry.get("binary"):
        p.update(activation_threshold=0.5, deactivation_threshold=0.5, smoothing=None,
                 threshold_override=False)
        return p
    ov = (thresholds or {}).get(entry["id"])
    if ov is None:
        ov = (thresholds or {}).get(entry.get("family"))
    if ov is not None:
        if isinstance(ov, dict):
            act = ov.get("activation", p["activation_threshold"])
            deact = ov.get("deactivation", max(0.01, act - (p["activation_threshold"]
                                                           - p["deactivation_threshold"])))
        else:
            act = float(ov)
            deact = max(0.01, act - (p["activation_threshold"] - p["deactivation_threshold"]))
        p["activation_threshold"], p["deactivation_threshold"] = round(act, 4), round(deact, 4)
    p["threshold_override"] = ov is not None
    p["smoothing"] = PLUGIN_ALPHA if params.get("smoothing") else None
    return p


# ------------------------------------------------------------------ sources


def source_files(meta: dict) -> dict[str, str]:
    """source id -> wav name: `input`, then every candidate with an output."""
    out = {}
    if (meta.get("input") or {}).get("file"):
        out["input"] = meta["input"]["file"]
    for c in meta.get("candidates") or []:
        if c.get("output"):
            out[c["id"]] = c["output"]
    return out


def resolve_sources(meta: dict, sources, ticked=None) -> list[str]:
    """`all`, `ticked` (input + the ticked NC candidates this run has), or an
    explicit list. Unknown ids raise: a typo must not silently shrink the
    matrix."""
    have = source_files(meta)
    if sources in (None, "all", ["all"]):
        return list(have)
    if sources in ("ticked", ["ticked"]):
        want = ["input"] + [c for c in (ticked or []) if c != "input"]
        return [s for s in dict.fromkeys(want) if s in have]
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in have]
    if unknown:
        raise ValueError(f"no audio for sources {unknown} in run {meta.get('id')}; "
                         f"have {list(have)}")
    return list(dict.fromkeys(sources))


def resolve_vads(vads) -> list[dict]:
    reg = registry()
    if vads in (None, "all", ["all"]):
        return reg
    if isinstance(vads, str):
        vads = [v.strip() for v in vads.split(",") if v.strip()]
    ids = {e["id"]: e for e in reg}
    fams: dict[str, list] = {}
    for e in reg:
        fams.setdefault(e["family"], []).append(e)
    out, unknown = [], []
    for v in vads:
        if v in ids:
            out.append(ids[v])
        elif v in fams:          # a family id selects every rate variant
            out.extend(fams[v])
        else:
            unknown.append(v)
    if unknown:
        raise ValueError(f"unknown VADs {unknown}; known: {sorted(ids)}")
    return list({e["id"]: e for e in out}.values())


# ------------------------------------------------------------ fingerprints
# Two identities travel with every stored number, and a row is only current
# while both still hold:
#
# - the SOURCE: size + mtime + sha256 of the wav the VAD read (and, for
#   lkaic-flag, of the flag file). A reprocessed candidate rewrites <cid>.wav
#   under the same name; without this its old VAD rows would keep ranking.
# - the INFERENCE: what actually computed the probabilities: the real sha256
#   of every model file the loader opens (not the registry's declared hash;
#   that is fetch_vads.py's supply-chain check), the version of every package
#   that holds code or embedded weights, the sha256 of those embedded weights
#   (livekit's native .so, TEN's library, Earshot's and WebRTC's extension
#   modules, vadonnx's bundled frontends), the loader's semantic config, the
#   resampler, and a schema version for changes in this file.

BENCH_SCHEMA = 2   # 2: resampling moved to soxr QQ (was rtc QUICK, which jitters)

_PACKAGES = {
    "lk_inference": ["livekit-agents", "livekit-local-inference", "livekit"],
    "silero_onnx": ["onnxruntime"],
    "firered": ["onnxruntime", "kaldi-native-fbank", "kaldiio"],
    "ten": ["ten-vad"],
    "earshot": ["earshot"],
    "vadonnx": ["vadonnx", "onnxruntime", "kaldi-native-fbank"],
    "fsmn8k": ["vadonnx", "onnxruntime", "kaldi-native-fbank"],
    "webrtc": ["webrtcvad-wheels"],
    "aic_vad": ["aic-sdk"],
    "lkaic_flag": [],
    "legacy_spans": [],
}
# embedded weights / native code, as (importable module, glob inside its dir)
_RESOURCES = {
    "lk_inference": [("livekit.local_inference", "_native*")],
    "ten": [("ten_vad", "lib/macOS/ten_vad.framework/Versions/A/ten_vad"),
            ("ten_vad", "lib/Linux/x64/libten_vad.so")],
    "earshot": [("earshot", "_earshot*")],
    "webrtc": [("_webrtcvad", None)],
    "vadonnx": [("vadonnx", "data/*")],
    "fsmn8k": [("vadonnx", "data/*")],
}
_CONFIG_KEYS = ("rate", "hop_s", "state_api", "onnx", "cmvn", "mvn", "mode", "vadonnx_name",
                "model_id", "model_file", "binary")

_sha_cache: dict[tuple, str] = {}


def file_sha256(path: Path) -> str:
    """sha256, cached per (path, size, mtime) so a UI refresh re-hashes nothing."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def source_fp(path: Path) -> dict:
    st = path.stat()
    return {"file": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "sha256": file_sha256(path)}


def fp_current(fp: dict | None, path: Path) -> bool:
    """Does `path` still hold the audio `fp` was taken from? Content, not
    timestamps: a touch() is not a change, a same-size rewrite is."""
    if not fp or not path.is_file():
        return False
    return path.stat().st_size == fp.get("size") and file_sha256(path) == fp.get("sha256")


def _resource_files(kind: str) -> list[Path]:
    import importlib.util

    out = []
    for mod, pattern in _RESOURCES.get(kind, []):
        spec = importlib.util.find_spec(mod)
        if spec is None:
            continue
        if pattern is None:
            out.append(Path(spec.origin))
            continue
        base = Path(next(iter(spec.submodule_search_locations or []), Path(spec.origin).parent))
        out += sorted(p for p in base.glob(pattern) if p.is_file())
    return out


def inference_fp(entry: dict) -> dict:
    import importlib.metadata

    from .loaders import RESAMPLER

    def ver(dist):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return None

    files = {rel: file_sha256(MODELS_DIR / rel) for rel in (entry.get("files") or {})
             if (MODELS_DIR / rel).is_file()}
    return {
        "schema": BENCH_SCHEMA, "loader": entry.get("loader"),
        "config": {k: entry[k] for k in _CONFIG_KEYS if k in entry},
        "files": files,
        "packages": {d: ver(d) for d in _PACKAGES.get(entry.get("loader"), []) + ["soxr"]},
        "resources": {p.name: file_sha256(p) for p in _resource_files(entry.get("loader"))},
        "resampler": RESAMPLER,
    }


def fp_key(fp: dict) -> str:
    return hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------ probabilities


def probs_paths(run_dir: Path, sid: str, vid: str) -> tuple[Path, Path]:
    # spelled out, not with_suffix(): ".vadp" IS the suffix of "<sid>.<vid>.vadp",
    # so with_suffix would replace it and name the files <sid>.<vid>.npy
    return run_dir / f"{sid}.{vid}.vadp.npy", run_dir / f"{sid}.{vid}.vadp.json"


def load_probs(run_dir: Path, sid: str, entry: dict, wav: Path | None = None) -> dict | None:
    """Stored probabilities, or None if absent or stale: made from other audio
    (source fingerprint), by another model / package / loader configuration
    (inference fingerprint), or not the payload this sidecar describes.

    The last check is what makes a cache pair trustworthy under concurrent
    writers (the UI and a backfill, say): the sidecar carries the sha256 of the
    exact .vadp.npy it was published with, and an .npy that does not hash to it
    is a miss, never a hit. See save_probs for the publication order.
    """
    npy, js = probs_paths(run_dir, sid, entry["id"])
    if not npy.is_file() or not js.is_file():
        return None
    try:
        side = json.loads(js.read_text())
    except json.JSONDecodeError:
        return None
    if wav is not None and not fp_current(side.get("source_fp"), wav):
        return None
    if side.get("inference_key") != fp_key(inference_fp(entry)):
        return None
    try:
        payload = npy.read_bytes()
    except OSError:
        return None
    if not side.get("payload_sha256") or hashlib.sha256(payload).hexdigest() != side["payload_sha256"]:
        return None                     # an older/newer .npy under this sidecar: a miss
    side["probs"] = np.load(io.BytesIO(payload)).astype(np.float32)
    return side


class StaleSource(RuntimeError):
    """The source audio changed while its probabilities were being computed."""


@contextlib.contextmanager
def _pair_lock(run_dir: Path, sid: str, vid: str):
    """Exclusive, cross-process lock on one (source, VAD) cache pair."""
    with open(run_dir / f".{sid}.{vid}.vadp.lock", "a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def save_probs(run_dir: Path, sid: str, entry: dict, res: dict, wav: Path, dur: float,
               src_fp: dict | None = None) -> dict:
    """Publish one (source, VAD) cache pair so that a reader can never accept a
    payload the sidecar does not describe, and an older job can never replace a
    newer one.

    `src_fp` is the fingerprint of the audio the probabilities were computed
    FROM (taken before it was read). Under a per-pair flock: if the wav on disk
    no longer matches it, the job is older than the audio and nothing is
    published (StaleSource). Otherwise the .npy is written to a temp name and
    os.replace()d, and the sidecar, which names the payload's sha256, is
    written the same way LAST. A reader that catches the pair between the two
    replaces sees a hash mismatch, which load_probs treats as a miss.
    """
    npy, js = probs_paths(run_dir, sid, entry["id"])
    buf = io.BytesIO()
    np.save(buf, res["probs"].astype(np.float16))
    payload = buf.getvalue()
    ifp = inference_fp(entry)
    src_fp = src_fp or source_fp(wav)
    side = {
        "vad": entry["id"], "source": sid, "rate": entry.get("rate"),
        "hop_s": res["hop_s"], "t0_s": res["t0_s"], "n": int(len(res["probs"])),
        "duration_s": round(dur, 3), "binary": bool(entry.get("binary")),
        "cpu_ms_per_s": res["cpu_ms_per_s"], "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_fp": src_fp, "inference_fp": ifp, "inference_key": fp_key(ifp),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "generation": uuid.uuid4().hex,
    }
    with _pair_lock(run_dir, sid, entry["id"]):
        if not fp_current(src_fp, wav):
            raise StaleSource(f"{wav.name} changed while {entry['id']} ran on it; not published")
        _atomic_write(npy, payload)
        _atomic_write(js, json.dumps(side, indent=1).encode())
    side["probs"] = res["probs"]
    return side


# ----------------------------------------------------- ai-coustics flags


FLAG_MIN_COVERAGE = 0.99
_FLAG_MIN_NUM, _FLAG_MIN_DEN = 99, 100   # the same floor, exactly


def _coverage_ok(result: dict) -> bool:
    """valid / total >= 99%, decided on the exact integer counts: the stored
    ratio is rounded for display, and 24,749 / 25,000 (98.996%) rounds to
    0.99 but is below the floor. Rows written before the counts existed fall
    back to the ratio."""
    v, t = result.get("valid_frames"), result.get("total_frames")
    if isinstance(v, int) and isinstance(t, int):
        return t > 0 and v * _FLAG_MIN_DEN >= _FLAG_MIN_NUM * t
    return float(result.get("flag_coverage") or 0.0) >= FLAG_MIN_COVERAGE


def lkaic_flags(meta: dict, run_dir: Path, sid: str) -> dict | None:
    """Ticket 1's per-frame ai-coustics flag for an lkaic-* candidate, or None
    when this source has none.

    Contract (ticket 1, 2026-09-29): candidate["aicvad"] = {file, frame_s,
    t0_s, missing_frames, dtype, encoding}; frame k covers [t0_s + k*frame_s,
    (k+1)*frame_s) of <cid>.wav time. The file is int8: 1 speech, 0 not
    speech, -1 no flag on that frame. An older np.bool_ file means every
    frame had a flag.

    A missing frame must never become a confident "no speech": that would hand
    the plugin a phantom it never rejected, or a miss it never made. For the
    segmenter it holds the previous frame's decision (the flag simply did not
    change, as far as anyone knows), and the row carries flag_coverage and the
    missing stretches, which score() uses to keep it out of every ranking.
    """
    c = next((x for x in meta.get("candidates") or [] if x.get("id") == sid), None)
    info = (c or {}).get("aicvad")
    if not info or not info.get("file"):
        return None
    f = run_dir / info["file"]
    if not f.is_file():
        return None
    raw = np.load(f).reshape(-1)
    if raw.dtype == np.bool_:
        flags, valid = raw.astype(np.float32), np.ones(len(raw), dtype=bool)
    else:
        valid = raw >= 0
        flags = (raw == 1).astype(np.float32)
    held, last = flags.copy(), None
    for i in range(len(held)):          # hold the last known decision over a gap
        if valid[i]:
            last = held[i]
        elif last is not None:
            held[i] = last
    if not valid[:1].all() and valid.any():   # a gap at the start takes the first known one
        first = int(np.argmax(valid))
        held[:first] = held[first]
    hop, t0 = float(info["frame_s"]), float(info.get("t0_s") or 0.0)
    missing = []
    for i in np.flatnonzero(~valid):
        s = t0 + i * hop
        if missing and abs(missing[-1][1] - s) < 1e-9:
            missing[-1][1] = s + hop
        else:
            missing.append([s, s + hop])
    return {"probs": held, "hop_s": hop, "t0_s": t0, "rate": None, "cpu_ms_per_s": None,
            "probs_file": info["file"],
            # exact counts decide eligibility; the ratio is for display only
            "valid_frames": int(valid.sum()), "total_frames": int(len(valid)),
            "flag_coverage": round(float(valid.mean()), 4) if len(valid) else 0.0,
            "missing_spans": [[round(float(a), 3), round(float(b), 3)] for a, b in missing],
            "aux_fp": source_fp(f)}


# ------------------------------------------------------ legacy Silero spans


LEGACY_FIXED_WHY = ("fixed: the spans the old per-run Silero pass recorded with this run, "
                    "under its own parameters, so not comparable with the bench's rows")


def legacy_label(v: dict) -> str:
    """What produced a recorded span set. Records written before 2026-09-29 carry
    no model field: every one of them came from livekit-plugins-silero 1.6.7
    (uv.lock since 2026-08-01), which bundled the Silero v6.0 ONNX, so that is
    what they are labelled, as an inference (see vad.model_info)."""
    if v.get("model"):
        return f"{v['model']} · livekit-plugins-silero {v.get('silero_plugin_version') or '?'} (recorded)"
    return "Silero v6.0 · livekit-plugins-silero 1.6.7 (inferred from the 2026-08-01 lock)"


def legacy_spans(meta: dict, sid: str) -> dict | None:
    """The span record the old per-run Silero pass stored for one source, or None."""
    if sid == "input":
        v = meta.get("input_vad")
    else:
        c = next((x for x in meta.get("candidates") or [] if x.get("id") == sid), None)
        v = (c or {}).get("vad")
    if not isinstance(v, dict) or v.get("error") or v.get("segments") is None:
        return None
    return v


def evaluate_fixed(v: dict, entry: dict, truth, truth_dur: float, file_dur: float,
                   src_fp: dict | None = None, known_nonspeech=None, ignore=None) -> dict:
    """A legacy-silero row: the recorded spans exactly, no segmenter, scored
    against the marks like any other row."""
    segs = [[float(a), float(b)] for a, b in v["segments"]]
    rp = v.get("params") or {}
    p = {k: float(rp[k]) for k in PARAM_KEYS if rp.get(k) is not None}
    p.setdefault("prefix_padding_duration", float(config.VAD_PREFIX_PADDING_DURATION))
    # the old plugin path always smoothed (ExpFilter); recorded for the thr column
    p.update(threshold_override=False, smoothing=PLUGIN_ALPHA)
    out = {
        "vad": entry["id"], "label": legacy_label(v), "rate": rp.get("sample_rate"),
        "hop_s": None, "t0_s": 0.0, "n_frames": None, "probs_file": None,
        "cpu_ms_per_s": None, "binary": False, "kind": entry.get("kind"), "fixed": True,
        "params": p, "segments": segs, **summary(segs, file_dur),
        "source_fp": src_fp, "inference_key": None,
    }
    out.update(score(out, truth, truth_dur, known_nonspeech, ignore))
    return out


# ------------------------------------------------------------------ scoring


def _truth_duration(meta: dict) -> float:
    d = float((meta.get("input") or {}).get("duration_s") or 0)
    return d or float((meta.get("input_vad") or {}).get("duration_s") or 0)


def evaluate(rec: dict, entry: dict, params: dict, thresholds: dict | None,
             truth, truth_dur: float, file_dur: float, src_fp: dict | None = None,
             known_nonspeech=None, ignore=None) -> dict:
    """Segment one stored probability record and score it."""
    p = effective_params(params, entry, thresholds)
    segs = segment(rec["probs"], rec["hop_s"], t0_s=rec.get("t0_s") or 0.0,
                   duration_s=file_dur, smoothing=p["smoothing"],
                   **{k: p[k] for k in ("activation_threshold", "deactivation_threshold",
                                        "min_speech_duration", "min_silence_duration")})
    out = {
        "vad": entry["id"], "label": entry.get("label"), "rate": rec.get("rate"),
        "hop_s": rec["hop_s"], "t0_s": rec.get("t0_s") or 0.0, "n_frames": int(len(rec["probs"])),
        "probs_file": rec.get("probs_file"), "cpu_ms_per_s": rec.get("cpu_ms_per_s"),
        "binary": bool(entry.get("binary")), "kind": entry.get("kind"),
        "params": p, "segments": segs, **summary(segs, file_dur),
        "source_fp": src_fp, "inference_key": rec.get("inference_key"),
    }
    for k in ("flag_coverage", "valid_frames", "total_frames", "missing_spans", "aux_fp"):
        if k in rec:
            out[k] = rec[k]
    out.update(score(out, truth, truth_dur, known_nonspeech, ignore))
    return out


def _touches(spans, windows) -> bool:
    return any(a < d and b > c for a, b in spans or [] for c, d in windows or [])


def score(result: dict, truth, truth_dur: float, known_nonspeech=None, ignore=None) -> dict:
    """Frame + event scores for a result's segments (pure; rescore() reuses it),
    and whether the row may rank at all."""
    segs = result.get("segments")
    # Events are scored on turns the segmenter could actually split: marks whose
    # gap is below this row's own min_silence are one turn to it (coalesce_truth).
    # Frame scores stay on the raw marks, which is what they measure.
    min_sil = float((result.get("params") or {}).get("min_silence_duration") or 0.0)
    turns = coalesce_truth(truth, min_sil, known_nonspeech) if truth else []
    out = {
        "frame": vad.score_spans(truth, segs, truth_dur, ignore) if truth else None,
        "events": score_events(turns, segs, ignore, known_nonspeech),
        "truth_turns": len(turns) if truth else None,
        "truth_turns_raw": len(truth) if truth else None,
        "rank_excluded": None,
    }
    if "flag_coverage" in result:
        miss = result.get("missing_spans") or []
        if not _coverage_ok(result):
            out["rank_excluded"] = (f"ai-coustics flags cover {result['flag_coverage']:.1%} "
                                    f"of frames (< {FLAG_MIN_COVERAGE:.0%})")
        elif _touches(miss, truth):
            out["rank_excluded"] = "a frame with no ai-coustics flag falls inside marked speech"
        elif _touches(miss, spans_of(known_nonspeech)):
            out["rank_excluded"] = "a frame with no ai-coustics flag falls inside a known non-speech window"
    if result.get("fixed") and not out["rank_excluded"]:
        out["rank_excluded"] = LEGACY_FIXED_WHY
    return out


def rescore(meta: dict) -> None:
    """Re-score every stored VAD-bench result against meta["truth_spans"] (and
    meta["known_nonspeech"], meta["ignore_spans"]), in place. Called from vad.score_run, so re-marking
    speech refreshes these numbers along with every candidate's truth_score."""
    vb = meta.get("vad_bench") or {}
    truth = meta.get("truth_spans") or []
    kns = meta.get("known_nonspeech") or []
    ign = meta.get("ignore_spans") or []
    dur = _truth_duration(meta)
    for per in (vb.get("results") or {}).values():
        for r in per.values():
            if r.get("segments") is not None:
                r.update(score(r, truth, dur, kns, ign))


def _seg_key(segments, pad) -> str:
    return hashlib.sha1(json.dumps([segments, round(float(pad), 4)]).encode()).hexdigest()[:12]


def _row_pad(row: dict, default: float = 0.5) -> float:
    """The prefix padding a row's cuts are made with: its own parameters (the
    bench's for a segmented row, the recorded ones for a legacy-silero row)."""
    return float((row.get("params") or {}).get("prefix_padding_duration", default))


def _inside(span, windows) -> bool:
    a, b = span
    return any(c <= a and b <= d for c, d in windows or [])


def hyp_text(rec: dict, ignore=None) -> str:
    """What a VAD-cut WER scores: every cut's text in time order, minus the cuts
    the VAD opened entirely inside an ignore span (label uncertain: neither
    credited nor charged). Transcripts made before cuts were stored only have
    the join, which is scored as it is."""
    cuts = rec.get("cuts")
    if cuts is None:
        return rec.get("text") or ""
    return " ".join(c["text"] for c in cuts
                    if c.get("text") and not _inside(c.get("span") or [c["start"], c["end"]], ignore))


def rescore_wer(meta: dict) -> int:
    """Re-derive every stored VAD-cut WER from the run's current reference
    (meta["script"]) and ignore spans, in place. No STT: the cut texts are
    stored. Returns how many transcripts were scored."""
    script = (meta.get("script") or "").strip()
    ign = meta.get("ignore_spans") or []
    n = 0
    for per in ((meta.get("vad_bench") or {}).get("results") or {}).values():
        for r in per.values():
            for s in (r.get("stt") or {}).values():
                if s.get("error"):
                    continue
                s["wer"] = scoring.wer(script, hyp_text(s, ign)) if script else None
                n += 1
    return n


def cut_cache(results: dict, run_dir: Path, names) -> dict[str, dict]:
    """Every cut text already stored on the run that is still valid, per read
    file: {file name: {stt.cut_id: {text, confidence, language}}}.

    Valid = the file still holds the audio the transcript read (sha256, not
    timestamps) and the transcript was made under the current stt.cache_key().
    Transcripts from before the key was stored carry none and are taken as
    current: the key is new, the endpoint is not. A stale row's cuts count too:
    staleness is about the row's segments, and a cut with the same bounds on the
    same audio is the same request whatever row it came from."""
    key = stt.cache_key()
    sha = {n: file_sha256(run_dir / n) for n in names if (run_dir / n).is_file()}
    out: dict[str, dict] = {n: {} for n in sha}
    for per in results.values():
        for r in per.values():
            for rec in (r.get("stt") or {}).values():
                name = rec.get("read")
                if (name not in sha or rec.get("error") or rec.get("cuts") is None
                        or (rec.get("read_fp") or {}).get("sha256") != sha[name]
                        or rec.get("stt_key", key) != key):
                    continue
                for c in rec["cuts"]:
                    if "text" in c:
                        out[name][stt.cut_id(c["start"], c["end"])] = {
                            "text": c["text"], "confidence": c.get("confidence"),
                            "language": c.get("language") or rec.get("language")}
    return out


def _stt_read_name(files: dict, sid: str, mode: str) -> str | None:
    return files.get(sid) if mode == "same" else (files.get("none") or files.get("input"))


def mark_stale(meta: dict, run_dir: Path) -> int:
    """Flag every result, and every VAD-cut transcript, whose audio is no longer
    what is on disk. In place; returns how many rows are stale.

    A stale row keeps its numbers for inspection but is greyed in the table and
    left out of every ranking and overlay. A row cleared by a parameter change
    it could not follow (resegment-only with no fresh probabilities) is stale
    with no segments at all.
    """
    vb = meta.get("vad_bench") or {}
    files = source_files(meta)
    pad = float((vb.get("params") or {}).get("prefix_padding_duration", 0.5))
    n = 0
    for sid, per in (vb.get("results") or {}).items():
        wav = run_dir / files[sid] if sid in files else None
        for r in per.values():
            if r.get("segments") is None:
                n += bool(r.get("stale"))
                continue
            why = None
            if wav is None:
                why = "this source no longer has audio in the run"
            elif not r.get("source_fp"):
                why = "computed before rows carried a source fingerprint; re-run the bench"
            elif not fp_current(r.get("source_fp"), wav):
                why = "the source audio changed since this row was computed; re-run the bench"
            elif r.get("aux_fp") and not fp_current(r["aux_fp"], run_dir / r["aux_fp"]["file"]):
                why = "the ai-coustics flag file changed since this row was computed"
            if why:
                r["stale"], r["stale_reason"] = True, why
                n += 1
            else:
                r.pop("stale", None)
                r.pop("stale_reason", None)
            key = _seg_key(r["segments"], _row_pad(r, pad))
            for mode, s in (r.get("stt") or {}).items():
                name = _stt_read_name(files, sid, mode)
                if s.get("seg_key") is None:
                    bad = "made before transcripts carried their cut list and audio fingerprint"
                elif s.get("seg_key") != key:
                    bad = "the cut list changed since this transcript"
                elif not name or not fp_current(s.get("read_fp"), run_dir / name):
                    bad = "the audio STT read changed since this transcript"
                else:
                    bad = None
                if bad:
                    s["stale"], s["stale_reason"] = True, bad
                else:
                    s.pop("stale", None)
                    s.pop("stale_reason", None)
    return n


def fresh_view(run_id: str) -> dict:
    """meta["vad_bench"] as it stands against the audio on disk now: staleness
    re-checked in memory, nothing saved. What the page renders."""
    meta = store.load_meta(run_id)
    mark_stale(meta, store.run_dir(run_id))
    return meta.get("vad_bench") or {}


# ---------------------------------------------------------------- the run


def _read_mono(wav: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(wav, dtype="int16", always_2d=True)
    return np.ascontiguousarray(data[:, 0]), rate


def _stale_row(vid: str, why: str) -> dict:
    return {"vad": vid, "segments": None, "stale": True, "stale_reason": why}


def _carry_stt(res: dict, prev: dict, params: dict) -> dict:
    """Keep a pair's VAD-cut transcripts only while the cut list is unchanged.
    Whether the audio STT read is unchanged is checked per transcript by
    mark_stale (read_fp), since `raw` reads another file than the VAD did."""
    key = _seg_key(res["segments"], _row_pad(res, params["prefix_padding_duration"]))
    kept = {m: s for m, s in (prev.get("stt") or {}).items() if s.get("seg_key") == key}
    if kept:
        res["stt"] = kept
    res["seg_key"] = key
    return res


def _backup_once(run_dir: Path) -> None:
    backup = run_dir / "meta.pre-vadbench.json"
    if not backup.exists():
        shutil.copy2(run_dir / "meta.json", backup)


def _resegment_rest(meta: dict, run_dir: Path, touched: set, p: dict, th: dict, counts: dict) -> None:
    """One parameter set per run: every OTHER stored pair follows the new
    parameters and overrides (milliseconds; no model runs). A pair that cannot
    (its probabilities are gone or stale) is cleared to a stale row rather than
    left scored under the old parameters."""
    vb = meta["vad_bench"]
    files = source_files(meta)
    reg = {e["id"]: e for e in registry()}
    truth, kns, tdur = meta.get("truth_spans") or [], meta.get("known_nonspeech") or [], _truth_duration(meta)
    ign = meta.get("ignore_spans") or []
    for sid, per in vb.get("results", {}).items():
        if sid not in files:
            continue
        wav = run_dir / files[sid]
        file_dur = None
        for vid, prev in list(per.items()):
            e = reg.get(vid)
            if (sid, vid) in touched or prev.get("segments") is None or e is None:
                continue
            # recorded spans keep the parameters they were recorded with
            if prev.get("fixed") or e["loader"] == "legacy_spans":
                continue
            if prev.get("params") == effective_params(p, e, th):
                continue
            try:
                rec = (lkaic_flags(meta, run_dir, sid) if e["loader"] == "lkaic_flag"
                       else load_probs(run_dir, sid, e, wav))
                if rec is None:
                    per[vid] = _stale_row(vid, "parameters changed and this pair's probabilities "
                                               "are missing or stale: re-run inference")
                    counts["stale"] = counts.get("stale", 0) + 1
                    continue
                rec.setdefault("probs_file", probs_paths(run_dir, sid, vid)[0].name)
                if file_dur is None:
                    info = sf.info(wav)
                    file_dur = info.frames / info.samplerate
                per[vid] = _carry_stt(evaluate(rec, e, p, th, truth, tdur, file_dur,
                                               source_fp(wav), kns, ign), prev, p)
                counts["resegmented"] += 1
            except Exception as ex:  # noqa: BLE001
                per[vid] = _stale_row(vid, f"re-segmentation failed: {type(ex).__name__}: {ex}")
                counts["errors"] += 1


def run_bench(run_id: str, *, vads=None, sources=None, ticked=None, params=None,
              thresholds=None, force: bool = False, resegment_only: bool = False,
              progress=None) -> dict:
    """Infer (or reuse) probabilities for every (source, VAD), segment, score,
    and merge into meta["vad_bench"]. Synchronous and CPU-bound: the server
    calls it through asyncio.to_thread.

    Inference and segmentation run WITHOUT the run's lock. The merge takes it
    (store.locked_meta): a fresh read of meta.json, only this job's pairs
    replaced, every other pair re-segmented to this job's parameters, stale
    rows flagged, one atomic write. So two jobs on disjoint pairs, a CLI and
    the UI, or a bench job and a set of new hand marks, never lose each
    other's results.

    Idempotent: a (source, VAD) whose stored probabilities are fresh is not
    re-inferred unless `force`. `resegment_only` never runs a model; a pair
    without fresh probabilities becomes a stale row (excluded from rankings)
    instead of keeping spans made under other parameters.
    """
    t_start = time.time()
    meta = store.load_meta(run_id)
    run_dir = store.run_dir(run_id)
    files = source_files(meta)
    sids = resolve_sources(meta, sources, ticked)
    entries = resolve_vads(vads)
    p = clean_params(params, meta)
    vb0 = meta.get("vad_bench") or {}
    th = dict(vb0.get("thresholds") or {})
    if thresholds is not None:
        th = clean_thresholds(thresholds)
    truth, kns = meta.get("truth_spans") or [], meta.get("known_nonspeech") or []
    ign = meta.get("ignore_spans") or []
    truth_dur = _truth_duration(meta)
    counts = {"inferred": 0, "cached": 0, "resegmented": 0, "fixed": 0, "unavailable": 0,
              "skipped": 0, "errors": 0}
    total, done = len(sids) * len(entries), 0
    updates: dict[tuple[str, str], dict] = {}

    for sid in sids:
        wav = run_dir / files[sid]
        # fingerprint BEFORE the read, re-checked after: sfp must describe the
        # exact samples inference sees, or a reprocess landing mid-read would
        # publish probabilities of one file under the fingerprint of another
        sfp = source_fp(wav)
        pcm, rate = _read_mono(wav)
        if not fp_current(sfp, wav):
            for e in entries:
                updates[(sid, e["id"])] = _stale_row(
                    e["id"], f"{wav.name} changed while it was being read; re-run the bench")
                counts["stale"] = counts.get("stale", 0) + 1
                done += 1
            continue
        file_dur = len(pcm) / rate if rate else 0.0
        for e in entries:
            vid, stage = e["id"], None
            try:
                if not e["available"]:
                    updates[(sid, vid)] = {"vad": vid, "unavailable": e["unavailable_reason"]}
                    stage = "unavailable"
                elif e["loader"] == "lkaic_flag":
                    rec = lkaic_flags(meta, run_dir, sid)
                    if rec is None:
                        updates[(sid, vid)] = {"vad": vid, "unavailable": "no flags for this source"}
                        stage = "unavailable"
                    else:
                        stage = "resegmented"
                elif e["loader"] == "legacy_spans":
                    v = legacy_spans(meta, sid)
                    if v is None:
                        updates[(sid, vid)] = {"vad": vid, "unavailable":
                                               "no spans were recorded with the run for this source"}
                        stage = "unavailable"
                    else:
                        updates[(sid, vid)] = evaluate_fixed(v, e, truth, truth_dur, file_dur,
                                                             sfp, kns, ign)
                        stage = "fixed"
                else:
                    rec = load_probs(run_dir, sid, e, wav) if not force else None
                    if rec is not None:
                        stage = "resegmented" if resegment_only else "cached"
                    elif resegment_only:
                        updates[(sid, vid)] = _stale_row(
                            vid, "re-segment only, but there are no fresh probabilities for this "
                                 "pair: run the bench (inference) first")
                        stage = "skipped"
                    else:
                        rec = save_probs(run_dir, sid, e, infer(e, pcm, rate), wav, file_dur,
                                         src_fp=sfp)
                        stage = "inferred"
                    if rec is not None:
                        rec["probs_file"] = probs_paths(run_dir, sid, vid)[0].name
                if stage in ("inferred", "cached", "resegmented"):
                    updates[(sid, vid)] = evaluate(rec, e, p, th, truth, truth_dur, file_dur,
                                                   sfp, kns, ign)
            except StaleSource as ex:
                updates[(sid, vid)] = _stale_row(vid, f"{ex}; re-run the bench")
                stage = "stale"
            except Exception as ex:  # noqa: BLE001 - one model must not sink the matrix
                updates[(sid, vid)] = {"vad": vid, "error": f"{type(ex).__name__}: {ex}"}
                stage = "errors"
            counts[stage] = counts.get(stage, 0) + 1
            done += 1
            if progress:
                progress({"type": "vad_bench", "run_id": run_id, "done": done, "total": total,
                          "source": sid, "vad": vid, "stage": stage})

    with store.locked_meta(run_id) as fresh:
        _backup_once(run_dir)
        vb = fresh.setdefault("vad_bench", {})
        results = vb.setdefault("results", {})
        for (sid, vid), res in updates.items():
            prev = results.get(sid, {}).get(vid) or {}
            if res.get("segments") is not None:
                res = _carry_stt(res, prev, p)
            results.setdefault(sid, {})[vid] = res
        vb["params"], vb["thresholds"], vb["version"] = p, th, BENCH_SCHEMA
        _resegment_rest(fresh, run_dir, set(updates), p, th, counts)
        n_stale = mark_stale(fresh, run_dir)
        rescore(fresh)
        log = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "mode": "resegment" if resegment_only else "infer",
               "vads": [e["id"] for e in entries], "sources": sids, "force": bool(force),
               "params": p, "thresholds": th, "elapsed_s": round(time.time() - t_start, 1),
               "stale_rows": n_stale, **counts}
        vb.setdefault("runs", []).append(log)
    return log


# ---------------------------------------------------------------- VAD-cut WER


def _wav16(wav: Path, tmpdir: Path) -> Path:
    """stt.transcribe_turns wants PIPELINE_RATE audio; input.wav is usually 48 kHz."""
    info = sf.info(wav)
    if info.samplerate == config.WHISPER_SAMPLE_RATE and info.channels == 1:
        return wav
    out = tmpdir / f"{wav.stem}.16k.wav"
    if not out.exists():
        audio = scoring.load_16k(wav)
        sf.write(out, (np.clip(audio, -1, 1) * 32767).astype(np.int16),
                 config.WHISPER_SAMPLE_RATE, subtype="PCM_16")
    return out


async def run_stt(run_id: str, *, vads=None, sources=None, modes=STT_MODES,
                  progress=None, force: bool = False) -> dict:
    """VAD-cut WER per (source, VAD): one /recognize request per turn, the
    existing segmented-STT path (stt.transcribe_turns), in two read modes:

    - same: STT reads the audio the VAD ran on (the source itself);
    - raw:  STT reads `none` (or input.wav when the run has no `none`),
            production's split gate: VAD on the gate audio, STT on raw.

    Expensive (one request per turn per pair per mode), so never automatic.
    STT_CONCURRENCY pairs run at once. Each transcript is stored with the cut
    list (seg_key) and the fingerprint of the file it read (read_fp); it goes
    stale when either changes. The STT runs without the lock; the merge takes it
    and discards a transcript whose row was re-segmented or whose audio changed
    in the meantime. WER is scored in the merge, from the reference as it is
    then (rescore_wer), so references saved while the job ran are the ones used.

    Cuts are cached (cut_cache): a cut with the same bounds on the same audio,
    under the same STT endpoint, reuses the text already stored, so a re-run
    after a reference edit sends nothing and one after a re-segment sends only
    the cuts that moved. `force` sends every cut again.
    """
    t_start = time.time()
    meta = store.load_meta(run_id)
    run_dir = store.run_dir(run_id)
    vb = meta.get("vad_bench") or {}
    mark_stale(meta, run_dir)
    results = vb.get("results") or {}
    files = source_files(meta)
    modes = [m for m in (modes or STT_MODES) if m in STT_MODES]
    want_s = set(resolve_sources(meta, sources)) if sources not in (None, "all") else None
    want_v = {e["id"] for e in resolve_vads(vads)} if vads not in (None, "all") else None
    pad = float((vb.get("params") or {}).get("prefix_padding_duration", 0.5))

    todo = [(sid, vid, r) for sid, per in results.items() for vid, r in per.items()
            if r.get("segments") is not None and not r.get("stale")
            and (want_s is None or sid in want_s) and (want_v is None or vid in want_v)]
    got_all: list[tuple[str, str, str, dict]] = []
    counts = {"done": 0, "errors": 0, "sent": 0, "reused": 0}
    skey = stt.cache_key()
    with tempfile.TemporaryDirectory(prefix="vadbench-stt-") as tmp:
        tmpdir = Path(tmp)
        # each read file resampled once, before any pair starts: two pairs
        # racing on one file would both write its 16 kHz copy
        names = {_stt_read_name(files, sid, m) for sid, _v, _r in todo for m in modes} - {None}
        wav16 = {n: await asyncio.to_thread(_wav16, run_dir / n, tmpdir) for n in sorted(names)}
        # shared by every pair reading the same file, and filled as the job goes:
        # `none` read as same and as raw is the same request twice
        caches = {} if force else await asyncio.to_thread(cut_cache, results, run_dir, names)
        sem = asyncio.Semaphore(STT_CONCURRENCY)

        async def one(sid: str, vid: str, r: dict) -> None:
            rpad = _row_pad(r, pad)
            key = _seg_key(r["segments"], rpad)
            async with sem:
                for mode in modes:
                    name = _stt_read_name(files, sid, mode)
                    rec = {"read": name, "seg_key": key, "stt_key": skey}
                    try:
                        if not name:
                            raise FileNotFoundError("the run has no audio for this read mode")
                        rec["read_fp"] = source_fp(run_dir / name)
                        rec.update(await stt.transcribe_turns(
                            wav16[name], r["segments"], rpad,
                            cache=None if force else caches.setdefault(name, {})))
                        counts["sent"] += rec["sent"]
                        counts["reused"] += rec["reused"]
                    except Exception as ex:  # noqa: BLE001
                        rec["error"] = f"{type(ex).__name__}: {ex}"
                        counts["errors"] += 1
                    got_all.append((sid, vid, mode, rec))
            counts["done"] += 1
            if progress:
                await progress({"type": "vad_bench", "run_id": run_id, "done": counts["done"],
                                "total": len(todo), "source": sid, "vad": vid, "stage": "stt"})

        await asyncio.gather(*(one(*t) for t in todo))

    def merge() -> dict:
        discarded = 0
        with store.locked_meta(run_id) as fresh:
            _backup_once(run_dir)
            fvb = fresh.setdefault("vad_bench", {})
            fpad = float((fvb.get("params") or {}).get("prefix_padding_duration", 0.5))
            for sid, vid, mode, rec in got_all:
                row = (fvb.get("results") or {}).get(sid, {}).get(vid)
                if (not row or row.get("segments") is None
                        or _seg_key(row["segments"], _row_pad(row, fpad)) != rec["seg_key"]):
                    discarded += 1          # re-segmented while this STT was running
                    continue
                row.setdefault("stt", {})[mode] = rec
                row["seg_key"] = rec["seg_key"]
            mark_stale(fresh, run_dir)
            rescore_wer(fresh)
            log = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "stt", "modes": modes,
                   "pairs": counts["done"], "errors": counts["errors"], "discarded": discarded,
                   "sent": counts["sent"], "reused": counts["reused"], "force": bool(force),
                   "elapsed_s": round(time.time() - t_start, 1)}
            fvb.setdefault("runs", []).append(log)
        return log

    return await asyncio.to_thread(merge)
