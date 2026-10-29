"""Do restore_orphans.py, measure_live_offsets.py, sweep_round2.py and
report_round2.py do what they claim, with every outside dependency faked?

Same rules as scripts/check_round2.py, whose sandbox this reuses (importing it
sets DATA_DIR to a temp dir before nc_bench.config loads and asserts RUNS_DIR is
inside it): nothing under the real data/runs or data/incidents, no network, no
STT, no LiveKit, no model inference, short synthetic audio (<= 2 s).

    .venv/bin/python scripts/check_round2_report.py
"""

import asyncio
import contextlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_round2 as c2  # noqa: E402  (sandbox + shared helpers)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from nc_bench import config, scoring, store, stt, vad  # noqa: E402
from nc_bench.vads import bench, by_id  # noqa: E402

import measure_live_offsets as mlo  # noqa: E402
import report_round2 as rr  # noqa: E402
import restore_orphans as ro  # noqa: E402
import sweep_round2 as sw  # noqa: E402

assert config.RUNS_DIR.is_relative_to(c2._SANDBOX)
Env, quiet, raises, fresh_runs = c2.Env, c2.quiet, c2.raises, c2.fresh_runs

RATE = 16000


# ================================================================= synthetic audio

def bursty(dur: float, rate: int, seed: int) -> np.ndarray:
    """Speech-like and aperiodic: random 80-300 ms bursts, random 80-400 ms gaps."""
    rng = np.random.default_rng(seed)
    n = int(dur * rate)
    env = np.zeros(n)
    t = int(0.02 * rate)
    while t < n:
        length = int(rng.uniform(0.08, 0.3) * rate)
        amp = rng.uniform(0.3, 0.9)
        seg = np.sqrt(np.hanning(length)) * amp
        m = min(length, n - t)
        env[t: t + m] = seg[:m]
        t += length + int(rng.uniform(0.08, 0.4) * rate)
    x = env * rng.standard_normal(n) + 0.003 * rng.standard_normal(n)
    return (0.7 * x / np.abs(x).max()).astype(np.float32)


X = bursty(2.0, RATE, 11)


def delayed(x: np.ndarray, ms: float) -> np.ndarray:
    """The same audio `ms` late, same length."""
    n = int(ms / 1000 * RATE)
    return np.concatenate([np.zeros(n, np.float32), x])[: len(x)]


def head_cut(x: np.ndarray, ms: float) -> np.ndarray:
    """A stream that attached `ms` late: starts that far into the input, shorter."""
    return x[int(ms / 1000 * RATE):]


def write(path: Path, x: np.ndarray, rate: int = RATE) -> None:
    sf.write(path, x.astype(np.float32), rate, subtype="PCM_16")


def wav_of(tmp: Path, name: str, x: np.ndarray, rate: int = RATE, old: bool = True) -> Path:
    p = tmp / name
    write(p, x, rate)
    if old:
        os.utime(p, (OLD_TS, OLD_TS))
    return p


OLD_TS = time.mktime(time.strptime("2026-08-01", "%Y-%m-%d"))
BEFORE = "2026-08-10"


# =================================================================== restore_orphans: 1

def check_measure(env):
    tmp = Path(__import__("tempfile").mkdtemp(dir=c2._SANDBOX))
    inp = wav_of(tmp, "input.wav", X)
    for rate in (16000, 48000):
        def out_of(x, name, rate=rate):
            import soxr
            y = x if rate == RATE else soxr.resample(x, RATE, rate).astype(np.float32)
            return wav_of(tmp, f"{name}{rate}.wav", y, rate)

        g = ro.measure(inp, out_of(delayed(X, 70), "late"))
        assert abs(g["lag_ms"] - 70) <= 10 and abs(g["len_diff_ms"]) <= 10, (rate, g)
        assert g["prominence"] >= 0.15 and g["mtime"].startswith("2026-08-01"), g

        g = ro.measure(inp, out_of(head_cut(X, 150), "cut"))
        assert abs(g["lag_ms"] + 150) <= 10 and abs(g["len_diff_ms"] + 150) <= 10, (rate, g)
        assert g["prominence"] >= 0.15, g

        # aligned: on input time
        g = ro.measure(inp, out_of(X, "same"))
        assert g["lag_ms"] == 0 and abs(g["len_diff_ms"]) <= 1 and g["env_corr"] > 0.99, g

    # heavily attenuated + noisy copy: prominence is at the peak, so it holds
    rng = np.random.default_rng(5)
    quiet_copy = 0.03 * X + 0.0004 * rng.standard_normal(len(X)).astype(np.float32)
    g = ro.measure(inp, wav_of(tmp, "quiet.wav", quiet_copy))
    assert g["lag_ms"] == 0 and g["prominence"] >= 0.15, g
    noisy = X + 0.05 * rng.standard_normal(len(X)).astype(np.float32)
    g = ro.measure(inp, wav_of(tmp, "noisy.wav", noisy))
    assert g["prominence"] >= 0.15, g

    # unrelated noise: no clear peak
    for seed in (1, 2, 3):
        g = ro.measure(inp, wav_of(tmp, f"n{seed}.wav", 0.1 * np.random.default_rng(seed).standard_normal(len(X))))
        assert g["prominence"] < 0.15, (seed, g)
    # a different speech-like signal is not a match either
    g = ro.measure(inp, wav_of(tmp, "other.wav", bursty(2.0, RATE, 99)))
    assert g["prominence"] < 0.15, g


# =================================================================== restore_orphans: 2

def gm(lag=0.0, len_diff=0.0, prom=0.5, mtime="2026-08-01T00:00:00"):
    return {"lag_ms": lag, "len_diff_ms": len_diff, "prominence": prom, "mtime": mtime}


def check_gate():
    g = ro.gate(gm(20, 10), None, BEFORE)
    assert g == {"ok": True, "rule": "input-time", "reason": None}, g
    ref = {"median_lag_ms": -180.0, "median_len_diff_ms": -210.0}
    g = ro.gate(gm(-170, -200), ref, BEFORE)
    assert g["ok"] and g["rule"] == "candidate-relative" and g["reason"] is None, g
    # input time wins when both hold
    assert ro.gate(gm(0, 0), {"median_lag_ms": 0.0, "median_len_diff_ms": 0.0}, BEFORE)["rule"] == "input-time"
    # both fail
    g = ro.gate(gm(-170, -200), {"median_lag_ms": -400.0, "median_len_diff_ms": -400.0}, BEFORE)
    assert not g["ok"] and g["rule"] is None and "neither" in g["reason"] and "-400" in g["reason"], g
    # lag fits, length does not (and vice versa)
    assert not ro.gate(gm(-170, -300), ref, BEFORE)["ok"]
    assert not ro.gate(gm(-260, -200), ref, BEFORE)["ok"]
    assert not ro.gate(gm(0, 80), None, BEFORE)["ok"] and not ro.gate(gm(80, 0), None, BEFORE)["ok"]
    # no reference and an off-time lag
    g = ro.gate(gm(-170, -200), None, BEFORE)
    assert not g["ok"] and g["rule"] is None and "no other run has it" in g["reason"], g
    # low prominence
    g = ro.gate(gm(0, 0, prom=0.14), None, BEFORE)
    assert not g["ok"] and "prominence" in g["reason"] and g["rule"] == "input-time", g
    assert ro.gate(gm(0, 0, prom=0.15), None, BEFORE)["ok"]
    # written on/after the reprocess date
    for mt in ("2026-08-10T00:00:00", "2026-09-01T12:00:00"):
        g = ro.gate(gm(0, 0, mtime=mt), None, BEFORE)
        assert not g["ok"] and "not before" in g["reason"], g
    assert ro.gate(gm(0, 0, mtime="2026-08-09T23:59:59"), None, BEFORE)["ok"]
    # every reason is reported, not just the first
    g = ro.gate(gm(-170, -200, prom=0.01, mtime="2026-09-01T00:00:00"), None, BEFORE)
    assert g["reason"].count(";") == 2, g


# =================================================================== restore_orphans: 3

def add_run(rid: str, source: str, cands: list[dict], files: dict[str, np.ndarray], *,
            note: str = "", script: str = "", old: bool = True, extra: dict | None = None) -> Path:
    d = config.RUNS_DIR / rid
    d.mkdir(parents=True)
    wav_of(d, "input.wav", X, old=old)
    for name, x in files.items():
        wav_of(d, name, x, old=old)
    meta = {"id": rid, "created_at": "2026-08-04 13:00:00", "source": source, "status": "done",
            "note": note, "script": script,
            "input": {"file": "input.wav", "duration_s": 2.0, "sample_rate": RATE},
            "candidates": cands, **(extra or {})}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return d


def rec(cid: str, **kw) -> dict:
    return {"id": cid, "status": "done", "output": f"{cid}.wav", **kw}


def check_reference():
    fresh_runs()
    add_run("r-target", "phone", [], {"X.wav": X})
    add_run("r-b", "phone", [rec("X")], {"X.wav": delayed(X, 70)})
    add_run("r-c-web", "web", [rec("X")], {"X.wav": head_cut(X, 200)})
    add_run("_scratch-1", "phone", [rec("X")], {"X.wav": head_cut(X, 400)})
    add_run("r-e-restored", "phone", [rec("X", restored_orphan=True)], {"X.wav": head_cut(X, 300)})
    add_run("r-f-nofile", "phone", [rec("X")], {})
    add_run("r-g-nooutput", "phone", [{"id": "X", "status": "error"}], {})
    (config.RUNS_DIR / "r-h-nometa").mkdir()
    r = ro.reference("X", "r-target", "phone")
    assert r is not None and r["runs"] == 1, r          # only r-b
    assert abs(r["median_lag_ms"] - 70) <= 10 and abs(r["median_len_diff_ms"]) <= 10, r
    # the run itself is never its own reference
    r2 = ro.reference("X", "r-b", "phone")
    assert r2 is None, "r-target holds no X, so with r-b excluded nothing is left on phone"
    w = ro.reference("X", "r-target", "web")
    assert w["runs"] == 1 and abs(w["median_lag_ms"] + 200) <= 10, w
    assert ro.reference("X", "r-target", None) is None       # no unlabelled-surface runs
    assert ro.reference("nope", "r-target", "phone") is None
    # two usable runs: the median of both
    add_run("r-b2", "phone", [rec("X")], {"X.wav": delayed(X, 90)})
    r = ro.reference("X", "r-target", "phone")
    assert r["runs"] == 2 and abs(r["median_lag_ms"] - 80) <= 10, r
    # an upload is placed by its note
    add_run("r-up", "upload", [rec("X")], {"X.wav": delayed(X, 50)}, note="lab clip (phone)")
    assert ro.reference("X", "r-target", "phone")["runs"] == 3
    assert ro._surface({"source": "upload", "note": "x (WEB)"}) == "web"
    assert ro._surface({"source": "upload", "note": "plain"}) is None


# =================================================================== restore_orphans: 4

def fake_scoring(env):
    calls = {"stt": [], "score": [], "vad": []}

    async def fake_stt(wav):
        calls["stt"].append(Path(wav).name)
        return {"text": "hello world", "language": "en"}

    def fake_score(wav, input_scores, script, text):
        calls["score"].append(Path(wav).name)
        return {"wer": scoring.wer(script, text) if script else None, "fake": True}

    async def fake_vad(wav, p=None, source=None):
        calls["vad"].append(Path(wav).name)
        return {"segments": [[0.1, 0.5]], "n": 1}

    env.attr(stt, "transcribe", fake_stt)
    env.attr(scoring, "score_output", fake_score)
    env.attr(vad, "analyze", fake_vad)
    env.attr(config, "WHISPER_BOOST_VOLUME", False)
    return calls


def run_main(env, mod, *argv):
    env.attr(sys, "argv", [mod.__file__, *argv])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mod.main()
    return out.getvalue()


def check_restore_orphans_main(env):
    fresh_runs()
    calls = fake_scoring(env)
    noise = 0.1 * np.random.default_rng(4).standard_normal(len(X)).astype(np.float32)
    # a second run holds aic-quail-l's usual offset: 180 ms into the input
    add_run("r-ref", "phone", [rec("aic-quail-l")], {"aic-quail-l.wav": head_cut(X, 180)})
    existing = [rec("none", scores={"wer": None}), rec("krisp-viva", rail="cloud_replay")]
    d = add_run("r-target", "phone", existing,
                {"none.wav": X, "krisp-viva.wav": X,
                 "krisp-bvc.wav": X,                              # good, on input time
                 "aic-quail-l.wav": head_cut(X, 170),             # off-time, matches its own offset
                 "krisp-nc.wav": noise,                           # unrelated
                 "stray.wav": X},                                 # not a candidate id
                script="hello world")
    for n in ("krisp-bvc-tel",):
        wav_of(d, f"{n}.wav", X, old=False)                       # good audio, written too late
    before_meta = json.loads((d / "meta.json").read_text())
    files_before = {p.name: p.read_bytes() for p in d.glob("*.wav")}

    # dry run: reports, changes nothing
    quiet(run_main, env, ro, "--run", "r-target", "--before", BEFORE, "--dry-run")
    assert json.loads((d / "meta.json").read_text()) == before_meta and calls["stt"] == []

    out = run_main(env, ro, "--run", "r-target", "--before", BEFORE)
    m = store.load_meta("r-target")
    got = {c["id"]: c for c in m["candidates"]}
    # existing records byte-identical
    for c in before_meta["candidates"]:
        assert json.dumps(got[c["id"]], sort_keys=True) == json.dumps(c, sort_keys=True), c["id"]
    assert [c["id"] for c in m["candidates"]][:2] == ["none", "krisp-viva"]
    assert set(got) == {"none", "krisp-viva", "krisp-bvc", "aic-quail-l"}, sorted(got)
    good = got["krisp-bvc"]
    assert good["restored_orphan"] is True and good["status"] == "done" and good["output"] == "krisp-bvc.wav"
    assert good["rail"] == "live" and good["label"] and good["restored_at"]
    for k in ("chain", "chain_spec", "latency", "cost", "proc_ms", "rtf", "processed_at"):
        assert k in good and good[k] is None, k
    chk = good["restored_check"]
    assert set(chk) == {"lag_ms", "len_diff_ms", "ref_median_lag_ms", "ref_median_len_diff_ms",
                        "ref_runs", "prominence", "env_corr", "rule"}, chk
    assert chk["rule"] == "input-time" and chk["lag_ms"] == 0 and chk["ref_median_lag_ms"] is None
    assert good["duration_s"] == 2.0 and good["stt"]["text"] == "hello world" and good["scores"]["fake"]
    assert good["stt_setting"]["boost_volume"] is False and good["vad"]["segments"] == [[0.1, 0.5]]
    ref = got["aic-quail-l"]
    assert ref["rail"] == "live" and ref["restored_check"]["rule"] == "candidate-relative"
    assert abs(ref["restored_check"]["lag_ms"] + 170) <= 10 and abs(ref["restored_check"]["ref_median_lag_ms"] + 180) <= 10
    assert ref["restored_check"]["ref_runs"] == 1
    log = m["backfill_round2"]["restored_orphans"]
    assert len(log["passes"]) == 1 and log["restored"] == ["aic-quail-l", "krisp-bvc"], log
    bf = log["passes"][0]
    assert bf["restored"] == ["aic-quail-l", "krisp-bvc"], bf["restored"]
    kept = {k["id"]: k["reason"] for k in bf["kept_orphaned"]}
    assert {k["id"] for k in log["kept_orphaned"]} == set(kept)
    assert set(kept) == {"krisp-nc", "krisp-bvc-tel"}, kept
    assert "prominence" in kept["krisp-nc"] and "not before" in kept["krisp-bvc-tel"], kept
    assert bf["not_candidates"] == ["stray.wav"] and bf["before"] == BEFORE
    assert set(bf["checks"]) == {"aic-quail-l", "krisp-bvc", "krisp-bvc-tel", "krisp-nc"}
    assert "krisp-nc" not in got and "krisp-bvc-tel" not in got and "stray.wav" not in json.dumps(m["candidates"])
    # nothing on disk was touched
    assert {p.name: p.read_bytes() for p in d.glob("*.wav")} == files_before
    assert sorted(calls["stt"]) == ["aic-quail-l.wav", "krisp-bvc.wav"], calls
    # scored with a fake wer against the script
    assert good["scores"]["wer"]["wer"] == 0.0

    # a second run restores nothing new and leaves the restored records alone
    recs_before = json.dumps(m["candidates"], sort_keys=True)
    calls["stt"].clear()
    run_main(env, ro, "--run", "r-target", "--before", BEFORE)
    m2 = store.load_meta("r-target")
    assert json.dumps(m2["candidates"], sort_keys=True) == recs_before
    assert calls["stt"] == []
    log2 = m2["backfill_round2"]["restored_orphans"]
    # a second run appends a pass that restored nothing; the audit trail of the first stays
    assert len(log2["passes"]) == 2 and log2["passes"][0] == bf and log2["passes"][1]["restored"] == [], log2["passes"]
    assert log2["restored"] == ["aic-quail-l", "krisp-bvc"], log2["restored"]
    assert {k["id"] for k in log2["kept_orphaned"]} == set(kept)
    assert {k["id"] for k in log2["passes"][1]["kept_orphaned"]} == set(kept)
    # a dry run appends nothing
    quiet(run_main, env, ro, "--run", "r-target", "--before", BEFORE, "--dry-run")
    assert len(store.load_meta("r-target")["backfill_round2"]["restored_orphans"]["passes"]) == 2

    # refusals
    env.attr(config, "WHISPER_BOOST_VOLUME", True)
    raises(SystemExit, run_main, env, ro, "--run", "r-target", "--before", BEFORE, contains="WHISPER_BOOST_VOLUME")
    env.attr(config, "WHISPER_BOOST_VOLUME", False)
    raises(SystemExit, run_main, env, ro, "--run", "r-target", "--before", "2026-07-01", contains="input.wav itself")


# ======================================================================== measure_live_offsets

def check_measure_live_offsets(env):
    fresh_runs()
    recs = [rec("none"), rec("krisp-bvc-tel"), rec("aic-quail-l", rail="live"),
            rec("krisp-viva", rail="cloud_replay"), {"id": "krisp-bvc", "status": "error"},
            rec("krisp-nc")]                         # live id, but its wav is missing
    files = {"none.wav": X, "krisp-bvc-tel.wav": head_cut(X, 150), "aic-quail-l.wav": head_cut(X, 100),
             "krisp-viva.wav": X}
    d1 = add_run("r1", "phone", recs, files)
    # a live id whose record is a cloud replay: never stamped
    d2 = add_run("r2", "phone", [rec("krisp-nc", rail="cloud_replay")], {"krisp-nc.wav": delayed(X, 30)})
    d3 = add_run("_scratch", "phone", [rec("krisp-nc")], {"krisp-nc.wav": X})
    d4 = add_run("r4", "phone", [rec("krisp-nc")], {"krisp-nc.wav": delayed(X, 30)},
                 extra={"weird": {"keep": [1, 2, 3]}})
    snap = {d.name: json.loads((d / "meta.json").read_text()) for d in (d1, d2, d3, d4)}

    quiet(run_main, env, mlo, "--dry-run")
    assert {d.name: json.loads((d / "meta.json").read_text()) for d in (d1, d2, d3, d4)} == snap

    out = run_main(env, mlo)
    assert "r1 krisp-bvc-tel" in out and "r2 krisp-nc" not in out and "_scratch" not in out
    now = {d.name: store.load_meta(d.name) for d in (d1, d2, d3, d4)}
    got = {c["id"]: c for c in now["r1"]["candidates"]}
    o = got["krisp-bvc-tel"]["live_rail_offset"]
    assert abs(o["lag_ms"] + 150) <= 10 and set(o) == {"lag_ms", "len_diff_ms", "env_corr", "prominence", "measured_at"}, o
    assert abs(got["aic-quail-l"]["live_rail_offset"]["lag_ms"] + 100) <= 10
    for cid in ("none", "krisp-viva", "krisp-bvc", "krisp-nc"):
        assert "live_rail_offset" not in got[cid], cid
    assert "live_rail_offset" not in now["r2"]["candidates"][0], "cloud_replay rail is not live"
    assert "live_rail_offset" not in now["_scratch"]["candidates"][0]
    assert abs(now["r4"]["candidates"][0]["live_rail_offset"]["lag_ms"] - 30) <= 10
    # every other key untouched
    for rid in ("r1", "r2", "_scratch", "r4"):
        a, b = json.loads(json.dumps(snap[rid])), json.loads(json.dumps(now[rid]))
        for c in b["candidates"]:
            c.pop("live_rail_offset", None)
        assert a == b, rid


# =============================================================================== sweep_round2

def probs_of(*runs, n=None):
    """probs_of((0.9, 20), (0, 10), ...) -> float32 array."""
    return np.concatenate([np.full(k, v, np.float32) for v, k in runs])


HOP = 0.032
LK = by_id()["lk-inference"]
WEBRTC = by_id()["webrtc-3"]
LKAIC = by_id()["lkaic-flag"]


def cell_meta(truth, kns=None, dur=4.0, **kw):
    return {"truth_spans": truth, "known_nonspeech": kns or [], "ignore_spans": [],
            "input": {"duration_s": dur}, **kw}


def check_cell():
    # frames 10-29 speech, 60-79 speech, 100-109 a click
    p = probs_of((0, 10), (0.9, 20), (0, 30), (0.9, 20), (0, 20), (0.9, 10), (0, 15))
    rec = {"probs": p, "hop_s": HOP, "t0_s": 0.0}
    meta = cell_meta([[0.3, 1.0], [1.9, 2.6]], [{"name": "click", "start": 3.2, "end": 3.55}])
    c = sw.cell(meta, rec, LK, 0.5, 0.05, 0.4, 4.0, True)
    segs = c["segments"]
    want = [[0.32, 0.96], [1.92, 2.56], [3.2, 3.52]]
    assert len(segs) == 3 and all(abs(a - b) < 0.005 for s, w in zip(segs, want) for a, b in zip(s, w)), segs
    assert (c["turns"], c["hit"], c["missed"], c["phantom"], c["merged"]) == (2, 2, 0, 1, 0), c
    assert c["n_segments"] == 3 and c["nonspeech_hits"] == {"click": True} and c["ignored"] == 0
    # hand check: 4.0 s file, marked speech 1.4 s, segments cover 1.28 s of it -> 0.13 s missed
    # (the marks stick out 0.02 s left of each span... 0.12 + 0.01 rounding), the click is 0.32 s of false alarm
    assert c["rank_excluded"] is None and c["miss_s"] == 0.13 and c["fa_s"] == 0.32 and c["agree"] == 0.8875, c
    assert c["overreach_s"] == 0.0
    # no segments stored unless asked
    assert "segments" not in sw.cell(meta, rec, LK, 0.5, 0.05, 0.4, 4.0, False)
    # activation too high: nothing opens, both turns missed, click closed
    c = sw.cell(meta, rec, LK, 0.95, 0.05, 0.4, 4.0, True)
    assert c["segments"] == [] and (c["hit"], c["missed"], c["phantom"]) == (0, 2, 0)
    assert c["nonspeech_hits"] == {"click": False}
    # min_speech longer than the click drops it, but not the 0.64 s turns
    c = sw.cell(meta, rec, LK, 0.5, 0.35, 0.4, 4.0, True)
    assert c["n_segments"] == 2 and c["nonspeech_hits"] == {"click": False} and c["phantom"] == 0, c
    # min_silence changes the truth coalescing: a 0.9 s gap between marks merges at 1.0
    meta2 = cell_meta([[0.3, 1.0], [1.9, 2.6]])
    assert sw.cell(meta2, rec, LK, 0.5, 0.05, 1.0, 4.0, False)["turns"] == 1
    assert sw.cell(meta2, rec, LK, 0.5, 0.05, 0.4, 4.0, False)["turns"] == 2

    # deactivation == activation: a 0.55 plateau after speech holds the turn open at
    # 0.5/0.5 and closes it at 0.6/0.6 (a fixed 0.5 deactivation would keep holding it)
    q = probs_of((0, 5), (0.9, 10), (0.55, 20), (0, 30))
    r2 = {"probs": q, "hop_s": HOP, "t0_s": 0.0}
    hold = sw.cell(cell_meta([[0.16, 1.0]]), r2, LK, 0.5, 0.05, 0.4, 3.0, True)["segments"]
    shut = sw.cell(cell_meta([[0.16, 1.0]]), r2, LK, 0.6, 0.05, 0.4, 3.0, True)["segments"]
    assert hold[0][1] > 0.9 and shut[0][1] < 0.6, (hold, shut)
    # a binary VAD ignores the activation
    r3 = {"probs": probs_of((0, 5), (0.7, 20), (0, 30)), "hop_s": HOP, "t0_s": 0.0}
    assert sw.cell(cell_meta([]), r3, LK, 0.8, 0.05, 0.4, 3.0, True)["segments"] == []
    b = sw.cell(cell_meta([]), r3, WEBRTC, 0.8, 0.05, 0.4, 3.0, True)
    assert len(b["segments"]) == 1, b
    # non-speech hits come from the segments even with no marked speech (PR-54 B)
    kns = [{"name": "throat", "start": 0.2, "end": 0.6}, {"name": "far", "start": 2.0, "end": 2.5}]
    e = sw.cell(cell_meta([], kns), r3, LK, 0.5, 0.05, 0.4, 3.0, True)
    assert e["nonspeech_hits"] == {"throat": True, "far": False} and e["turns"] is None and e["hit"] is None
    assert e["n_segments"] == 1 and e["agree"] is None
    # t0_s shifts the window grid
    shifted = sw.cell(cell_meta([]), {**r3, "t0_s": -0.1}, LK, 0.5, 0.05, 0.4, 3.0, True)["segments"][0]
    plain = sw.cell(cell_meta([]), r3, LK, 0.5, 0.05, 0.4, 3.0, True)["segments"][0]
    assert abs(plain[0] - shifted[0] - 0.1) < 0.005, (plain, shifted)


def sweep_run(with_incident=False, script="") -> tuple[str, Path]:
    """A run with 2 s wavs and stored probabilities for gates `none` and `hush`, a
    wav with no probabilities (`nofp`), and truth marks."""
    rid, d = store.new_run("phone", "sweep")
    files = {}
    for g in ("none", "hush", "nofp"):
        c2.short_wav(d / f"{g}.wav", 2.0, RATE)
        files[g] = f"{g}.wav"
    c2.short_wav(d / "input.wav", 2.0, RATE)
    with store.locked_meta(rid) as m:
        m["input"] = {"file": "input.wav", "duration_s": 2.0, "sample_rate": RATE}
        m["truth_spans"] = [[0.3, 1.0]]
        m["known_nonspeech"] = [{"name": "click", "start": 1.5, "end": 1.8}]
        m["script"] = script
        m["candidates"] = [{"id": g, "status": "done", "output": f} for g, f in files.items()]
        m["vad_bench"] = {"params": {"activation_threshold": 0.5}, "results": {"none": {"lk-inference": {"marker": 1}}}}
        if with_incident:
            m["incident"] = {"id": "PR-T"}
    p = probs_of((0, 10), (0.9, 22), (0, 14), (0.9, 9), (0, 7))     # 62 frames
    for g in ("none", "hush"):
        bench.save_probs(d, g, LK, {"probs": p, "hop_s": HOP, "t0_s": 0.0, "cpu_ms_per_s": 1.0},
                         d / f"{g}.wav", 2.0)
    return rid, d


def sweep_out(fn, *a):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*a)
    return [json.loads(ln) for ln in out.getvalue().splitlines() if ln.startswith("{")]


def check_sweep(env):
    fresh_runs()
    rid, d = sweep_run()
    vb_before = json.dumps(store.load_meta(rid)["vad_bench"], sort_keys=True)
    log = sweep_out(sw.sweep, [rid], ["none", "hush", "nofp", "ghost"], ["lk-inference"], False, False)
    assert log[0]["new_cells"] == 144, log
    assert sorted(log[0]["missing"]) == ["ghost: no audio", "nofp x lk-inference: no fresh probabilities"], log
    m = store.load_meta(rid)
    s = m[sw.KEY]
    assert len(s["cells"]) == 144 and s["at"] and s["grid"]["continuity"] == [0.5, 0.05, 0.4]
    assert sorted(s["missing"]) == sorted(log[0]["missing"])
    k = json.dumps(["none", "lk-inference", 0.5, 0.05, 0.4])
    assert k in s["cells"] and sw.ckey("none", "lk-inference", 0.5, 0.05, 0.4) == k
    assert json.dumps(["hush", "lk-inference", 0.8, 0.25, 0.55]) in s["cells"]
    cont = s["cells"][k]
    assert cont["turns"] == 1 and cont["hit"] == 1 and cont["missed"] == 0 and cont["n_segments"] == 2
    assert "segments" not in cont and cont["nonspeech_hits"] == {"click": True}
    # the 2nd burst (1.47-1.76 s) sits on the click: a phantom; production's numbers are untouched
    assert cont["phantom"] == 1
    assert json.dumps(m["vad_bench"], sort_keys=True) == vb_before

    # resumable: nothing new the second time, and the cells are not rewritten
    at = s["at"]
    log = sweep_out(sw.sweep, [rid], ["none", "hush"], ["lk-inference"], False, False)
    assert log[0]["new_cells"] == 0
    assert store.load_meta(rid)[sw.KEY]["at"] == at
    # cells kept unless --force: change the probabilities, no-force keeps, force recomputes
    p2 = probs_of((0, 10), (0.9, 40), (0, 12))
    bench.save_probs(d, "none", LK, {"probs": p2, "hop_s": HOP, "t0_s": 0.0, "cpu_ms_per_s": 1.0},
                     d / "none.wav", 2.0)
    sweep_out(sw.sweep, [rid], ["none"], ["lk-inference"], False, False)
    assert store.load_meta(rid)[sw.KEY]["cells"][k] == cont
    log = sweep_out(sw.sweep, [rid], ["none"], ["lk-inference"], True, False)
    assert log[0]["new_cells"] == 72
    assert store.load_meta(rid)[sw.KEY]["cells"][k]["n_segments"] == 1
    assert len(store.load_meta(rid)[sw.KEY]["cells"]) == 144       # hush cells still there
    # dry run stores nothing
    rid2, _ = sweep_run()
    log = sweep_out(sw.sweep, [rid2], ["none"], ["lk-inference"], False, True)
    assert log[0]["new_cells"] == 72 and sw.KEY not in store.load_meta(rid2)
    # incident runs keep the segments; a stale probability file is not used
    rid3, d3 = sweep_run(with_incident=True)
    sweep_out(sw.sweep, [rid3], ["none"], ["lk-inference"], False, False)
    assert "segments" in store.load_meta(rid3)[sw.KEY]["cells"][k]
    write(d3 / "none.wav", X)                         # audio rewritten: the stored probs are stale
    os.utime(d3 / "none.wav", (OLD_TS, OLD_TS))
    rid4 = rid3
    log = sweep_out(sw.sweep, [rid4], ["none"], ["lk-inference"], True, False)
    assert log[0]["new_cells"] == 0 and log[0]["missing"] == ["none x lk-inference: no fresh probabilities"], log
    # unknown VAD ids fail loudly
    raises(KeyError, sw.sweep, [rid], ["none"], ["no-such-vad"], False, False)


def lkaic_run(valid_frac: float) -> tuple[str, Path]:
    rid, d = store.new_run("phone", "lkaic")
    c2.short_wav(d / "input.wav", 2.0, RATE)
    c2.short_wav(d / "lkaic-x.wav", 2.0, RATE)
    flags = np.zeros(40, np.int8)
    flags[6:20] = 1
    flags[int(40 * valid_frac):] = -1
    np.save(d / "lkaic-x.aicvad.npy", flags)
    with store.locked_meta(rid) as m:
        m["input"] = {"file": "input.wav", "duration_s": 2.0, "sample_rate": RATE}
        m["truth_spans"] = [[0.3, 1.0]]
        m["candidates"] = [{"id": "lkaic-x", "status": "done", "output": "lkaic-x.wav",
                            "aicvad": {"file": "lkaic-x.aicvad.npy", "frame_s": 0.05, "t0_s": 0.0}}]
    return rid, d


def check_sweep_lkaic_coverage(env):
    fresh_runs()
    bad, _ = lkaic_run(0.9)
    log = sweep_out(sw.sweep, [bad], ["lkaic-x"], ["lkaic-flag"], False, False)
    assert log[0]["new_cells"] == 0 and log[0]["missing"] == ["lkaic-x x lkaic-flag: no fresh probabilities"], log
    assert sw.KEY not in store.load_meta(bad)
    good, d = lkaic_run(1.0)
    log = sweep_out(sw.sweep, [good], ["lkaic-x"], ["lkaic-flag"], False, False)
    assert log[0]["new_cells"] == 72 and log[0]["missing"] == [], log
    cells = store.load_meta(good)[sw.KEY]["cells"]
    # a binary detector ignores activation: every activation gives the same segments
    keys = [k for k in cells if json.loads(k)[3:] == [0.05, 0.4]]
    assert len({json.dumps(cells[k]["n_segments"]) + json.dumps(cells[k]["hit"]) for k in keys}) == 1
    assert cells[sw.ckey("lkaic-x", "lkaic-flag", 0.5, 0.05, 0.4)]["hit"] == 1
    # exactly 99 %: kept; a hair under: skipped (integer counts, not the rounded ratio)
    m = {"candidates": [{"id": "lkaic-x", "aicvad": {"file": "lkaic-x.aicvad.npy", "frame_s": 0.05}}]}
    flags = np.ones(100, np.int8)
    flags[0] = -1
    np.save(d / "lkaic-x.aicvad.npy", flags)
    assert bench._coverage_ok(bench.lkaic_flags(m, d, "lkaic-x"))
    flags[1] = -1
    np.save(d / "lkaic-x.aicvad.npy", flags)
    assert not bench._coverage_ok(bench.lkaic_flags(m, d, "lkaic-x"))
    # the -1 frames hold the previous decision, they never become "no speech"
    r = bench.lkaic_flags({"candidates": [{"id": "s", "aicvad": {"file": "s.npy", "frame_s": 0.05}}]},
                          d, "s") if (np.save(d / "s.npy", np.array([1, 1, -1, -1, 0], np.int8)) or True) else None
    assert list(r["probs"]) == [1, 1, 1, 1, 0] and r["missing_spans"] == [[0.1, 0.2]], r


def edit_side(d: Path, sid: str, fn) -> None:
    js = d / f"{sid}.lk-inference.vadp.json"
    side = json.loads(js.read_text())
    fn(side)
    side["inference_key"] = bench.fp_key(side["inference_fp"])      # so load_probs itself misses
    js.write_text(json.dumps(side))


LINUX_SO = "_native.cpython-312-x86_64-linux-gnu.so"


def check_probs_other_platform(env):
    fresh_runs()
    rid, d = sweep_run()
    wav = d / "none.wav"
    stored = bench.load_probs(d, "none", LK, wav)
    assert stored is not None and "other_platform" not in stored
    js0 = (d / "none.lk-inference.vadp.json").read_text()

    def reset():
        (d / "none.lk-inference.vadp.json").write_text(js0)

    def linux(side):
        side["inference_fp"]["resources"] = {LINUX_SO: "a" * 64}

    def probe():
        return sw._probs({}, d, "none", LK, wav)

    # accepts: only `resources` differs, with different file names
    edit_side(d, "none", linux)
    assert bench.load_probs(d, "none", LK, wav) is None
    got = probe()
    assert got is not None and got["other_platform"] == [LINUX_SO], got
    assert got["probs"].shape == stored["probs"].shape and np.allclose(got["probs"], stored["probs"])
    assert got["hop_s"] == stored["hop_s"]
    # and a sweep runs on it
    log = sweep_out(sw.sweep, [rid], ["none"], ["lk-inference"], False, False)
    assert log[0]["new_cells"] == 72 and log[0]["missing"] == [], log

    def rejected(edit, why):
        reset()
        edit_side(d, "none", lambda s: (linux(s), edit(s)))
        assert probe() is None, why

    rejected(lambda s: s["inference_fp"]["packages"].update(soxr="0.0.0"), "package version")
    rejected(lambda s: s["inference_fp"]["files"].update({"m.onnx": "b" * 64}), "model file")
    rejected(lambda s: s["inference_fp"]["config"].update(hop_s=0.5), "config")
    rejected(lambda s: s["inference_fp"].update(schema=-1), "schema")
    rejected(lambda s: s.update(payload_sha256="0" * 64), "payload hash")
    # same resource names as this platform's, different hashes: a real change here
    reset()
    now = bench.inference_fp(LK)["resources"]
    edit_side(d, "none", lambda s: s["inference_fp"].update(resources={k: "c" * 64 for k in now}))
    assert probe() is None
    # the source changed
    reset()
    edit_side(d, "none", linux)
    assert probe() is not None
    write(wav, X)
    assert probe() is None
    # no cache pair at all
    assert sw._probs_other_platform(d, "nope", LK, wav) is None


def check_run_wer(env):
    fresh_runs()
    calls: list[int] = []            # samples per /recognize request, in call order
    fail_next = [False]

    async def fake_recognize(mono, rate):
        i = len(calls)
        calls.append(len(mono))
        if fail_next[0]:
            fail_next[0] = False
            raise RuntimeError("stt exploded")
        return {"text": "hello" if i % 2 == 0 else "world", "language": "en", "confidence": 0.8}

    env.attr(stt, "_recognize", fake_recognize)
    rid, d = sweep_run(script="hello world")
    noscript, _ = sweep_run(script="")
    combo = ["hush:lk-inference:0.5:0.05:0.4"]
    key = sw.ckey("hush", "lk-inference", 0.5, 0.05, 0.4)

    def run(runs, combos, force=False, dry=False):
        return sweep_out(sw.run_wer, runs, combos, force, dry)

    # the cuts production would send: pad 0.5 s before, min_silence after, one per segment, not merged
    stored = bench.load_probs(d, "hush", LK, d / "hush.wav")
    segs = sw.cell(store.load_meta(rid), stored, LK, 0.5, 0.05, 0.4, 2.0, True)["segments"]
    cuts = [[max(0.0, s - sw.PAD), min(2.0, e + 0.4)] for s, e in segs]
    want = [int(e * RATE) - int(s * RATE) for s, e in cuts]
    assert len(cuts) == 2 and cuts[1][0] < cuts[0][1], cuts          # they overlap: the point of the test

    log = run([rid, noscript], combo)
    assert log == [{"run": rid, "combos": 1}], log            # the no-script run is skipped silently
    assert sw.WER_KEY not in store.load_meta(noscript)
    got = store.load_meta(rid)[sw.WER_KEY][key]
    assert got["same"]["read"] == "hush.wav" and got["raw"]["read"] == "none.wav", got
    assert got["same"]["wer"] == scoring.wer("hello world", "hello world") and got["segments_n"] == 2
    assert got["merged"] is False and got["pad_s"] == sw.PAD and got["trail_s"] == 0.4, got
    assert got["same"]["mode"] == "per-segment" and got["same"]["turns"] == 2
    assert got["same"]["text"] == "hello world" and got["same"]["confidence"] == 0.8
    assert got["cell"]["turns"] == 1 and "segments" not in got["cell"]
    assert calls == want * 2, (calls, want)        # two requests per mode, the overlapping slices as cut
    # resumable
    calls.clear()
    assert run([rid], combo) == [{"run": rid, "combos": 0}] and calls == []
    run([rid], combo, force=True)
    assert len(calls) == 4
    # dry run
    calls.clear()
    assert run([rid], combo, force=True, dry=True) == [{"run": rid, "combos": 1}] and calls == []
    # an exception is recorded, not raised, and the other mode still runs
    fail_next[0] = True
    combo2 = ["hush:lk-inference:0.6:0.05:0.4"]
    run([rid], combo2)
    g2 = store.load_meta(rid)[sw.WER_KEY][sw.ckey("hush", "lk-inference", 0.6, 0.05, 0.4)]
    assert g2["same"]["error"] == "RuntimeError: stt exploded" and g2["same"]["read"] == "hush.wav", g2
    assert g2["raw"]["wer"] is not None and "error" not in g2["raw"]
    # a gate the run has no audio for is not a combo; one with no probabilities is an error entry
    assert run([rid], ["ghost:lk-inference:0.5:0.05:0.4"]) == [{"run": rid, "combos": 0}]
    run([rid], ["nofp:lk-inference:0.5:0.05:0.4"])
    assert store.load_meta(rid)[sw.WER_KEY][sw.ckey("nofp", "lk-inference", 0.5, 0.05, 0.4)] == {"error": "no fresh probabilities"}
    # the earlier combo is untouched by later ones
    assert store.load_meta(rid)[sw.WER_KEY][key]["same"]["read"] == "hush.wav"


def check_per_segment(env):
    """_per_segment: one recognize per cut, slices exactly as given, never merged."""
    tmp = Path(__import__("tempfile").mkdtemp(dir=c2._SANDBOX))
    ramp = np.arange(2 * RATE, dtype=np.int16)                       # every sample identifiable
    sf.write(tmp / "ramp.wav", ramp, RATE, subtype="PCM_16")
    seen: list[np.ndarray] = []
    replies = iter([("one", 1.0), ("", 0.9), ("three", 0.5)])

    async def fake_recognize(mono, rate):
        seen.append(mono.copy())
        text, conf = next(replies)
        return {"text": text, "language": "en" if text else None, "confidence": conf}

    env.attr(stt, "_recognize", fake_recognize)
    cuts = [[0.0, 1.0], [0.5, 1.5], [1.0, 1.01], [0.5, 2.0]]           # overlap, an empty reply, a 10 ms sliver
    got = asyncio.run(sw._per_segment(tmp / "ramp.wav", cuts))
    assert len(seen) == 3, "the 10 ms sliver is skipped, the two overlapping cuts are both sent"
    assert np.array_equal(seen[0], ramp[0:RATE]) and np.array_equal(seen[1], ramp[8000:24000])
    assert np.array_equal(seen[2], ramp[8000:32000])
    assert got["text"] == "one three" and got["language"] == "en" and got["mode"] == "per-segment"
    assert got["turns"] == 4, "turns counts the cuts, not the requests"
    # confidence weighted by cut length, and only over cuts that returned text: (1.0 * 1.0 + 1.5 * 0.5) / 2.5
    assert abs(got["confidence"] - 0.7) < 1e-9, got
    # a cut past the end of the file is just the tail; nothing to send -> empty text, no crash
    seen.clear()
    got = asyncio.run(sw._per_segment(tmp / "ramp.wav", [[5.0, 6.0]]))
    assert seen == [] and got["text"] == "" and got["confidence"] is None and got["turns"] == 1


# ================================================================================ report_round2

def cr(cid: str, wer: float | None = None, text: str = "", rail: str | None = None,
       status: str = "done", **kw) -> dict:
    r = {"id": cid, "status": status, "output": f"{cid}.wav", "stt": {"text": text}, **kw}
    if wer is not None:
        r["scores"] = {"wer": {"wer": wer}}
    if rail:
        r["rail"] = rail
    return r


def scored(segments, truth, kns=None, ign=None, dur=10.0, min_sil=0.4, **extra) -> dict:
    """A stored vad_bench row, scored by the bench's own score()."""
    params = {"activation_threshold": 0.5, "deactivation_threshold": 0.5, "min_speech_duration": 0.05,
              "min_silence_duration": min_sil}
    res = {"segments": segments, "params": params}
    return {**res, **bench.score(res, truth, dur, kns, ign), **extra}


def rmeta(source: str | None, cands: list[dict], *, vb: dict | None = None, truth=None, script="",
          dur: float = 10.0, note: str = "", **kw) -> dict:
    m = {"source": source, "note": note, "script": script, "candidates": cands,
         "truth_spans": truth or [], "input": {"duration_s": dur}, **kw}
    if vb is not None:
        m["vad_bench"] = {"params": {"min_silence_duration": 0.4}, "results": vb}
    return m


def md_header(text: str) -> list[str]:
    return [c.strip() for c in text.strip().splitlines()[0].strip().strip("|").split("|")]


def md_rows(text: str) -> list[list[str]]:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in text.splitlines() if ln.startswith("|")]
    return [r for r in rows if not set("".join(r)) <= set("-")][1:]      # header and rule out


def sections_of(text: str) -> dict[str, str]:
    out = {}
    for part in text.split("### ")[1:]:
        head, _, body = part.partition("\n")
        out[head] = body
    return out


def check_report_helpers():
    # collapse: longest run of one 1-3-gram repeated back to back
    assert rr.collapse("bank, bank, bank, bank, bank, bank") == 6
    assert rr.collapse("Bank bank BANK bank bank") == 5
    assert rr.collapse("I want to check my balance please, thank you very much") < 5
    assert rr.collapse("hello hello world") == 2
    assert rr.collapse("thank you sir " * 5) == 5
    assert rr.collapse("so " + "thank you sir " * 4 + "bye") == 4
    assert rr.collapse("a b a b a b a b a b") == 5
    assert rr.collapse("") == 0 and rr.collapse(None) == 0
    # the literal report label "bank, bank ×6" is two words and a number, not six repeats
    assert rr.collapse("bank, bank ×6") == 2

    # agg
    a = rr.agg([-0.1, -0.2, 0.1, 0.0])
    assert a["n"] == 4 and abs(a["mean"] + 0.05) < 1e-9 and a["worst"] == 0.1
    assert (a["W"], a["L"]) == (2, 1), a                      # the 0.0 tie counts in neither
    h = rr.agg([-0.1, -0.2, 0.1, 0.0], higher_better=True)
    assert h["worst"] == -0.2 and (h["W"], h["L"]) == (1, 2), h
    assert rr.agg([]) == {"n": 0}
    assert rr.agg([0.0, 0.0])["W"] == 0 and rr.agg([0.0, 0.0])["L"] == 0
    assert rr.agg([3.0])["median"] == 3.0

    # ranked: a candidate in fewer than half of the runs is left out; ties break on id
    per = {"a": [1.0, 2.0], "b": [0.5], "c": [3.0], "d": [0.5, 0.5]}
    assert [c for c, _ in rr.ranked(per, 4)] == ["d", "a"]           # >= 2 of 4
    assert [c for c, _ in rr.ranked(per, 3)] == ["d", "a"]           # 1 < 1.5
    assert [c for c, _ in rr.ranked(per, 2)] == ["b", "d", "a", "c"]  # b, d tie at 0.5 -> by id
    assert [c for c, _ in rr.ranked(per, 2, top=2)] == ["b", "d"]
    assert [c for c, _ in rr.ranked({"x": [0.1, 0.3], "y": [0.5, 0.5]}, 2, higher_better=True)] == ["y", "x"]
    assert [c for c, _ in rr.ranked({"x": [], "y": [1.0]}, 0)] == ["y"]
    assert len(rr.ranked({str(i): [i] for i in range(30)}, 1, top=None)) == 30

    # surface
    assert rr.surface({"source": "phone"}) == "phone" and rr.surface({"source": "web"}) == "web"
    assert rr.surface({"source": "upload", "note": "clip (phone)"}) == "phone"
    assert rr.surface({"source": "upload", "note": "Clip (WEB) 2"}) == "web"
    assert rr.surface({"source": "upload", "note": "clip"}) is None and rr.surface({}) is None
    assert rr.surface({"source": "phone", "incident": {"id": "PR-42"}}) is None
    assert rr.surface({"source": "upload", "note": "(phone)", "incident": {"id": "x"}}) is None

    # small ones
    assert rr.is_live_rail("krisp-nc", {}) and not rr.is_live_rail("krisp-nc", {"rail": "cloud_replay"})
    assert not rr.is_live_rail("aicsdk-x", {}) and not rr.is_live_rail("krisp-viva", {})
    assert rr.label("aic-vf-s") == "`aic-vf-s` (VF 2.1 / historical)"
    assert rr.label("x", {"restored_orphan": True}) == "`x` (restored)"
    assert rr.wer_of({"status": "error", "scores": {"wer": {"wer": 0.1}}}) is None
    assert rr.wer_of({"status": "done", "scores": {"wer": None}}) is None
    assert rr.wer_of({"status": "done", "scores": {"wer": {"wer": 0.25}}}) == 0.25
    assert rr.wer_of(None) is None
    assert rr.fmt(None) == "—" and rr.fmt(0.5, "+.1%") == "+50.0%"
    rows = md_rows(rr.md(["a", "b"], [[1, 2], [3, 4]]))
    assert rows == [["1", "2"], ["3", "4"]]


def check_report_vb_row():
    good = {"segments": [[1, 2]]}
    vb = lambda **kw: {"vad_bench": {"results": {"s": {"v": {**good, **kw}}}}}  # noqa: E731
    assert rr.vb_row(vb(), "s", "v") == good
    assert rr.vb_row(vb(stale=True), "s", "v") is None
    assert rr.vb_row(vb(error="boom"), "s", "v") is None
    assert rr.vb_row(vb(unavailable=True), "s", "v") is None
    assert rr.vb_row(vb(rank_excluded="coverage 90%"), "s", "v") is None
    assert rr.vb_row(vb(segments=None), "s", "v") is None
    assert rr.vb_row(vb(segments=[]), "s", "v") == {"segments": []}          # zero segments is a result
    assert rr.vb_row(vb(), "nope", "v") is None and rr.vb_row(vb(), "s", "nope") is None
    assert rr.vb_row({}, "s", "v") is None and rr.vb_row({"vad_bench": None}, "s", "v") is None


def stt_dataset() -> dict[str, dict]:
    bank = "bank " * 6
    return {
        "p1": rmeta("phone", [cr("none", .2, "a b c"), cr("X", .1, "a b c"), cr("Y", .3, "a b"),
                              cr("krisp-nc", 0.0, "a b c"), cr("krisp-bvc", 0.0, "a b c"),
                              cr("krisp-nc-replay", .15, "a b", rail="cloud_replay"),
                              cr("cloud-raw", .2, "a b c", rail="cloud_replay")], script="a b c"),
        "p2": rmeta("phone", [cr("none", .5, bank), cr("X", .1, "a b c")], script="a b c"),
        "p3": rmeta("phone", [cr("none", .4, "a b c"), cr("X", .3, "no " * 6), cr("Y", .2, "a b c"),
                              cr("krisp-bvc", .1, "a b c", rail="cloud_replay"),
                              cr("krisp-nc-replay", .1, "a", rail="cloud_replay"),
                              cr("cloud-raw", .3, "a b c", rail="cloud_replay")], script="a b c"),
        "p4-noscript": rmeta("phone", [cr("none", .4, "a")], script=""),
        "p5-nower": rmeta("phone", [cr("none", None, "a"), cr("X", .1, "a")], script="a b c"),
        "w1": rmeta("web", [cr("none", .1, "a b c"), cr("Z", .3, "a b c"), cr("W", .05, "a b c")], script="a b c"),
        "u1": rmeta("upload", [cr("none", .1, "a")], script="a", note="unlabelled"),
    }


def check_report_nc_stt():
    runs = stt_dataset()
    runs["p6-junk"] = rmeta("phone", [cr("none", 6.5, "a b c"), cr("X", .1, "a b c")], script="a b c")
    out = rr.sec_nc_stt(runs)
    secs = sections_of(out)
    phone = next(b for h, b in secs.items() if h.startswith("Phone — STT"))
    assert "(2 valid runs" in next(h for h in secs if h.startswith("Phone — STT")), list(secs)
    # p2's passthrough collapsed: excluded, with the reason; p4/p5 too
    excl = out[out.index("Excluded runs"):]
    assert "`p2`" in excl and "passthrough transcript collapsed (x6)" in excl
    assert "`p4-noscript`" in excl and "no reference script" in excl
    assert "`p5-nower`" in excl and "no passthrough WER" in excl
    assert "`p6-junk`" in excl and "passthrough WER 6.50 > 1.0 (junk transcript)" in excl
    edge = rr.sec_nc_stt({"e": rmeta("phone", [cr("none", 1.0, "a b c"), cr("X", .1, "a b c")], script="a b c")})
    assert "(1 valid runs" in edge and "junk" not in edge, "a passthrough WER of exactly 1.0 is not junk"
    assert "`u1`" in excl and "upload with no surface label" in excl and excl.count("`u1`") == 1
    # p3's collapsed X transcript is dropped (not just p2's), and named with its repeat count
    drop = out[out.index("Candidate transcripts dropped"):]
    assert "`p3`" in drop and "`X`" in drop and "x6" in drop and "`p2`" not in drop
    ranks = md_rows(phone.split("\n\n")[0])
    ids = [r[1] for r in ranks]
    assert "`none`" not in ids
    # live rail never ranked, but a replay-rail record under a live id is
    assert "`krisp-nc`" not in ids and "`krisp-bvc`" in ids and "`krisp-nc-replay`" in ids, ids
    # X: p1 only (p2 excluded, p3 collapsed) -> -0.1; Y: (+0.1, -0.2) -> mean -0.05
    row = {r[1]: r for r in ranks}
    assert row["`X`"][2] == "1" and row["`X`"][3] == "-0.100", row["`X`"]
    assert row["`Y`"][2] == "2" and row["`Y`"][3] == "-0.050" and row["`Y`"][6:8] == ["1", "1"], row["`Y`"]
    # sorted best (most negative mean) first
    means = [float(r[3]) for r in ranks]
    assert means == sorted(means), means
    # against cloud-raw: krisp-nc-replay (.15-.2, .1-.3) -> -0.125, and not cloud-raw itself
    raw = out[out.index("cloud replay rail against"):]
    raw = raw[: raw.index("### ")] if "### " in raw else raw
    assert "`krisp-nc-replay` | 2 | -0.125" in raw and "`cloud-raw`" not in raw.split("\n", 2)[2].split("Excluded")[0]
    # web: W beats Z
    web = md_rows(next(b for h, b in secs.items() if h.startswith("Web — STT")).split("\n\n")[0])
    assert [r[1] for r in web] == ["`W`", "`Z`"], web
    # passthrough mean shown for the valid runs: (.2 + .4) / 2
    assert "median 0.300, mean 0.300" in next(h for h in secs if h.startswith("Phone — STT"))
    assert [h for h in ("#", "candidate", "n", "median ΔWER", "mean", "worst", "W", "L")] == \
        md_header(phone.split("\n\n")[0])
    # ranked by median, not mean: X's deltas are -0.3, -0.3, +0.9 (mean +0.1), Y's are 0
    skew = {f"s{i}": rmeta("phone", [cr("none", .3, "a b c"), cr("X", x, "a b c"), cr("Y", .3, "a b c")],
                           script="a b c") for i, x in enumerate((0.0, 0.0, 1.2))}
    sk = sections_of(rr.sec_nc_stt(skew))
    hdr = next(h for h in sk if h.startswith("Phone"))
    rows = md_rows(sk[hdr].split("\n\n")[0])
    assert [r[1] for r in rows] == ["`X`", "`Y`"], rows
    assert rows[0][3] == "-0.300" and rows[0][4] == "+0.100", rows[0]
    assert "median 0.300, mean 0.300" in hdr
    # nothing valid: no crash
    assert "Phone — STT" in rr.sec_nc_stt({"x": rmeta("phone", [cr("none", .4, "a")], script="")})


def vad_dataset() -> dict[str, dict]:
    truth = [[1.0, 2.0], [4.0, 5.0]]
    perfect = [[1.0, 2.0], [4.0, 5.0]]

    def vbs(**segs):
        return {sid: {"lk-inference": scored(s, truth)} for sid, s in segs.items()}

    out = {}
    for rid in ("p1", "p2"):
        out[rid] = rmeta("phone", [cr("none"), cr("X"), cr("Y"), cr("krisp-nc"),
                                   cr("krisp-bvc", rail="cloud_replay")],
                         truth=truth,
                         vb=vbs(none=perfect, X=[[1.0, 2.0]], Y=perfect + [[7.0, 8.0]],
                                **{"krisp-nc": perfect, "krisp-bvc": perfect + [[7, 8]]}))
        out[rid]["vad_bench"]["results"]["input"] = {"lk-inference": scored(perfect, truth),
                                                       "webrtc-3": scored(perfect + [[7, 8], [9, 9.5]], truth)}
        out[rid]["vad_bench"]["results"]["none"]["webrtc-3"] = scored([[1.0, 5.0]], truth)
        out[rid]["vad_bench"]["results"]["X"]["webrtc-3"] = scored([[1.0, 2.0]], truth)
    return out


def check_report_nc_vad_and_matrix():
    runs = vad_dataset()
    out = rr.sec_nc_vad(runs)
    secs = sections_of(out)
    miss = md_rows(next(b for h, b in secs.items() if h.startswith("Phone — missed turns")))
    ids = [r[1] for r in miss]
    assert "`krisp-nc`" not in out and "`krisp-bvc`" in ids, ids     # live never, replay-under-live id yes
    assert ids.index("`Y`") < ids.index("`X`"), "lowest missed turns first"
    row = {r[1]: r for r in miss}
    assert row["`X`"][3] == "+1.00" and row["`Y`"][3] == "+0.00" and row["`X`"][2] == "2", row
    phan = md_rows(next(b for h, b in secs.items() if h.startswith("Phone — phantom")))
    assert [r[1] for r in phan][0] == "`X`" and {r[1] for r in phan} == {"`X`", "`Y`", "`krisp-bvc`"}, phan
    agree = md_rows(next(b for h, b in secs.items() if h.startswith("Phone — frame agreement")))
    vals = [float(r[3].rstrip("%")) for r in agree]
    assert vals == sorted(vals, reverse=True), f"higher-is-better table must sort high to low: {vals}"
    assert not any(h.startswith("Web") for h in secs)

    cells = rr._matrix(runs, "phone")
    assert ("krisp-nc", "lk-inference") not in cells and ("krisp-bvc", "lk-inference") in cells
    assert ("input", "lk-inference") in cells
    c = cells[("X", "lk-inference")]
    assert c == {"runs": 2, "turns": 4, "missed": 2, "phantom": 0, "merged": 0, "over": 0.0}, c
    assert cells[("Y", "lk-inference")]["phantom"] == 2
    assert cells[("none", "webrtc-3")]["merged"] == 4, cells[("none", "webrtc-3")]   # one segment spans both turns, 2 runs
    text = rr.sec_matrix(runs)
    assert "Pareto front" in text and "`krisp-nc` →" not in text
    # stale / errored / excluded rows do not enter
    runs["p1"]["vad_bench"]["results"]["X"]["lk-inference"]["stale"] = True
    assert rr._matrix(runs, "phone")[("X", "lk-inference")]["runs"] == 1


def check_report_pareto():
    cells = {("g1", "v"): {"runs": 2, "missed": 1, "phantom": 5},
             ("g2", "v"): {"runs": 2, "missed": 2, "phantom": 5},     # dominated by g1
             ("g3", "v"): {"runs": 2, "missed": 0, "phantom": 9},     # front: fewest missed
             ("g4", "v"): {"runs": 2, "missed": 1, "phantom": 5},     # equal to g1: neither dominates
             ("g5", "v"): {"runs": 1, "missed": 0, "phantom": 0},     # would dominate all, too few runs
             ("g6", "v"): {"runs": 2, "missed": 3, "phantom": 1}}     # front: fewest phantoms
    front = [k for k, _ in rr.pareto(cells, 2)]
    assert front == [("g3", "v"), ("g1", "v"), ("g4", "v"), ("g6", "v")], front
    assert [k for k, _ in rr.pareto(cells, 1)] == [("g5", "v")]
    assert rr.pareto({}, 1) == []
    assert [k for k, _ in rr.pareto({("a", "v"): {"runs": 3, "missed": 0, "phantom": 0}}, 3)] == [("a", "v")]


def check_report_tail():
    tail = rr.TAIL_RUN
    truth = [[1.0, 2.0]]
    segs = [[1.0, 2.0], [24.0, 24.5]]
    runs = {tail: rmeta("phone", [cr("hush")], truth=truth, dur=30.0,
                        vb={"hush": {"lk-inference": scored(segs, truth, dur=30.0)}}),
            "other": rmeta("phone", [cr("hush")], truth=truth, dur=30.0,
                           vb={"hush": {"lk-inference": scored(segs, truth, dur=30.0)}})}
    on = {r[0]: r for r in rr._vad_table(runs, "phone", "hush", tail=True)}["`lk-inference`"]
    off = {r[0]: r for r in rr._vad_table(runs, "phone", "hush", tail=False)}["`lk-inference`"]
    assert on[4] == 2 and off[4] == 1, (on, off)              # only TAIL_RUN's segment after 23.8 s is dropped
    assert on[3] == off[3] == 0 and on[1] == off[1] == 2
    assert rr._matrix(runs, "phone", True)[("hush", "lk-inference")]["phantom"] == 2
    assert rr._matrix(runs, "phone", False)[("hush", "lk-inference")]["phantom"] == 1
    # a segment before 23.8 s in the tail run keeps counting
    runs[tail]["vad_bench"]["results"]["hush"]["lk-inference"] = scored([[1, 2], [10, 10.5]], truth, dur=30.0)
    off = {r[0]: r for r in rr._vad_table(runs, "phone", "hush", tail=False)}["`lk-inference`"]
    assert off[4] == 2, off
    # a segment straddling 23.8 s is ignored too (it touches the ignore window)
    runs[tail]["vad_bench"]["results"]["hush"]["lk-inference"] = scored([[1, 2], [23.0, 24.0]], truth, dur=30.0)
    off = {r[0]: r for r in rr._vad_table(runs, "phone", "hush", tail=False)}["`lk-inference`"]
    assert off[4] == 1, off
    text = rr.sec_vad(runs)
    assert "unmarked tail" in text and text.count("### Phone") >= 2


def check_report_incidents():
    thr = {"name": "throat-clear-B", "start": 17.9, "end": 18.4}
    a_truth = [[24.0, 24.5]]
    a_kns = [{"name": "click-A", "start": 38.42, "end": 39.07}, {"name": "hangup-clunk-A", "start": 64.79, "end": 64.88}]
    sounds = ([{"cls": "detected", "start": 1, "end": 2}, {"cls": "detected", "start": 3, "end": 4},
               {"cls": "loud_missed", "start": 10, "end": 11}, {"cls": "loud_missed", "start": 12, "end": 13},
               {"cls": "quiet_echo", "start": 20, "end": 21}])
    b_segs = [[12.6, 13.0], [18.0, 18.2], [30.0, 31.0]]
    a_segs = [[24.0, 24.5], [38.5, 38.6]]
    c_segs = [[0.9, 2.1], [10.2, 10.8]]
    runs = {
        "ra": rmeta("phone", [cr("none")], truth=a_truth, dur=70.0, known_nonspeech=a_kns,
                    incident={"id": "PR-54 A"}, vb={"none": {"lk-inference": scored(a_segs, a_truth, a_kns, dur=70.0)}}),
        "rb": rmeta("phone", [cr("none")], truth=[], dur=30.0, known_nonspeech=[thr], incident={"id": "PR-54 B"},
                    vb={"none": {"lk-inference": scored(b_segs, [], [thr], dur=30.0)}}),
        "rc": rmeta("phone", [cr("none")], truth=[[1, 2], [3, 4], [10, 11]], dur=30.0,
                    incident={"id": "PR-42", "sounds": sounds},
                    vb={"none": {"lk-inference": scored(c_segs, [[1, 2], [3, 4], [10, 11]], dur=30.0)}}),
    }
    rows = md_rows(rr.sec_incidents(runs))
    assert len(rows) == 1 and len(rows[0]) == 13, rows
    r = rows[0]
    assert r[0] == "`none` → `lk-inference`"
    assert r[1] == "1/1" and r[2] == "OPEN" and r[3] == "no", r           # A: click opened, clunk not
    # B: from the segments alone (no marks): throat-clear opened, 2 of 3 segments outside the greeting
    assert r[6] == "OPEN" and r[7] == "2" and r[8] == "3", r
    # PR-42 from incident.sounds: 1 of 2 detected kept, 1 of 2 loud-missed caught
    assert r[9] == "1/2" and r[10] == "1/2", r
    # B closed: only the greeting segment. Production's VAD (lk-inference on this source) is silent
    # on the throat-clear too, so B does not discriminate here
    runs["rb"]["vad_bench"]["results"]["none"]["lk-inference"] = scored([[12.6, 13.0]], [], [thr], dur=30.0)
    r = md_rows(rr.sec_incidents(runs))[0]
    assert r[6] == "n/a: prod VAD silent here" and r[7] == "0" and r[8] == "1", r
    # where lk-inference opens on it: another VAD that stays closed "rejects", one that opens is OPEN
    res = runs["rb"]["vad_bench"]["results"]["none"]
    res["lk-inference"] = scored(b_segs, [], [thr], dur=30.0)
    res["webrtc-3"] = scored([[12.6, 13.0]], [], [thr], dur=30.0)
    res["ten"] = scored([[18.0, 18.2]], [], [thr], dur=30.0)
    rows = {r[0]: r for r in md_rows(rr.sec_incidents(
        runs, combos=[("none", "lk-inference"), ("none", "webrtc-3"), ("none", "ten")]))}
    assert rows["`none` → `lk-inference`"][6] == "OPEN"
    assert rows["`none` → `webrtc-3`"][6] == "rejects" and rows["`none` → `ten`"][6] == "OPEN", rows
    del res["webrtc-3"], res["ten"]
    res["lk-inference"] = scored([[12.6, 13.0]], [], [thr], dur=30.0)
    # a segment touching only the greeting edge (12.5, 16.3) from outside does not count as inside
    runs["rb"]["vad_bench"]["results"]["none"]["lk-inference"] = scored([[16.3, 16.6]], [], [thr], dur=30.0)
    assert md_rows(rr.sec_incidents(runs))[0][7] == "1"
    # missing incidents and a `combos` filter
    rows = md_rows(rr.sec_incidents({"rb": runs["rb"]}))
    assert rows[0][1:6] == ["—"] * 5 and rows[0][9:] == ["—"] * 4, rows
    assert md_rows(rr.sec_incidents(runs, combos=[("none", "nope")])) == []
    assert md_rows(rr.sec_incidents({})) == []
    # "A phantoms excl. click/clunk": a segment only on click-A (a_segs) is OPEN but not counted
    r = md_rows(rr.sec_incidents(runs))[0]
    assert r[2] == "OPEN" and r[4] == "0", r
    # one more phantom elsewhere counts, and a clunk-only segment still does not
    both = a_segs + [[50.0, 50.4], [64.8, 64.85]]
    runs["ra"]["vad_bench"]["results"]["none"]["lk-inference"] = scored(both, a_truth, a_kns, dur=70.0)
    r = md_rows(rr.sec_incidents(runs))[0]
    assert r[2] == "OPEN" and r[3] == "OPEN" and r[4] == "1", r
    assert "A phantoms excl. click/clunk" in rr.sec_incidents(runs)
    # default combos: SWEEP_GATES x SWEEP_VADS order, then input/none/hush; absent combos and
    # all-"—" rows (a stale row on every incident) are dropped
    for sid in ("input", "hush", "none", "dpdfnet2-8k"):
        for r_ in runs.values():
            r_["vad_bench"]["results"][sid] = {"lk-inference": scored(c_segs, [[1, 2]], dur=30.0)}
    runs["rb"]["vad_bench"]["results"]["dpdfnet2-8k"]["lk-inference"]["stale"] = True
    runs["ra"]["vad_bench"]["results"]["dpdfnet2-8k"]["lk-inference"]["stale"] = True
    runs["rc"]["vad_bench"]["results"]["dpdfnet2-8k"]["lk-inference"]["stale"] = True
    assert [r[0] for r in md_rows(rr.sec_incidents(runs))] == [
        "`hush` → `lk-inference`", "`none` → `lk-inference`", "`input` → `lk-inference`"]


def live_dataset() -> dict[str, dict]:
    truth = [[1.0, 2.0]]
    vb = {"none": {"lk-inference": scored([[0.02, 0.10], [1.0, 2.0]], truth, dur=5.0)},
          "krisp-bvc-tel": {"lk-inference": scored([[0.85, 1.85]], truth, dur=5.0)},
          "krisp-nc": {"lk-inference": scored([[1.03, 2.03]], truth, dur=5.0)},
          "krisp-bvc": {"lk-inference": scored([[0.85, 1.85]], truth, dur=5.0)}}
    off = lambda lag: {"lag_ms": lag, "len_diff_ms": lag, "env_corr": .9, "prominence": .5}  # noqa: E731
    return {"lr1": rmeta("phone", [cr("none"), cr("krisp-bvc-tel", live_rail_offset=off(-150.0)),
                                   cr("krisp-nc", live_rail_offset=off(30.0)),
                                   cr("krisp-bvc", rail="cloud_replay", live_rail_offset=off(-150.0)),
                                   cr("aic-quail-l")],           # live, no offset stored yet
                         truth=truth, dur=5.0, vb=vb)}


def check_report_live_rail():
    runs = live_dataset()
    before = json.dumps(runs, sort_keys=True)
    out = rr.sec_live_rail(runs)
    assert json.dumps(runs, sort_keys=True) == before, "the report must never write back"
    assert "`krisp-bvc`" not in out.replace("krisp-bvc-tel", ""), "a cloud_replay record under a live id is not a live-rail row"
    assert "| `krisp-bvc-tel` | 1 | -150 | -150 | -150 |" in out and "| `krisp-nc` | 1 | +30 | +30 | +30 |" in out, out
    table = out[out.index("Under `"):]
    rows = md_rows(table)
    assert [(r[1], r[2]) for r in rows] == [("`krisp-bvc-tel`", "stored"), ("`krisp-bvc-tel`", "corrected"),
                                            ("`krisp-nc`", "stored"), ("`krisp-nc`", "corrected")], rows
    st, co, nst, nco = rows
    # lag -150: stored, the early stream hits its turn (no miss) and has no phantom vs none's 1
    assert st[3:] == ["1", "+0.00", "-1.00", "-0.150"], st
    # corrected: segments +0.15 s onto input time (onset lag 0), the head [0, 0.15] ignored so none's
    # head segment [0.02, 0.10] is no longer a phantom: 0 - 0
    assert co[3:] == ["1", "+0.00", "+0.00", "+0.000"], co
    # lag +30: shifted -0.03, no head: the baseline keeps its head phantom
    assert nst[3:] == ["1", "+0.00", "-1.00", "+0.030"], nst
    assert nco[3:] == ["1", "+0.00", "-1.00", "+0.000"], nco
    # the corrected row differs from the stored one
    assert st != co
    # no offsets at all
    for r in runs.values():
        for c in r["candidates"]:
            c.pop("live_rail_offset", None)
    assert rr.sec_live_rail(runs) == "(no live-rail offsets stored yet)"
    # an offset but no marked speech, or no surface: skipped without a crash
    runs = live_dataset()
    runs["lr1"]["truth_spans"] = []
    assert "Under `" not in rr.sec_live_rail(runs)
    runs = live_dataset()
    runs["lr1"]["source"] = "upload"
    assert "Under `" not in rr.sec_live_rail(runs)
    # a segment shifted entirely off the start is dropped, not written as a negative span
    runs = live_dataset()
    runs["lr1"]["vad_bench"]["results"]["krisp-nc"]["lk-inference"] = scored([[0.0, 0.02]], [[1.0, 2.0]], dur=5.0)
    rr.sec_live_rail(runs)


def check_report_live_rail_skips_rows_without_events():
    runs = live_dataset()
    res = runs["lr1"]["vad_bench"]["results"]
    res["krisp-nc"]["lk-inference"] = {"segments": [[1.03, 2.03]]}                 # no events at all
    res["krisp-bvc-tel"]["lk-inference"]["events"] = {k: None for k in res["krisp-bvc-tel"]["lk-inference"]["events"]}
    out = rr.sec_live_rail(runs)
    assert "Under `" not in out, out          # both live rows dropped: no table, no crash
    # the offsets table is still printed
    assert "| `krisp-nc` | 1 | +30 | +30 | +30 |" in out
    # a baseline without events skips every live row of the run
    runs = live_dataset()
    runs["lr1"]["vad_bench"]["results"]["none"]["lk-inference"] = {"segments": [[1, 2]]}
    assert "Under `" not in rr.sec_live_rail(runs)
    # only the row with events is dropped, the other stays
    runs = live_dataset()
    runs["lr1"]["vad_bench"]["results"]["krisp-nc"]["lk-inference"] = {"segments": [[1.03, 2.03]]}
    rows = md_rows(rr.sec_live_rail(runs)[rr.sec_live_rail(runs).index("Under `"):])
    assert {r[1] for r in rows} == {"`krisp-bvc-tel`"}


def check_report_vad_wer():
    def w(wer, text="a b"):
        return {"wer": {"wer": wer}, "text": text}

    other = json.dumps(["none", "lk-inference", 0.6, 0.05, 0.3])
    third = json.dumps(["hush", "silero-v4.0-8k", 0.5, 0.05, 0.4])

    def run(base, oth_raw, oth_same, prod_same, extra=None):
        cuts = {rr.PROD_COMBO: {"raw": w(base), "same": w(prod_same)},
                other: {"raw": w(oth_raw), "same": w(oth_same)}}
        if extra:
            cuts[third] = extra
        return rmeta("phone", [], script="a b", vad_cut_wer_round2=cuts)

    runs = {"p1": run(.2, .1, .3, .2, extra={"same": w(.05, "no " * 6), "raw": w(.25)}),
            "p2": run(.4, .3, .5, .4),
            "p5-junk": run(1.5, .1, .1, .1),                                                          # baseline WER > 1.0
            "p3": rmeta("phone", [], script="a b", vad_cut_wer_round2={
                rr.PROD_COMBO: {"raw": w(.5, "bank " * 6), "same": w(.5)}, other: {"raw": w(.1)}}),     # base collapsed
            "p4": rmeta("phone", [], script="a b", vad_cut_wer_round2={other: {"raw": w(.1)}}),        # no base
            "w1": rmeta("web", [], vad_cut_wer_round2={rr.PROD_COMBO: {"raw": w(.2)}}),
            "u": rmeta("upload", [], vad_cut_wer_round2={rr.PROD_COMBO: {"raw": w(.2)}})}
    out = rr.sec_vad_wer(runs)
    secs = sections_of(out)
    head = next(h for h in secs if h.startswith("Phone"))
    assert "(2 runs;" in head, head
    rows = md_rows(secs[head].split("\n\n")[0])
    lab = lambda r: (r[0], r[1])  # noqa: E731
    assert [lab(r) for r in rows] == [
        ("`hush` → `silero-v4.0-8k` @ 0.5/0.05/0.4", "same"),   # -0.15: the looping transcript is counted, not dropped
        ("`none` → `lk-inference` @ 0.6/0.05/0.3", "raw"),      # -0.1
        ("`hush` → `lk-inference` @ 0.6/0.05/0.3", "same"),     # 0 (production itself; ties keep same, raw)
        ("`hush` → `lk-inference` @ 0.6/0.05/0.3", "raw"),      # 0
        ("`hush` → `silero-v4.0-8k` @ 0.5/0.05/0.4", "raw"),    # +0.05
        ("`none` → `lk-inference` @ 0.6/0.05/0.3", "same")], rows   # +0.1
    assert rows[0][2] == "1" and rows[0][4] == "-0.150" and rows[0][9] == "1", rows[0]   # n 1, one loop
    assert [r[9] for r in rows[1:]] == ["0"] * 5
    top = rows[1]
    assert top[2] == "2" and top[3] == "0.200" and top[4] == "-0.100" and top[7:9] == ["2", "0"], top
    assert md_header(secs[head].split("\n\n")[0])[-1] == "loops"
    body = secs[head]
    assert "\"loops\":" in body and "They are counted in the WER (1 on this surface)." in body, body
    assert "dropped" not in body
    # excluded baselines, listed once each, and their other combos do not count
    ex = md_rows(body[body.index("Runs excluded"):].split("\n\n", 1)[1].split("\n\n")[0])
    assert {r[0]: r[2] for r in ex} == {"`p3`": "production baseline collapsed (x6)",
                                       "`p5-junk`": "production baseline WER 1.50 > 1.0"}, ex
    assert "`p4`" not in body
    web = md_rows(next(b for h, b in secs.items() if h.startswith("Web")).split("\n\n")[0])
    assert {r[2] for r in web} == {"1"} and {r[4] for r in web} == {"+0.000"}, web   # production against itself
    assert rr.sec_vad_wer({"x": rmeta("phone", [])}) == "(no VAD-cut WER stored yet)"


def check_report_tail_section():
    assert rr.sec_tail({}) == "(run missing)"
    truth = [[1.0, 2.0]]
    m = rmeta("phone", [cr("hush")], truth=truth, dur=30.0,
              vb={"hush": {"lk-inference": scored([[1.0, 2.0], [24.0, 24.5]], truth, dur=30.0),
                           "ten": scored([[10.0, 10.5]], truth, dur=30.0)},
                  "none": {"earshot": scored([[1.0, 2.0]], truth, dur=30.0)}})
    rows = {r[0]: r for r in md_rows(rr.sec_tail({rr.TAIL_RUN: m}))}
    # phantoms as stored, with the tail ignored, segments in the tail, missed turns
    assert rows["`hush` → `lk-inference`"][1:] == ["1", "0", "1", "0"], rows
    assert rows["`hush` → `ten`"][1:] == ["1", "1", "0", "1"], rows            # before 23.8 s: still a phantom
    assert rows["`none` → `earshot`"][1:] == ["0", "0", "0", "0"], rows
    assert "`hush` → `silero-v4.0-8k`" not in rows
    assert md_rows(rr.sec_tail({rr.TAIL_RUN: rmeta("phone", [])})) == []


def check_report_sweep_and_load(env):
    def cell(missed, phantom, merged=0, turns=5, **kw):
        return {"missed": missed, "phantom": phantom, "merged": merged, "turns": turns, **kw}

    PROD, CONT, OTHER = (0.6, 0.05, 0.3), (0.5, 0.05, 0.4), (0.7, 0.1, 0.55)
    ck = lambda g, v, p: sw.ckey(g, v, *p)  # noqa: E731
    lk = "lk-inference"
    a_cells = {ck("hush", lk, PROD): cell(1, 1), ck("hush", lk, CONT): cell(2, 2), ck("hush", lk, OTHER): cell(0, 1),
               ck("none", lk, PROD): cell(2, 2), ck("none", lk, (0.3, 0.05, 0.3)): cell(0, 0),   # 2nd: run a only
               ck("krisp-viva", lk, PROD): cell(0, 0, turns=None), ck("silero-v4.0-8k", lk, PROD): cell(0, 0, rank_excluded="x")}
    b_cells = {ck("hush", lk, PROD): cell(1, 0), ck("hush", lk, CONT): cell(1, 3), ck("hush", lk, OTHER): cell(1, 0),
               ck("none", lk, PROD): cell(3, 3)}
    truth = [[1, 2]]
    bpt = ck("hush", lk, OTHER)
    runs = {
        "a": rmeta("phone", [], truth=truth, vad_sweep_round2={"cells": a_cells}),
        "b": rmeta("phone", [], truth=truth, vad_sweep_round2={"cells": b_cells}),
        "w": rmeta("web", [], truth=truth, vad_sweep_round2={"cells": {ck("hush", lk, PROD): cell(4, 0)}}),
        "nomarks": rmeta("phone", [], vad_sweep_round2={"cells": {ck("hush", lk, PROD): cell(9, 9)}}),
        "c": {},
        "iA": rmeta("upload", [], incident={"id": "PR-54 A"}, vad_sweep_round2={"cells": {bpt: cell(
            0, 0, hit=1, turns=1, nonspeech_hits={"click-A": True, "hangup-clunk-A": False})}}),
        "iB": rmeta("upload", [], incident={"id": "PR-54 B"}, vad_sweep_round2={"cells": {bpt: cell(
            0, 0, turns=None, nonspeech_hits={"throat-clear-B": False})}}),
        "i42": rmeta("upload", [], incident={"id": "PR-42", "sounds": [
            {"cls": "detected", "start": 1, "end": 2}, {"cls": "loud_missed", "start": 10, "end": 11}]},
            vad_sweep_round2={"cells": {bpt: cell(0, 0, segments=[[0.9, 2.1]])}}),
    }
    out = rr.sec_sweep(runs)
    secs = sections_of(out)
    ph = md_rows(next(b for h, b in secs.items() if h.startswith("Phone")))
    assert [r[0] for r in ph] == ["`hush` → `lk-inference`", "`none` → `lk-inference`"], ph   # best score first
    h, n = ph
    # pooled hush: prod 2/1, cont 3/5, other 1/1 -> the best is `other`, 2 of 3 points within +1
    assert h[1] == "0.7/0.1/0.55" and h[2:5] == ["1", "1", "0"] and h[5] == "2/3", h
    assert h[6] == "2/1/0" and h[7] == "3/5/0", h                       # at 0.6/0.05/0.3 and at 0.5/0.05/0.4
    assert h[8] == "A 1/1 click OPEN clunk no; B throat no; PR-42 1/21 +0/7 orphans 0", h
    # none: only points present in every run count, so run a's lone 0/0 point does not win
    assert n[1] == "0.6/0.05/0.3" and n[2:5] == ["5", "5", "0"] and n[5] == "1/1" and n[7] == "—", n
    assert "krisp-viva" not in out and "silero" not in out             # turns None / rank_excluded cells skipped
    web = md_rows(next(b for h, b in secs.items() if h.startswith("Web")))
    assert web[0][2:4] == ["4", "0"] and web[0][8] == "", web
    assert "9" not in "".join(r[2] for r in ph), "a run without marks is not pooled"
    # ties go to the set nearest production's parameters
    tie = {"a": rmeta("phone", [], truth=truth, vad_sweep_round2={"cells": {
        ck("hush", lk, (0.3, 0.05, 0.3)): cell(1, 0), ck("hush", lk, (0.8, 0.25, 0.55)): cell(1, 0),
        ck("hush", lk, (0.7, 0.05, 0.3)): cell(1, 0)}})}
    assert md_rows(rr.sec_sweep(tie))[0][1] == "0.7/0.05/0.3"
    assert rr.sec_sweep({"a": {}}) == "(no sweep stored yet)"
    assert md_rows(rr.sec_sweep(tie))[0][5] == "3/3"
    # load_runs: skips _scratch, dirs without meta.json; sorted
    fresh_runs()
    for n in ("b", "a", "_scratch"):
        (config.RUNS_DIR / n).mkdir()
        (config.RUNS_DIR / n / "meta.json").write_text(json.dumps({"id": n}))
    (config.RUNS_DIR / "nometa").mkdir()
    assert list(rr.load_runs()) == ["a", "b"] and list(rr.load_runs(config.RUNS_DIR)) == ["a", "b"]
    assert rr.load_runs(config.RUNS_DIR)["a"] == {"id": "a"}


def check_report_no_crash_on_sparse_meta(env):
    """Every section over metas missing most keys, and over the full synthetic set."""
    sparse = {
        "r0": {},
        "r1": {"source": "phone", "candidates": None, "input": None, "vad_bench": None},
        "r2": {"source": "phone", "script": "a b", "candidates": [{"id": "none"}, {"id": "X", "status": "done"}],
               "truth_spans": [[1, 2]], "input": {}, "vad_bench": {"results": {"none": {"lk-inference": {"segments": [[1, 2]]}}}}},
        "r3": {"source": "web", "candidates": [{"id": "krisp-nc", "live_rail_offset": {"lag_ms": -100}},
                                               {"id": "none"}],
               "truth_spans": [[1, 2]], "input": {"duration_s": 5},
               "vad_bench": {"results": {"none": {"lk-inference": scored([[1, 2]], [[1, 2]])},
                                         "krisp-nc": {"lk-inference": scored([[1, 2]], [[1, 2]])}}}},
        "r4": {"incident": {"id": "PR-54 B"}, "vad_bench": {"results": {"none": {"lk-inference": {"segments": []}}}}},
        "r5": {"incident": {"id": "PR-42"}, "vad_bench": {"results": {"none": {"lk-inference": {"segments": []}}}}},
    }
    failures = []
    datasets = {"sparse": sparse, "stt": stt_dataset(), "vad": vad_dataset(), "live": live_dataset()}
    for dname, runs in datasets.items():
        for name, fn in rr.SECTIONS.items():
            try:
                out = fn(runs)
                assert isinstance(out, str)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{dname}/{name}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


def check_report_cli(env):
    fresh_runs()
    assert "report_round2.py --section coverage" in run_main(env, rr, "--section", "coverage")
    assert "report_round2.py --section nc-stt" in run_main(env, rr, "--section", "nc-stt", "--section", "sweep")


def check_report_list_flag(env):
    out = run_main(env, rr, "--list")
    assert "nc-stt" in out and "live-rail" in out and "incidents" in out


# ====================================================================== runner

KNOWN_BUGS: set[str] = set()


def main() -> None:
    checks = [(n, f) for n, f in globals().items() if n.startswith("check_") and callable(f)]
    started = time.perf_counter()
    failed, known = [], []
    try:
        for name, fn in checks:
            env = Env()
            t0 = time.perf_counter()
            try:
                fn(env) if fn.__code__.co_varnames[: fn.__code__.co_argcount] == ("env",) else fn()
            except (Exception, SystemExit):  # noqa: BLE001
                if name in KNOWN_BUGS:
                    known.append(name)
                    print(f"KNOWN-BUG {name}\n{traceback.format_exc().splitlines()[-1]}")
                    continue
                failed.append(name)
                print(f"FAIL {name} ({time.perf_counter() - t0:.1f}s)\n{traceback.format_exc()}")
            else:
                print(f"OK {name} ({time.perf_counter() - t0:.1f}s)")
            finally:
                env.close()
    finally:
        shutil.rmtree(c2._SANDBOX, ignore_errors=True)
    total = time.perf_counter() - started
    print(f"\n{len(checks) - len(failed) - len(known)}/{len(checks)} passed, {len(failed)} failed, "
          f"{len(known)} known-bug, {total:.1f}s")
    if failed:
        sys.exit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
