"""Do the round-2 CLIs (scripts/backfill_round2.py, scripts/import_incidents.py)
do what they claim, with every outside dependency faked?

No network, no LiveKit, no STT server, no licence key, no model inference.
DATA_DIR points at a temp dir before nc_bench.config is imported (load_dotenv does
not override a variable that is already set) and the script refuses to go on
unless config.RUNS_DIR is inside it, so the real data/runs/ and data/incidents/
cannot be touched whatever .env says.

Covers
- backfill_round2: the replay aliases, resolve_cands, _done, process() with a fake
  server._process_run (skip, stt_setting/backfill tagging, untouched records,
  pass log, cloud retries and the halt), run_stt() with a fake stt.transcribe;
- import_incidents helpers: _ncc_locate, _active, _labels, _find.
  The build_* functions need the real recordings and are not run. The script
  itself is kept out of the repository (it reads private call recordings), so
  these four checks are skipped when it isn't there.

    .venv/bin/python scripts/check_round2.py
"""

import asyncio
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SANDBOX = Path(tempfile.mkdtemp(prefix="nc-check-round2-")).resolve()
for _k, _v in {
    "LIVEKIT_URL": "wss://bogus.invalid",
    "LIVEKIT_API_KEY": "bogus-key",
    "LIVEKIT_API_SECRET": "bogus-secret",
    "AIC_LICENSE_KEY": "",
    "AIC_MODEL_ID": "",
    "AIC_MODEL_DIR": str(_SANDBOX / "aic-models"),
    "WHISPER_URL": "http://127.0.0.1:9/never",
    "WHISPER_BOOST_VOLUME": "false",
    "DATA_DIR": str(_SANDBOX / "data"),
}.items():
    os.environ[_k] = _v

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nc_bench import config, scoring, store, stt  # noqa: E402

# the hard rule of this file: never anywhere near the real recordings
assert config.RUNS_DIR.is_relative_to(_SANDBOX), f"RUNS_DIR escaped the sandbox: {config.RUNS_DIR}"
assert config.DATA_DIR.is_relative_to(_SANDBOX), f"DATA_DIR escaped the sandbox: {config.DATA_DIR}"
config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

import backfill_round2 as bf  # noqa: E402
from nc_bench import server  # noqa: E402

try:
    import import_incidents as ii  # noqa: E402
except ImportError:   # local-only script, not in the repository
    ii = None
_NEEDS_II = {"check_ncc_locate", "check_active", "check_labels", "check_find"}

assert bf.LOG.is_relative_to(_SANDBOX) and (ii is None or ii.INC.is_relative_to(_SANDBOX))

_MISSING = object()


class Env:
    """Monkeypatching that undoes itself."""

    def __init__(self):
        self.undo = []

    def attr(self, obj, name, value):
        self.undo.append((obj, name, getattr(obj, name, _MISSING)))
        setattr(obj, name, value)

    def close(self):
        for obj, name, old in reversed(self.undo):
            setattr(obj, name, old) if old is not _MISSING else delattr(obj, name)


def quiet(fn, *a, **kw):
    """Run fn with the CLI's JSON log lines swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def raises(exc, fn, *a, contains: str = "", **kw):
    try:
        quiet(fn, *a, **kw)
    except exc as e:
        assert contains in str(e), f"{type(e).__name__} message {e!r} lacks {contains!r}"
        return e
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


def fresh_runs() -> None:
    assert config.RUNS_DIR.is_relative_to(_SANDBOX)
    shutil.rmtree(config.RUNS_DIR, ignore_errors=True)
    config.RUNS_DIR.mkdir(parents=True)


def short_wav(path: Path, dur: float = 0.1, rate: int = 16000) -> None:
    sf.write(path, np.zeros(int(dur * rate), dtype=np.int16), rate, subtype="PCM_16")


def make_run(script: str = "", cands: list[dict] | None = None, files: list[str] | None = None) -> str:
    rid, d = store.new_run("upload", "check_round2")
    short_wav(d / "input.wav")
    for f in files if files is not None else [c["output"] for c in cands or [] if c.get("output")]:
        short_wav(d / f)
    with store.locked_meta(rid) as m:
        m["script"] = script
        m["status"] = "done"
        m["candidates"] = cands or []
    return rid


def rec_of(rid: str, cid: str) -> dict | None:
    return next((c for c in store.load_meta(rid)["candidates"] if c.get("id") == cid), None)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ============================================================ fakes for process()

OFFLINE = ["off-a", "off-b"]


def fake_known() -> dict[str, dict]:
    """A registry with every id the backfill sets name, all available."""
    k: dict[str, dict] = {}
    for cid in [c for c in bf.CLOUD_SET if c not in bf.REPLAY_ALIASES]:
        k[cid] = {"id": cid, "label": cid, "lk_model": "M:" + cid, "rail": "cloud",
                  "available": True, "unavailable_reason": ""}
    for cid in ("krisp-nc", "krisp-bvc", "krisp-bvc-tel"):
        k[cid] = {"id": cid, "label": cid, "lk_model": "KRISP:" + cid, "rail": "cloud",
                  "available": True, "unavailable_reason": ""}
    k["aic-quail-l"] = {"id": "aic-quail-l", "label": "AIC (historical)", "lk_model": "AIC:Q",
                        "rail": "cloud", "available": False, "historical": True,
                        "unavailable_reason": "historical"}
    for cid in OFFLINE:
        k[cid] = {"id": cid, "label": cid, "chain": [cid], "rail": "offline",
                  "available": True, "unavailable_reason": ""}
    return k


class FakeProcessRun:
    """Stands in for server._process_run. Writes records the way the real one
    does: real server._settle + _put, under store.locked_meta, files staged."""

    def __init__(self, known: dict, plan: dict[str, list[str]] | None = None):
        self.known = known
        self.plan = {k: list(v) for k, v in (plan or {}).items()}   # cid -> outcomes, then "ok"
        self.calls: list[list[str]] = []
        self.use_stt = False   # transcribe each success through server.stt.transcribe, like the real one

    async def __call__(self, run_id, ids, live_results=None, script="", concurrency=1):
        assert live_results is None and concurrency == 1, (live_results, concurrency)
        self.calls.append(list(ids))
        rd = store.run_dir(run_id)
        known = server._candidates_by_id()    # includes the replay aliases
        attempts = rd / "_attempts"
        attempts.mkdir(exist_ok=True)
        for cid in ids:
            outcome = (self.plan.get(cid) or ["ok"]).pop(0) if self.plan.get(cid) else "ok"
            if outcome == "ok":
                staged_path = attempts / f"{cid}.wav"
                short_wav(staged_path)
                new = {"id": cid, "label": known[cid]["label"], "status": "done",
                       "output": f"{cid}.wav", "processed_at": now(),
                       "rail": "cloud_replay" if known[cid]["rail"] == "cloud" else "offline",
                       "scores": {"wer": None}}
                staged = {f"{cid}.wav": staged_path}
                if self.use_stt:
                    try:
                        new["stt"] = await server.stt.transcribe(staged_path)
                    except Exception as e:  # noqa: BLE001 - what the real pipeline records
                        new["stt_error"] = str(e)
            else:
                new = {"id": cid, "status": "error", "error": outcome, "processed_at": now()}
                staged = {}
            with store.locked_meta(run_id) as m:
                old = next((c for c in m["candidates"] if c.get("id") == cid), None)
                server._put(m["candidates"], server._settle(rd, old, new, staged), list(ids))
        shutil.rmtree(attempts, ignore_errors=True)


def patched(env: Env, plan=None) -> tuple[dict, FakeProcessRun]:
    known = fake_known()
    env.attr(server, "_candidates_by_id", lambda: {k: dict(v) for k, v in known.items()})
    fake = FakeProcessRun(known, plan)
    env.attr(server, "_process_run", fake)
    return known, fake


def snapshot(rid: str, cid: str):
    d = store.run_dir(rid)
    r = rec_of(rid, cid)
    return json.dumps(r, sort_keys=True), (d / r["output"]).read_bytes() if r.get("output") else None


# ================================================================== backfill: 1

def check_replay_aliases(env):
    orig = server._candidates_by_id
    env.attr(server, "_candidates_by_id", orig)   # restore whatever _server() wraps
    base = orig()
    by = quiet(bf._server)._candidates_by_id()
    assert set(bf.REPLAY_ALIASES) <= set(by), sorted(by)
    for alias, (cid, label) in bf.REPLAY_ALIASES.items():
        e = by[alias]
        assert e["id"] == alias and e["replay_alias_of"] == cid
        assert e["lk_model"] == base[cid]["lk_model"], (alias, e["lk_model"])
        assert "(cloud replay)" in e["label"] and e["label"] == label
        assert e["rail"] == "cloud"
        # the real one is unchanged, and carries no alias marker
        assert by[cid] == base[cid], cid
        assert "replay_alias_of" not in by[cid] and "(cloud replay)" not in by[cid]["label"]
    # the wrapper is idempotent: a second _server() call changes nothing
    again = quiet(bf._server)._candidates_by_id()
    assert set(again) == set(by)


def check_resolve_cands():
    known = fake_known()
    known.update({
        "aicsdk-a": {"id": "aicsdk-a", "rail": "offline", "available": True},
        "aicsdk-b": {"id": "aicsdk-b", "rail": "offline", "available": True},
        "aicsdk-down": {"id": "aicsdk-down", "rail": "offline", "available": False},
        "aicsdk-hist": {"id": "aicsdk-hist", "rail": "offline", "available": True, "historical": True},
    })
    assert bf.resolve_cands("cloud", known) == bf.CLOUD_SET
    assert bf.resolve_cands("aicsdk", known) == ["aicsdk-a", "aicsdk-b"]
    inc = bf.resolve_cands("incident-offline", known)
    assert set(inc) == {"off-a", "off-b", "aicsdk-a", "aicsdk-b"} and inc == bf.resolve_cands("offline-all", known)
    assert bf.resolve_cands("off-a, krisp-nc ,", known) == ["off-a", "krisp-nc"]
    e = raises(SystemExit, bf.resolve_cands, "off-a,nope", known, contains="nope")
    assert "off-a" not in str(e)
    raises(SystemExit, bf.resolve_cands, "krisp-nc-replay", known, contains="krisp-nc-replay")  # aliases are not in a raw registry


# ================================================================== backfill: 2

def check_done():
    d = tempfile.mkdtemp(dir=_SANDBOX)
    d = Path(d)
    (d / "a.wav").write_bytes(b"x")
    ok = {"status": "done", "output": "a.wav", "rail": "cloud_replay"}
    for cloud in (False, True):
        assert not bf._done(None, d, cloud)
        assert not bf._done({}, d, cloud)
        assert not bf._done({**ok, "status": "error"}, d, cloud)
        assert not bf._done({**ok, "output": None}, d, cloud)
        assert not bf._done({**ok, "output": "missing.wav"}, d, cloud)
    assert bf._done(ok, d, False) and bf._done(ok, d, True)
    live = {**ok, "rail": "live"}
    assert bf._done(live, d, False), "an offline candidate does not care about the rail"
    assert not bf._done(live, d, True), "a live-rail record is not a cloud replay"
    assert not bf._done({**ok, "rail": None}, d, True)
    assert not bf._done({"status": "done", "output": "a.wav"}, d, True)


# ================================================================== backfill: 3

def check_process_skip_tag_untouched(env):
    fresh_runs()
    known, fake = patched(env)
    pre = [
        {"id": "krisp-nc", "status": "done", "output": "krisp-nc.wav", "rail": "live",
         "processed_at": "2026-09-20T10:00:00", "stt": {"text": "old"}},
        {"id": "aic-quail-l", "status": "done", "output": "aic-quail-l.wav", "rail": "cloud",
         "historical": True, "processed_at": "2026-09-01T10:00:00"},
        {"id": "cloud-raw", "status": "done", "output": "cloud-raw.wav", "rail": "cloud_replay",
         "processed_at": "2026-09-28T10:00:00"},
    ]
    rid = make_run("hello world", pre)
    before = {c["id"]: snapshot(rid, c["id"]) for c in pre}
    quiet(bf.process, [rid], "cloud-raw,krisp-nc-replay,lkaic-vf22-s", 2, False, False)
    assert fake.calls == [["krisp-nc-replay", "lkaic-vf22-s"]], fake.calls   # cloud-raw skipped
    for cid in ("krisp-nc-replay", "lkaic-vf22-s"):
        r = rec_of(rid, cid)
        assert r["status"] == "done" and r["backfill"] == bf.PASS, r
        assert r["stt_setting"]["boost_volume"] is False and r["stt_setting"]["whole_file"] is True
        assert r["stt_setting"]["pass"] == bf.PASS
    # a live-rail krisp-nc and a historical aic-quail-l are byte-identical, so is the skipped cloud-raw
    for cid, snap in before.items():
        assert snapshot(rid, cid) == snap, cid
        assert "backfill" not in rec_of(rid, cid) and "stt_setting" not in rec_of(rid, cid)
    passes = store.load_meta(rid)["backfill_round2"]["passes"]
    assert len(passes) == 1 and passes[0]["ids"] == ["krisp-nc-replay", "lkaic-vf22-s"]
    assert passes[0]["failed"] == {} and passes[0]["attempt"] == 0 and passes[0]["cands"].startswith("cloud-raw")
    assert store.load_meta(rid)["backfill_round2"]["cloud_replay"]["retries"] == 0

    # everything done: a second pass is a no-op, no call, no new pass entry
    quiet(bf.process, [rid], "cloud-raw,krisp-nc-replay,lkaic-vf22-s", 2, False, False)
    assert len(fake.calls) == 1 and len(store.load_meta(rid)["backfill_round2"]["passes"]) == 1

    # --force re-runs them
    quiet(bf.process, [rid], "krisp-nc-replay", 2, True, False)
    assert fake.calls[-1] == ["krisp-nc-replay"]
    assert len(store.load_meta(rid)["backfill_round2"]["passes"]) == 2


def check_process_dry_run_and_unavailable(env):
    fresh_runs()
    known, fake = patched(env)
    rid = make_run("x", [])
    quiet(bf.process, [rid], "off-a", 2, False, True)
    assert fake.calls == [] and "backfill_round2" not in store.load_meta(rid)
    known["off-a"]["available"] = False
    known["off-a"]["unavailable_reason"] = "no model"
    raises(SystemExit, bf.process, [rid], "off-a", 2, False, False, contains="no model")
    assert fake.calls == []
    # the live krisp-nc record does not count for the alias, and the offline record
    # of a still-done candidate is skipped without the rail check
    known["off-a"]["available"] = True
    quiet(bf.process, [rid], "off-a", 2, False, False)
    assert fake.calls == [["off-a"]] and rec_of(rid, "off-a")["rail"] == "offline"
    quiet(bf.process, [rid], "off-a", 2, False, False)
    assert fake.calls == [["off-a"]]


def check_live_record_does_not_satisfy_alias(env):
    """Same run, live-rail krisp-nc done: the alias id is still todo."""
    fresh_runs()
    known, fake = patched(env)
    rid = make_run("", [{"id": "krisp-nc", "status": "done", "output": "krisp-nc.wav",
                         "rail": "live", "processed_at": "2026-09-20T10:00:00"}])
    quiet(bf.process, [rid], "krisp-nc-replay", 2, False, False)
    assert fake.calls == [["krisp-nc-replay"]]
    assert rec_of(rid, "krisp-nc")["rail"] == "live" and rec_of(rid, "krisp-nc-replay")["rail"] == "cloud_replay"


def check_process_no_script_run_skips_stt(env):
    """No reference script: the real transcribe is never called, fresh records carry a
    "skipped" stt_error. With a script the real one is used."""
    fresh_runs()
    env.attr(stt, "transcribe", stt.transcribe)   # process() rebinds it and never puts it back
    real = FakeStt({"off-a": "hello", "off-b": "world"})
    env.attr(bf, "_real_transcribe", real)
    known, fake = patched(env)
    fake.use_stt = True
    noscript = make_run("", [])
    quiet(bf.process, [noscript], "off-a,off-b", 2, False, False)
    assert real.calls == [], real.calls
    for cid in ("off-a", "off-b"):
        r = rec_of(noscript, cid)
        assert r["status"] == "done" and r["backfill"] == bf.PASS
        assert r["stt_error"].startswith("skipped") and "stt" not in r, r
    assert server.stt.transcribe is bf._no_script_stt
    scripted = make_run("hello world", [])
    quiet(bf.process, [scripted], "off-a,off-b", 2, False, False)
    assert sorted(real.calls) == ["off-a.wav", "off-b.wav"], real.calls
    r = rec_of(scripted, "off-a")
    assert r["stt"]["text"] == "hello" and "stt_error" not in r
    assert server.stt.transcribe is real, "a scripted run gets the real transcribe back"
    # whitespace-only script counts as none
    blank = make_run("  \n", [])
    n = len(real.calls)
    quiet(bf.process, [blank], "off-a", 2, False, False)
    assert len(real.calls) == n and rec_of(blank, "off-a")["stt_error"].startswith("skipped")


# ================================================================== backfill: 4

def check_retry_cloud_recovers(env):
    fresh_runs()
    known, fake = patched(env, {"cloud-raw": ["boom-1", "boom-2"]})
    rid = make_run("", [])
    quiet(bf.process, [rid], "cloud-raw", 2, False, False)   # no SystemExit
    b = store.load_meta(rid)["backfill_round2"]
    assert [p["attempt"] for p in b["passes"]] == [0, 1, 2], b
    assert list(b["passes"][0]["failed"]) == ["cloud-raw"] and b["passes"][0]["failed"]["cloud-raw"] == "boom-1"
    assert b["passes"][1]["failed"] == {"cloud-raw": "boom-2"} and b["passes"][2]["failed"] == {}
    assert b["cloud_replay"]["retries"] == 2 and b["cloud_replay"]["failed"] == {}
    r = rec_of(rid, "cloud-raw")
    assert r["status"] == "done" and r["backfill"] == bf.PASS and "last_attempt" not in r
    assert fake.calls == [["cloud-raw"]] * 3


def check_retry_with_existing_record_keeps_it(env):
    """A --force re-run over a good record: the failure lands in last_attempt, the
    retry replaces the record and tags it."""
    fresh_runs()
    known, fake = patched(env, {"cloud-raw": ["boom"]})
    rid = make_run("", [{"id": "cloud-raw", "status": "done", "output": "cloud-raw.wav",
                         "rail": "cloud_replay", "processed_at": "2026-09-01T00:00:00", "marker": 1}])
    quiet(bf.process, [rid], "cloud-raw", 2, True, False)
    b = store.load_meta(rid)["backfill_round2"]
    assert len(b["passes"]) == 2 and b["passes"][0]["failed"] == {"cloud-raw": "boom"}
    r = rec_of(rid, "cloud-raw")
    assert r["backfill"] == bf.PASS and "marker" not in r and "last_attempt" not in r


def check_retry_exhausted_halts(env):
    fresh_runs()
    known, fake = patched(env, {"cloud-raw": ["e1", "e2", "e3", "e4"]})
    rid = make_run("", [])
    e = raises(SystemExit, bf.process, [rid], "cloud-raw", 2, False, False, contains="cloud replay failed 3x")
    assert rid in str(e)
    b = store.load_meta(rid)["backfill_round2"]
    assert len(b["passes"]) == 3 and b["cloud_replay"]["retries"] == 2
    assert b["cloud_replay"]["failed"] == {"cloud-raw": "e3"}, b["cloud_replay"]
    assert len(fake.calls) == 3
    r = rec_of(rid, "cloud-raw")
    assert r["status"] == "error" and "backfill" not in r
    # retries=0: one attempt, halts straight away
    fresh_runs()
    known, fake = patched(env, {"cloud-raw": ["e1"]})
    rid = make_run("", [])
    raises(SystemExit, bf.process, [rid], "cloud-raw", 0, False, False, contains="1x")
    assert store.load_meta(rid)["backfill_round2"]["cloud_replay"]["retries"] == 0


def check_offline_failure_not_retried(env):
    fresh_runs()
    known, fake = patched(env, {"off-a": ["kaboom"]})
    rid = make_run("", [])
    quiet(bf.process, [rid], "off-a,off-b", 2, False, False)   # no halt
    assert fake.calls == [["off-a", "off-b"]], fake.calls
    b = store.load_meta(rid)["backfill_round2"]
    assert len(b["passes"]) == 1 and b["passes"][0]["failed"] == {"off-a": "kaboom"}
    assert "cloud_replay" not in b
    assert rec_of(rid, "off-a")["status"] == "error" and rec_of(rid, "off-b")["backfill"] == bf.PASS


def check_mixed_only_cloud_retried(env):
    fresh_runs()
    known, fake = patched(env, {"off-a": ["kaboom"], "cloud-raw": ["boom"]})
    rid = make_run("", [])
    quiet(bf.process, [rid], "off-a,cloud-raw", 2, False, False)   # cloud recovers, offline is only a warning
    assert fake.calls == [["off-a", "cloud-raw"], ["cloud-raw"]], fake.calls
    b = store.load_meta(rid)["backfill_round2"]
    assert len(b["passes"]) == 2 and b["cloud_replay"] == {**b["cloud_replay"], "retries": 1, "failed": {}}
    assert rec_of(rid, "off-a")["status"] == "error" and rec_of(rid, "cloud-raw")["backfill"] == bf.PASS


def check_offline_failure_warned_after_cloud_retries(env):
    """Offline fails in pass 1, cloud recovers in pass 2: no halt, and the warning
    still names the offline id."""
    fresh_runs()
    known, fake = patched(env, {"off-a": ["kaboom"], "cloud-raw": ["boom"]})
    rid = make_run("", [])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        bf.process([rid], "off-a,cloud-raw", 2, False, False)
    warns = [json.loads(ln) for ln in out.getvalue().splitlines() if '"warn"' in ln]
    assert len(warns) == 1 and warns[0]["run"] == rid and warns[0]["failed"] == {"off-a": "kaboom"}, warns
    assert fake.calls == [["off-a", "cloud-raw"], ["cloud-raw"]]
    # nothing failed: no warning
    fresh_runs()
    known, fake = patched(env)
    rid = make_run("", [])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        bf.process([rid], "off-a,cloud-raw", 2, False, False)
    assert '"warn"' not in out.getvalue()


# ================================================================== backfill: 5

class FakeStt:
    def __init__(self, texts: dict[str, str | Exception]):
        self.texts = texts
        self.calls: list[str] = []

    async def __call__(self, wav: Path):
        self.calls.append(Path(wav).name)
        v = self.texts[Path(wav).stem]
        if isinstance(v, Exception):
            raise v
        return {"text": v, "language": "en"}


def stt_run() -> str:
    cands = [
        {"id": "a", "status": "done", "output": "a.wav", "stt": {"text": "hello word old"},
         "scores": {"wer": {"wer": 0.9}}},
        {"id": "b", "status": "done", "output": "b.wav", "stt": {"text": "already new"},
         "scores": {"wer": {"wer": 0.1}},
         "stt_setting": {"boost_volume": False, "whole_file": True, "pass": bf.PASS}},
        {"id": "c", "status": "done", "output": "c-missing.wav", "stt": {"text": "no file"}},
        {"id": "d", "status": "done", "output": "d.wav", "stt": {"text": "will fail"},
         "scores": {"wer": {"wer": 0.4}}},
        {"id": "e", "status": "error", "error": "x"},      # no output at all
        {"id": "f", "status": "done", "output": "f.wav", "stt": {"text": "no scores"}},
    ]
    return make_run("hello world", cands, files=["a.wav", "b.wav", "d.wav", "f.wav"])


def check_run_stt(env):
    fresh_runs()
    env.attr(config, "WHISPER_BOOST_VOLUME", False)
    fake = FakeStt({"a": "hello word", "d": RuntimeError("stt down"), "f": "hello world"})
    env.attr(bf, "_real_transcribe", fake)
    rid = stt_run()
    b_before = snapshot(rid, "b")
    c_before = snapshot_no_file(rid, "c")

    quiet(bf.run_stt, [rid], False, True)   # dry run: nothing happens
    assert fake.calls == [] and "stt_v1" not in rec_of(rid, "a")

    quiet(bf.run_stt, [rid], False, False)
    assert sorted(fake.calls) == ["a.wav", "d.wav", "f.wav"], fake.calls
    a = rec_of(rid, "a")
    assert a["stt_v1"]["text"] == "hello word old" and a["stt_v1"]["wer"] == {"wer": 0.9}
    assert a["stt_v1"]["setting"] == bf.V1_SETTING and a["stt_v1"]["error"] is None
    assert a["stt"] == {"text": "hello word", "language": "en"}
    assert a["scores"]["wer"] == scoring.wer("hello world", "hello word") and a["scores"]["wer"]["wer"] == 0.5
    assert a["stt_setting"]["boost_volume"] is False and "stt_error" not in a
    # already at the new setting: not re-transcribed, not touched, no stt_v1
    assert snapshot(rid, "b") == b_before and "stt_v1" not in rec_of(rid, "b")
    # missing output file and error records are skipped
    assert snapshot_no_file(rid, "c") == c_before and "stt_v1" not in rec_of(rid, "c")
    assert "stt_v1" not in rec_of(rid, "e") and "stt_setting" not in rec_of(rid, "e")
    # transcribe raised: stt_error set, old transcript preserved in stt_v1, current stt dropped
    d = rec_of(rid, "d")
    assert d["stt_error"] == "stt down" and "stt" not in d
    assert d["stt_v1"]["text"] == "will fail" and d["stt_v1"]["wer"] == {"wer": 0.4}
    assert d["scores"]["wer"] == scoring.wer("hello world", "")
    assert "stt_setting" not in d, "an errored record is not stamped, so it stays retryable"
    # no scores dict: still transcribed, and no scores key invented
    f = rec_of(rid, "f")
    assert f["stt"]["text"] == "hello world" and f["stt_v1"]["wer"] is None and "scores" not in f

    # a second plain run retries only the errored record (still failing here) and
    # touches nothing else, overwriting no stt_v1
    others = {c: snapshot(rid, c) for c in ("a", "b", "f")}
    fake.calls.clear()
    quiet(bf.run_stt, [rid], False, False)
    assert fake.calls == ["d.wav"], fake.calls
    assert {c: snapshot(rid, c) for c in others} == others
    d = rec_of(rid, "d")
    assert d["stt_error"] == "stt down" and d["stt_v1"]["text"] == "will fail" and "stt_setting" not in d
    # it recovers: stamped, stt_error gone, stt_v1 still the original v1 transcript
    fake.texts["d"] = "hello world"
    fake.calls.clear()
    quiet(bf.run_stt, [rid], False, False)
    assert fake.calls == ["d.wav"], fake.calls
    d = rec_of(rid, "d")
    assert "stt_error" not in d and d["stt"]["text"] == "hello world" and d["stt_setting"]["boost_volume"] is False
    assert d["stt_v1"]["text"] == "will fail" and d["stt_v1"]["wer"] == {"wer": 0.4}
    assert d["scores"]["wer"]["wer"] == 0.0
    # now everything is stamped: idempotent
    dump = json.dumps(store.load_meta(rid)["candidates"], sort_keys=True)
    fake.calls.clear()
    quiet(bf.run_stt, [rid], False, False)
    assert fake.calls == []
    assert json.dumps(store.load_meta(rid)["candidates"], sort_keys=True) == dump
    # --force re-transcribes the tagged ones but never rewrites stt_v1
    fake.texts.update({"a": "different", "b": "b text", "d": "recovered", "f": "x"})
    fake.calls.clear()
    quiet(bf.run_stt, [rid], True, False)
    assert sorted(fake.calls) == ["a.wav", "b.wav", "d.wav", "f.wav"], fake.calls
    a2 = rec_of(rid, "a")
    assert a2["stt_v1"]["text"] == "hello word old" and a2["stt"]["text"] == "different"
    assert "stt_v1" not in rec_of(rid, "b"), "a record already at the new setting never gets an stt_v1"
    assert rec_of(rid, "d")["stt_v1"]["text"] == "will fail" and "stt_error" not in rec_of(rid, "d")


def snapshot_no_file(rid: str, cid: str) -> str:
    return json.dumps(rec_of(rid, cid), sort_keys=True)


def check_run_stt_errored_record_is_retried(env):
    """An STT error leaves the record unstamped, so a plain re-run retries it."""
    fresh_runs()
    env.attr(config, "WHISPER_BOOST_VOLUME", False)
    fake = FakeStt({"a": RuntimeError("down")})
    env.attr(bf, "_real_transcribe", fake)
    rid = make_run("hello world", [{"id": "a", "status": "done", "output": "a.wav", "stt": {"text": "t"}}])
    quiet(bf.run_stt, [rid], False, False)
    r = rec_of(rid, "a")
    assert r["stt_error"] == "down" and "stt_setting" not in r and r["stt_v1"]["text"] == "t"
    fake.texts["a"] = "hello world"
    quiet(bf.run_stt, [rid], False, False)
    assert fake.calls == ["a.wav", "a.wav"], fake.calls
    r = rec_of(rid, "a")
    assert "stt_error" not in r and r["stt"]["text"] == "hello world" and r["stt_setting"]["boost_volume"] is False
    assert r["stt_v1"]["text"] == "t", "stt_v1 is written once"
    quiet(bf.run_stt, [rid], False, False)
    assert len(fake.calls) == 2


def check_run_stt_skips_and_refusals(env):
    fresh_runs()
    fake = FakeStt({"a": "hello"})
    env.attr(bf, "_real_transcribe", fake)
    env.attr(config, "WHISPER_BOOST_VOLUME", False)
    cand = {"id": "a", "status": "done", "output": "a.wav", "stt": {"text": "t"}}
    noscript = make_run("", [dict(cand)])
    blank = make_run("   \n", [dict(cand)])
    quiet(bf.run_stt, [noscript, blank], False, False)
    assert fake.calls == []
    assert "stt_v1" not in rec_of(noscript, "a") and rec_of(blank, "a")["stt"] == {"text": "t"}
    # boost on: refuse before reading anything
    env.attr(config, "WHISPER_BOOST_VOLUME", True)
    rid = make_run("hello", [dict(cand)])
    raises(SystemExit, bf.run_stt, [rid], False, False, contains="WHISPER_BOOST_VOLUME")
    raises(SystemExit, bf.run_stt, [], True, True, contains="WHISPER_BOOST_VOLUME")
    assert fake.calls == [] and rec_of(rid, "a")["stt"] == {"text": "t"}


def check_run_stt_reprocessed_meanwhile(env):
    """If the candidate's output changed while transcribing, the transcript is dropped."""
    fresh_runs()
    env.attr(config, "WHISPER_BOOST_VOLUME", False)
    rid = make_run("hello world", [{"id": "a", "status": "done", "output": "a.wav", "stt": {"text": "t"}}])

    async def swap(wav):
        with store.locked_meta(rid) as m:
            m["candidates"][0]["output"] = "a2.wav"
        return {"text": "of the old audio"}

    env.attr(bf, "_real_transcribe", swap)
    quiet(bf.run_stt, [rid], False, False)
    r = rec_of(rid, "a")
    assert r["stt"] == {"text": "t"} and "stt_v1" not in r and "stt_setting" not in r


# ============================================================ import_incidents: 6-8

def noise(n: int, seed: int, amp: float = 3000) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * amp).astype(np.int16)


def check_ncc_locate():
    rate = 16000
    clip = noise(int(0.6 * rate), 1, 8000)
    for offset in (0, 12345, 25000):
        track = noise(3 * rate, 2, 500)
        track[offset: offset + len(clip)] = (clip * 0.3).astype(np.int16)   # a different gain
        k, v = ii._ncc_locate(track, clip)
        assert abs(k - offset) <= 1 and v > 0.99, (offset, k, v)
    # at the exact same gain and with an int16 clip against an int16 track
    track = noise(rate, 3, 500)
    track[4000: 4000 + len(clip) // 2] = clip[: len(clip) // 2]
    k, v = ii._ncc_locate(track, clip[: len(clip) // 2])
    assert abs(k - 4000) <= 1 and v > 0.99, (k, v)
    # no clip in there: the score is clearly lower than a real match
    k, v = ii._ncc_locate(noise(2 * rate, 5, 500), clip)
    assert v < 0.5, v


def burst(x: np.ndarray, rate: int, a: float, b: float, amp: float = 3000, seed: int = 0) -> None:
    i, j = round(a * rate), round(b * rate)
    x[i:j] = noise(j - i, seed + int(a * 1000), amp)


def check_active():
    rate = 16000
    x = np.zeros(3 * rate, dtype=np.int16)
    assert ii._active(x, rate, 0.0, 3.0) == []                    # digital silence
    burst(x, rate, 1.0, 1.3)
    out = ii._active(x, rate, 0.0, 3.0)
    assert len(out) == 1 and abs(out[0][0] - 1.0) <= 0.02 and abs(out[0][1] - 1.3) <= 0.02, out
    # a window offset changes the origin but not the answer
    out2 = ii._active(x, rate, 0.5, 2.0)
    assert len(out2) == 1 and abs(out2[0][0] - 1.0) <= 0.02 and abs(out2[0][1] - 1.3) <= 0.02, out2
    # a region outside [a0, a1) is not reported
    assert ii._active(x, rate, 2.0, 3.0) == []
    # two bursts 100 ms apart merge, 300 ms apart do not
    y = np.zeros(3 * rate, dtype=np.int16)
    burst(y, rate, 1.0, 1.2)
    burst(y, rate, 1.3, 1.5)
    m = ii._active(y, rate, 0.0, 3.0)
    assert len(m) == 1 and abs(m[0][0] - 1.0) <= 0.02 and abs(m[0][1] - 1.5) <= 0.02, m
    z = np.zeros(3 * rate, dtype=np.int16)
    burst(z, rate, 1.0, 1.2)
    burst(z, rate, 1.5, 1.7)
    s = ii._active(z, rate, 0.0, 3.0)
    assert len(s) == 2 and abs(s[0][1] - 1.2) <= 0.02 and abs(s[1][0] - 1.5) <= 0.02, s
    # the threshold: a -70 dBFS burst is below the default -60, above a -80 one
    q = np.zeros(2 * rate, dtype=np.int16)
    burst(q, rate, 0.5, 0.8, amp=10)     # ~ -70 dBFS RMS
    assert ii._active(q, rate, 0, 2) == [] and len(ii._active(q, rate, 0, 2, thr_dbfs=-80.0)) == 1
    # 24 kHz works the same
    r24 = 24000
    w = np.zeros(2 * r24, dtype=np.int16)
    burst(w, r24, 0.4, 0.6)
    o = ii._active(w, r24, 0, 2)
    assert len(o) == 1 and abs(o[0][0] - 0.4) <= 0.02 and abs(o[0][1] - 0.6) <= 0.02, o


def spec(**over) -> dict:
    s = {"id": "PR-T", "truth": [[1.00049, 2.0]], "nonspeech": [{"name": "click", "start": 3.0, "end": 3.2}],
         "ignore": [[4.0, 4.5049]], "provenance": {"call_log": "abc", "room": "r"}}
    s.update(over)
    return s


def check_labels():
    meta = {"input": {"duration_s": 6.0},
            "candidates": [{"id": "x", "vad": {"segments": [[1.0, 2.0], [3.0, 3.1]]}},
                           {"id": "y", "vad": {"error": "boom"}}, {"id": "z"}]}
    ii._labels(meta, spec())
    assert meta["truth_spans"] == [[1.0, 2.0]]
    assert meta["known_nonspeech"] == [{"name": "click", "start": 3.0, "end": 3.2}]
    assert meta["ignore_spans"] == [[4.0, 4.505]]
    inc = meta["incident"]
    assert inc["id"] == "PR-T" and inc["call_log"] == "abc" and inc["room"] == "r" and inc["labelled_at"]
    assert meta["candidates"][0]["truth_score"] is not None       # score_run ran
    assert meta["candidates"][1]["truth_score"] is None
    # empty truth (PR-54 B): all three keys written, nothing crashes
    m2 = {"input": {"duration_s": 6.0}, "candidates": []}
    ii._labels(m2, spec(truth=[], ignore=[]))
    assert m2["truth_spans"] == [] and m2["ignore_spans"] == [] and m2["known_nonspeech"][0]["name"] == "click"
    # overlapping classes: refuse, and write nothing
    for bad in (spec(ignore=[[1.5, 2.5]]),                                    # speech vs ignore
                spec(nonspeech=[{"name": "n", "start": 1.9, "end": 2.4}]),    # speech vs nonspeech
                spec(ignore=[[3.1, 3.5]])):                                   # nonspeech vs ignore
        m3 = {"input": {"duration_s": 6.0}, "candidates": []}
        e = raises(SystemExit, ii._labels, m3, bad, contains="PR-T: label classes overlap")
        assert "overlaps" in str(e)
        assert "truth_spans" not in m3 and "incident" not in m3
    # touching edges are not an overlap (strict)
    m4 = {"input": {"duration_s": 6.0}, "candidates": []}
    ii._labels(m4, spec(ignore=[[2.0, 2.5]]))


def check_find():
    fresh_runs()
    assert ii._find("PR-T") is None
    r1, _ = store.new_run("phone", "one")
    r2, d2 = store.new_run("phone", "two")
    with store.locked_meta(r1) as m:
        m["incident"] = {"id": "PR-OTHER"}
    with store.locked_meta(r2) as m:
        m["incident"] = {"id": "PR-T"}
    (config.RUNS_DIR / "empty-dir").mkdir()
    bad = config.RUNS_DIR / "0000-corrupt"
    bad.mkdir()
    (bad / "meta.json").write_text("{not json")
    assert ii._find("PR-T") == r2 and ii._find("PR-OTHER") == r1 and ii._find("PR-missing") is None
    # labelling then finding round-trips through a real saved meta
    with store.locked_meta(r1) as m:
        m["input"] = {"duration_s": 6.0}
        ii._labels(m, spec(id="PR-RT"))
    assert ii._find("PR-RT") == r1 and ii._find("PR-OTHER") is None


# ====================================================================== runner

def main() -> None:
    checks = [(n, f) for n, f in globals().items() if n.startswith("check_") and callable(f)]
    if ii is None:
        print(f"SKIP {sorted(_NEEDS_II)}: scripts/import_incidents.py is not in this checkout")
        checks = [(n, f) for n, f in checks if n not in _NEEDS_II]
    started = time.perf_counter()
    failed = []
    try:
        for name, fn in checks:
            env = Env()
            t0 = time.perf_counter()
            try:
                fn(env) if fn.__code__.co_varnames[: fn.__code__.co_argcount] == ("env",) else fn()
            except (Exception, SystemExit):  # noqa: BLE001
                failed.append(name)
                print(f"FAIL {name} ({time.perf_counter() - t0:.1f}s)\n{traceback.format_exc()}")
            else:
                print(f"OK {name} ({time.perf_counter() - t0:.1f}s)")
            finally:
                env.close()
    finally:
        shutil.rmtree(_SANDBOX, ignore_errors=True)
    total = time.perf_counter() - started
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed, {len(failed)} failed, {total:.1f}s")
    if failed:
        sys.exit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
