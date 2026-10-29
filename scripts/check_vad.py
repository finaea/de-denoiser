"""Do the stored VAD spans actually land on the speech?

Worth a script because the span bounds are derived (END_OF_SPEECH carries an end
timestamp and a duration, not a start), and a systematic offset would be
invisible: the highlight would look plausible on every waveform while sitting a
fixed distance from the audio it claims to mark. Every downstream reading — "NC
ate this word's onset", "NC added a turn" — would then be wrong in the same
direction, which is worse than no highlight at all.

Uses real speech, not tones: silero scores synthetic tones near zero, so a tone
test would pass by finding nothing.

    .venv/bin/python scripts/check_vad.py
"""

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import config, store, vad  # noqa: E402

PAD_S = 3.0  # silence welded on each side; spans must stay inside the speech


def _find_speech() -> Path | None:
    """A real recorded run's passthrough — known-good speech with known bounds.

    Skips `_`-prefixed scratch copies: `_` sorts after every dated id, so
    without this a scratch copy of some older run would silently become the
    speech under test.
    """
    for d in sorted(config.RUNS_DIR.iterdir(), reverse=True):
        if d.name.startswith("_"):
            continue
        wav = d / "none.wav"
        if wav.is_file() and wav.stat().st_size > 300_000:
            return wav
    return None


SILERO_V60_SHA = "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004"


def _reproduce(wav: Path, stored: dict, meta: dict) -> tuple[dict, str]:
    """Stored spans re-derived through the VAD bench with the ONNX that made them.

    A record written before 2026-09-29 carries no onnx_sha256; every such
    record came from livekit-plugins-silero 1.6.7 (uv.lock since 2026-08-01),
    which bundles Silero v6.0, so that is the model assumed for it.
    """
    from nc_bench import vads
    from nc_bench.vads.segmenter import PLUGIN_ALPHA, segment, summary

    sha = stored.get("onnx_sha256") or SILERO_V60_SHA
    p = dict(stored["params"])
    p.setdefault("sample_rate", 16000)   # recorded before the rate was selectable
    p = vad.clean(p, meta.get("source"))
    fam = next((e["family"] for e in vads.registry() if e["loader"] == "silero_onnx"
                and sha in {s.get("sha256") for s in (e.get("files") or {}).values()}), None)
    assert fam, f"no Silero in vad_candidates.json has the stored ONNX sha {sha[:8]}…"
    vid = f"{fam}-{p['sample_rate'] // 1000}k"
    data, rate = sf.read(wav, dtype="int16", always_2d=True)
    x = np.ascontiguousarray(data[:, 0])
    r = vads.infer(vid, x, rate)
    segs = segment(r["probs"], r["hop_s"], t0_s=r["t0_s"], duration_s=len(x) / rate,
                   smoothing=PLUGIN_ALPHA, **{k: p[k] for k in (
                       "activation_threshold", "deactivation_threshold",
                       "min_speech_duration", "min_silence_duration")})
    how = "stored sha" if stored.get("onnx_sha256") else "inferred from lock"
    return {"segments": segs, **summary(segs, len(x) / rate)}, f"{vid} ({how})"


async def main() -> None:
    src = _find_speech()
    if src is None:
        print("SKIP: no recorded run to use as speech")
        return
    if not config.VAD_ENABLED:
        print("SKIP: VAD_ENABLED is false")
        return

    speech, rate = sf.read(src, dtype="float32", always_2d=True)
    speech = speech[:, 0]
    pad = np.zeros(int(PAD_S * rate), dtype=np.float32)
    padded = np.concatenate([pad, speech, pad])
    tmp = Path(tempfile.mkdtemp(prefix="vadcheck-"))
    try:
        probe = tmp / "padded.wav"
        sf.write(probe, padded, rate)

        base = await vad.analyze(src)
        shifted = await vad.analyze(probe)
        assert base and shifted, "VAD returned nothing"

        print(f"source        {src.name}  {base['duration_s']}s")
        print(f"  spans {base['n']:>3}  speech {base['speech_s']:>7.2f}s  "
              f"coverage {base['coverage']:.1%}")
        print(f"padded (+{PAD_S}s silence each side)  {shifted['duration_s']}s")
        print(f"  spans {shifted['n']:>3}  speech {shifted['speech_s']:>7.2f}s  "
              f"coverage {shifted['coverage']:.1%}")

        assert base["n"] > 0, "no speech found in a real recording"

        # 1. every span sits inside the file
        for s, e in shifted["segments"]:
            assert 0 <= s < e <= shifted["duration_s"] + 0.01, f"span {s}-{e} out of bounds"
        print("OK   every span is inside the file and non-empty")

        # 2. NOTHING may be marked in the welded-on silence. prefix padding
        #    back-dates a start by up to VAD_PREFIX_PADDING_DURATION, so the
        #    leading bound allows exactly that much and no more.
        slack = config.VAD_PREFIX_PADDING_DURATION + 0.05
        first = min(s for s, _ in shifted["segments"])
        last = max(e for _, e in shifted["segments"])
        print(f"     first span starts {first:.2f}s (silence ends {PAD_S}s, "
              f"prefix padding allows {slack:.2f}s back-dating)")
        print(f"     last span ends    {last:.2f}s (speech ends "
              f"{PAD_S + len(speech) / rate:.2f}s)")
        assert first >= PAD_S - slack, (
            f"a span starts {PAD_S - first:.2f}s into pure silence — the bounds are shifted"
        )
        assert last <= PAD_S + len(speech) / rate + 0.35, (
            f"a span runs {last - (PAD_S + len(speech) / rate):.2f}s past the end of "
            "the speech — the bounds are shifted"
        )
        print("OK   no span leaks into the silence padding (bounds are not shifted)")

        # 3. padding must not change what was found in the speech itself
        assert abs(shifted["speech_s"] - base["speech_s"]) < 0.8, (
            f"padding changed total speech by "
            f"{abs(shifted['speech_s'] - base['speech_s']):.2f}s"
        )
        print("OK   total speech is stable when the same audio is re-positioned")

        # 4. audio that ENDS MID-SPEECH must still report that final turn.
        #    No END_OF_SPEECH ever fires for a turn the recording cuts off, so
        #    reading only END events drops it — silently, and completely: a 13 s
        #    web call that is 68% speech by probability reported zero spans.
        #    Cut 60% of the way into the LONGEST span the VAD itself found, so the
        #    file provably ends mid-speech — truncating at "the last loud sample"
        #    is not good enough, that tail can be breath or noise the VAD rightly
        #    ignores, and the test then passes for the wrong reason.
        s0, e0 = max(base["segments"], key=lambda se: se[1] - se[0])
        cut_at = s0 + 0.6 * (e0 - s0)
        cut = speech[: int(cut_at * rate)]
        mid = tmp / "cut-mid-speech.wav"
        sf.write(mid, cut, rate)
        cm = await vad.analyze(mid)
        cut_dur = len(cut) / rate
        print(f"     cut inside a detected span ({s0:.2f}-{e0:.2f}s) at {cut_at:.2f}s")
        assert cm["n"] > 0, "a file ending mid-speech reported no spans at all"
        last_end = max(e for _, e in cm["segments"])
        print(f"     ends mid-speech ({cut_dur:.2f}s, no trailing silence): "
              f"{cm['n']} spans, last ends {last_end:.2f}s")
        assert last_end >= cut_dur - 0.25, (
            f"the final turn was dropped: last span ends {last_end:.2f}s of {cut_dur:.2f}s"
        )
        print("OK   the turn a recording cuts off is still reported")

        # 5. silence alone must produce nothing
        quiet = tmp / "quiet.wav"
        sf.write(quiet, np.zeros(int(5 * rate), dtype=np.float32), rate)
        q = await vad.analyze(quiet)
        assert q["n"] == 0, f"found {q['n']} spans in 5 s of digital silence"
        print("OK   5 s of silence yields no spans")

        # 6. stored spans on disk still match a fresh analysis at the SAME params.
        #    Re-derived from what the run actually stored rather than from
        #    defaults(): the rate now follows the run's source, so a phone run's
        #    stored spans are not expected to match the global default.
        meta = store.load_meta(src.parent.name)
        stored = next(
            (c.get("vad") for c in meta.get("candidates") or [] if c.get("id") == "none"),
            None,
        )
        if stored and stored.get("params"):
            # Re-derived with the model that PRODUCED the stored spans, not with
            # whatever plugin is installed now: a plugin bump swaps the bundled
            # ONNX (1.6.7 = Silero v6.0, 1.8.3 = v6.2), and comparing across that
            # tests the bump, not the spans. The VAD bench holds every Silero
            # release, and scripts/check_vad_parity.py proves its path equals the
            # plugin's (same ONNX + the plugin's ExpFilter), so it stands in for
            # plugin versions that are no longer installed.
            fresh, model = _reproduce(src, stored, meta)
            # NOT exact equality: the stored spans came from the plugin path,
            # whose in-stream resampling is not repeatable at 8 kHz (the same
            # file gives 2-3 distinct span sets across runs, measured
            # 2026-09-29), so a borderline span can flicker in or out. Tolerate
            # that jitter; anything larger is a real algorithm change.
            d_speech = abs(fresh["speech_s"] - stored["speech_s"])
            d_n = abs(fresh["n"] - stored["n"])
            assert d_speech <= 0.6 and d_n <= 1, (
                f"stored spans drifted beyond inference jitter: "
                f"Δspeech {d_speech:.2f}s, Δspans {d_n} (re-derived with {model})"
            )
            exact = fresh["segments"] == stored["segments"]
            print(f"OK   spans on disk reproduce within inference jitter with {model} "
                  f"(rate {stored['params'].get('sample_rate', 16000)}"
                  f"{', exact' if exact else f', Δ{d_speech:.2f}s/{d_n} span'})")
        else:
            print("     (no stored spans yet — press 're-run VAD on this run')")

        print("\ncheck_vad passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
