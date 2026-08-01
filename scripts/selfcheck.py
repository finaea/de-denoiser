"""Assert-based self-check: run synthetic tone+noise through every available
candidate chain and verify shape, level, and noise reduction. No server, no
network beyond Hecttor's license check. Run: .venv/bin/python scripts/selfcheck.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import config  # noqa: E402
from nc_bench.pipeline import run_chain  # noqa: E402
from nc_bench.processors import chain_available  # noqa: E402
import json  # noqa: E402

RATE = 48_000
DUR = 3.0


def make_input(path: Path) -> None:
    t = np.arange(int(RATE * DUR)) / RATE
    speechish = 0.35 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2.5 * t) > 0)
    rng = np.random.default_rng(7)
    noise = 0.08 * rng.standard_normal(len(t)).astype(np.float32)
    sf.write(path, (speechish + noise).astype(np.float32), RATE)


def gap_rms(path: Path) -> float:
    """RMS in the speech gaps (odd half-cycles of the 2.5 Hz gate)."""
    x, r = sf.read(path, dtype="float32")
    t = np.arange(len(x)) / r
    gaps = np.sin(2 * np.pi * 2.5 * t) <= -0.5  # well inside the silent half
    return float(np.sqrt((x[gaps] ** 2).mean()))


class _EchoSession:
    """Stands in for an ONNX session: returns the frame unchanged + caches as-is."""

    def __init__(self, spec_name: str):
        self._spec_name = spec_name

    def run(self, _outputs, feeds):  # noqa: ANN001 - mirrors ort's signature
        spec = feeds[self._spec_name]
        caches = [v for k, v in feeds.items() if k != self._spec_name]
        return [spec, *caches]


def check_spec_reconstruction() -> None:
    """With the model bypassed, analysis -> OLA must reproduce the input exactly.

    This is what proves the window choice (vorbis / hann-sqrt / hann) and the
    window-squared normalisation are right per model — a wrong window still
    "cuts noise", it just eats the speech with it.
    """
    from nc_bench.processors.spec_onnx import _MODELS, SpecOnnxProcessor, spec_onnx_available

    rng = np.random.default_rng(3)
    checked = 0
    for name in _MODELS:
        ok, why = spec_onnx_available(name)
        if not ok:
            print(f"SKIP recon {name}: {why}")
            continue
        p = SpecOnnxProcessor(name)
        p._sess = _EchoSession(p._spec_name)
        x = (0.2 * rng.standard_normal(p.rate)).astype(np.float32)
        step = max(1, p.rate // 50)  # 20 ms blocks, like the pipeline
        outs = [p.process_block(x[i : i + step]) for i in range(0, len(x), step)]
        outs.append(p.flush())
        y = np.concatenate([o for o in outs if len(o)])
        assert len(y) == len(x), f"{name}: recon length {len(y)} != {len(x)}"
        err = float(np.abs(y - x).max())
        assert err < 1e-4, f"{name}: reconstruction error {err:.2e} (window/OLA wrong?)"
        print(f"OK   recon {name}: max err {err:.2e}")
        checked += 1
    assert checked >= 1, "no spec-onnx model available to check"


# Measured output shift, in ms, of each candidate against its input (2026-07-31,
# probe below). 0 = the wrapper emits sample-aligned audio. Non-zero is the
# model holding audio back internally: DPDFNet's deep-filter stage 40 ms, DTLN's
# canonical overlap-add loop 24 ms, ffmpeg's arnndn one 48 kHz frame.
_EXPECTED_LAG_MS = {
    "fastenhancer-t": 0.0,
    "fastenhancer-s": 0.0,
    "fastenhancer-l": 0.0,
    "gtcrn": 0.0,
    "ulunas": 0.0,
    "dpdfnet2": 40.0,
    "dpdfnet2-8k": 40.0,
    "dtln": 24.0,
    "rnnoise-sh": 10.0,
    # 10 ms of internal shift + its 10 ms frame = the ~20 ms its README claims
    "hush": 10.0,
}


def check_alignment(tmp: Path, candidates: list[dict]) -> None:
    """Pin each candidate's output shift against the input.

    Scoring compares gap windows by time, so a shift that grows silently (a
    forgotten lookahead-skip: FastEnhancer's raw output lags by n_fft - hop,
    256 samples for -T and 412 for -L) would quietly misalign every metric.

    Uses modulated white noise, not the tone-based input above: a tone's
    correlation peak repeats every period, so it cannot resolve a lag.
    """
    from nc_bench.pipeline import run_chain

    rng = np.random.default_rng(11)
    n = int(RATE * 2.5)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * np.arange(n) / RATE)
    x = (0.15 * env * rng.standard_normal(n)).astype(np.float32)
    src = tmp / "align_input.wav"
    sf.write(src, x, RATE)

    by_id = {c["id"]: c for c in candidates}
    for cid, expected_ms in _EXPECTED_LAG_MS.items():
        cand = by_id.get(cid)
        if not cand or not chain_available(cand["chain"])[0]:
            continue
        out = tmp / f"align_{cid}.wav"
        timing = run_chain(src, cand["chain"], out)
        y, r = sf.read(out, dtype="float32")
        xr = soxr.resample(x, RATE, r).astype(np.float32)
        span = min(len(xr), len(y), 2 * r)
        c = np.correlate(y[:span] - y[:span].mean(), xr[:span] - xr[:span].mean(), mode="full")
        lag_ms = (int(np.argmax(c)) - (span - 1)) / r * 1000
        # 2 ms of slack for resampler group delay; a missed skip is 16-26 ms
        assert abs(lag_ms - expected_ms) <= 2, (
            f"{cid}: output shift {lag_ms:.1f} ms, expected {expected_ms:.1f} ms"
        )
        print(
            f"OK   align {cid}: shift {lag_ms:.1f} ms "
            f"(reported algo delay {timing['latency']['algo_delay_ms']} ms)"
        )


def main() -> None:
    candidates = json.loads(config.CANDIDATES_FILE.read_text())
    tmp = Path(tempfile.mkdtemp(prefix="nc-selfcheck-"))
    inp = tmp / "input.wav"
    make_input(inp)
    base_gap = gap_rms(inp)
    print(f"input: {inp}  gap-rms={base_gap:.4f}")

    ran = 0
    for cand in candidates:
        if "lk_model" in cand:
            print(f"SKIP {cand['id']}: live-rail (Cloud) candidate — needs a live session")
            continue
        ok, why = chain_available(cand["chain"])
        if not ok:
            print(f"SKIP {cand['id']}: {why}")
            continue
        out = tmp / f"{cand['id']}.wav"
        try:
            timing = run_chain(inp, cand["chain"], out)
        except RuntimeError as e:
            # e.g. Hecttor license/machine-binding failures — the server shows
            # these as per-candidate errors; here they're a skip, not a crash
            print(f"SKIP {cand['id']}: {e}")
            continue
        x, r = sf.read(out, dtype="float32")
        assert r == config.PIPELINE_RATE, f"{cand['id']}: wrong rate {r}"
        expected = int(DUR * config.PIPELINE_RATE)
        assert abs(len(x) - expected) < config.PIPELINE_RATE * 0.2, (
            f"{cand['id']}: length {len(x)} vs expected {expected}"
        )
        assert np.abs(x).max() > 1e-4, f"{cand['id']}: output is silence"
        g = gap_rms(out)
        if not cand["chain"]:
            # the passthrough candidate is the bar: it already drops the noise
            # above 8 kHz (48k input -> 16k output), so comparing a real NC to
            # the *input* would let a do-nothing chain pass (it did: Hush with
            # atten_lim_db=0 is a passthrough, and only this caught it)
            passthrough_gap = g
        else:
            assert g < 0.9 * passthrough_gap, (
                f"{cand['id']}: gap rms {g:.4f} is not meaningfully below the "
                f"passthrough's {passthrough_gap:.4f} — is this chain doing anything?"
            )
        print(
            f"OK   {cand['id']}: {timing['proc_ms']} ms, rtf={timing['rtf']}, "
            f"gap-rms {base_gap:.4f} -> {g:.4f}"
        )
        ran += 1

    assert ran >= 1, "no candidate could run"

    check_spec_reconstruction()
    check_alignment(tmp, candidates)

    # ---- scoring stack ----
    from nc_bench import scoring

    # WER: exact metric, exact expectations
    w = scoring.wer("the quick brown fox", "the quick brown fox")
    assert w["wer"] == 0.0, w
    w = scoring.wer("the quick brown fox", "the quack fox jumps")
    assert w["wer"] == 0.75 and w["ref_words"] == 4, w  # sub, del, ins
    assert scoring.wer("", "anything") is None

    # Gap windows must never land on speech. A real inbound call (2026-08-01)
    # had silero score p=0.02 during -16 dBFS talking, so 1.4 s of speech was
    # called a gap and every "noise cut" then measured speech destruction.
    fs = config.PIPELINE_RATE
    rng2 = np.random.default_rng(9)
    quiet = (1e-4 * rng2.standard_normal(2 * fs)).astype(np.float32)
    t = np.arange(3 * fs) / fs
    loud = (0.4 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )
    probe = np.concatenate([quiet, loud, quiet])
    loud_span = (2.0, 5.0)
    for g0, g1 in scoring.gap_windows(probe):
        assert g1 <= loud_span[0] or g0 >= loud_span[1], (
            f"gap ({g0}, {g1}) overlaps the loud region {loud_span} — gap gating is inverted"
        )
    seg_db = scoring.gap_rms_db(probe, scoring.gap_windows(probe))
    assert seg_db is not None and seg_db < -60, f"quiet gaps read {seg_db} dBFS"
    # ...and it must abstain rather than guess when speech barely clears the
    # noise, which is where a fixed threshold would start reporting nonsense
    t2 = np.arange(8 * fs) / fs
    gate = np.sin(2 * np.pi * 0.25 * t2) > 0
    near = (0.3 * np.sin(2 * np.pi * 180 * t2) * gate
            + 0.19 * rng2.standard_normal(len(t2))).astype(np.float32)  # ~4 dB apart
    assert not scoring.gap_windows(near), "gaps claimed on audio with no usable separation"
    print(f"gap gating OK: gaps avoid the loud region ({seg_db} dBFS), abstains at 4 dB SNR")

    # Measured band: the whole point is telling a phone call from a mic
    # recording when both arrive as 48 kHz wavs, so check both ends.
    rng = np.random.default_rng(5)
    wide = (0.2 * rng.standard_normal(RATE * 2)).astype(np.float32)
    narrow = soxr.resample(soxr.resample(wide, RATE, 8000), 8000, RATE).astype(np.float32)
    bw_wide = scoring.bandwidth_hz(wide, RATE)
    bw_narrow = scoring.bandwidth_hz(narrow, RATE)
    assert bw_wide > 16_000, f"full-band read as {bw_wide} Hz"
    assert 3000 < bw_narrow < 4600, f"8 kHz round trip read as {bw_narrow} Hz"
    print(f"band OK: full-band {bw_wide} Hz, 8 kHz round trip {bw_narrow} Hz")

    # DNSMOS + VAD gap plumbing on the synthetic input (semantics need real
    # speech; here we assert shapes, ranges, and that gaps exist in tone+noise)
    s_in = scoring.score_input(inp)
    d = s_in["dnsmos"]
    assert all(1.0 <= d[k] <= 5.0 for k in ("sig", "bak", "ovrl")), d
    assert s_in["gap_rms_db"] is None or s_in["gap_rms_db"] < 0
    out_scores = scoring.score_output(
        tmp / "dtln.wav", s_in, reference_script="hello world", transcript="hello word"
    )
    assert "dnsmos" in out_scores and out_scores["wer"]["wer"] == 0.5
    print(f"scoring OK: input dnsmos={d}, gaps={s_in['gap_total_s']}s, "
          f"dtln noise_reduction_db={out_scores['noise_reduction_db']}")

    print(f"\nself-check passed ({ran} candidates ran)")


if __name__ == "__main__":
    main()
