"""Assert-based self-check: run synthetic tone+noise through every available
candidate chain and verify shape, level, and noise reduction. No server, no
network beyond Hecttor's license check. Run: .venv/bin/python scripts/selfcheck.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

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
        if cand["chain"]:  # any real NC should cut noise in the gaps
            assert g < base_gap, f"{cand['id']}: gap rms {g:.4f} not below input {base_gap:.4f}"
        print(
            f"OK   {cand['id']}: {timing['proc_ms']} ms, rtf={timing['rtf']}, "
            f"gap-rms {base_gap:.4f} -> {g:.4f}"
        )
        ran += 1

    assert ran >= 1, "no candidate could run"

    # ---- scoring stack ----
    from nc_bench import scoring

    # WER: exact metric, exact expectations
    w = scoring.wer("the quick brown fox", "the quick brown fox")
    assert w["wer"] == 0.0, w
    w = scoring.wer("the quick brown fox", "the quack fox jumps")
    assert w["wer"] == 0.75 and w["ref_words"] == 4, w  # sub, del, ins
    assert scoring.wer("", "anything") is None

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
