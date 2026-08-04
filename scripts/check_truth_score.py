"""Does "VAD vs my marked speech" actually mean what the column says?

Worth a script because the number is a single percentage that will be used to
rank candidates, and every way it can be wrong still produces a plausible
percentage: swap miss for false-alarm and an onset-shaving chain looks clean;
get the frame rounding wrong and every candidate drifts the same direction so the
ranking still looks sensible. The degenerate cases below pin it from both ends.

    .venv/bin/python scripts/check_truth_score.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nc_bench import store, vad  # noqa: E402
from nc_bench.server import _merge_spans  # noqa: E402


def close(a, b, tol=0.02):
    return abs(a - b) <= tol


def main() -> None:
    DUR = 10.0
    TRUTH = [[1.0, 3.0], [6.0, 8.0]]  # 4 s of speech in a 10 s file
    MARKED = 4.0

    # --- 1. perfect agreement ---
    s = vad.score_spans(TRUTH, TRUTH, DUR)
    assert close(s["agree"], 1.0), s
    assert close(s["miss_s"], 0.0) and close(s["fa_s"], 0.0), s
    assert close(s["f1"], 1.0) and close(s["marked_s"], MARKED), s
    print(f"OK   identical spans      agree {s['agree']:.3f}  miss {s['miss_s']}  fa {s['fa_s']}")

    # --- 2. VAD found nothing: all marked speech is missed, no false alarms.
    #        This is the case that matters most — a chain that silences you.
    s = vad.score_spans(TRUTH, [], DUR)
    assert close(s["miss_s"], MARKED), s
    assert close(s["fa_s"], 0.0), s
    assert close(s["agree"], (DUR - MARKED) / DUR), s
    print(f"OK   VAD found nothing    agree {s['agree']:.3f}  miss {s['miss_s']}  fa {s['fa_s']}")

    # --- 3. VAD says the whole file is speech: nothing missed, all silence is a
    #        false alarm. The mirror of case 2 — if these two are swapped
    #        anywhere, this assertion is what catches it.
    s = vad.score_spans(TRUTH, [[0.0, DUR]], DUR)
    assert close(s["miss_s"], 0.0), s
    assert close(s["fa_s"], DUR - MARKED), s
    print(f"OK   VAD says all speech  agree {s['agree']:.3f}  miss {s['miss_s']}  fa {s['fa_s']}")

    # --- 4. half of each marked region found: exactly half missed ---
    s = vad.score_spans(TRUTH, [[1.0, 2.0], [6.0, 7.0]], DUR)
    assert close(s["miss_s"], MARKED / 2), s
    assert close(s["fa_s"], 0.0), s
    print(f"OK   half of each region  agree {s['agree']:.3f}  miss {s['miss_s']}  fa {s['fa_s']}")

    # --- 5. disjoint: VAD speech exactly where I marked silence ---
    s = vad.score_spans(TRUTH, [[3.5, 5.5]], DUR)
    assert close(s["miss_s"], MARKED), s
    assert close(s["fa_s"], 2.0), s
    assert s["f1"] == 0.0, s
    print(f"OK   completely disjoint  agree {s['agree']:.3f}  miss {s['miss_s']}  fa {s['fa_s']}")

    # --- 6. no marks at all -> no score, not a fake 100% ---
    assert vad.score_spans([], TRUTH, DUR) is None
    assert vad.score_spans(TRUTH, None, DUR) is None
    assert vad.score_spans(TRUTH, TRUTH, 0) is None
    print("OK   unmarked / unmeasured runs score None rather than a misleading 100%")

    # --- 7. a shifted VAD must be penalised, and by roughly the shift.
    #        Guards the whole point of check_vad.py from the other side: if span
    #        bounds ever drift again, this column has to notice.
    for shift in (0.1, 0.5):
        moved = [[a + shift, b + shift] for a, b in TRUTH]
        s = vad.score_spans(TRUTH, moved, DUR)
        expect = 2 * shift * len(TRUTH)  # each region loses its head and gains a tail
        assert close(s["miss_s"] + s["fa_s"], expect, 0.05), (shift, s)
        print(f"OK   VAD shifted {shift:+.1f}s     agree {s['agree']:.3f}  "
              f"miss {s['miss_s']}  fa {s['fa_s']}")

    # --- 8. score_run wires it to real stored data, and stale scores clear ---
    target = None
    for meta in store.list_runs():
        if any(c.get("vad", {}).get("segments") for c in meta.get("candidates") or []):
            target = meta
            break
    if target is None:
        print("\nSKIP: no run with stored VAD spans to exercise score_run")
        return

    dur = float((target.get("input") or {}).get("duration_s") or 0)
    base = next((c for c in target["candidates"] if c["id"] == "none"), None)
    # Stand in for hand marks with the passthrough's own spans: not a real ground
    # truth, but it makes "how far is each candidate from the untouched leg"
    # exactly computable, so score_run's plumbing is verifiable end to end.
    target["truth_spans"] = _merge_spans(base["vad"]["segments"], dur)
    vad.score_run(target)
    assert base["truth_score"]["agree"] == 1.0, "passthrough vs its own spans is not 1.0"
    scored = [c for c in target["candidates"] if c.get("truth_score")]
    assert len(scored) > 1, "score_run only scored one candidate"
    print(f"\nOK   score_run scored {len(scored)} candidates on run {target['id']}")
    worst = sorted(scored, key=lambda c: c["truth_score"]["agree"])[:3]
    for c in worst:
        t = c["truth_score"]
        print(f"       {c['id']:<32} agree {t['agree']:.3f}  missed {t['miss_s']:.2f}s")

    # clearing the marks must clear the scores, not leave the old ones behind
    target["truth_spans"] = []
    vad.score_run(target)
    assert all(c.get("truth_score") is None for c in target["candidates"]), (
        "clearing the marks left stale scores on disk"
    )
    print("OK   clearing the marks clears every stored score")

    print("\ncheck_truth_score passed")


if __name__ == "__main__":
    main()
