"""Headless live-rail check: publish synthetic tone+noise into a real
LiveKit Cloud room (as the 'web user') and let the Recorder capture the raw
subscriber stream plus one Krisp live-rail stream. Verifies room join, track
subscribe, Cloud-authenticated NC AudioStream, and buffer plumbing.

Run: .venv/bin/python scripts/live_loopback_test.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit import rtc  # noqa: E402

from nc_bench.recorder import RECORD_RATE, Recorder  # noqa: E402

DUR = 5.0


async def main() -> None:
    events = []

    async def emit(ev):
        if ev.get("type") == "session":
            print("event:", ev)
        events.append(ev)

    from nc_bench import lk_cloud

    lk_cloud.preload()
    lk_model = sys.argv[1] if len(sys.argv) > 1 else "NC"  # e.g. BVC or AIC:QUAIL_L
    cid = f"live-{lk_model.replace(':', '-').lower()}"
    rec = Recorder(emit=emit, live_candidates=[{"id": cid, "lk_model": lk_model}])
    join = await rec.start_web()
    print("room:", join["room"])

    # publish tone+noise as the web user (what the browser would do)
    pub_room = rtc.Room()
    await pub_room.connect(join["livekit_url"], join["token"])
    source = rtc.AudioSource(RECORD_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await pub_room.local_participant.publish_track(track)

    n_frame = RECORD_RATE // 100  # 10 ms
    t_all = np.arange(int(RECORD_RATE * DUR)) / RECORD_RATE
    tone = 0.3 * np.sin(2 * np.pi * 300 * t_all) * (np.sin(2 * np.pi * 1.5 * t_all) > 0)
    noise = 0.05 * np.random.default_rng(3).standard_normal(len(t_all))
    audio = ((tone + noise) * 32767).astype(np.int16)
    for i in range(0, len(audio) - n_frame, n_frame):
        frame = rtc.AudioFrame(
            data=audio[i : i + n_frame].tobytes(),
            sample_rate=RECORD_RATE,
            num_channels=1,
            samples_per_channel=n_frame,
        )
        await source.capture_frame(frame)
    await asyncio.sleep(0.5)

    tmp = Path(tempfile.mkdtemp(prefix="nc-loopback-"))
    input_meta, live = await rec.stop(tmp / "input.wav")
    await pub_room.disconnect()

    print("input:", input_meta)
    for cid, res in live.items():
        if "error" in res:
            print(f"live {cid}: ERROR {res['error']}")
        else:
            a = res["audio"]
            print(f"live {cid}: {len(a) / RECORD_RATE:.2f}s captured, rms={np.sqrt((a.astype(float) ** 2).mean()):.0f}")

    assert input_meta["file"] is not None, "raw recording is empty"
    assert input_meta["duration_s"] > DUR * 0.6, f"raw too short: {input_meta['duration_s']}"
    krisp = live[cid]
    assert "audio" in krisp, f"live NC stream failed: {krisp.get('error')}"
    assert len(krisp["audio"]) > RECORD_RATE * DUR * 0.6, "live NC capture too short"

    # if the Cloud filter failed to initialize the stream silently passes
    # through raw audio — catch that by requiring the output to differ
    import soundfile as sf

    raw, _ = sf.read(tmp / "input.wav", dtype="int16")
    raw_rms = np.sqrt((raw.astype(float) ** 2).mean())
    k_rms = np.sqrt((krisp["audio"].astype(float) ** 2).mean())
    print(f"raw rms={raw_rms:.0f}  krisp rms={k_rms:.0f}")
    assert abs(k_rms - raw_rms) / max(raw_rms, 1) > 0.02, (
        "krisp output is (near-)identical to raw — Cloud NC filter did not "
        "initialize (check the [livekit-nc] warning in stderr)"
    )
    levels = [e for e in events if e.get("type") == "level"]
    assert levels, "no level events emitted"
    print(f"\nloopback test passed ({len(levels)} level events)")


if __name__ == "__main__":
    asyncio.run(main())
