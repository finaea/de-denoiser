"""Prove the recorder does not start until the call is actually answered.

A LiveKit SIP participant joins the room while the phone is still RINGING and
can publish early media (ringback), so `track_subscribed` is not proof of an
answer. Without the gate a phone run opens with several seconds of ring tone,
which lands in the recording, gets scored, and quietly corrupts every candidate.

Simulates that exact shape with a normal publisher standing in for the SIP leg:
publish a track, send LOUD tone while "ringing", open the gate, then send a
QUIET tone as "speech". The recording must contain only the quiet tone.

    .venv/bin/python scripts/check_answer_gate.py
"""

import asyncio
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit import api, rtc  # noqa: E402

from nc_bench import config, recorder  # noqa: E402

RATE = 48_000
RING_S, TALK_S = 3.0, 4.0
RING_AMP, TALK_AMP = 0.50, 0.05


def _http() -> str:
    return config.LIVEKIT_URL.replace("wss://", "https://").replace("ws://", "http://")


async def main() -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not getattr(config, name):
            print(f"SKIP: {name} is not set in .env")
            return

    # Unique agent name and room: a bench server on :8777 registers the same
    # default agent name, and LiveKit load-balances the job to an arbitrary
    # worker — the other one drops it (not armed) and this test would see
    # nothing at all. Do not "simplify" this to the default name.
    tag = uuid.uuid4().hex[:6]
    recorder.AGENT_NAME = f"nc-gate-probe-{tag}"
    room_name, ident = f"nc-answer-gate-{tag}", "sip_callee"

    lk = api.LiveKitAPI(_http(), config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
    await lk.room.create_room(api.CreateRoomRequest(name=room_name))

    b = recorder._bridge
    b.reset()

    async def emit(ev: dict) -> None:
        if ev.get("type") == "session":
            print(f"     -> {ev.get('state')}")

    b.emit = emit
    b.main_loop = asyncio.get_running_loop()
    b.target_room, b.target_identity, b.armed = room_name, ident, True

    recorder.ensure_worker()
    await asyncio.sleep(3)
    await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(agent_name=recorder.AGENT_NAME, room=room_name)
    )

    token = (
        api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity(ident)
        .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True))
        .to_jwt()
    )
    room = rtc.Room()
    await room.connect(config.LIVEKIT_URL, token)
    src = rtc.AudioSource(RATE, 1)
    await room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track("sip", src),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    for _ in range(40):
        if b.claimed:
            break
        await asyncio.sleep(0.5)
    assert b.claimed, "job never claimed the session — setup failed, not the gate"
    await asyncio.sleep(2)

    async def send(seconds: float, amp: float, freq: float) -> None:
        n, t = RATE // 100, 0
        for _ in range(int(seconds * 100)):
            k = np.arange(t, t + n) / RATE
            t += n
            frame = (amp * 32767 * np.sin(2 * np.pi * freq * k)).astype(np.int16)
            await src.capture_frame(rtc.AudioFrame(frame.tobytes(), RATE, 1, n))
            await asyncio.sleep(0.008)

    print(f"  {RING_S}s ringback at amp {RING_AMP} (gate shut)")
    await send(RING_S, RING_AMP, 440)
    during_ring = sum(len(x) for x in b.raw) / RATE
    # capture_frame queues, so let the wire drain before opening the gate —
    # otherwise queued ringback lands after the answer and the test blames the
    # gate for its own pacing. A real ringback simply stops when you pick up.
    await asyncio.sleep(3.0)
    drained = sum(len(x) for x in b.raw) / RATE

    print("  answering")
    b.answered.set()
    await asyncio.sleep(0.3)
    print(f"  {TALK_S}s speech at amp {TALK_AMP} (gate open)")
    await send(TALK_S, TALK_AMP, 900)

    b.stop_flag.set()
    await asyncio.to_thread(b.finished.wait, 10.0)
    audio = np.concatenate(b.raw) if b.raw else np.zeros(0, np.int16)
    dur = len(audio) / RATE
    peak = float(np.abs(audio).max()) / 32768 if len(audio) else 0.0

    print(f"\n  buffered while ringing: {during_ring:.2f}s (after drain {drained:.2f}s)")
    print(f"  recorded: {dur:.2f}s  peak {peak:.3f}")
    assert during_ring < 0.05, f"{during_ring:.2f}s recorded before the answer — gate leaked"
    assert drained < 0.05, f"{drained:.2f}s arrived before the answer — gate leaked"
    assert dur > 1.0, f"only {dur:.2f}s after the answer — gate is stuck shut"
    assert peak < RING_AMP / 2, f"peak {peak:.3f} is ringback-loud — the ring got in"
    print("\ncheck_answer_gate passed — ring excluded, post-answer audio present")

    await room.disconnect()
    await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
    await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())
