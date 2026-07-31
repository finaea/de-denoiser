"""Probe: does LiveKit Cloud initialize the Krisp filter for a proper
agents-framework participant on this project?

Runs a minimal livekit-agents worker (agent_name=nc-bench-probe), creates a
room, dispatches the agent into it, publishes tone+noise, and inside the job
opens an AudioStream with noise_cancellation=NC(). Prints whether the filter
initialized (the [livekit-nc] code=209 warning means NO).

Run: .venv/bin/python scripts/agents_nc_probe.py
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit import api, rtc  # noqa: E402
from livekit import agents  # noqa: E402
from livekit.plugins import noise_cancellation  # noqa: E402

from nc_bench import config  # noqa: E402

RATE = 48_000
DUR = 4.0
ROOM = "nc-probe-room"
result: dict = {}  # shared with the job via THREAD executor


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    print("PROBE: job connected to", ctx.room.name)

    fut: asyncio.Future[rtc.Track] = asyncio.get_event_loop().create_future()

    @ctx.room.on("track_subscribed")
    def on_track(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO and not fut.done():
            fut.set_result(track)

    for p in ctx.room.remote_participants.values():
        for pub in p.track_publications.values():
            if pub.track is not None and not fut.done():
                fut.set_result(pub.track)

    track = await asyncio.wait_for(fut, timeout=30)
    stream = rtc.AudioStream(
        track, sample_rate=RATE, num_channels=1,
        noise_cancellation=noise_cancellation.NC(),
    )
    bufs = []
    try:
        async with asyncio.timeout(DUR + 3):
            async for ev in stream:
                bufs.append(np.frombuffer(ev.frame.data, dtype=np.int16).copy())
    except TimeoutError:
        pass
    await stream.aclose()
    audio = np.concatenate(bufs) if bufs else np.zeros(0, dtype=np.int16)
    result["nc_rms"] = float(np.sqrt((audio.astype(float) ** 2).mean())) if len(audio) else None
    result["seconds"] = len(audio) / RATE
    result["done"] = True


async def publisher():
    lkapi = api.LiveKitAPI(
        config.LIVEKIT_URL.replace("wss://", "https://"),
        config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET,
    )
    await lkapi.room.create_room(api.CreateRoomRequest(name=ROOM))
    await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(agent_name="nc-bench-probe", room=ROOM)
    )
    token = (
        api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity("probe-pub")
        .with_grants(api.VideoGrants(room_join=True, room=ROOM, can_publish=True))
        .to_jwt()
    )
    room = rtc.Room()
    await room.connect(config.LIVEKIT_URL, token)
    source = rtc.AudioSource(RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(track)
    t = np.arange(int(RATE * DUR)) / RATE
    tone = 0.3 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
    noise = 0.05 * np.random.default_rng(3).standard_normal(len(t))
    audio = ((tone + noise) * 32767).astype(np.int16)
    raw_rms = float(np.sqrt((audio.astype(float) ** 2).mean()))
    result["raw_rms"] = raw_rms
    n = RATE // 100
    for i in range(0, len(audio) - n, n):
        await source.capture_frame(rtc.AudioFrame(
            data=audio[i : i + n].tobytes(), sample_rate=RATE,
            num_channels=1, samples_per_channel=n,
        ))
    for _ in range(30):  # job runs in a THREAD executor, poll the shared dict
        if result.get("done"):
            break
        await asyncio.sleep(1.0)
    await room.disconnect()
    await lkapi.room.delete_room(api.DeleteRoomRequest(room=ROOM))
    await lkapi.aclose()


async def main():
    from livekit.agents.worker import AgentServer

    from livekit.agents.job import JobExecutorType

    server = AgentServer(
        ws_url=config.LIVEKIT_URL,
        api_key=config.LIVEKIT_API_KEY,
        api_secret=config.LIVEKIT_API_SECRET,
        job_executor_type=JobExecutorType.THREAD,
    )
    server.rtc_session(entrypoint, agent_name="nc-bench-probe")
    worker_task = asyncio.create_task(asyncio.to_thread(server.run, devmode=True))
    await asyncio.sleep(4)  # let the worker register
    await publisher()
    print("\nPROBE RESULT:", result)
    if result.get("nc_rms") is not None and result.get("raw_rms"):
        diff = abs(result["nc_rms"] - result["raw_rms"]) / result["raw_rms"]
        print(f"rms diff vs raw: {diff:.1%} -> filter {'ACTIVE' if diff > 0.02 else 'NOT ACTIVE (passthrough)'}")
    worker_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
