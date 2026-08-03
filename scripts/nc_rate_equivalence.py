"""Does the AudioStream sample rate change what the Cloud NC filter does?

The bench attaches its live-rail candidates with rtc.AudioStream(sample_rate=
48000). ai-handler attaches the same Krisp filter at 16000 on its live rail
(livekit_worker.py: AudioStream(..., sample_rate=16000)) and at the agents
default of 24000 on the AgentSession path. If the filter runs before that
resample, all three see identical audio and a bench result transfers to
production unchanged; if it runs after, the bench is measuring a different
thing and its rate should be matched to production.

Attaches all three rates to the SAME live track at the same time, then compares
the outputs at a common 16 kHz.

    .venv/bin/python scripts/nc_rate_equivalence.py [MODEL]   # default BVCTelephony
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit import api, rtc  # noqa: E402

from nc_bench import config, lk_cloud  # noqa: E402

RATES = [16_000, 24_000, 48_000]
PUB_RATE = 48_000
DUR = 6.0
AGENT = "nc-rate-probe"


def _http(u: str) -> str:
    return u.replace("wss://", "https://").replace("ws://", "http://")


def source_signal(n: int, rate: int, wideband: bool) -> np.ndarray:
    """Speech-ish harmonics + steady noise.

    Narrowband (default) is band-limited through 8 kHz like a PSTN leg. Pass
    --wideband to keep real content above 5 kHz, which is what tells a genuine
    rate-dependent filter difference apart from ratios measured on an empty band.
    """
    t = np.arange(n) / rate
    rng = np.random.default_rng(11)
    voiced = np.sin(2 * np.pi * 0.6 * t) > 0
    harmonics = range(1, 40) if wideband else range(1, 6)
    speech = 0.3 * voiced * sum((1.0 / k) * np.sin(2 * np.pi * 180 * k * t) for k in harmonics)
    x = speech + 0.05 * rng.standard_normal(n)
    if wideband:
        return (x / max(np.abs(x).max(), 1e-9) * 0.5).astype(np.float32)
    return soxr.resample(soxr.resample(x, rate, 8000), 8000, rate).astype(np.float32)


# label -> (rate, use_nc). "48k-nc-B" is a second filter instance at a rate we
# already measure: it calibrates how much two runs differ for reasons that have
# nothing to do with the rate, so cross-rate differences can be read against it.
STREAMS = {f"{r // 1000}k-{'nc' if nc else 'raw'}": (r, nc) for r in RATES for nc in (True, False)}
STREAMS["48k-nc-B"] = (48_000, True)
captured: dict[str, list[np.ndarray]] = {k: [] for k in STREAMS}
done = asyncio.Event()


async def entrypoint(ctx) -> None:
    await ctx.connect()
    model = next((a for a in sys.argv[1:] if not a.startswith("-")), "BVCTelephony")
    track_fut: asyncio.Future = asyncio.get_running_loop().create_future()

    @ctx.room.on("track_subscribed")
    def _on(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO and not track_fut.done():
            track_fut.set_result(track)

    track = await track_fut

    async def consume(label: str, rate: int, use_nc: bool):
        stream = rtc.AudioStream(
            track, sample_rate=rate, num_channels=1,
            noise_cancellation=lk_cloud.build(model) if use_nc else None,
        )
        try:
            async for ev in stream:
                captured[label].append(np.frombuffer(ev.frame.data, dtype=np.int16).copy())
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    tasks = [asyncio.create_task(consume(k, *v)) for k, v in STREAMS.items()]
    await asyncio.sleep(DUR + 1.0)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    done.set()


async def main() -> None:
    lk_cloud.preload()
    from livekit.agents.job import JobExecutorType
    from livekit.agents.worker import AgentServer

    server = AgentServer(
        ws_url=config.LIVEKIT_URL, api_key=config.LIVEKIT_API_KEY,
        api_secret=config.LIVEKIT_API_SECRET, job_executor_type=JobExecutorType.THREAD,
    )
    server.rtc_session(entrypoint, agent_name=AGENT)
    import threading

    threading.Thread(
        target=lambda: asyncio.run(server.run(devmode=True)), daemon=True
    ).start()
    await asyncio.sleep(3)

    room_name = "nc-rate-probe-room"
    lk = api.LiveKitAPI(_http(config.LIVEKIT_URL), config.LIVEKIT_API_KEY,
                        config.LIVEKIT_API_SECRET)
    await lk.room.create_room(api.CreateRoomRequest(name=room_name))
    await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(agent_name=AGENT, room=room_name))

    token = (api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
             .with_identity("sip_probe")
             .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True))
             .to_jwt())
    room = rtc.Room()
    await room.connect(config.LIVEKIT_URL, token)
    src = rtc.AudioSource(PUB_RATE, 1)
    await room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track("caller", src),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

    wideband = "--wideband" in sys.argv
    src_file = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--file=")), None)
    if src_file:
        # Real speech, not synthetic harmonics: a speech model fed non-speech
        # adapts erratically, and the instance-to-instance control blows up to
        # the point where nothing can be concluded.
        import soundfile as sf
        raw_in, in_rate = sf.read(src_file, dtype="float32", always_2d=True)
        audio = soxr.resample(raw_in.mean(axis=1), in_rate, PUB_RATE).astype(np.float32)
    else:
        audio = source_signal(int(PUB_RATE * DUR), PUB_RATE, wideband)
    step = PUB_RATE // 100
    for i in range(0, len(audio) - step, step):
        chunk = (np.clip(audio[i:i + step], -1, 1) * 32767).astype(np.int16)
        await src.capture_frame(rtc.AudioFrame(chunk.tobytes(), PUB_RATE, 1, step))

    await asyncio.wait_for(done.wait(), timeout=30)
    await room.disconnect()
    await lk.aclose()

    def at16k(label: str):
        buf = captured[label]
        if not buf:
            return None
        rate = STREAMS[label][0]
        x = np.concatenate(buf).astype(np.float32) / 32768
        return soxr.resample(x, rate, 16_000).astype(np.float32)

    def ltas_db(y: np.ndarray) -> np.ndarray:
        """Long-term average spectrum in 8 bands up to 8 kHz — what the filter
        did to each part of the band, immune to stream start offsets."""
        n = 1024
        frames = np.array([y[i:i + n] for i in range(0, len(y) - n, n)])
        p = (np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2).mean(axis=0)
        edges = np.linspace(0, len(p), 9).astype(int)
        return np.array([10 * np.log10(max(p[a:b].mean(), 1e-20)) for a, b in
                         zip(edges[:-1], edges[1:])])

    model = next((a for a in sys.argv[1:] if not a.startswith("-")), "BVCTelephony")
    print(f"\nmodel: {model}  source: {'wideband' if '--wideband' in sys.argv else 'narrowband'}")
    print("  stream     attenuation vs its own raw, per 1 kHz band (dB)            total")
    profiles = {}
    for label, (r, nc) in STREAMS.items():
        if not nc:
            continue
        raw, out = at16k(f"{r // 1000}k-raw"), at16k(label)
        if raw is None or out is None:
            print(f"  {label:9s} (missing frames)")
            continue
        db = lambda y: 20 * np.log10(max(float(np.sqrt((y ** 2).mean())), 1e-9))  # noqa: E731
        bands = ltas_db(raw) - ltas_db(out)
        profiles[label] = bands
        print(f"  {label:9s} " + " ".join(f"{b:5.1f}" for b in bands)
              + f"   {db(raw) - db(out):5.1f} dB")
    if "48k-nc" in profiles and "48k-nc-B" in profiles:
        control = float(np.abs(profiles["48k-nc"] - profiles["48k-nc-B"]).max())
        print(f"\n  control: two filter instances at the SAME rate differ by up to "
              f"{control:.1f} dB in a band")
        for label in ("16k-nc", "24k-nc"):
            if label in profiles:
                d = float(np.abs(profiles[label] - profiles["48k-nc"]).max())
                verdict = "within" if d <= control * 1.5 else "BEYOND"
                print(f"  {label} vs 48k-nc: up to {d:.1f} dB — {verdict} instance variance")


if __name__ == "__main__":
    asyncio.run(main())
