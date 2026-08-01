"""Subscriber-side recorder, implemented as an embedded livekit-agents worker.

Why agents and not a plain rtc join: LiveKit Cloud only initializes the Krisp
noise-cancellation filter for participants running inside an agents-framework
job (verified with scripts/agents_nc_probe.py — a plain rtc participant gets
`code=209 noise cancellation was not initialized`, an agents job gets
`code=200 active`). So the server registers an AgentServer with a THREAD job
executor (same process — buffers are shared directly) and explicitly
dispatches it into the room to record.

Web call:  create a fresh room, dispatch the recorder agent, hand the browser
           a publish token; the job records the browser's mic track.
Phone call: with AGENT_NAME set to the name in the project's SIP dispatch rule
            (`inbound-agent` by default, see config.LK_AGENT_NAME), LiveKit
            dispatches the job into the call's room itself the moment the room
            exists — so the session is *armed* before the call lands and the job
            adopts whatever room it arrives in. Polling for a room that gained a
            SIP participant remains as the fallback for a rule pointing at some
            other agent. Either way the first inbound call wins, and a claim
            flag stops a second job from double-recording.

The job records the raw track at 48 kHz mono s16 (plus one extra AudioStream
per ticked cloud live-rail candidate, e.g. Krisp BVC) and pushes ~20 ms
RMS/peak events to the UI websocket.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import soundfile as sf
from livekit import api, rtc

from . import config, lk_cloud

RECORD_RATE = 48_000
_LEVEL_WINDOW = int(RECORD_RATE * 0.02)  # 20 ms
AGENT_NAME = config.LK_AGENT_NAME

EventCb = Callable[[dict], Awaitable[None]]


def _http_url() -> str:
    return config.LIVEKIT_URL.replace("wss://", "https://").replace("ws://", "http://")


def _browser_token(identity: str, room: str) -> str:
    return (
        api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(room_join=True, room=room, can_subscribe=True, can_publish=True)
        )
        .to_jwt()
    )


def _is_sip(p) -> bool:
    kind = getattr(p, "kind", None)
    if kind is not None:
        try:
            from livekit.protocol.models import ParticipantInfo

            if kind == ParticipantInfo.Kind.SIP:
                return True
        except Exception:
            pass
    return str(getattr(p, "identity", "")).startswith("sip_")


# ------------------------------------------------------------------ bridge
# Shared state between the FastAPI loop and the agents job thread. One
# session at a time (enforced by the server), so a module singleton is fine.


class _Bridge:
    def __init__(self):
        self.reset()
        self.main_loop: asyncio.AbstractEventLoop | None = None
        self.emit: EventCb | None = None

    def reset(self):
        # armed: a session is open, so an incoming job belongs to us
        # claimed: a job is already recording — later jobs must not double-record
        self.armed = False
        self.adopt_any = False
        self.claimed = False
        self.claim_lock = threading.Lock()
        self.target_room: str | None = None
        self.target_identity: str | None = None  # None = first SIP/non-agent
        self.live: list[dict] = []
        self.stop_flag = threading.Event()
        self.finished = threading.Event()
        self.raw: list[np.ndarray] = []
        self.live_bufs: dict[str, list[np.ndarray]] = {}
        self.live_errors: dict[str, str] = {}
        self.diag: dict = {}  # room / participant / SIP attributes of what we recorded

    def emit_threadsafe(self, ev: dict) -> None:
        if self.emit is not None and self.main_loop is not None:
            asyncio.run_coroutine_threadsafe(self.emit(ev), self.main_loop)


_bridge = _Bridge()

# ------------------------------------------------------------ agent worker

_server_started = False
_server_thread: threading.Thread | None = None
_server_obj = None
_server_loop: asyncio.AbstractEventLoop | None = None
_server_lock = threading.Lock()
# last few worker-level events (job offered / accepted / rejected), so a phone
# call that never records can be told apart from one that never arrived
_worker_events: deque = deque(maxlen=30)


def _note(event: str, **kw) -> None:
    _worker_events.append({"t": time.strftime("%H:%M:%S"), "event": event, **kw})


async def _entrypoint(ctx) -> None:
    await ctx.connect()
    b = _bridge
    room = ctx.room.name
    if not b.armed:
        _note("job dropped", room=room, why="no session open — press Start before dialing")
        return
    with b.claim_lock:
        if b.claimed:
            _note("job dropped", room=room, why="another job already recording")
            return
        if b.target_room is None:
            # phone mode: the SIP dispatch rule sent us straight into the
            # call's room, so adopt it — nothing else knows the name yet
            if not b.adopt_any:
                _note("job dropped", room=room, why="session is not expecting a phone call")
                return
            b.target_room = room
            b.emit_threadsafe(
                {"type": "session", "state": "call_found",
                 "room": room, "participant": "(dispatched by SIP rule)"}
            )
        elif room != b.target_room:
            _note("job dropped", room=room, why=f"foreign room (expecting {b.target_room})")
            return
        b.claimed = True
    _note("job recording", room=room)

    loop = asyncio.get_running_loop()
    track_fut: asyncio.Future[tuple[rtc.Track, str]] = loop.create_future()

    def consider(track: rtc.Track, participant) -> None:
        if track_fut.done() or track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        ident = participant.identity
        if b.target_identity is not None:
            if ident != b.target_identity:
                return
        elif not _is_sip(participant):
            return
        track_fut.set_result((track, ident))

    @ctx.room.on("track_subscribed")
    def on_track(track, pub, participant):
        consider(track, participant)

    for participant in ctx.room.remote_participants.values():
        for pub in participant.track_publications.values():
            if pub.track is not None:
                consider(pub.track, participant)

    # wait for the target track, abortable by stop
    while not track_fut.done():
        if b.stop_flag.is_set():
            b.finished.set()
            return
        await asyncio.sleep(0.1)
    track, ident = track_fut.result()

    b.emit_threadsafe({"type": "session", "state": "recording", "room": ctx.room.name,
                       "participant": ident})
    speaker = next(
        (p for p in ctx.room.remote_participants.values() if p.identity == ident), None
    )
    b.diag = {
        "room": ctx.room.name,
        "participant": ident,
        "participant_kind": str(getattr(speaker, "kind", "")),
        # SIP participants carry sip.callStatus / sip.callID / numbers here —
        # the difference between "the call was up and quiet" and "it never
        # actually answered" is only visible in these
        "attributes": dict(getattr(speaker, "attributes", {}) or {}),
    }

    async def publish_silence():
        """Send silence toward the caller for the whole recording.

        A recorder has nothing to say, but a SIP leg that receives no RTP at all
        may never send any either: PBXs behind NAT latch onto the first inbound
        stream. Without this the call connects and the recording comes back
        bit-exactly zero — which is what happened before this existed. A live
        agent hides the problem by greeting the caller immediately.
        """
        source = rtc.AudioSource(RECORD_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("nc-bench-silence", source)
        await ctx.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        frame = np.zeros(RECORD_RATE // 50, dtype=np.int16)  # 20 ms
        try:
            while True:
                await source.capture_frame(
                    rtc.AudioFrame(frame.tobytes(), RECORD_RATE, 1, len(frame))
                )
        except asyncio.CancelledError:
            pass

    async def consume_raw():
        t0 = time.monotonic()
        level_acc = np.zeros(0, dtype=np.int16)
        stream = rtc.AudioStream(track, sample_rate=RECORD_RATE, num_channels=1)
        try:
            async for event in stream:
                samples = np.frombuffer(event.frame.data, dtype=np.int16)
                b.raw.append(samples.copy())
                level_acc = np.concatenate([level_acc, samples])
                while len(level_acc) >= _LEVEL_WINDOW:
                    win = level_acc[:_LEVEL_WINDOW].astype(np.float64)
                    level_acc = level_acc[_LEVEL_WINDOW:]
                    b.emit_threadsafe({
                        "type": "level",
                        "t": round(time.monotonic() - t0, 3),
                        "rms": int(np.sqrt((win**2).mean())),
                        "peak": int(np.abs(win).max()),
                    })
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    async def consume_live(cand: dict):
        cid = cand["id"]
        try:
            opts = lk_cloud.build(cand["lk_model"])
            stream = rtc.AudioStream(
                track, sample_rate=RECORD_RATE, num_channels=1, noise_cancellation=opts
            )
        except Exception as e:
            b.live_errors[cid] = f"could not attach cloud NC stream: {e}"
            return
        bufs = b.live_bufs.setdefault(cid, [])
        try:
            async for event in stream:
                bufs.append(np.frombuffer(event.frame.data, dtype=np.int16).copy())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            b.live_errors[cid] = str(e)
        finally:
            await stream.aclose()

    tasks = [asyncio.create_task(consume_raw()), asyncio.create_task(publish_silence())]
    tasks += [asyncio.create_task(consume_live(c)) for c in b.live]

    try:
        while not b.stop_flag.is_set():
            await asyncio.sleep(0.1)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        b.finished.set()


async def _on_request(req) -> None:
    """Accept every job offered to us, and record that it was offered.

    Without this trail a failed phone call is indistinguishable from a call that
    never arrived: the job log is the only place that difference shows up.
    """
    job = getattr(req, "job", None)
    room = getattr(getattr(job, "room", None), "name", "?")
    _note("job offered", room=room, armed=_bridge.armed)
    await req.accept()


def worker_alive() -> bool:
    return _server_thread is not None and _server_thread.is_alive()


def worker_events() -> list[dict]:
    return list(_worker_events)


def shutdown_worker(timeout: float = 5.0) -> None:
    """Unregister on the way out.

    A killed process leaves its registration behind until LiveKit's own timeout,
    and LiveKit load-balances across everything registered under an agent name —
    so the first call after a restart can be handed to the corpse and simply
    fail. Closing cleanly makes a restart safe immediately.
    """
    global _server_started
    if _server_obj is None or _server_loop is None or not worker_alive():
        return
    try:
        asyncio.run_coroutine_threadsafe(_server_obj.aclose(), _server_loop).result(timeout)
    except Exception:
        pass
    _server_started = False


def ensure_worker() -> None:
    """Register the agents worker, reviving it if its thread has died.

    Called at server startup rather than on Start: registration takes a moment,
    and a SIP dispatch that finds no worker means LiveKit never creates the room
    — the call just fails, leaving no trace on this side at all. The liveness
    check matters for the same reason: if the worker loop dies (dropped signal
    connection, an unhandled error) a "started" flag alone would keep claiming
    all is well while every inbound call quietly failed.
    """
    global _server_started, _server_thread, _server_obj
    with _server_lock:
        if _server_started and worker_alive():
            return
        from livekit.agents.job import JobExecutorType
        from livekit.agents.worker import AgentServer

        server = AgentServer(
            ws_url=config.LIVEKIT_URL,
            api_key=config.LIVEKIT_API_KEY,
            api_secret=config.LIVEKIT_API_SECRET,
            job_executor_type=JobExecutorType.THREAD,
        )
        server.rtc_session(_entrypoint, agent_name=AGENT_NAME, on_request=_on_request)
        _server_obj = server

        def _run() -> None:
            global _server_loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _server_loop = loop
            try:
                loop.run_until_complete(server.run(devmode=True))
            finally:
                loop.close()

        # run() is a coroutine — give the worker its own loop in its own thread
        _server_thread = threading.Thread(
            target=_run,
            daemon=True,
            name="nc-agent-server",
        )
        _server_thread.start()
        _server_started = True


async def _dispatch(lk: api.LiveKitAPI, room: str) -> None:
    """Dispatch the recorder agent; retries while the worker registers."""
    last: Exception | None = None
    for _ in range(8):
        try:
            await lk.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=room)
            )
            return
        except Exception as e:
            last = e
            await asyncio.sleep(1.0)
    raise RuntimeError(f"could not dispatch recorder agent: {last}")


# ---------------------------------------------------------------- session


class Recorder:
    """One recording session (web or phone). Single-use."""

    def __init__(self, emit: EventCb, live_candidates: list[dict] | None = None):
        _bridge.reset()
        _bridge.emit = emit
        _bridge.main_loop = asyncio.get_running_loop()
        _bridge.live = live_candidates or []
        self._emit = emit
        self._poll_task: asyncio.Task | None = None
        self._lk: api.LiveKitAPI | None = None
        self.room_name: str | None = None

    def _api(self) -> api.LiveKitAPI:
        if self._lk is None:
            self._lk = api.LiveKitAPI(
                _http_url(), config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET
            )
        return self._lk

    # ---------------------------------------------------------------- web

    async def start_web(self) -> dict:
        ensure_worker()
        self.room_name = f"nc-bench-{uuid.uuid4().hex[:8]}"
        identity = "nc-web-user"
        _bridge.target_room = self.room_name
        _bridge.target_identity = identity
        _bridge.armed = True
        lk = self._api()
        await lk.room.create_room(api.CreateRoomRequest(name=self.room_name))
        await _dispatch(lk, self.room_name)
        return {
            "room": self.room_name,
            "livekit_url": config.LIVEKIT_URL,
            "token": _browser_token(identity, self.room_name),
        }

    # -------------------------------------------------------------- phone

    async def start_phone(self) -> None:
        # Armed *before* the call arrives, because with AGENT_NAME matching the
        # project's SIP dispatch rule the job lands on us the moment the room is
        # created — earlier than any polling could notice it. The poller below
        # stays as the fallback for when the rule points at a different agent.
        _bridge.armed = True
        _bridge.adopt_any = True
        ensure_worker()
        lk = self._api()
        baseline = {r.name for r in (await lk.room.list_rooms(api.ListRoomsRequest())).rooms}
        await self._emit({"type": "session", "state": "waiting_call"})

        async def poll():
            while True:
                if _bridge.target_room is not None:
                    return  # a dispatched job already claimed the call
                rooms = (await lk.room.list_rooms(api.ListRoomsRequest())).rooms
                for r in rooms:
                    if r.name in baseline or r.num_participants == 0:
                        continue
                    parts = (
                        await lk.room.list_participants(
                            api.ListParticipantsRequest(room=r.name)
                        )
                    ).participants
                    sip = next((p for p in parts if _is_sip(p)), None)
                    if sip is not None:
                        self.room_name = r.name
                        _bridge.target_room = r.name
                        _bridge.target_identity = sip.identity
                        await self._emit({
                            "type": "session", "state": "call_found",
                            "room": r.name, "participant": sip.identity,
                        })
                        await _dispatch(lk, r.name)
                        return
                await asyncio.sleep(1.0)

        self._poll_task = asyncio.create_task(poll())

    # ---------------------------------------------------------------- stop

    async def stop(self, out_wav: Path) -> tuple[dict, dict]:
        """Tear down and write the recording.

        Returns (input metadata, live-rail results): live results map
        candidate id -> {"audio": int16 ndarray @48k} or {"error": str}.
        """
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass

        _bridge.stop_flag.set()
        if _bridge.claimed:
            # the job sets finished after draining its streams
            await asyncio.to_thread(_bridge.finished.wait, 10.0)
        _bridge.armed = False
        if self._lk is not None:
            try:
                await self._lk.aclose()
            except Exception:
                pass

        raw_all = np.concatenate(_bridge.raw) if _bridge.raw else np.zeros(0, np.int16)
        raw_rms = float(np.sqrt((raw_all.astype(np.float64) ** 2).mean())) if len(raw_all) else 0.0

        live: dict[str, dict] = {}
        for cand in _bridge.live:
            cid = cand["id"]
            if cid in _bridge.live_errors:
                live[cid] = {"error": _bridge.live_errors[cid]}
            elif _bridge.live_bufs.get(cid):
                audio = np.concatenate(_bridge.live_bufs[cid])
                # When Cloud refuses the filter (code=209 in server logs) the
                # stream silently passes raw audio through — detect and refuse
                # to present passthrough as an NC result.
                rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))
                if raw_rms > 0 and abs(rms - raw_rms) / raw_rms < 0.02:
                    live[cid] = {"error": (
                        "output is (near-)identical to raw — LiveKit Cloud did not "
                        "initialize the NC filter for this session (code=209; check "
                        "the project's noise-cancellation enablement in the Cloud dashboard)"
                    )}
                else:
                    live[cid] = {"audio": audio}
            else:
                live[cid] = {"error": "no frames captured on the live NC stream"}

        if not _bridge.raw:
            return {"file": None, "duration_s": 0.0, "sample_rate": RECORD_RATE,
                    "diag": _bridge.diag}, live
        audio = np.concatenate(_bridge.raw)
        sf.write(out_wav, audio, RECORD_RATE)
        # A leg that was up but carried no audio: bit-exact zeros when no media
        # arrived at all (-180 dBFS), ~-107 dBFS when the stream is there but
        # empty. Real speech is above -50; -80 separates them with room to spare,
        # and scoring silence would waste the whole run.
        level_db = 20 * np.log10(max(float(np.sqrt((audio.astype(np.float64) ** 2).mean())),
                                     1e-9) / 32768)
        return {
            "file": out_wav.name,
            "diag": _bridge.diag,
            "level_dbfs": round(level_db, 1),
            "silent": bool(len(audio) and level_db < -80),
            "duration_s": round(len(audio) / RECORD_RATE, 2),
            "sample_rate": RECORD_RATE,
        }, live
