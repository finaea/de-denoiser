"""Verify the outbound phone path end-to-end without ringing anyone.

Everything up to the dial is real — the room is created on the configured
LiveKit project, the recorder agent is really dispatched into it, and
`stop()` really deletes the room. Only `create_sip_participant` is
intercepted, so no call is placed and no trunk minutes are spent.

What it would otherwise take a real call to notice:
  - the SIP request carrying the wrong trunk, room, or identity
  - `krisp_enabled` drifting to True, which would silently clean the audio
    the whole bench exists to measure
  - a dial failure vanishing instead of surfacing as `call_failed`
  - `stop()` leaving the room (and therefore the call) up

    .venv/bin/python scripts/check_outbound.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit import api  # noqa: E402

from nc_bench import config, recorder  # noqa: E402


def _http() -> str:
    return config.LIVEKIT_URL.replace("wss://", "https://").replace("ws://", "http://")


async def _listed(lk: api.LiveKitAPI, name: str) -> list[str]:
    """Unfiltered list_rooms is eventually consistent — query the name."""
    res = await lk.room.list_rooms(api.ListRoomsRequest(names=[name]))
    return [r.name for r in res.rooms]


async def main() -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LK_SIP_TRUNK_ID"):
        if not getattr(config, name):
            print(f"SKIP: {name} is not set in .env")
            return
    # A number is required by the code path but never dialled.
    config.LK_SIP_CALL_TO = config.LK_SIP_CALL_TO or "+60000000000"

    events: list[dict] = []
    captured: dict = {}

    async def emit(ev: dict) -> None:
        events.append(ev)

    real = api.sip_service.SipService.create_sip_participant

    async def intercept(self, create, **kw):
        captured["req"] = create
        raise RuntimeError("intercepted by check_outbound — no call placed")

    api.sip_service.SipService.create_sip_participant = intercept
    lk = api.LiveKitAPI(_http(), config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
    rec = recorder.Recorder(emit=emit)
    try:
        info = await rec.start_phone()
        assert info["room"] == rec.room_name
        await asyncio.sleep(3)  # let the backgrounded dial task run and fail

        req = captured.get("req")
        assert req is not None, "create_sip_participant was never called"
        assert req.sip_trunk_id == config.LK_SIP_TRUNK_ID, "wrong trunk"
        assert req.sip_call_to == config.LK_SIP_CALL_TO, "wrong number"
        assert req.room_name == rec.room_name, "dialling into a foreign room"
        assert req.participant_identity == _bridge_identity(), "identity the job waits for"
        assert req.wait_until_answered is True, "a failed call must raise, not go quiet"
        assert req.krisp_enabled is False, "Krisp on the SIP leg would corrupt every result"
        assert req.ringing_timeout.seconds == config.LK_SIP_RINGING_TIMEOUT_S
        print(f"OK   SIP request: trunk={req.sip_trunk_id} room={req.room_name} "
              f"identity={req.participant_identity} krisp={req.krisp_enabled} "
              f"ring={req.ringing_timeout.seconds}s")

        assert await _listed(lk, rec.room_name), "room was not created"
        dispatched = [d.agent_name for d in await lk.agent_dispatch.list_dispatch(rec.room_name)]
        assert recorder.AGENT_NAME in dispatched, f"recorder not dispatched: {dispatched}"
        print(f"OK   room created and recorder dispatched as {recorder.AGENT_NAME!r}")

        failed = [e for e in events if e.get("state") == "call_failed"]
        assert failed, f"dial failure did not surface as call_failed: {events}"
        print(f"OK   dial failure surfaced: {failed[0]['error']}")

        meta, _ = await rec.stop(Path("/tmp/nc_check_outbound.wav"))
        assert meta["file"] is None, "a call that never connected must record nothing"
        await asyncio.sleep(2)
        left = await _listed(lk, rec.room_name)
        assert not left, f"stop() must delete the room (= hang up the leg); still up: {left}"
        print("OK   stop() hung up and cleaned the room up")
        print("\ncheck_outbound passed — no call was placed")
    finally:
        api.sip_service.SipService.create_sip_participant = real
        try:
            await lk.room.delete_room(api.DeleteRoomRequest(room=rec.room_name or "x"))
        except Exception:
            pass
        await lk.aclose()


def _bridge_identity() -> str:
    return recorder._bridge.target_identity or ""


if __name__ == "__main__":
    asyncio.run(main())
