"""LiveKit Cloud noise-cancellation models (Krisp NC/BVC/BVCTelephony, and
ai-coustics variants where the installed plugin exposes them).

These are NOT offline processors: the plugin's FrameProcessor authenticates
through the Cloud room the track lives in, so they only run on the live rail —
the recorder opens one extra AudioStream per ticked model while the session
records. They are unavailable for uploads / history reprocessing.
"""

from __future__ import annotations

from . import config


def preload() -> None:
    """Import (and thereby register) the cloud NC plugins on the MAIN thread.

    Both plugins register an rtc plugin at import time, and the rtc layer
    rejects registration from other threads — recorder jobs run in a THREAD
    executor, so a lazy first import there would fail with
    'Plugins must be registered on the main thread'.
    """
    for mod in ("noise_cancellation", "ai_coustics"):
        try:
            __import__(f"livekit.plugins.{mod}")
        except ImportError:
            pass


def available(model: str = "") -> tuple[bool, str]:
    if ".livekit.cloud" not in config.LIVEKIT_URL:
        return False, "LIVEKIT_URL is not a LiveKit Cloud project (cloud NC is Cloud-gated)"
    try:
        if model.startswith("AIC:"):
            from livekit.plugins import ai_coustics  # noqa: F401
        else:
            from livekit.plugins import noise_cancellation  # noqa: F401
    except ImportError as e:
        return False, f"plugin not installed: {e.name}"
    return True, ""


def model_names() -> list[str]:
    """Model factory names exposed by the installed plugin version."""
    try:
        from livekit.plugins import noise_cancellation as nc
    except ImportError:
        return []
    return [
        n
        for n in dir(nc)
        if not n.startswith("_") and n[0].isupper() and callable(getattr(nc, n))
    ]


def build(model: str):
    """Return NoiseCancellationOptions for rtc.AudioStream(noise_cancellation=...).

    model is either a Krisp factory name ("NC" | "BVC" | "BVCTelephony") or an
    ai-coustics enhancer prefixed "AIC:" (e.g. "AIC:QUAIL_VF_S") — the latter
    authenticates/meters through LiveKit Cloud by default.
    """
    if model.startswith("AIC:"):
        from livekit.plugins import ai_coustics

        name = model[4:]
        enhancer = getattr(ai_coustics.EnhancerModel, name, None)
        if enhancer is None:
            raise RuntimeError(
                f"unknown ai-coustics model '{name}' "
                f"(has: {[m.name for m in ai_coustics.EnhancerModel]})"
            )
        return ai_coustics.audio_enhancement(model=enhancer)

    from livekit.plugins import noise_cancellation as nc

    factory = getattr(nc, model, None)
    if factory is None:
        raise RuntimeError(
            f"model '{model}' not exposed by installed plugin (has: {model_names()})"
        )
    return factory()
