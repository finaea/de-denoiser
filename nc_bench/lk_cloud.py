"""LiveKit-hosted noise-cancellation models, as `rtc.AudioStream(noise_cancellation=...)`
filters.

- Krisp NC / BVC / BVCTelephony (`livekit-plugins-noise-cancellation`): native
  audio filters, named by their factory: `"NC"`, `"BVC"`, `"BVCTelephony"`.
- Krisp VIVA (`livekit-plugins-krisp`): Python FrameProcessors, `"KRISP:VIVA"`
  (`voice_isolation()`) and `"KRISP:VIVA_TEL"` (`voice_isolation_telephony()`).
- ai-coustics (`livekit-plugins-ai-coustics`, the official plugin):
  `"AIC:<EnhancerModel name>"`, e.g. `"AIC:QUAIL_VF_L"`. Each processed frame
  carries the enhancer's speech flag in its userdata (`AIC_VAD_KEY`).
- `"RAW"`: no filter at all, the control that isolates what the Cloud round
  trip itself does to the audio.

None of these run offline: each authenticates through the Cloud room its track
lives in. They run on two rails — the live rail (recorder.py, one extra stream
per candidate while a session records) and the cloud replay rail
(cloud_replay.py, which publishes a stored input.wav into a fresh room so uploads
and old runs can be processed too).

A candidate may carry `lk_options` for the ai-coustics filter: `bypass` and
`enhancement_level` (ModelParameters) and `vad_sensitivity` (VadSettings, valid
1.0–15.0 in plugin 0.3.2).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import config

RAW = "RAW"
# the key the ai-coustics plugin stamps its per-frame speech flag under
# (ai_coustics.FRAME_USERDATA_AIC_VAD_ATTRIBUTE; a constant here so callers need
# not import the plugin to read it)
AIC_VAD_KEY = "lk.aic-vad"
# How <cid>.aicvad.npy stores that flag, one int8 per frame. A frame the
# enhancer stamped nothing on is -1, never 0: 0 is a confident "not speech".
AICVAD_ENCODING = "int8: 1 speech, 0 not speech, -1 no flag on that frame"

_KRISP_VIVA = {"KRISP:VIVA": "voice_isolation", "KRISP:VIVA_TEL": "voice_isolation_telephony"}
_AIC_OPTIONS = {"bypass", "enhancement_level", "vad_sensitivity"}
_VAD_SENSITIVITY = (1.0, 15.0)

_PACKAGES = (
    "livekit",
    "livekit-plugins-noise-cancellation",
    "livekit-plugins-krisp",
    "livekit-plugins-ai-coustics",
)


def preload() -> None:
    """Import (and thereby register) the cloud NC plugins on the MAIN thread.

    Each plugin registers itself at import time, and the rtc layer rejects
    registration from other threads — recorder jobs run in a THREAD executor, so
    a lazy first import there would fail with 'Plugins must be registered on the
    main thread'.
    """
    for mod in ("noise_cancellation", "ai_coustics", "krisp"):
        try:
            __import__(f"livekit.plugins.{mod}")
        except ImportError:
            pass


def _plugin(model: str) -> str | None:
    if model == RAW:
        return None
    if model.startswith("AIC:"):
        return "ai_coustics"
    if model.startswith("KRISP:"):
        return "krisp"
    return "noise_cancellation"


def is_aic(model: str) -> bool:
    """ai-coustics streams carry a per-frame speech flag worth storing."""
    return model.startswith("AIC:")


def expects_raw(model: str, options: dict | None = None) -> bool:
    """Whether the output is meant to be the raw audio: the control, or an
    ai-coustics filter with bypass=1 or enhancement_level=0 (kept for its VAD flag).

    "Meant to" is not "is": plugin 0.3.2 ignores bypass=1.0 on the Voice Focus
    models (output bit-identical to normal enhancement), so cloud_replay checks it.
    """
    if model == RAW:
        return True
    if not is_aic(model):
        return False
    opts = options or {}
    return (float(opts.get("bypass") or 0) >= 1.0
            or (opts.get("enhancement_level") is not None
                and float(opts["enhancement_level"]) == 0.0))


def changes_audio(model: str, options: dict | None = None) -> bool:
    """Whether a working filter must make the output differ from the raw stream.
    For the ones that should not, "output ≈ raw" is no evidence of a refused filter.
    """
    return not expects_raw(model, options)


def plugin_versions() -> dict[str, str]:
    out = {}
    for pkg in _PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _check_options(model: str, options: dict | None) -> str:
    """Why the options are invalid for this model, or '' if they are fine."""
    if not options:
        return ""
    if not is_aic(model):
        return f"lk_options are only supported for ai-coustics models, not {model}"
    unknown = set(options) - _AIC_OPTIONS
    if unknown:
        return f"unknown lk_options {sorted(unknown)} (known: {sorted(_AIC_OPTIONS)})"
    sens = options.get("vad_sensitivity")
    if sens is not None and not _VAD_SENSITIVITY[0] <= float(sens) <= _VAD_SENSITIVITY[1]:
        return f"vad_sensitivity {sens} outside {_VAD_SENSITIVITY[0]}–{_VAD_SENSITIVITY[1]}"
    for key in ("bypass", "enhancement_level"):
        v = options.get(key)
        if v is not None and not 0.0 <= float(v) <= 1.0:
            return f"{key} {v} outside 0.0–1.0"
    return ""


def available(model: str = "", options: dict | None = None) -> tuple[bool, str]:
    if ".livekit.cloud" not in config.LIVEKIT_URL:
        return False, "LIVEKIT_URL is not a LiveKit Cloud project (cloud NC is Cloud-gated)"
    if not (config.LIVEKIT_API_KEY and config.LIVEKIT_API_SECRET):
        return False, "LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set in .env"
    why = _check_options(model, options)
    if why:
        return False, why
    plugin = _plugin(model)
    if plugin is None:
        return True, ""
    try:
        mod = __import__(f"livekit.plugins.{plugin}", fromlist=["_"])
    except ImportError as e:
        return False, f"plugin not installed: {e.name}"
    if plugin == "ai_coustics" and getattr(mod.EnhancerModel, model[4:], None) is None:
        return False, f"ai-coustics plugin has no model '{model[4:]}'"
    if plugin == "krisp" and model not in _KRISP_VIVA:
        return False, f"unknown Krisp VIVA model '{model}' (known: {sorted(_KRISP_VIVA)})"
    if plugin == "noise_cancellation" and model not in model_names():
        return False, f"model '{model}' not exposed by installed plugin (has: {model_names()})"
    return True, ""


def model_names() -> list[str]:
    """Krisp NC factory names exposed by the installed plugin version."""
    try:
        from livekit.plugins import noise_cancellation as nc
    except ImportError:
        return []
    return [
        n
        for n in dir(nc)
        if not n.startswith("_") and n[0].isupper() and callable(getattr(nc, n))
    ]


def build(model: str, options: dict | None = None):
    """The `noise_cancellation=` argument for rtc.AudioStream: NoiseCancellationOptions
    (Krisp NC family), a FrameProcessor (Krisp VIVA, ai-coustics), or None (RAW).

    Build one per stream — the FrameProcessors are stateful.
    """
    why = _check_options(model, options)
    if why:
        raise RuntimeError(why)
    if model == RAW:
        return None

    if is_aic(model):
        from livekit.plugins import ai_coustics

        name = model[4:]
        enhancer = getattr(ai_coustics.EnhancerModel, name, None)
        if enhancer is None:
            raise RuntimeError(
                f"unknown ai-coustics model '{name}' "
                f"(has: {[m.name for m in ai_coustics.EnhancerModel]})"
            )
        opts = options or {}
        params = None
        if opts.get("bypass") is not None or opts.get("enhancement_level") is not None:
            params = ai_coustics.ModelParameters(
                bypass=None if opts.get("bypass") is None else float(opts["bypass"]),
                enhancement_level=(None if opts.get("enhancement_level") is None
                                   else float(opts["enhancement_level"])),
            )
        sens = opts.get("vad_sensitivity")
        return ai_coustics.audio_enhancement(
            model=enhancer,
            vad_settings=ai_coustics.VadSettings(
                speech_hold_duration=None,
                sensitivity=None if sens is None else float(sens),
                minimum_speech_duration=None,
            ),
            model_parameters=params,
        )

    if model.startswith("KRISP:"):
        from livekit.plugins import krisp

        factory = _KRISP_VIVA.get(model)
        if factory is None:
            raise RuntimeError(f"unknown Krisp VIVA model '{model}' (known: {sorted(_KRISP_VIVA)})")
        return getattr(krisp, factory)()

    from livekit.plugins import noise_cancellation as nc

    factory = getattr(nc, model, None)
    if factory is None:
        raise RuntimeError(
            f"model '{model}' not exposed by installed plugin (has: {model_names()})"
        )
    return factory()
