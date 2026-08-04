"""All runtime configuration comes from .env (see .env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


PORT = int(_get("PORT", "8777"))
DATA_DIR = (PROJECT_ROOT / _get("DATA_DIR", "./data")).resolve()
RUNS_DIR = DATA_DIR / "runs"

LIVEKIT_URL = _get("LIVEKIT_URL")
LIVEKIT_API_KEY = _get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = _get("LIVEKIT_API_SECRET")

# The agent name the recorder registers under. Phone mode dials *out* and
# dispatches the recorder into its own room explicitly, so this deliberately does
# NOT match the project's inbound SIP dispatch rule — sharing that name would make
# the bench steal ai-handler's real inbound calls (LiveKit load-balances jobs
# across every worker registered under a name).
LK_AGENT_NAME = _get("LK_AGENT_NAME", "nc-bench-recorder")

# ---- outbound phone call: the bench dials you, you pick up ----
# Outbound trunk to place the call on, and the number to ring.
LK_SIP_TRUNK_ID = _get("LK_SIP_TRUNK_ID")
LK_SIP_CALL_TO = _get("LK_SIP_CALL_TO")
LK_SIP_RINGING_TIMEOUT_S = int(_get("LK_SIP_RINGING_TIMEOUT_S", "45"))

WHISPER_URL = _get("WHISPER_URL")
WHISPER_SAMPLE_RATE = int(_get("WHISPER_SAMPLE_RATE", "16000"))
WHISPER_BOOST_VOLUME = _get("WHISPER_BOOST_VOLUME", "true").lower() == "true"

HECTTOR_API_KEY = _get("HECTTOR_API_KEY")
HECTTOR_MODEL = _get("HECTTOR_MODEL", "coda-vi-1.0")
HECTTOR_ENHANCER_WEIGHT = float(_get("HECTTOR_ENHANCER_WEIGHT", "1.0"))
HECTTOR_SAMPLE_RATE = int(_get("HECTTOR_SAMPLE_RATE", "16000"))
HECTTOR_CHUNK_MS = int(_get("HECTTOR_CHUNK_MS", "20"))

# ---- Silero VAD ----
# Same model file and the same livekit-agents state machine ai-handler runs, so a
# span here is a turn production's segmentation would cut.
#
# The THRESHOLDS deliberately no longer match ai-handler's config.example.ini
# [livekit] (min_speech 0.12, min_silence 0.55). These are tighter — 0.05 accepts
# shorter blips as speech and 0.4 ends a turn on a shorter pause — which resolves
# the timeline into more, smaller spans and makes it visible when a chain shaves
# an onset or swallows a pause. Comparing against ai-handler means putting 0.12 /
# 0.55 back, in .env or in the per-run panel.
VAD_ENABLED = _get("VAD_ENABLED", "true").lower() == "true"
VAD_ACTIVATION_THRESHOLD = float(_get("VAD_ACTIVATION_THRESHOLD", "0.50"))
VAD_DEACTIVATION_THRESHOLD = float(_get("VAD_DEACTIVATION_THRESHOLD", "0.50"))
VAD_MIN_SPEECH_DURATION = float(_get("VAD_MIN_SPEECH_DURATION", "0.05"))
VAD_MIN_SILENCE_DURATION = float(_get("VAD_MIN_SILENCE_DURATION", "0.40"))
# Not set by ai-handler, so this is the plugin's own default. It back-dates each
# span's start, which is why a highlight can begin before audible speech.
VAD_PREFIX_PADDING_DURATION = float(_get("VAD_PREFIX_PADDING_DURATION", "0.5"))

MODELS_DIR = (PROJECT_ROOT / _get("MODELS_DIR", "./models")).resolve()
DTLN_MODEL_DIR = (PROJECT_ROOT / _get("DTLN_MODEL_DIR", "./models/dtln")).resolve()
# ffmpeg's arnndn needs a weights file; sh = "somnolent hogwash" (general speech)
RNNOISE_MODEL = _get("RNNOISE_MODEL", "sh")
# Hush: max attenuation in dB — 100 is their "unlimited" default, and 0 means
# *no* attenuation (i.e. passthrough), so don't set this to 0 expecting "off"
HUSH_ATTEN_LIM_DB = float(_get("HUSH_ATTEN_LIM_DB", "100"))

AIC_LICENSE_KEY = _get("AIC_LICENSE_KEY")
AIC_MODEL_ID = _get("AIC_MODEL_ID")
AIC_MODEL_DIR = (PROJECT_ROOT / _get("AIC_MODEL_DIR", "./models/aic")).resolve()

CANDIDATES_FILE = PROJECT_ROOT / "candidates.json"
STATIC_DIR = PROJECT_ROOT / "static"

# The rate every candidate's output is written at, and what STT expects.
PIPELINE_RATE = 16_000

RUNS_DIR.mkdir(parents=True, exist_ok=True)
