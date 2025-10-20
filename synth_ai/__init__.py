# Environment exports - moved from synth-env
from synth_ai.environments import *  # noqa
import synth_ai.environments as environments  # expose module name for __all__

from ._docs_message import DOCS_MESSAGE

try:
    from synth_ai.lm.core.main import LM  # Moved from zyk to lm for better organization
except Exception:  # allow minimal imports (e.g., tracing) without LM stack
    LM = None  # type: ignore
try:
    from synth_ai.lm.provider_support.anthropic import Anthropic, AsyncAnthropic
except Exception:  # optional in minimal environments
    Anthropic = AsyncAnthropic = None  # type: ignore

# Provider support exports - moved from synth-sdk to synth_ai/lm
try:
    from synth_ai.lm.provider_support.openai import AsyncOpenAI, OpenAI
except Exception:
    AsyncOpenAI = OpenAI = None  # type: ignore

# For LLMs
try:
    from .main import SynthAI
except Exception:
    SynthAI = None

# Legacy tracing v1 is not required for v3 usage and can be unavailable in minimal envs.
tracing = None  # type: ignore
EventPartitionElement = RewardSignal = SystemTrace = TrainingQuestion = None  # type: ignore
trace_event_async = trace_event_sync = upload = None  # type: ignore

SDK_HELP = DOCS_MESSAGE

__doc__ = SDK_HELP


def help() -> str:
    """Return the docs-first directive for the Synth-AI SDK."""
    return SDK_HELP

__version__ = "0.2.9.dev14"
__all__ = [
    "LM",
    "OpenAI",
    "AsyncOpenAI",
    "Anthropic",
    "AsyncAnthropic",
    "environments",
    "help",
    "SynthAI"
]  # Explicitly define public API (v1 tracing omitted in minimal env)
