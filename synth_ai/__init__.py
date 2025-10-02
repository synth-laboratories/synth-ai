"""
Synth AI - Software for aiding the best and multiplying the will.
"""

# Environment exports - moved from synth-env
from synth_ai.environments import *  # noqa
import synth_ai.environments as environments  # expose module name for __all__
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

# Legacy tracing v1 is not required for v3 usage and can be unavailable in minimal envs.
tracing = None  # type: ignore
EventPartitionElement = RewardSignal = SystemTrace = TrainingQuestion = None  # type: ignore
trace_event_async = trace_event_sync = upload = None  # type: ignore

__version__ = "0.2.6.dev4"
__all__ = [
    "LM",
    "OpenAI",
    "AsyncOpenAI",
    "Anthropic",
    "AsyncAnthropic",
    "environments",
]  # Explicitly define public API (v1 tracing omitted in minimal env)
