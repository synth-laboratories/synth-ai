"""
Synth AI - Software for aiding the best and multiplying the will.
"""

# Environment exports - moved from synth-env
from synth_ai.environments import *  # noqa
from synth_ai.lm.core.main import LM  # Moved from zyk to lm for better organization
from synth_ai.lm.provider_support.anthropic import Anthropic, AsyncAnthropic

# Provider support exports - moved from synth-sdk to synth_ai/lm
from synth_ai.lm.provider_support.openai import AsyncOpenAI, OpenAI

# Tracing exports - moved from synth-sdk (deprecated v1)
from synth_ai.tracing_v1 import *  # noqa
from synth_ai.tracing_v1.abstractions import (
    EventPartitionElement,
    RewardSignal,
    SystemTrace,
    TrainingQuestion,
)
from synth_ai.tracing_v1.decorators import trace_event_async, trace_event_sync
from synth_ai.tracing_v1.upload import upload

__version__ = "0.2.1.dev0"
__all__ = [
    "LM",
    "tracing",
    "OpenAI",
    "AsyncOpenAI",
    "Anthropic",
    "AsyncAnthropic",
    "environments",
    # v1 tracing legacy API re-exports
    "EventPartitionElement",
    "RewardSignal",
    "SystemTrace",
    "TrainingQuestion",
    "trace_event_async",
    "trace_event_sync",
    "upload",
]  # Explicitly define public API
