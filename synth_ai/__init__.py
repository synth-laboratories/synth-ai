"""
Synth AI - Software for aiding the best and multiplying the will.
"""

from synth_ai.lm.core.main import LM  # Moved from zyk to lm for better organization

# Tracing exports - moved from synth-sdk
from synth_ai.tracing import *  # noqa
from synth_ai.tracing.abstractions import (
    EventPartitionElement,
    SystemTrace,
    TrainingQuestion,
    RewardSignal,
)
from synth_ai.tracing.decorators import trace_event_async, trace_event_sync
from synth_ai.tracing.upload import upload

# Provider support exports - moved from synth-sdk to synth_ai/lm
from synth_ai.lm.provider_support.openai import OpenAI, AsyncOpenAI
from synth_ai.lm.provider_support.anthropic import Anthropic, AsyncAnthropic

# Environment exports - moved from synth-env
from synth_ai.environments import *  # noqa

__version__ = "0.1.9"
__all__ = [
    "LM",
    "tracing",
    "OpenAI",
    "AsyncOpenAI",
    "Anthropic",
    "AsyncAnthropic",
    "environments",
]  # Explicitly define public API
