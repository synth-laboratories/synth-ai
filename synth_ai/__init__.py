"""
Synth AI - Software for aiding the best and multiplying the will.
"""

from __future__ import annotations

from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path

try:  # Prefer the installed package metadata when available
    __version__ = _metadata.version("synth-ai")
except PackageNotFoundError:  # Fallback to pyproject version for editable installs
    try:
        import tomllib as _toml  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - legacy interpreter guard
        import tomli as _toml  # type: ignore[no-redef]

    try:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as fh:
            _pyproject = _toml.load(fh)
        __version__ = str(_pyproject["project"]["version"])
    except Exception:
        __version__ = "0.0.0.dev0"

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

# Judge API contract schemas
from synth_ai.judge_schemas import (
    JudgeScoreRequest,
    JudgeScoreResponse,
    JudgeOptions,
    JudgeTaskApp,
    JudgeTracePayload,
    ReviewPayload,
    CriterionScorePayload,
)

# Legacy tracing v1 is not required for v3 usage and can be unavailable in minimal envs.
tracing = None  # type: ignore
EventPartitionElement = RewardSignal = SystemTrace = TrainingQuestion = None  # type: ignore
trace_event_async = trace_event_sync = upload = None  # type: ignore

__all__ = [
    "LM",
    "OpenAI",
    "AsyncOpenAI",
    "Anthropic",
    "AsyncAnthropic",
    "environments",
    # Judge API contracts
    "JudgeScoreRequest",
    "JudgeScoreResponse",
    "JudgeOptions",
    "JudgeTaskApp",
    "JudgeTracePayload",
    "ReviewPayload",
    "CriterionScorePayload",
]  # Explicitly define public API (v1 tracing omitted in minimal env)
