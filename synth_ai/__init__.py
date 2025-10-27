from __future__ import annotations

import importlib
from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any, cast

import synth_ai.environments as environments  # expose module name for __all__
from synth_ai.environments import *  # noqa
from synth_ai.judge_schemas import (
    CriterionScorePayload,
    JudgeOptions,
    JudgeScoreRequest,
    JudgeScoreResponse,
    JudgeTaskApp,
    JudgeTracePayload,
    ReviewPayload,
)

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

def _optional_import(module_path: str) -> Any | None:
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


_lm_module = _optional_import("synth_ai.lm.core.main")
LM = cast(Any, _lm_module).LM if _lm_module and hasattr(_lm_module, "LM") else None  # type: ignore[attr-defined]

_anthropic_module = _optional_import("synth_ai.lm.provider_support.anthropic")
Anthropic = (
    cast(Any, _anthropic_module).Anthropic
    if _anthropic_module and hasattr(_anthropic_module, "Anthropic")
    else None
)  # type: ignore[attr-defined]
AsyncAnthropic = (
    cast(Any, _anthropic_module).AsyncAnthropic
    if _anthropic_module and hasattr(_anthropic_module, "AsyncAnthropic")
    else None
)  # type: ignore[attr-defined]

_openai_module = _optional_import("synth_ai.lm.provider_support.openai")
AsyncOpenAI = (
    cast(Any, _openai_module).AsyncOpenAI
    if _openai_module and hasattr(_openai_module, "AsyncOpenAI")
    else None
)  # type: ignore[attr-defined]
OpenAI = (
    cast(Any, _openai_module).OpenAI
    if _openai_module and hasattr(_openai_module, "OpenAI")
    else None
)  # type: ignore[attr-defined]

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
