from __future__ import annotations

from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path

# Install log filter as early as possible to suppress noisy codex_otel logs
try:
    from synth_ai.core.log_filter import install_log_filter
    install_log_filter()
except Exception:
    # Silently fail if log filter can't be installed
    pass

# Judge schemas moved to sdk/judging/schemas.py
from synth_ai.sdk.judging.schemas import (
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

# Legacy tracing v1 is not required for v3 usage and can be unavailable in minimal envs.
tracing = None  # type: ignore
EventPartitionElement = RewardSignal = SystemTrace = TrainingQuestion = None  # type: ignore
trace_event_async = trace_event_sync = upload = None  # type: ignore

__all__ = [
    # Judge API contracts
    "JudgeScoreRequest",
    "JudgeScoreResponse",
    "JudgeOptions",
    "JudgeTaskApp",
    "JudgeTracePayload",
    "ReviewPayload",
    "CriterionScorePayload",
]  # Explicitly define public API (v1 tracing omitted in minimal env)
