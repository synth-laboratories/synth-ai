"""Synth AI - Policy optimization and evaluation platform.

This package provides the public API for Synth AI. Most users should import
directly from this module:

    from synth_ai import (
        # Optimization
        PolicyOptimizationJob,
        GraphOptimizationJob,

        # Evaluation
        EvalJob,

        # Task Apps
        InProcessTaskApp,
        LocalAPIClient,
        create_task_app,

        # Clients
        VerifierClient,
        InferenceClient,
    )

For data types, import from synth_ai.data:

    from synth_ai.data import SessionTrace, Rubric, Criterion
"""

from __future__ import annotations

import importlib
from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Install log filter as early as possible to suppress noisy codex_otel logs
try:
    from synth_ai.core.utils.log_filter import install_log_filter

    install_log_filter()
except Exception:
    # Silently fail if log filter can't be installed
    pass

# Version resolution
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


# Type hints for IDE support
if TYPE_CHECKING:
    from synth_ai.sdk.eval import EvalJob, EvalJobConfig
    from synth_ai.sdk.graphs import GraphCompletionsClient, GraphTarget, VerifierClient
    from synth_ai.sdk.graphs.verifier_schemas import (
        CriterionScorePayload,
        ReviewPayload,
        VerifierOptions,
        VerifierScoreRequest,
        VerifierScoreResponse,
        VerifierTaskApp,
        VerifierTracePayload,
    )
    from synth_ai.sdk.inference import InferenceClient
    from synth_ai.sdk.localapi import (
        InProcessTaskApp,
        LocalAPIClient,
        LocalAPIConfig,
        TaskAppConfig,
        create_task_app,
    )
    from synth_ai.sdk.optimization import GraphOptimizationJob, PolicyOptimizationJob

    # Legacy aliases
    PromptLearningJob = PolicyOptimizationJob
    GraphEvolveJob = GraphOptimizationJob


__all__ = [
    # Optimization (canonical names)
    "PolicyOptimizationJob",
    "GraphOptimizationJob",
    # Legacy aliases
    "PromptLearningJob",
    "GraphEvolveJob",
    # Evaluation
    "EvalJob",
    "EvalJobConfig",
    # Task Apps
    "InProcessTaskApp",
    "LocalAPIClient",
    "LocalAPIConfig",
    "TaskAppConfig",
    "create_task_app",
    # Clients
    "VerifierClient",
    "GraphCompletionsClient",
    "GraphTarget",
    "InferenceClient",
    # Verifier schemas
    "VerifierScoreRequest",
    "VerifierScoreResponse",
    "VerifierOptions",
    "VerifierTaskApp",
    "VerifierTracePayload",
    "ReviewPayload",
    "CriterionScorePayload",
]

# Lazy loading map: name -> (module, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    # Optimization
    "PolicyOptimizationJob": ("synth_ai.sdk.optimization", "PolicyOptimizationJob"),
    "GraphOptimizationJob": ("synth_ai.sdk.optimization", "GraphOptimizationJob"),
    "PromptLearningJob": ("synth_ai.sdk.optimization", "PolicyOptimizationJob"),
    "GraphEvolveJob": ("synth_ai.sdk.optimization", "GraphOptimizationJob"),
    # Evaluation
    "EvalJob": ("synth_ai.sdk.eval", "EvalJob"),
    "EvalJobConfig": ("synth_ai.sdk.eval", "EvalJobConfig"),
    # Task Apps
    "InProcessTaskApp": ("synth_ai.sdk.localapi", "InProcessTaskApp"),
    "LocalAPIClient": ("synth_ai.sdk.localapi", "LocalAPIClient"),
    "LocalAPIConfig": ("synth_ai.sdk.localapi", "LocalAPIConfig"),
    "TaskAppConfig": ("synth_ai.sdk.localapi", "TaskAppConfig"),
    "create_task_app": ("synth_ai.sdk.localapi", "create_task_app"),
    # Clients
    "VerifierClient": ("synth_ai.sdk.graphs", "VerifierClient"),
    "GraphCompletionsClient": ("synth_ai.sdk.graphs", "GraphCompletionsClient"),
    "GraphTarget": ("synth_ai.sdk.graphs", "GraphTarget"),
    "InferenceClient": ("synth_ai.sdk.inference", "InferenceClient"),
    # Verifier schemas
    "VerifierScoreRequest": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierScoreRequest"),
    "VerifierScoreResponse": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierScoreResponse"),
    "VerifierOptions": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierOptions"),
    "VerifierTaskApp": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierTaskApp"),
    "VerifierTracePayload": ("synth_ai.sdk.graphs.verifier_schemas", "VerifierTracePayload"),
    "ReviewPayload": ("synth_ai.sdk.graphs.verifier_schemas", "ReviewPayload"),
    "CriterionScorePayload": ("synth_ai.sdk.graphs.verifier_schemas", "CriterionScorePayload"),
}


def __getattr__(name: str) -> Any:
    """Lazy load SDK exports on first access."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except Exception as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
