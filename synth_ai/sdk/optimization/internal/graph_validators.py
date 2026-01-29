"""TOML schema + validation for Graph Opt (GraphGen) jobs.

Graph Opt jobs are JSON-dataset-first, but for convenience we also
support a small TOML wrapper that points at a GraphGenTaskSet JSON file plus a few
optimization knobs.

Example `graph.toml`:

```toml
[graph]
dataset = "my_tasks.json"          # required (path to GraphGenTaskSet JSON)
policy_models = ["gpt-4o-mini"]    # required (list of models)
rollout_budget = 200              # optional
proposer_effort = "medium"        # optional: low|medium|high
auto_start = true                 # optional

[graph.metadata]
session_id = "sess_123"
parent_job_id = "graph_opt_parent"
population_size = 4
num_generations = 5
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from .graphgen_models import GraphGenJobConfig, GraphGenTaskSet

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.graph_validators.") from exc


@dataclass(slots=True)
class GraphTomlResult:
    """Normalized graph job config loaded from TOML."""

    dataset_path: Path
    dataset: GraphGenTaskSet
    config: GraphGenJobConfig
    auto_start: bool
    metadata: Dict[str, Any]
    initial_prompt: Optional[str] = None


class GraphTomlValidationError(Exception):
    """Raised when a graph TOML config is invalid."""

    def __init__(self, message: str, errors: List[Dict[str, Any]]) -> None:
        self.message = message
        self.errors = errors
        super().__init__(message)


def validate_graph_job_section(
    section: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> GraphTomlResult:
    """Validate a `[graph]` TOML section and return normalized config."""
    if synth_ai_py is None or not hasattr(synth_ai_py, "validate_graph_job_section"):
        raise GraphTomlValidationError(
            "Graph TOML validation failed",
            [
                {
                    "field": "validation",
                    "error": "Rust core validation required; synth_ai_py unavailable",
                }
            ],
        )

    result = synth_ai_py.validate_graph_job_section(
        section,
        str(base_dir) if base_dir else None,
    )
    errors = result.get("errors", [])
    if errors:
        raise GraphTomlValidationError("Graph TOML validation failed", errors)

    payload = result.get("result") or {}
    dataset_path = payload.get("dataset_path")
    dataset = payload.get("dataset")
    config = payload.get("config")
    auto_start = payload.get("auto_start", True)
    metadata = payload.get("metadata", {})
    initial_prompt = payload.get("initial_prompt")

    if dataset_path is None or dataset is None or config is None:
        raise GraphTomlValidationError(
            "Graph TOML validation failed",
            [{"field": "dataset", "error": "Dataset is required"}],
        )

    return GraphTomlResult(
        dataset_path=Path(dataset_path),
        dataset=GraphGenTaskSet.model_validate(dataset),
        config=GraphGenJobConfig.model_validate(config),
        auto_start=bool(auto_start),
        metadata=cast(Dict[str, Any], metadata),
        initial_prompt=str(initial_prompt) if initial_prompt is not None else None,
    )


def load_graph_job_toml(path: str | Path) -> GraphTomlResult:
    """Load and validate a graph job TOML file."""
    if synth_ai_py is None or not hasattr(synth_ai_py, "load_graph_job_toml"):
        raise GraphTomlValidationError(
            "Graph TOML validation failed",
            [
                {
                    "field": "validation",
                    "error": "Rust core validation required; synth_ai_py unavailable",
                }
            ],
        )

    path = Path(path).expanduser().resolve()
    result = synth_ai_py.load_graph_job_toml(str(path))
    errors = result.get("errors", [])
    if errors:
        raise GraphTomlValidationError("Graph TOML validation failed", errors)

    payload = result.get("result") or {}
    dataset_path = payload.get("dataset_path")
    dataset = payload.get("dataset")
    config = payload.get("config")
    auto_start = payload.get("auto_start", True)
    metadata = payload.get("metadata", {})
    initial_prompt = payload.get("initial_prompt")

    if dataset_path is None or dataset is None or config is None:
        raise GraphTomlValidationError(
            "Graph TOML validation failed",
            [{"field": "dataset", "error": "Dataset is required"}],
        )

    return GraphTomlResult(
        dataset_path=Path(dataset_path),
        dataset=GraphGenTaskSet.model_validate(dataset),
        config=GraphGenJobConfig.model_validate(config),
        auto_start=bool(auto_start),
        metadata=cast(Dict[str, Any], metadata),
        initial_prompt=str(initial_prompt) if initial_prompt is not None else None,
    )


def validate_graph_job_payload(payload: Dict[str, Any]) -> None:
    """Validate a graph job payload (matching backend create request).

    Expected keys:
      - dataset: GraphGenTaskSet dict
      - policy_models, rollout_budget, proposer_effort
      - optional verifier_model/verifier_provider
      - optional metadata (population_size/num_generations)
    """
    if synth_ai_py is None or not hasattr(synth_ai_py, "validate_graph_job_payload"):
        raise GraphTomlValidationError(
            "Graph job validation failed",
            [
                {
                    "field": "validation",
                    "error": "Rust core validation required; synth_ai_py unavailable",
                }
            ],
        )

    errors = synth_ai_py.validate_graph_job_payload(payload)
    if errors:
        raise GraphTomlValidationError("Graph job validation failed", errors)
    return


__all__ = [
    "GraphTomlResult",
    "GraphTomlValidationError",
    "validate_graph_job_section",
    "validate_graph_job_payload",
    "load_graph_job_toml",
]
