"""TOML schema + validation for ADAS/Graphs jobs.

Graphs jobs (aka ADAS jobs) are JSON-dataset-first, but for convenience we also
support a small TOML wrapper that points at an GraphGenTaskSet JSON file plus a few
optimization knobs.

Example `graph.toml`:

```toml
[graph]
dataset = "my_tasks.json"          # required (path to GraphGenTaskSet JSON)
policy_model = "gpt-4o-mini"       # optional
rollout_budget = 200              # optional
proposer_effort = "medium"        # optional: low|medium|high
auto_start = true                 # optional

[graph.metadata]
session_id = "sess_123"
parent_job_id = "adas_parent"
population_size = 4
num_generations = 5
```
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast, Literal

from .graphgen_models import GraphGenJobConfig, GraphGenTaskSet, load_graphgen_taskset
from .graphgen_validators import GraphGenValidationError, validate_graphgen_job_config


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
    errors: List[Dict[str, Any]] = []

    if not isinstance(section, dict):
        raise GraphTomlValidationError(
            "Graph config must be a TOML table",
            [{"field": "graph", "error": "graph section must be a table"}],
        )

    dataset_ref = section.get("dataset_path") or section.get("dataset")
    if not dataset_ref or not isinstance(dataset_ref, str):
        errors.append(
            {
                "field": "graph.dataset",
                "error": "dataset (path) is required",
                "suggestion": "Set graph.dataset = \"my_tasks.json\"",
            }
        )
        dataset_path = None
        dataset = None
    else:
        dataset_path = Path(dataset_ref).expanduser()
        if base_dir and not dataset_path.is_absolute():
            dataset_path = (base_dir / dataset_path).resolve()
        try:
            dataset = load_graphgen_taskset(dataset_path)
        except FileNotFoundError:
            errors.append(
                {
                    "field": "graph.dataset",
                    "error": f"Dataset file not found: {dataset_path}",
                }
            )
            dataset = None
        except ValueError as e:
            errors.append(
                {
                    "field": "graph.dataset",
                    "error": f"Invalid GraphGenTaskSet JSON: {e}",
                }
            )
            dataset = None

    policy_model = section.get("policy_model") or section.get("model")
    rollout_budget = section.get("rollout_budget") or section.get("budget")
    proposer_effort = section.get("proposer_effort") or section.get("effort")

    try:
        config = GraphGenJobConfig(
            policy_model=str(policy_model) if policy_model is not None else "gpt-4o-mini",
            policy_provider=section.get("policy_provider"),
            rollout_budget=int(rollout_budget) if rollout_budget is not None else 100,
            proposer_effort=cast(Literal["low", "medium", "high"], str(proposer_effort)) if proposer_effort is not None else "medium",
            judge_model=section.get("judge_model"),
            judge_provider=section.get("judge_provider"),
            population_size=section.get("population_size", 4),
            num_generations=section.get("num_generations"),
        )
    except Exception as e:
        errors.append({"field": "graph", "error": f"Invalid graph config fields: {e}"})
        config = GraphGenJobConfig()

    auto_start = bool(section.get("auto_start", True))
    metadata = cast(Dict[str, Any], section.get("metadata") if isinstance(section.get("metadata"), dict) else {})
    initial_prompt = section.get("initial_prompt")

    if dataset is not None:
        try:
            validate_graphgen_job_config(config, dataset)
        except GraphGenValidationError as e:
            errors.extend(e.errors)

    if errors:
        raise GraphTomlValidationError("Graph TOML validation failed", errors)

    assert dataset_path is not None and dataset is not None
    return GraphTomlResult(
        dataset_path=dataset_path,
        dataset=dataset,
        config=config,
        auto_start=auto_start,
        metadata=metadata,
        initial_prompt=str(initial_prompt) if initial_prompt is not None else None,
    )


def load_graph_job_toml(path: str | Path) -> GraphTomlResult:
    """Load and validate a graph job TOML file."""
    path = Path(path).expanduser().resolve()
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    section = cfg.get("graph") or cfg.get("adas") or {}
    return validate_graph_job_section(section, base_dir=path.parent)


def validate_graph_job_payload(payload: Dict[str, Any]) -> None:
    """Validate a graph/ADAS job payload (matching backend create request).

    Expected keys:
      - dataset: GraphGenTaskSet dict
      - policy_model, rollout_budget, proposer_effort
      - optional judge_model/judge_provider
      - optional metadata (population_size/num_generations)
    """
    errors: List[Dict[str, Any]] = []

    dataset_raw = payload.get("dataset")
    dataset: Optional[GraphGenTaskSet]
    if isinstance(dataset_raw, GraphGenTaskSet):
        dataset = dataset_raw
    elif isinstance(dataset_raw, dict):
        try:
            dataset = GraphGenTaskSet.model_validate(dataset_raw)
        except Exception as e:
            errors.append({"field": "dataset", "error": f"Invalid GraphGenTaskSet: {e}"})
            dataset = None
    else:
        errors.append({"field": "dataset", "error": "dataset must be a dict"})
        dataset = None

    metadata = cast(Dict[str, Any], payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {})

    try:
        config = GraphGenJobConfig(
            policy_model=str(payload.get("policy_model") or "gpt-4o-mini"),
            policy_provider=payload.get("policy_provider"),
            rollout_budget=int(payload.get("rollout_budget") or 100),
            proposer_effort=cast(Literal["low", "medium", "high"], str(payload.get("proposer_effort") or "medium")),
            judge_model=payload.get("judge_model"),
            judge_provider=payload.get("judge_provider"),
            population_size=metadata.get("population_size", 4),
            num_generations=metadata.get("num_generations"),
        )
    except Exception as e:
        errors.append({"field": "config", "error": f"Invalid config fields: {e}"})
        config = GraphGenJobConfig()

    if dataset is not None:
        try:
            validate_graphgen_job_config(config, dataset)
        except GraphGenValidationError as e:
            errors.extend(e.errors)

    if errors:
        raise GraphTomlValidationError("Graph job validation failed", errors)


__all__ = [
    "GraphTomlResult",
    "GraphTomlValidationError",
    "validate_graph_job_section",
    "validate_graph_job_payload",
    "load_graph_job_toml",
]
