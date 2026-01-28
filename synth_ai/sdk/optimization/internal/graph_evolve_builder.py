"""Helpers for building Graph Evolve jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from synth_ai.core.rust_core.urls import ensure_api_base
from synth_ai.core.utils.urls import BACKEND_URL_API
from synth_ai.data.enums import GraphType

from .graphgen_models import (
    GraphGenJobConfig as GraphEvolveJobConfig,
)
from .graphgen_models import (
    GraphGenTaskSet as GraphEvolveTaskSet,
)

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.graph_evolve_builder.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "build_graph_evolve_config"):
        raise RuntimeError("Rust core graph evolve builders required; synth_ai_py is unavailable.")
    return synth_ai_py


def parse_graph_evolve_dataset(
    dataset: str | Path | Dict[str, Any] | GraphEvolveTaskSet,
) -> GraphEvolveTaskSet:
    rust = _require_rust()
    if isinstance(dataset, (str, Path)):
        parsed = rust.load_graph_evolve_dataset(str(dataset))
        return GraphEvolveTaskSet.model_validate(parsed)
    if isinstance(dataset, dict):
        parsed = rust.parse_graph_evolve_dataset(dataset)
        return GraphEvolveTaskSet.model_validate(parsed)
    if isinstance(dataset, GraphEvolveTaskSet):
        parsed = rust.parse_graph_evolve_dataset(
            dataset.model_dump(mode="json", exclude_none=False)
        )
        return GraphEvolveTaskSet.model_validate(parsed)
    raise TypeError(
        f"dataset must be a file path, dict, or GraphEvolveTaskSet, got {type(dataset)}"
    )


def resolve_graph_evolve_credentials(
    *,
    backend_url: Optional[str],
    api_key: Optional[str],
) -> tuple[str, str]:
    if not backend_url:
        backend_url = BACKEND_URL_API
    backend_url = ensure_api_base(backend_url)
    if not api_key:
        api_key = os.environ.get("SYNTH_API_KEY")
        if not api_key:
            raise ValueError(
                "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
            )
    return backend_url, api_key


def normalize_policy_models(policy_models: str | List[str]) -> List[str]:
    rust = _require_rust()
    policy_models_list = [policy_models] if isinstance(policy_models, str) else list(policy_models)
    return list(rust.normalize_graph_evolve_policy_models(policy_models_list))


def build_graph_evolve_config(
    *,
    policy_models: List[str],
    rollout_budget: int,
    proposer_effort: Literal["low", "medium", "high"],
    judge_model: Optional[str],
    judge_provider: Optional[str],
    population_size: int,
    num_generations: Optional[int],
    problem_spec: Optional[str],
    target_llm_calls: Optional[int],
    graph_type: Optional[Literal["policy", "verifier", "rlm"]],
    initial_graph_id: Optional[str],
) -> GraphEvolveJobConfig:
    rust = _require_rust()
    graph_type_value = graph_type.value if isinstance(graph_type, GraphType) else graph_type
    config = rust.build_graph_evolve_config(
        policy_models,
        rollout_budget,
        proposer_effort,
        judge_model,
        judge_provider,
        population_size,
        num_generations,
        problem_spec,
        target_llm_calls,
        graph_type_value,
        initial_graph_id,
    )
    return GraphEvolveJobConfig.model_validate(config)


def build_placeholder_dataset() -> GraphEvolveTaskSet:
    rust = _require_rust()
    payload = rust.build_graph_evolve_placeholder_dataset()
    return GraphEvolveTaskSet.model_validate(payload)


__all__ = [
    "parse_graph_evolve_dataset",
    "resolve_graph_evolve_credentials",
    "normalize_policy_models",
    "build_graph_evolve_config",
    "build_placeholder_dataset",
]
