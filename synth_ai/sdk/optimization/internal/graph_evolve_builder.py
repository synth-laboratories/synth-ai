"""Helpers for building Graph Evolve jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from synth_ai.core.utils.urls import BACKEND_URL_API
from synth_ai.data.enums import GraphType

from .graphgen_models import (
    GraphGenJobConfig as GraphEvolveJobConfig,
)
from .graphgen_models import (
    GraphGenTask as GraphEvolveTask,
)
from .graphgen_models import (
    GraphGenTaskSet as GraphEvolveTaskSet,
)
from .graphgen_models import (
    GraphGenTaskSetMetadata as GraphEvolveTaskSetMetadata,
)
from .graphgen_models import (
    load_graphgen_taskset as load_graph_evolve_taskset,
)
from .graphgen_models import (
    parse_graphgen_taskset as parse_graph_evolve_taskset,
)


def parse_graph_evolve_dataset(
    dataset: str | Path | Dict[str, Any] | GraphEvolveTaskSet,
) -> GraphEvolveTaskSet:
    if isinstance(dataset, (str, Path)):
        return load_graph_evolve_taskset(dataset)
    if isinstance(dataset, dict):
        return parse_graph_evolve_taskset(dataset)
    if isinstance(dataset, GraphEvolveTaskSet):
        return dataset
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
    if not api_key:
        api_key = os.environ.get("SYNTH_API_KEY")
        if not api_key:
            raise ValueError(
                "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
            )
    return backend_url, api_key


def normalize_policy_models(policy_models: str | List[str]) -> List[str]:
    policy_models_list = [policy_models] if isinstance(policy_models, str) else list(policy_models)
    if not policy_models_list:
        raise ValueError("policy_models must contain at least one model")
    return policy_models_list


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
    if not initial_graph_id:
        raise ValueError(
            "initial_graph_id is required for Graph Evolve. De-novo graph generation is disabled."
        )

    graph_type_enum = GraphType(graph_type) if graph_type else GraphType.POLICY

    return GraphEvolveJobConfig(
        policy_models=policy_models,
        rollout_budget=rollout_budget,
        proposer_effort=proposer_effort,
        verifier_model=judge_model,
        verifier_provider=judge_provider,
        population_size=population_size,
        num_generations=num_generations,
        problem_spec=problem_spec,
        target_llm_calls=target_llm_calls,
        graph_type=graph_type_enum,
        initial_graph_id=initial_graph_id,
    )


def build_placeholder_dataset() -> GraphEvolveTaskSet:
    return GraphEvolveTaskSet(
        metadata=GraphEvolveTaskSetMetadata(name="(resumed job)"),
        tasks=[GraphEvolveTask(id="placeholder", input={})],
    )


__all__ = [
    "parse_graph_evolve_dataset",
    "resolve_graph_evolve_credentials",
    "normalize_policy_models",
    "build_graph_evolve_config",
    "build_placeholder_dataset",
]
