"""Simplified config expansion with versioned defaults.

This module allows users to specify minimal configs that auto-expand into
full configs using smart defaults. All defaults are versioned and frozen
to ensure reproducibility.

## Minimal GEPA Config (6 required fields)

    ```toml
    [prompt_learning]
    algorithm = "gepa"
    task_app_url = "http://localhost:8001"
    total_seeds = 200
    proposer_effort = "LOW"
    proposer_output_tokens = "FAST"
    num_generations = 10
    children_per_generation = 5

    # Optional budget constraints (if omitted -> uses account balance)
    max_cost_usd = 10.0
    ```

## Minimal Eval Config (2 required fields)

    ```toml
    [eval]
    task_app_url = "http://localhost:8103"
    seeds = [0, 1, 2, 3, 4]
    ```

## Versioning

All defaults are frozen in versioned dataclasses. To pin a specific version:

    ```toml
    defaults_version = "v1"  # Optional: locks behavior forever
    ```

If not specified, the latest version is used. The expanded config includes
`_defaults_version` to track which version was applied.

## Overriding Defaults

Any field can be overridden by specifying it explicitly:

    ```toml
    [prompt_learning]
    algorithm = "gepa"
    task_app_url = "http://localhost:8001"
    total_seeds = 200
    proposer_effort = "LOW"
    proposer_output_tokens = "FAST"
    num_generations = 10
    children_per_generation = 5

    # Override specific defaults
    population_size = 30
    ```

See Also:
    - Full config reference: job_configs.txt
    - SDK entry points: synth_ai.sdk.api.train.configs.prompt_learning
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

# =============================================================================
# VERSIONED DEFAULTS - Never modify existing versions, only add new ones
# =============================================================================


@dataclass(frozen=True)
class DefaultsV1:
    """v1 defaults - frozen, do not modify.

    This dataclass defines all auto-derived values for config expansion.
    Once released, these values must NEVER change. To update defaults,
    create a new version (DefaultsV2, etc.).

    Attributes:
        train_ratio: Fraction of total_seeds used for training (0.7 = 70%)
        rollout_budget: Maximum rollouts (effectively unlimited, rely on cost)
        rollout_max_concurrent: Parallel rollout limit
        mutation_rate: Probability of mutation in evolutionary search
        pop_size_min: Minimum population size
        pop_size_max: Maximum population size
        pop_size_divisor: n_train // divisor gives base population size
        num_generations: Number of evolutionary generations
        children_divisor: pop_size // divisor gives children per generation
        crossover_rate: Probability of crossover in evolutionary search
        selection_pressure: Pareto selection pressure
        archive_multiplier: archive_size = pop_size * multiplier
        pareto_eps: Epsilon for Pareto dominance comparison
        feedback_fraction: Fraction of archive used for feedback
        eval_max_concurrent: Maximum concurrent evaluations
        eval_timeout: Timeout per evaluation in seconds
    """

    version: str = "v1"

    # Seed split
    train_ratio: float = 0.7

    # Rollout
    rollout_budget: int = 100_000_000
    rollout_max_concurrent: int = 20

    # Mutation
    mutation_rate: float = 0.3

    # Population
    pop_size_min: int = 10
    pop_size_max: int = 30
    pop_size_divisor: int = 10  # n_train // divisor
    num_generations: int = 10
    children_divisor: int = 4  # pop_size // divisor, min 3
    crossover_rate: float = 0.5
    selection_pressure: float = 1.0

    # Archive
    archive_multiplier: int = 2  # pop_size * multiplier
    pareto_eps: float = 1e-6
    feedback_fraction: float = 0.5

    # Eval
    eval_max_concurrent: int = 20
    eval_timeout: float = 600.0


# Register all versions - latest is used when version not specified
DEFAULTS_REGISTRY: dict[str, DefaultsV1] = {
    "v1": DefaultsV1(),
}
LATEST_VERSION = "v1"


def get_defaults(version: str | None = None) -> DefaultsV1:
    """Get defaults for a specific version, or latest if not specified.

    Args:
        version: Version string (e.g., "v1"). If None, uses LATEST_VERSION.

    Returns:
        Frozen dataclass containing all default values.

    Raises:
        ValueError: If the specified version is not registered.

    Example:
        >>> d = get_defaults("v1")
        >>> d.train_ratio
        0.7
        >>> d = get_defaults()  # uses latest
        >>> d.version
        'v1'
    """
    version = version or LATEST_VERSION
    if version not in DEFAULTS_REGISTRY:
        available = list(DEFAULTS_REGISTRY.keys())
        raise ValueError(f"Unknown defaults version: {version}. Available: {available}")
    return DEFAULTS_REGISTRY[version]


# =============================================================================
# CONFIG EXPANSION
# =============================================================================


def expand_eval_config(minimal: dict[str, Any]) -> dict[str, Any]:
    """Expand minimal eval config to full config.

    Required fields:
        - task_app_url: URL of the task app
        - seeds: List of seeds or range dict

    Optional fields:
        - env_name: Environment name (defaults to app_id or "default")
        - app_id: Task app identifier
        - defaults_version: Pin to specific defaults version

    Args:
        minimal: Minimal config dict with required fields.

    Returns:
        Full config dict with all fields populated.

    Raises:
        ValueError: If required fields are missing.

    Example:
        >>> minimal = {
        ...     "task_app_url": "http://localhost:8103",
        ...     "seeds": [0, 1, 2, 3, 4],
        ... }
        >>> full = expand_eval_config(minimal)
        >>> full["max_concurrent"]
        5
        >>> full["timeout"]
        600.0
    """
    # Validate required fields
    if "task_app_url" not in minimal:
        raise ValueError("task_app_url is required")
    if "seeds" not in minimal:
        raise ValueError("seeds is required")

    d = get_defaults(minimal.get("defaults_version"))
    seeds = resolve_seeds(minimal["seeds"])

    return {
        "task_app_url": minimal["task_app_url"],
        "env_name": minimal.get("env_name", minimal.get("app_id", "default")),
        "app_id": minimal.get("app_id"),
        "seeds": seeds,
        "max_concurrent": min(d.eval_max_concurrent, len(seeds)),
        "timeout": d.eval_timeout,
        "policy": minimal.get("policy", {}),  # detected from task app if empty
        "_defaults_version": d.version,  # track which version was used
    }


def expand_gepa_config(minimal: dict[str, Any]) -> dict[str, Any]:
    """Expand minimal GEPA config to full config.

    Required fields:
        - task_app_url: URL of the task app
        - proposer_effort: "LOW_CONTEXT" | "LOW" | "MEDIUM" | "HIGH"
        - proposer_output_tokens: "RAPID" | "FAST" | "SLOW"
        - num_generations: Number of evolutionary generations
        - children_per_generation: Number of children per generation
        - One of:
            - total_seeds: Total number of seeds (auto-split 70/30)
            - train_seeds + validation_seeds: Explicit seed lists/ranges

    Optional fields:
        - env_name: Environment name
        - defaults_version: Pin to specific defaults version
        - population_size: Override auto-derived population size
        - max_cost_usd: Budget constraint in USD
        - max_rollouts: Budget constraint by rollout count
        - max_seconds: Budget constraint by time

    Args:
        minimal: Minimal config dict with required fields.

    Returns:
        Full config dict with all fields populated.

    Raises:
        ValueError: If required fields are missing.

    Example:
        >>> minimal = {
        ...     "task_app_url": "http://localhost:8001",
        ...     "total_seeds": 200,
        ...     "proposer_effort": "LOW",
        ...     "proposer_output_tokens": "FAST",
        ...     "num_generations": 10,
        ...     "children_per_generation": 5,
        ... }
        >>> full = expand_gepa_config(minimal)
        >>> full["gepa"]["population"]["initial_size"]
        14
        >>> full["gepa"]["evaluation"]["train_seeds"][:5]
        [0, 1, 2, 3, 4]
    """
    d = get_defaults(minimal.get("defaults_version"))

    # Validate required fields
    if "task_app_url" not in minimal:
        raise ValueError("task_app_url is required")
    if "proposer_effort" not in minimal:
        raise ValueError("proposer_effort is required")
    if "proposer_output_tokens" not in minimal:
        raise ValueError("proposer_output_tokens is required")
    if "num_generations" not in minimal:
        raise ValueError("num_generations is required")
    if "children_per_generation" not in minimal:
        raise ValueError("children_per_generation is required")

    # Handle total_seeds -> train/validation split
    if "total_seeds" in minimal:
        total = minimal["total_seeds"]
        split = int(total * d.train_ratio)
        train_seeds = list(range(0, split))
        val_seeds = list(range(split, total))
    elif "train_seeds" in minimal or "validation_seeds" in minimal:
        train_seeds = resolve_seeds(minimal.get("train_seeds", []))
        val_seeds = resolve_seeds(minimal.get("validation_seeds", []))
    else:
        raise ValueError("Either total_seeds or (train_seeds + validation_seeds) is required")

    # Validate seed counts
    n_train = len(train_seeds)
    n_val = len(val_seeds)

    if n_train == 0:
        raise ValueError("train_seeds cannot be empty")
    if n_val == 0:
        raise ValueError("validation_seeds cannot be empty")

    # Warn on small datasets
    if n_train < 20:
        warnings.warn(
            f"Small training set ({n_train} seeds). Consider using more data "
            "or specifying explicit train_seeds/validation_seeds.",
            UserWarning,
            stacklevel=2,
        )

    # Population parameters (num_generations and children_per_generation are required)
    pop_size = minimal.get(
        "population_size",
        max(d.pop_size_min, min(d.pop_size_max, n_train // d.pop_size_divisor)),
    )
    num_gens = minimal["num_generations"]
    children = minimal["children_per_generation"]

    # Build full config with defaults
    return {
        "algorithm": "gepa",
        "task_app_url": minimal["task_app_url"],
        "gepa": {
            "env_name": minimal.get("env_name", "default"),
            "proposer_effort": minimal["proposer_effort"],
            "proposer_output_tokens": minimal["proposer_output_tokens"],
            "evaluation": {
                "train_seeds": train_seeds,
                "validation_seeds": val_seeds,
            },
            "rollout": {
                "budget": d.rollout_budget,
                "max_concurrent": d.rollout_max_concurrent,
            },
            "mutation": {"rate": d.mutation_rate},
            "population": {
                "initial_size": pop_size,
                "num_generations": num_gens,
                "children_per_generation": children,
                "crossover_rate": d.crossover_rate,
                "selection_pressure": d.selection_pressure,
            },
            "archive": {
                "size": pop_size * d.archive_multiplier,
                "pareto_set_size": pop_size * d.archive_multiplier,
                "pareto_eps": d.pareto_eps,
                "feedback_fraction": d.feedback_fraction,
            },
        },
        "termination_config": build_termination_config(minimal),
        "_defaults_version": d.version,  # track which version was used
    }


def build_termination_config(minimal: dict[str, Any]) -> dict[str, Any] | None:
    """Build termination config from optional budget constraints.

    If no constraints are specified, returns None and the backend will
    auto-apply balance-based termination.

    Supported constraints:
        - max_cost_usd: Maximum spend in USD
        - max_rollouts: Maximum number of rollouts
        - max_seconds: Maximum time in seconds
        - max_trials: Maximum number of trials

    Args:
        minimal: Config dict potentially containing budget constraints.

    Returns:
        Termination config dict, or None if no constraints specified.

    Example:
        >>> build_termination_config({"max_cost_usd": 10.0})
        {'max_cost_usd': 10.0, 'max_trials': 100000, ...}
        >>> build_termination_config({})
        None
    """
    constraint_keys = ["max_cost_usd", "max_rollouts", "max_seconds", "max_trials"]
    has_constraint = any(k in minimal for k in constraint_keys)

    if not has_constraint:
        return None  # Backend will auto-apply balance-based termination

    return {
        "max_cost_usd": minimal.get("max_cost_usd", 1000.0),
        "max_trials": minimal.get("max_trials", 100000),
        "max_rollouts": minimal.get("max_rollouts"),
        "max_seconds": minimal.get("max_seconds"),
    }


def resolve_seeds(seeds_spec: list[int] | dict[str, int] | None) -> list[int]:
    """Resolve seed specification to list of integers.

    Accepts either:
        - A list of integers: [0, 1, 2, 3, 4]
        - A range dict: {"start": 0, "end": 5} -> [0, 1, 2, 3, 4]
        - A range dict with step: {"start": 0, "end": 10, "step": 2} -> [0, 2, 4, 6, 8]

    Args:
        seeds_spec: Seed specification (list or range dict).

    Returns:
        List of integers.

    Raises:
        ValueError: If seeds_spec is invalid format.

    Example:
        >>> resolve_seeds([0, 1, 2])
        [0, 1, 2]
        >>> resolve_seeds({"start": 0, "end": 5})
        [0, 1, 2, 3, 4]
        >>> resolve_seeds({"start": 0, "end": 10, "step": 2})
        [0, 2, 4, 6, 8]
    """
    if seeds_spec is None:
        return []
    if isinstance(seeds_spec, list):
        return list(seeds_spec)
    if isinstance(seeds_spec, dict):
        if "start" not in seeds_spec or "end" not in seeds_spec:
            raise ValueError(f"Range dict must have 'start' and 'end' keys, got: {seeds_spec}")
        start = seeds_spec["start"]
        end = seeds_spec["end"]
        step = seeds_spec.get("step", 1)
        return list(range(start, end, step))
    raise ValueError(f"Invalid seeds spec: {seeds_spec}. Expected list or range dict.")


def is_minimal_config(config: dict[str, Any]) -> bool:
    """Check if a config appears to be a minimal config needing expansion.

    A config is considered minimal if it has simplified top-level fields
    like `total_seeds` instead of nested structures.

    Args:
        config: Config dict to check.

    Returns:
        True if config appears to be minimal format.

    Example:
        >>> is_minimal_config({"total_seeds": 200, "proposer_effort": "LOW"})
        True
        >>> is_minimal_config({"gepa": {"evaluation": {"train_seeds": [...]}}})
        False
    """
    # Minimal config indicators
    minimal_indicators = ["total_seeds", "defaults_version"]

    # Full config indicators (nested structure)
    full_indicators = ["gepa", "mipro"]

    has_minimal = any(k in config for k in minimal_indicators)
    has_full = any(k in config for k in full_indicators)

    # If it has nested structure, it's a full config
    if has_full and not has_minimal:
        return False

    # If it has minimal indicators, it's minimal
    if has_minimal:
        return True

    # Check for flat structure (no deeply nested evaluation/population configs)
    # This catches cases like {"task_app_url": "...", "train_seeds": [...]}
    if "train_seeds" in config and "gepa" not in config:
        return True

    return False
