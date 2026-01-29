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

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for config expansion.") from exc

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


def _warn_on_unsupported_minimal_keys(
    minimal: dict[str, Any], allowed_keys: set[str], *, context: str
) -> None:
    ignored = sorted(k for k in minimal if k not in allowed_keys)
    if not ignored:
        return
    warnings.warn(
        f"{context} config includes unsupported keys that will be ignored: {', '.join(ignored)}",
        UserWarning,
        stacklevel=2,
    )


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
    return synth_ai_py.expand_eval_config(minimal)


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
    return synth_ai_py.expand_gepa_config(minimal)


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
    return list(synth_ai_py.resolve_seed_spec(seeds_spec))


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
    return bool(synth_ai_py.is_minimal_config(config))
