from __future__ import annotations

from collections.abc import Iterable, Sequence

from synth_ai.task import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutSafetyConfig,
)

DEFAULT_POLICY_NAME = "crafter-react"
DEFAULT_ENV_NAME = "crafter"


def parse_ops(spec: str | None) -> list[str] | None:
    """Parse a comma-separated operations string into a list."""

    if spec is None:
        return None
    ops = [op.strip() for op in spec.split(",") if op.strip()]
    if not ops:
        raise ValueError("Ops must contain at least one entry")
    return ops


def ops_from_pairs(max_llm_calls: int, *, cap: int | None = None) -> list[str]:
    """Return alternating agent/env ops for the requested number of LLM calls."""

    pairs = max(1, int(max_llm_calls or 0))
    if cap is not None:
        pairs = min(pairs, cap)
    ops: list[str] = []
    for _ in range(pairs):
        ops.extend(["agent", "env"])
    return ops


def build_rollout_request(
    *,
    seed: int,
    run_id: str,
    model: str,
    inference_url: str,
    ops: Sequence[str] | Iterable[str],
    inference_api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
    trace_format: str = "compact",
    return_trace: bool = False,
    policy_name: str = DEFAULT_POLICY_NAME,
    env_name: str = DEFAULT_ENV_NAME,
    max_policy_tokens: int | None = None,
    record_trajectories: bool = True,
) -> RolloutRequest:
    """Construct a RolloutRequest shared across local rollout utilities."""

    policy_config: dict[str, object] = {
        "model": model,
        "inference_url": inference_url,
    }
    if inference_api_key is not None:
        policy_config["api_key"] = inference_api_key
    if extra_headers:
        policy_config["extra_headers"] = extra_headers
    if max_policy_tokens is not None:
        policy_config["max_completion_tokens"] = max_policy_tokens
        policy_config["max_tokens"] = max_policy_tokens

    record_cfg = RolloutRecordConfig(
        trajectories=record_trajectories,
        trace_format=trace_format,
        return_trace=return_trace,
    )
    return RolloutRequest(
        run_id=run_id,
        env=RolloutEnvSpec(env_name=env_name, seed=seed, config={}),
        policy=RolloutPolicySpec(policy_name=policy_name, config=policy_config),
        ops=list(ops),
        record=record_cfg,
        on_done="reset",
        safety=RolloutSafetyConfig(),
    )


__all__ = [
    "DEFAULT_POLICY_NAME",
    "DEFAULT_ENV_NAME",
    "build_rollout_request",
    "ops_from_pairs",
    "parse_ops",
]
