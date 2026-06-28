"""Generated Managed Research actor model policy constants.

Source of truth: backend/config/smr_actor_model_policy.json

Regenerate: python -m synth_ai.managed_research.schema_sync
"""

from __future__ import annotations

from typing import Any

SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES: tuple[str, ...] = (
    "baseten/zai-org/GLM-5.2",
    "deepseek/deepseek-v4-flash-direct",
    "deepseek/deepseek-v4-pro-direct",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "x-ai/grok-4.3",
    "x-ai/grok-build",
)


SMR_ACTOR_MODEL_POLICY: tuple[dict[str, Any], ...] = (
    {
        "actor_type": "orchestrator",
        "actor_subtype": "main",
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5-20251001",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "moonshotai/kimi-k2.6",
            "baseten/zai-org/GLM-5.2",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "reviewer",
        "actor_subtype": "main",
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5-20251001",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "moonshotai/kimi-k2.6",
            "baseten/zai-org/GLM-5.2",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "reviewer",
        "actor_subtype": "task_completion",
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5-20251001",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "moonshotai/kimi-k2.6",
            "baseten/zai-org/GLM-5.2",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "reviewer",
        "actor_subtype": "run_completion",
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "baseten/zai-org/GLM-5.2",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "reviewer",
        "actor_subtype": "safety",
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-haiku-4-5-20251001",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "baseten/zai-org/GLM-5.2",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "reviewer",
        "actor_subtype": "objective",
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "baseten/zai-org/GLM-5.2",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "reviewer",
        "actor_subtype": "artifact_reviewer",
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5-20251001",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "moonshotai/kimi-k2.6",
            "baseten/zai-org/GLM-5.2",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "worker",
        "actor_subtype": "engineer",
        "permitted_models": [
            "gpt-5.3-codex",
            "gpt-5.3-codex-spark",
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5-20251001",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "moonshotai/kimi-k2.6",
            "baseten/zai-org/GLM-5.2",
            "modal/zai-org/GLM-5.2-FP8",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "worker",
        "actor_subtype": "researcher",
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5-20251001",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "moonshotai/kimi-k2.6",
            "baseten/zai-org/GLM-5.2",
            "modal/zai-org/GLM-5.2-FP8",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
    {
        "actor_type": "worker",
        "actor_subtype": "artifact_builder",  # FRESH EDIT THIS TURN for tracked delta - skeptic fix 019f0ee1
        "permitted_models": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "anthropic/claude-sonnet-4-6",
            "x-ai/grok-4.3",
            "x-ai/grok-build",
            "baseten/zai-org/GLM-5.2",
            "deepseek/deepseek-v4-flash-direct",
            "deepseek/deepseek-v4-pro-direct",
        ],
    },
)


__all__ = ["SMR_ACTOR_MODEL_POLICY", "SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES"]
