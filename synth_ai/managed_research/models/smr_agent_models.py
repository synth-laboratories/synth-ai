"""Generated public Managed Research agent model enum.

Source of truth: backend/config/smr_supported_models.json
"""

from __future__ import annotations

from enum import StrEnum


class SmrAgentModel(StrEnum):
    GPT_5_3_CODEX = "gpt-5.3-codex"
    GPT_5_3_CODEX_SPARK = "gpt-5.3-codex-spark"
    GPT_5_4 = "gpt-5.4"
    GPT_5_5 = "gpt-5.5"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"
    GPT_OSS_120B = "gpt-oss-120b"
    ANTHROPIC_CLAUDE_SONNET_4_6 = "anthropic/claude-sonnet-4-6"
    ANTHROPIC_CLAUDE_HAIKU_4_5_20251001 = "anthropic/claude-haiku-4-5-20251001"
    X_AI_GROK_4_1_FAST = "x-ai/grok-4.1-fast"
    X_AI_GROK_4_3 = "x-ai/grok-4.3"
    X_AI_GROK_4_20_BETA = "x-ai/grok-4.20-beta"
    MOONSHOTAI_KIMI_K2_6 = "moonshotai/kimi-k2.6"


SMR_AGENT_MODEL_VALUES: tuple[str, ...] = tuple(model.value for model in SmrAgentModel)


def coerce_smr_agent_model(
    value: SmrAgentModel | str | None,
    *,
    field_name: str = "agent_model",
) -> SmrAgentModel | None:
    if value is None:
        return None
    if isinstance(value, SmrAgentModel):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrAgentModel(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_AGENT_MODEL_VALUES)}. "
            "Backend preflight remains authoritative for model availability."
        ) from exc


__all__ = ["SMR_AGENT_MODEL_VALUES", "SmrAgentModel", "coerce_smr_agent_model"]
