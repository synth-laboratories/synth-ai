"""Generated Managed Research agent model enum.

Source of truth: backend/packages/smr/config/supported_models_catalog.py
"""

from __future__ import annotations

from enum import StrEnum


class SmrAgentModel(StrEnum):
    GPT_5_CODEX = "gpt-5-codex"
    GPT_5_3_CODEX = "gpt-5.3-codex"
    GPT_5_3_CODEX_SPARK = "gpt-5.3-codex-spark"
    GPT_5_5 = "gpt-5.5"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_6_SOL = "gpt-5.6-sol"
    GPT_5_6_TERRA = "gpt-5.6-terra"
    GPT_5_6_LUNA = "gpt-5.6-luna"
    NEMOTRON_SUPER = "nemotron-super"
    DEEPSEEK_DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"
    DEEPSEEK_DEEPSEEK_V4_PRO = "deepseek/deepseek-v4-pro"
    CURSOR_COMPOSER_2_5 = "cursor/composer-2.5"
    CURSOR_GPT_5 = "cursor/gpt-5"
    CURSOR_GROK_4_5 = "cursor/grok-4.5"
    CURSOR_SONNET_4 = "cursor/sonnet-4"
    X_AI_GROK_4_3 = "x-ai/grok-4.3"
    X_AI_GROK_BUILD = "x-ai/grok-build"
    MOONSHOTAI_KIMI_K2_6 = "moonshotai/kimi-k2.6"
    BASETEN_ZAI_ORG_GLM_5_2 = "baseten/zai-org/GLM-5.2"
    MODAL_ZAI_ORG_GLM_5_2_FP8 = "modal/zai-org/GLM-5.2-FP8"
    DEEPSEEK_DEEPSEEK_V4_FLASH_DIRECT = "deepseek/deepseek-v4-flash-direct"
    DEEPSEEK_DEEPSEEK_V4_PRO_DIRECT = "deepseek/deepseek-v4-pro-direct"
    DEEPSEEK_DEEPSEEK_CHAT = "deepseek/deepseek-chat"
    DEEPSEEK_DEEPSEEK_REASONER = "deepseek/deepseek-reasoner"
    GPT_5_4 = "gpt-5.4"


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
