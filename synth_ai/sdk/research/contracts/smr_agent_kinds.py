"""Public agent-harness enum with agent-kind compatibility aliases."""

from __future__ import annotations

from enum import StrEnum


class SmrAgentKind(StrEnum):
    CODEX = "codex"
    OPENCODE_SDK = "opencode_sdk"


SMR_AGENT_KIND_VALUES: tuple[str, ...] = tuple(value.value for value in SmrAgentKind)
_AGENT_KIND_ALIASES: dict[str, SmrAgentKind] = {
    "openai": SmrAgentKind.CODEX,
    "codex": SmrAgentKind.CODEX,
    "anthropic": SmrAgentKind.OPENCODE_SDK,
    "claude": SmrAgentKind.OPENCODE_SDK,
    "claude-code": SmrAgentKind.OPENCODE_SDK,
    "claude_code": SmrAgentKind.OPENCODE_SDK,
    "opencode": SmrAgentKind.OPENCODE_SDK,
    "open-code": SmrAgentKind.OPENCODE_SDK,
    "open_code": SmrAgentKind.OPENCODE_SDK,
    "opencode_sdk": SmrAgentKind.OPENCODE_SDK,
}


def coerce_smr_agent_kind(
    value: SmrAgentKind | str | None,
    *,
    field_name: str = "agent_kind",
) -> SmrAgentKind | None:
    if value is None:
        return None
    if isinstance(value, SmrAgentKind):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    alias = _AGENT_KIND_ALIASES.get(normalized.lower())
    if alias is not None:
        return alias
    raise ValueError(f"{field_name} must be one of: {', '.join(SMR_AGENT_KIND_VALUES)}")


__all__ = ["SMR_AGENT_KIND_VALUES", "SmrAgentKind", "coerce_smr_agent_kind"]
