"""Explicit API credential resolution.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass

from synth_ai.core.errors import AuthenticationError
from synth_ai.core.utils.env import get_api_key


@dataclass(frozen=True, slots=True, repr=False)
class ApiCredential:
    """A validated bearer credential whose value is never included in repr."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise AuthenticationError("Synth API credential must not be empty")

    def authorization_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.value}",
            "Content-Type": "application/json",
        }


def resolve_api_credential(api_key: str | None = None) -> ApiCredential:
    """Resolve one explicit or configured Synth API credential."""

    if api_key is not None and api_key.strip():
        return ApiCredential(api_key.strip())
    configured = get_api_key(required=False)
    if configured is None or not configured.strip():
        raise AuthenticationError("api_key is required (provide it or set SYNTH_API_KEY)")
    return ApiCredential(configured.strip())


__all__ = ["ApiCredential", "resolve_api_credential"]
