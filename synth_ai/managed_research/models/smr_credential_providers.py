"""Public Managed Research credential-provider enum."""

from __future__ import annotations

from enum import StrEnum


class SmrCredentialProvider(StrEnum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    TINKER = "tinker"


SMR_CREDENTIAL_PROVIDER_VALUES: tuple[str, ...] = tuple(
    provider.value for provider in SmrCredentialProvider
)


def coerce_smr_credential_provider(
    value: SmrCredentialProvider | str | None,
    *,
    field_name: str = "provider",
) -> SmrCredentialProvider | None:
    if value is None:
        return None
    if isinstance(value, SmrCredentialProvider):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrCredentialProvider(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_CREDENTIAL_PROVIDER_VALUES)}"
        ) from exc


__all__ = [
    "SMR_CREDENTIAL_PROVIDER_VALUES",
    "SmrCredentialProvider",
    "coerce_smr_credential_provider",
]
