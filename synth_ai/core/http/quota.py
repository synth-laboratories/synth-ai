"""Typed SMR quota refusal bodies (SYN-4844).

The backend sends every SMR project quota refusal (403 plan entitlement, 409
active-project cap, 409 nonterminal-run cap) as ``{"detail": {error_code,
message, service, stage, cause, scope, limit, used, remedy}}``. A body that
names a quota code but violates that shape is contract skew, never a guess.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from synth_ai.errors import ContractError, QuotaExceededError

from .response import _CAUSE_TOKEN

QUOTA_REFUSAL_CODES = frozenset(
    {
        "smr_project_plan_active_limit_reached",
        "smr_project_active_limit_reached",
        "smr_project_nonterminal_run_limit_reached",
    }
)


class QuotaRefusalDetail(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    error_code: str = Field(max_length=128)
    message: str = Field(max_length=1024)
    service: str = Field(max_length=32, pattern=_CAUSE_TOKEN)
    stage: str = Field(max_length=64, pattern=_CAUSE_TOKEN)
    cause: str = Field(max_length=64, pattern=_CAUSE_TOKEN)
    scope: Literal["organization", "project"]
    limit: StrictInt | None = Field(ge=0)
    used: StrictInt = Field(ge=0)
    remedy: str = Field(max_length=512)


class QuotaRefusalBody(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    detail: QuotaRefusalDetail


class _CodeOnly(BaseModel):
    model_config = ConfigDict(extra="ignore")

    detail: dict[str, object]


def quota_code(body: bytes) -> str | None:
    """The quota code a body names, or ``None`` when it names none."""
    try:
        code = _CodeOnly.model_validate_json(body).detail.get("error_code")
    except ValidationError:
        return None
    return code if isinstance(code, str) and code in QUOTA_REFUSAL_CODES else None


def decode_quota_refusal(
    status: int,
    body: bytes,
    *,
    operation: str | None = None,
    request_id: str | None = None,
) -> QuotaExceededError:
    """Decode a quota refusal; any other shape raises the typed ``ContractError``."""
    if status not in {403, 409}:
        raise ContractError("Quota refusals are 403 or 409", status=status)
    try:
        detail = QuotaRefusalBody.model_validate_json(body).detail
    except ValidationError as cause:
        raise ContractError(
            "Quota refusal body does not match its typed contract", status=status
        ) from cause
    if detail.error_code not in QUOTA_REFUSAL_CODES:
        raise ContractError("Unknown quota refusal code", status=status)
    return QuotaExceededError(
        f"{operation or 'request'} refused: {detail.error_code}: {detail.message}",
        code=detail.error_code,
        status=status,
        operation=operation,
        request_id=request_id,
        retryable=False,
        service=detail.service,
        stage=detail.stage,
        cause=detail.cause,
        scope=detail.scope,
        limit=detail.limit,
        used=detail.used,
        remedy=detail.remedy,
    )


__all__ = [
    "QUOTA_REFUSAL_CODES",
    "QuotaRefusalBody",
    "QuotaRefusalDetail",
    "decode_quota_refusal",
    "quota_code",
]
