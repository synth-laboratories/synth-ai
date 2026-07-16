"""SMR billing SDK namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.billing import (
    SmrBillingCatalog,
    SmrBillingDrawdown,
    SmrBillingPlanSnapshot,
    SmrBillingPreflight,
    SmrBillingPreflightRequest,
    SmrFactoryEffortBillingPreflightRequest,
    SmrManualBillingGrantPreview,
    SmrManualBillingGrantPreviewRequest,
)
from synth_ai.managed_research.models.promotions import (
    SmrPromotionDiscountPreview,
    SmrPromotionDiscountPreviewRequest,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace

_PREFLIGHT_REQUEST_FIELDS = frozenset(
    {
        "model_class",
        "estimated_customer_debit_microcents",
        "project_id",
    }
)

_FACTORY_EFFORT_PREFLIGHT_REQUEST_FIELDS = frozenset(
    {
        "model_class",
        "estimated_customer_debit_microcents",
    }
)

_MANUAL_GRANT_PREVIEW_REQUEST_FIELDS = frozenset(
    {
        "target_org_id",
        "grant_kind",
        "reason",
        "scope",
        "model_class",
        "window_kind",
        "amount_microcents",
        "resets_at",
        "expires_at",
        "campaign_id",
        "wallet_credit_microcents",
        "metadata",
    }
)


def _required_non_empty(value: object, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer when provided")
    return int(value)


def _metadata_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be an object when provided")
    return {str(key): item for key, item in value.items()}


def _preflight_request_from_mapping(
    request: Mapping[str, Any] | dict[str, Any],
    *,
    model_class: str,
    estimated_customer_debit_microcents: int,
    project_id: str | None,
) -> SmrBillingPreflightRequest:
    unknown = sorted(str(key) for key in request if key not in _PREFLIGHT_REQUEST_FIELDS)
    if unknown:
        raise ValueError("unknown SmrBillingPreflightRequest fields: " + ", ".join(unknown))
    raw_project_id = request.get("project_id", project_id)
    if raw_project_id is not None and not isinstance(raw_project_id, str):
        raise ValueError("project_id must be a string or None")
    return SmrBillingPreflightRequest(
        model_class=str(request.get("model_class", model_class)),
        estimated_customer_debit_microcents=int(
            request.get(
                "estimated_customer_debit_microcents",
                estimated_customer_debit_microcents,
            )
        ),
        project_id=raw_project_id,
    )


def _preflight_payload(
    request: SmrBillingPreflightRequest | Mapping[str, Any] | dict[str, Any] | None,
    *,
    model_class: str,
    estimated_customer_debit_microcents: int,
    project_id: str | None,
) -> dict[str, Any]:
    if isinstance(request, SmrBillingPreflightRequest):
        return request.to_wire()
    if isinstance(request, Mapping):
        return _preflight_request_from_mapping(
            request,
            model_class=model_class,
            estimated_customer_debit_microcents=estimated_customer_debit_microcents,
            project_id=project_id,
        ).to_wire()
    return SmrBillingPreflightRequest(
        model_class=model_class,
        estimated_customer_debit_microcents=estimated_customer_debit_microcents,
        project_id=project_id,
    ).to_wire()


def _factory_effort_preflight_request_from_mapping(
    request: Mapping[str, Any] | dict[str, Any],
    *,
    model_class: str,
    estimated_customer_debit_microcents: int,
) -> SmrFactoryEffortBillingPreflightRequest:
    unknown = sorted(
        str(key) for key in request if key not in _FACTORY_EFFORT_PREFLIGHT_REQUEST_FIELDS
    )
    if unknown:
        raise ValueError(
            "unknown SmrFactoryEffortBillingPreflightRequest fields: " + ", ".join(unknown)
        )
    return SmrFactoryEffortBillingPreflightRequest(
        model_class=str(request.get("model_class", model_class)),
        estimated_customer_debit_microcents=int(
            request.get(
                "estimated_customer_debit_microcents",
                estimated_customer_debit_microcents,
            )
        ),
    )


def _factory_effort_preflight_payload(
    request: (SmrFactoryEffortBillingPreflightRequest | Mapping[str, Any] | dict[str, Any] | None),
    *,
    model_class: str,
    estimated_customer_debit_microcents: int,
) -> dict[str, Any]:
    if isinstance(request, SmrFactoryEffortBillingPreflightRequest):
        return request.to_wire()
    if isinstance(request, Mapping):
        return _factory_effort_preflight_request_from_mapping(
            request,
            model_class=model_class,
            estimated_customer_debit_microcents=estimated_customer_debit_microcents,
        ).to_wire()
    return SmrFactoryEffortBillingPreflightRequest(
        model_class=model_class,
        estimated_customer_debit_microcents=estimated_customer_debit_microcents,
    ).to_wire()


def _manual_grant_preview_request_from_mapping(
    request: Mapping[str, Any] | dict[str, Any],
    *,
    target_org_id: str | None,
    grant_kind: str,
    reason: str | None,
    scope: str,
    model_class: str | None,
    window_kind: str | None,
    amount_microcents: int | None,
    resets_at: str | None,
    expires_at: str | None,
    campaign_id: str | None,
    wallet_credit_microcents: int | None,
    metadata: Mapping[str, object] | None,
) -> SmrManualBillingGrantPreviewRequest:
    unknown = sorted(str(key) for key in request if key not in _MANUAL_GRANT_PREVIEW_REQUEST_FIELDS)
    if unknown:
        raise ValueError(
            "unknown SmrManualBillingGrantPreviewRequest fields: " + ", ".join(unknown)
        )
    return SmrManualBillingGrantPreviewRequest(
        target_org_id=_required_non_empty(
            request.get("target_org_id", target_org_id),
            field_name="target_org_id",
        ),
        grant_kind=_required_non_empty(
            request.get("grant_kind", grant_kind),
            field_name="grant_kind",
        ),
        reason=_required_non_empty(
            request.get("reason", reason),
            field_name="reason",
        ),
        scope=_required_non_empty(request.get("scope", scope), field_name="scope"),
        model_class=_optional_string(request.get("model_class", model_class)),
        window_kind=_optional_string(request.get("window_kind", window_kind)),
        amount_microcents=_optional_int(
            request.get("amount_microcents", amount_microcents),
            field_name="amount_microcents",
        ),
        resets_at=_optional_string(request.get("resets_at", resets_at)),
        expires_at=_optional_string(request.get("expires_at", expires_at)),
        campaign_id=_optional_string(request.get("campaign_id", campaign_id)),
        wallet_credit_microcents=_optional_int(
            request.get("wallet_credit_microcents", wallet_credit_microcents),
            field_name="wallet_credit_microcents",
        ),
        metadata=_metadata_dict(request.get("metadata", metadata)),
    )


def _manual_grant_preview_payload(
    request: (SmrManualBillingGrantPreviewRequest | Mapping[str, Any] | dict[str, Any] | None),
    *,
    target_org_id: str | None,
    grant_kind: str,
    reason: str | None,
    scope: str,
    model_class: str | None,
    window_kind: str | None,
    amount_microcents: int | None,
    resets_at: str | None,
    expires_at: str | None,
    campaign_id: str | None,
    wallet_credit_microcents: int | None,
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    if isinstance(request, SmrManualBillingGrantPreviewRequest):
        return request.to_wire()
    if isinstance(request, Mapping):
        return _manual_grant_preview_request_from_mapping(
            request,
            target_org_id=target_org_id,
            grant_kind=grant_kind,
            reason=reason,
            scope=scope,
            model_class=model_class,
            window_kind=window_kind,
            amount_microcents=amount_microcents,
            resets_at=resets_at,
            expires_at=expires_at,
            campaign_id=campaign_id,
            wallet_credit_microcents=wallet_credit_microcents,
            metadata=metadata,
        ).to_wire()
    return SmrManualBillingGrantPreviewRequest(
        target_org_id=_required_non_empty(target_org_id, field_name="target_org_id"),
        grant_kind=_required_non_empty(grant_kind, field_name="grant_kind"),
        reason=_required_non_empty(reason, field_name="reason"),
        scope=_required_non_empty(scope, field_name="scope"),
        model_class=model_class,
        window_kind=window_kind,
        amount_microcents=amount_microcents,
        resets_at=resets_at,
        expires_at=expires_at,
        campaign_id=campaign_id,
        wallet_credit_microcents=wallet_credit_microcents,
        metadata=metadata,
    ).to_wire()


class BillingAPI(_ClientNamespace):
    """Plan, preflight, and debit drawdown reads for the shared SMR pool."""

    def catalog(self) -> SmrBillingCatalog:
        return SmrBillingCatalog.from_wire(
            self._client._request_json("GET", "/smr/billing/catalog")
        )

    def plan(self) -> SmrBillingPlanSnapshot:
        return SmrBillingPlanSnapshot.from_wire(
            self._client._request_json("GET", "/smr/billing/plan")
        )

    def preview_manual_grant(
        self,
        request: (
            SmrManualBillingGrantPreviewRequest | Mapping[str, Any] | dict[str, Any] | None
        ) = None,
        *,
        target_org_id: str | None = None,
        grant_kind: str = "make_good",
        reason: str | None = None,
        scope: str = "org",
        model_class: str | None = None,
        window_kind: str | None = None,
        amount_microcents: int | None = None,
        resets_at: str | None = None,
        expires_at: str | None = None,
        campaign_id: str | None = None,
        wallet_credit_microcents: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> SmrManualBillingGrantPreview:
        payload = _manual_grant_preview_payload(
            request,
            target_org_id=target_org_id,
            grant_kind=grant_kind,
            reason=reason,
            scope=scope,
            model_class=model_class,
            window_kind=window_kind,
            amount_microcents=amount_microcents,
            resets_at=resets_at,
            expires_at=expires_at,
            campaign_id=campaign_id,
            wallet_credit_microcents=wallet_credit_microcents,
            metadata=metadata,
        )
        return SmrManualBillingGrantPreview.from_wire(
            self._client._request_json(
                "POST",
                "/smr/billing/admin/grants/preview",
                json_body=payload,
            )
        )

    def preview_promotion_discount(
        self,
        request: SmrPromotionDiscountPreviewRequest | Mapping[str, Any],
    ) -> SmrPromotionDiscountPreview:
        """Preview draft promotion economics without mutation or enforcement."""

        typed_request = (
            request
            if isinstance(request, SmrPromotionDiscountPreviewRequest)
            else SmrPromotionDiscountPreviewRequest.from_wire(request)
        )
        return SmrPromotionDiscountPreview.from_wire(
            self._client._request_json(
                "POST",
                "/smr/promotions/admin/discount-preview",
                json_body=typed_request.to_wire(),
            )
        )

    def run_drawdown(self, run_id: str) -> SmrBillingDrawdown:
        return SmrBillingDrawdown.from_wire(
            self._client._request_json("GET", f"/smr/billing/runs/{run_id}/drawdown")
        )

    def factory_effort_drawdown(self, factory_effort_id: str) -> SmrBillingDrawdown:
        return SmrBillingDrawdown.from_wire(
            self._client._request_json(
                "GET",
                f"/smr/billing/factory-efforts/{factory_effort_id}/drawdown",
            )
        )

    def dev_environment_drawdown(self, dev_environment_id: str) -> SmrBillingDrawdown:
        return SmrBillingDrawdown.from_wire(
            self._client._request_json(
                "GET",
                f"/smr/billing/dev-environments/{dev_environment_id}/drawdown",
            )
        )

    def preflight_run(
        self,
        request: SmrBillingPreflightRequest | Mapping[str, Any] | dict[str, Any] | None = None,
        *,
        model_class: str = "premium",
        estimated_customer_debit_microcents: int = 0,
        project_id: str | None = None,
    ) -> SmrBillingPreflight:
        payload = _preflight_payload(
            request,
            model_class=model_class,
            estimated_customer_debit_microcents=estimated_customer_debit_microcents,
            project_id=project_id,
        )
        return SmrBillingPreflight.from_wire(
            self._client._request_json(
                "POST",
                "/smr/billing/runs/preflight",
                json_body=payload,
            )
        )

    def preflight_factory_effort(
        self,
        factory_effort_id: str,
        request: (
            SmrFactoryEffortBillingPreflightRequest | Mapping[str, Any] | dict[str, Any] | None
        ) = None,
        *,
        model_class: str = "premium",
        estimated_customer_debit_microcents: int = 0,
    ) -> SmrBillingPreflight:
        payload = _factory_effort_preflight_payload(
            request,
            model_class=model_class,
            estimated_customer_debit_microcents=estimated_customer_debit_microcents,
        )
        return SmrBillingPreflight.from_wire(
            self._client._request_json(
                "POST",
                f"/smr/billing/factory-efforts/{factory_effort_id}/preflight",
                json_body=payload,
            )
        )

    def preflight_dev_environment(
        self,
        dev_environment_id: str,
        request: (
            SmrFactoryEffortBillingPreflightRequest | Mapping[str, Any] | dict[str, Any] | None
        ) = None,
        *,
        model_class: str = "value",
        estimated_customer_debit_microcents: int = 0,
    ) -> SmrBillingPreflight:
        payload = _factory_effort_preflight_payload(
            request,
            model_class=model_class,
            estimated_customer_debit_microcents=estimated_customer_debit_microcents,
        )
        return SmrBillingPreflight.from_wire(
            self._client._request_json(
                "POST",
                f"/smr/billing/dev-environments/{dev_environment_id}/preflight",
                json_body=payload,
            )
        )


__all__ = ["BillingAPI"]
