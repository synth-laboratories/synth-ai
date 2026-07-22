"""SMR promotion registry wire models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from synth_ai.core.research._legacy.models.types import (
    _int_value,
    _optional_bool,
    _optional_int,
    _optional_string,
    _require_array,
    _require_mapping,
    _require_string,
    _string_list,
)


@dataclass(frozen=True)
class SmrPromotionBenefitSummary:
    credit_grant_cents: int | None
    allowed_runbooks: list[str]
    allowed_models: list[str]
    effective_plan: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrPromotionBenefitSummary:
        mapping = _require_mapping(payload, label="SMR promotion benefit summary")
        return cls(
            credit_grant_cents=_optional_int(mapping, "credit_grant_cents"),
            allowed_runbooks=_string_list(
                _require_array(mapping, "allowed_runbooks", label="allowed_runbooks"),
                label="allowed_runbooks",
            ),
            allowed_models=_string_list(
                _require_array(mapping, "allowed_models", label="allowed_models"),
                label="allowed_models",
            ),
            effective_plan=_optional_string(mapping, "effective_plan"),
        )


@dataclass(frozen=True)
class SmrPromotionCampaignPublic:
    campaign_id: str
    display_name: str
    description: str | None
    status: str
    claim_window_start_at: str
    claim_window_end_at: str
    grant_ttl_days: int
    benefits_summary: SmrPromotionBenefitSummary

    @classmethod
    def from_wire(cls, payload: object) -> SmrPromotionCampaignPublic:
        mapping = _require_mapping(payload, label="SMR promotion campaign")
        return cls(
            campaign_id=_require_string(
                mapping, "campaign_id", label="SMR promotion campaign.campaign_id"
            ),
            display_name=_require_string(
                mapping,
                "display_name",
                label="SMR promotion campaign.display_name",
            ),
            description=_optional_string(mapping, "description"),
            status=_require_string(mapping, "status", label="SMR promotion campaign.status"),
            claim_window_start_at=_require_string(
                mapping,
                "claim_window_start_at",
                label="SMR promotion campaign.claim_window_start_at",
            ),
            claim_window_end_at=_require_string(
                mapping,
                "claim_window_end_at",
                label="SMR promotion campaign.claim_window_end_at",
            ),
            grant_ttl_days=_int_value(mapping, "grant_ttl_days"),
            benefits_summary=SmrPromotionBenefitSummary.from_wire(
                mapping.get("benefits_summary") or {}
            ),
        )


@dataclass(frozen=True)
class SmrPromotionPublicCatalog:
    campaigns: list[SmrPromotionCampaignPublic]
    as_of: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrPromotionPublicCatalog:
        mapping = _require_mapping(payload, label="SMR promotion public catalog")
        return cls(
            campaigns=[
                SmrPromotionCampaignPublic.from_wire(item)
                for item in _require_array(mapping, "campaigns", label="campaigns")
            ],
            as_of=_require_string(mapping, "as_of", label="SMR promotion public catalog.as_of"),
        )


@dataclass(frozen=True)
class SmrPromotionGrantView:
    grant_id: str
    status: str
    claimed_at: str
    expires_at: str
    effective_plan: str | None
    credit_amount_cents: int

    @classmethod
    def from_wire(cls, payload: object) -> SmrPromotionGrantView:
        mapping = _require_mapping(payload, label="SMR promotion grant")
        return cls(
            grant_id=_require_string(mapping, "grant_id", label="SMR promotion grant.grant_id"),
            status=_require_string(mapping, "status", label="SMR promotion grant.status"),
            claimed_at=_require_string(
                mapping, "claimed_at", label="SMR promotion grant.claimed_at"
            ),
            expires_at=_require_string(
                mapping, "expires_at", label="SMR promotion grant.expires_at"
            ),
            effective_plan=_optional_string(mapping, "effective_plan"),
            credit_amount_cents=_int_value(mapping, "credit_amount_cents"),
        )


@dataclass(frozen=True)
class SmrOrgPromotionView:
    campaign_id: str
    display_name: str
    claim_window_open: bool
    eligible_to_claim: bool
    eligibility_reason: str | None
    grant: SmrPromotionGrantView | None
    benefits: SmrPromotionBenefitSummary

    @classmethod
    def from_wire(cls, payload: object) -> SmrOrgPromotionView:
        mapping = _require_mapping(payload, label="SMR org promotion view")
        grant_payload = mapping.get("grant")
        grant = (
            SmrPromotionGrantView.from_wire(grant_payload)
            if isinstance(grant_payload, Mapping)
            else None
        )
        claim_window_open = _optional_bool(mapping, "claim_window_open")
        eligible_to_claim = _optional_bool(mapping, "eligible_to_claim")
        return cls(
            campaign_id=_require_string(
                mapping,
                "campaign_id",
                label="SMR org promotion view.campaign_id",
            ),
            display_name=_require_string(
                mapping,
                "display_name",
                label="SMR org promotion view.display_name",
            ),
            claim_window_open=bool(claim_window_open),
            eligible_to_claim=bool(eligible_to_claim),
            eligibility_reason=_optional_string(mapping, "eligibility_reason"),
            grant=grant,
            benefits=SmrPromotionBenefitSummary.from_wire(mapping.get("benefits") or {}),
        )


@dataclass(frozen=True)
class SmrPromotionMineResponse:
    campaigns: list[SmrOrgPromotionView]

    @classmethod
    def from_wire(cls, payload: object) -> SmrPromotionMineResponse:
        mapping = _require_mapping(payload, label="SMR promotion mine response")
        return cls(
            campaigns=[
                SmrOrgPromotionView.from_wire(item)
                for item in _require_array(mapping, "campaigns", label="campaigns")
            ],
        )


@dataclass(frozen=True)
class SmrPromotionDiscountPreviewRequest:
    """Inputs for a backend-owned draft promotion economics preview."""

    campaign_id: str
    nominal_customer_debit_microcents: int
    provider_cost_pico_usd: int

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise ValueError("campaign_id is required")
        for field_name, value in (
            (
                "nominal_customer_debit_microcents",
                self.nominal_customer_debit_microcents,
            ),
            ("provider_cost_pico_usd", self.provider_cost_pico_usd),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    @classmethod
    def from_wire(cls, payload: object) -> SmrPromotionDiscountPreviewRequest:
        mapping = _require_mapping(
            payload,
            label="SMR promotion discount preview request",
        )
        expected_fields = {
            "campaign_id",
            "nominal_customer_debit_microcents",
            "provider_cost_pico_usd",
        }
        unknown_fields = sorted(key for key in mapping if key not in expected_fields)
        if unknown_fields:
            raise ValueError(
                "unknown SMR promotion discount preview request fields: "
                + ", ".join(unknown_fields)
            )
        return cls(
            campaign_id=_require_string(
                mapping,
                "campaign_id",
                label="SMR promotion discount preview request.campaign_id",
            ),
            nominal_customer_debit_microcents=_int_value(
                mapping,
                "nominal_customer_debit_microcents",
            ),
            provider_cost_pico_usd=_int_value(mapping, "provider_cost_pico_usd"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id.strip(),
            "nominal_customer_debit_microcents": (self.nominal_customer_debit_microcents),
            "provider_cost_pico_usd": self.provider_cost_pico_usd,
        }


@dataclass(frozen=True)
class SmrPromotionDiscountPreview:
    """Backend-authored draft economics; never an enforcement or activation receipt."""

    campaign_id: str
    campaign_status: str
    claim_window_start_at: str
    claim_window_end_at: str
    price_source: str
    eligible_usage: list[str]
    stacking_policy: str
    provider_model_exclusions: list[str]
    generated_at: str
    preview_formula_version: str
    enforcement_status: Literal["not_implemented"]
    customer_discount_percent: int
    nominal_customer_debit_microcents: int
    customer_discount_microcents: int
    customer_charge_microcents: int
    provider_cost_pico_usd: int
    synth_subsidy_microcents: int
    global_subsidy_cap_microcents: int
    org_subsidy_cap_microcents: int

    @classmethod
    def from_wire(cls, payload: object) -> SmrPromotionDiscountPreview:
        mapping = _require_mapping(payload, label="SMR promotion discount preview")
        enforcement_status = _require_string(
            mapping,
            "enforcement_status",
            label="SMR promotion discount preview.enforcement_status",
        )
        if enforcement_status != "not_implemented":
            raise ValueError(
                "SMR promotion discount preview.enforcement_status must be not_implemented"
            )
        eligible_usage = _string_list(
            _require_array(mapping, "eligible_usage", label="eligible_usage"),
            label="eligible_usage",
        )
        if not eligible_usage:
            raise ValueError("SMR promotion discount preview.eligible_usage is required")
        customer_discount_percent = _int_value(
            mapping,
            "customer_discount_percent",
        )
        if not 1 <= customer_discount_percent <= 100:
            raise ValueError(
                "SMR promotion discount preview.customer_discount_percent must be between 1 and 100"
            )
        amounts = {
            field_name: _int_value(mapping, field_name)
            for field_name in (
                "nominal_customer_debit_microcents",
                "customer_discount_microcents",
                "customer_charge_microcents",
                "provider_cost_pico_usd",
                "synth_subsidy_microcents",
                "global_subsidy_cap_microcents",
                "org_subsidy_cap_microcents",
            )
        }
        if any(value < 0 for value in amounts.values()):
            raise ValueError("SMR promotion discount preview amounts must be non-negative")
        if (
            amounts["global_subsidy_cap_microcents"] == 0
            or amounts["org_subsidy_cap_microcents"] == 0
        ):
            raise ValueError("SMR promotion discount preview subsidy caps must be positive")
        return cls(
            campaign_id=_require_string(
                mapping,
                "campaign_id",
                label="SMR promotion discount preview.campaign_id",
            ),
            campaign_status=_require_string(
                mapping,
                "campaign_status",
                label="SMR promotion discount preview.campaign_status",
            ),
            claim_window_start_at=_require_string(
                mapping,
                "claim_window_start_at",
                label="SMR promotion discount preview.claim_window_start_at",
            ),
            claim_window_end_at=_require_string(
                mapping,
                "claim_window_end_at",
                label="SMR promotion discount preview.claim_window_end_at",
            ),
            price_source=_require_string(
                mapping,
                "price_source",
                label="SMR promotion discount preview.price_source",
            ),
            eligible_usage=eligible_usage,
            stacking_policy=_require_string(
                mapping,
                "stacking_policy",
                label="SMR promotion discount preview.stacking_policy",
            ),
            provider_model_exclusions=_string_list(
                _require_array(
                    mapping,
                    "provider_model_exclusions",
                    label="provider_model_exclusions",
                ),
                label="provider_model_exclusions",
            ),
            generated_at=_require_string(
                mapping,
                "generated_at",
                label="SMR promotion discount preview.generated_at",
            ),
            preview_formula_version=_require_string(
                mapping,
                "preview_formula_version",
                label="SMR promotion discount preview.preview_formula_version",
            ),
            enforcement_status=enforcement_status,
            customer_discount_percent=customer_discount_percent,
            nominal_customer_debit_microcents=amounts["nominal_customer_debit_microcents"],
            customer_discount_microcents=amounts["customer_discount_microcents"],
            customer_charge_microcents=amounts["customer_charge_microcents"],
            provider_cost_pico_usd=amounts["provider_cost_pico_usd"],
            synth_subsidy_microcents=amounts["synth_subsidy_microcents"],
            global_subsidy_cap_microcents=amounts["global_subsidy_cap_microcents"],
            org_subsidy_cap_microcents=amounts["org_subsidy_cap_microcents"],
        )


@dataclass(frozen=True)
class SmrPromotionUpsertRequest:
    campaign_id: str
    display_name: str
    claim_window_start_at: str
    claim_window_end_at: str
    description: str | None = None
    status: str = "active"
    grant_ttl_days: int = 7
    campaign_credit_ceiling_cents: int | None = None
    concurrency_limit: int | None = None
    eligibility_kind: str = "signup_in_window"
    autumn_product_id: str | None = None

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "campaign_id": self.campaign_id,
            "display_name": self.display_name,
            "status": self.status,
            "claim_window_start_at": self.claim_window_start_at,
            "claim_window_end_at": self.claim_window_end_at,
            "grant_ttl_days": self.grant_ttl_days,
            "eligibility_kind": self.eligibility_kind,
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.campaign_credit_ceiling_cents is not None:
            payload["campaign_credit_ceiling_cents"] = self.campaign_credit_ceiling_cents
        if self.concurrency_limit is not None:
            payload["concurrency_limit"] = self.concurrency_limit
        if self.autumn_product_id is not None:
            payload["autumn_product_id"] = self.autumn_product_id
        return payload


@dataclass(frozen=True)
class SmrPromotionRetireResult:
    campaign_id: str
    status: str
    claim_window_end_at: str
    retired_at: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrPromotionRetireResult:
        mapping = _require_mapping(payload, label="SMR promotion retire result")
        return cls(
            campaign_id=_require_string(
                mapping,
                "campaign_id",
                label="SMR promotion retire result.campaign_id",
            ),
            status=_require_string(mapping, "status", label="SMR promotion retire result.status"),
            claim_window_end_at=_require_string(
                mapping,
                "claim_window_end_at",
                label="SMR promotion retire result.claim_window_end_at",
            ),
            retired_at=_require_string(
                mapping,
                "retired_at",
                label="SMR promotion retire result.retired_at",
            ),
        )


__all__ = [
    "SmrOrgPromotionView",
    "SmrPromotionBenefitSummary",
    "SmrPromotionCampaignPublic",
    "SmrPromotionDiscountPreview",
    "SmrPromotionDiscountPreviewRequest",
    "SmrPromotionGrantView",
    "SmrPromotionMineResponse",
    "SmrPromotionPublicCatalog",
    "SmrPromotionRetireResult",
    "SmrPromotionUpsertRequest",
]
