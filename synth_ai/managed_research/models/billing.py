"""SMR billing plan, preflight, and drawdown models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from synth_ai.managed_research.models.types import (
    _int_value,
    _optional_array,
    _optional_bool,
    _optional_int,
    _optional_object_dict,
    _optional_string,
    _require_array,
    _require_mapping,
    _require_string,
    _string_list,
)


def _required_bool(payload: Mapping[str, object], key: str, *, label: str) -> bool:
    value = _optional_bool(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_wire_value(payload: Mapping[str, object], key: str) -> object | None:
    if key not in payload:
        return None
    return payload[key]


def _required_string_list(
    payload: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> list[str]:
    return _string_list(_require_array(payload, key, label=label), label=label)


@dataclass(frozen=True)
class SmrBillingAllowanceWindow:
    model_class: str
    window_kind: str
    cap_microcents: int
    consumed_microcents: int
    remaining_microcents: int
    window_started_at: str
    resets_at: str
    source: str
    reset_anchor_id: str | None
    promo_campaign_id: str | None
    state: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingAllowanceWindow:
        mapping = _require_mapping(payload, label="SMR billing allowance window")
        return cls(
            model_class=_require_string(
                mapping,
                "model_class",
                label="SMR billing allowance window.model_class",
            ),
            window_kind=_require_string(
                mapping,
                "window_kind",
                label="SMR billing allowance window.window_kind",
            ),
            cap_microcents=_int_value(mapping, "cap_microcents"),
            consumed_microcents=_int_value(mapping, "consumed_microcents"),
            remaining_microcents=_int_value(mapping, "remaining_microcents"),
            window_started_at=_require_string(
                mapping,
                "window_started_at",
                label="SMR billing allowance window.window_started_at",
            ),
            resets_at=_require_string(
                mapping,
                "resets_at",
                label="SMR billing allowance window.resets_at",
            ),
            source=_require_string(
                mapping,
                "source",
                label="SMR billing allowance window.source",
            ),
            reset_anchor_id=_optional_string(mapping, "reset_anchor_id"),
            promo_campaign_id=_optional_string(mapping, "promo_campaign_id"),
            state=_optional_string(mapping, "state") or "active",
        )


@dataclass(frozen=True)
class SmrBillingPlanInfo:
    tier: str
    display_name: str
    legacy_alias: str
    billing_mode: str
    subscription_source: str
    renewal_or_roll_policy: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingPlanInfo:
        mapping = _require_mapping(payload, label="SMR billing plan info")
        return cls(
            tier=_require_string(mapping, "tier", label="SMR billing plan info.tier"),
            display_name=_require_string(
                mapping,
                "display_name",
                label="SMR billing plan info.display_name",
            ),
            legacy_alias=_require_string(
                mapping,
                "legacy_alias",
                label="SMR billing plan info.legacy_alias",
            ),
            billing_mode=_require_string(
                mapping,
                "billing_mode",
                label="SMR billing plan info.billing_mode",
            ),
            subscription_source=_require_string(
                mapping,
                "subscription_source",
                label="SMR billing plan info.subscription_source",
            ),
            renewal_or_roll_policy=_require_string(
                mapping,
                "renewal_or_roll_policy",
                label="SMR billing plan info.renewal_or_roll_policy",
            ),
        )

    @classmethod
    def fallback(
        cls,
        *,
        plan_tier: str,
        legacy_plan: str,
        billing_mode: str,
    ) -> SmrBillingPlanInfo:
        return cls(
            tier=plan_tier,
            display_name=plan_tier,
            legacy_alias=legacy_plan,
            billing_mode=billing_mode,
            subscription_source="unknown",
            renewal_or_roll_policy="unknown",
        )


@dataclass(frozen=True)
class SmrBillingWalletSource:
    credit_source: str
    campaign_id: str | None
    granted_microcents: int
    expires_at: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingWalletSource:
        mapping = _require_mapping(payload, label="SMR billing wallet source")
        return cls(
            credit_source=_require_string(
                mapping,
                "credit_source",
                label="SMR billing wallet source.credit_source",
            ),
            campaign_id=_optional_string(mapping, "campaign_id"),
            granted_microcents=_int_value(mapping, "granted_microcents"),
            expires_at=_optional_string(mapping, "expires_at"),
        )


@dataclass(frozen=True)
class SmrBillingWallet:
    balance_microcents: int
    expires_at: str | None
    sources: list[SmrBillingWalletSource]

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingWallet:
        mapping = _require_mapping(payload, label="SMR billing wallet")
        return cls(
            balance_microcents=_int_value(mapping, "balance_microcents"),
            expires_at=_optional_string(mapping, "expires_at"),
            sources=[
                SmrBillingWalletSource.from_wire(item)
                for item in _optional_array(mapping, "sources")
            ],
        )


@dataclass(frozen=True)
class SmrBillingResetGrant:
    reset_grant_id: str
    grant_kind: str
    status: str
    applies_to: str
    amount_microcents: int | None
    model_class: str | None
    window_kind: str | None
    resets_at: str | None
    expires_at: str | None
    campaign_id: str | None
    reason_label: str
    created_at: str
    applied_at: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingResetGrant:
        mapping = _require_mapping(payload, label="SMR billing reset grant")
        return cls(
            reset_grant_id=_require_string(
                mapping,
                "reset_grant_id",
                label="SMR billing reset grant.reset_grant_id",
            ),
            grant_kind=_require_string(
                mapping,
                "grant_kind",
                label="SMR billing reset grant.grant_kind",
            ),
            status=_require_string(
                mapping,
                "status",
                label="SMR billing reset grant.status",
            ),
            applies_to=_require_string(
                mapping,
                "applies_to",
                label="SMR billing reset grant.applies_to",
            ),
            amount_microcents=_optional_int(mapping, "amount_microcents"),
            model_class=_optional_string(mapping, "model_class"),
            window_kind=_optional_string(mapping, "window_kind"),
            resets_at=_optional_string(mapping, "resets_at"),
            expires_at=_optional_string(mapping, "expires_at"),
            campaign_id=_optional_string(mapping, "campaign_id"),
            reason_label=_require_string(
                mapping,
                "reason_label",
                label="SMR billing reset grant.reason_label",
            ),
            created_at=_require_string(
                mapping,
                "created_at",
                label="SMR billing reset grant.created_at",
            ),
            applied_at=_optional_string(mapping, "applied_at"),
        )


@dataclass(frozen=True)
class SmrBillingResetBank:
    available_count: int
    expiring_count: int
    grants: list[SmrBillingResetGrant]

    @classmethod
    def from_wire(cls, payload: object | None) -> SmrBillingResetBank:
        if payload is None:
            return cls(available_count=0, expiring_count=0, grants=[])
        mapping = _require_mapping(payload, label="SMR billing reset bank")
        return cls(
            available_count=_int_value(mapping, "available_count"),
            expiring_count=_int_value(mapping, "expiring_count"),
            grants=[
                SmrBillingResetGrant.from_wire(item)
                for item in _optional_array(mapping, "grants")
            ],
        )


@dataclass(frozen=True)
class SmrBillingPromotionGrant:
    grant_id: str
    campaign_id: str
    status: str
    claimed_at: str
    expires_at: str
    effective_plan: str | None
    credit_amount_cents: int

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingPromotionGrant:
        mapping = _require_mapping(payload, label="SMR billing promotion grant")
        return cls(
            grant_id=_require_string(
                mapping,
                "grant_id",
                label="SMR billing promotion grant.grant_id",
            ),
            campaign_id=_require_string(
                mapping,
                "campaign_id",
                label="SMR billing promotion grant.campaign_id",
            ),
            status=_require_string(
                mapping,
                "status",
                label="SMR billing promotion grant.status",
            ),
            claimed_at=_require_string(
                mapping,
                "claimed_at",
                label="SMR billing promotion grant.claimed_at",
            ),
            expires_at=_require_string(
                mapping,
                "expires_at",
                label="SMR billing promotion grant.expires_at",
            ),
            effective_plan=_optional_string(mapping, "effective_plan"),
            credit_amount_cents=_int_value(mapping, "credit_amount_cents"),
        )


@dataclass(frozen=True)
class SmrBillingPromotionCampaign:
    campaign_id: str
    display_name: str
    claim_window_open: bool
    eligible_to_claim: bool
    eligibility_reason: str | None
    benefits: dict[str, object]
    grant: SmrBillingPromotionGrant | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingPromotionCampaign:
        mapping = _require_mapping(payload, label="SMR billing promotion campaign")
        grant_payload = _optional_wire_value(mapping, "grant")
        return cls(
            campaign_id=_require_string(
                mapping,
                "campaign_id",
                label="SMR billing promotion campaign.campaign_id",
            ),
            display_name=_require_string(
                mapping,
                "display_name",
                label="SMR billing promotion campaign.display_name",
            ),
            claim_window_open=_required_bool(
                mapping,
                "claim_window_open",
                label="SMR billing promotion campaign.claim_window_open",
            ),
            eligible_to_claim=_required_bool(
                mapping,
                "eligible_to_claim",
                label="SMR billing promotion campaign.eligible_to_claim",
            ),
            eligibility_reason=_optional_string(mapping, "eligibility_reason"),
            benefits=_optional_object_dict(_optional_wire_value(mapping, "benefits")),
            grant=SmrBillingPromotionGrant.from_wire(grant_payload)
            if grant_payload is not None
            else None,
        )


@dataclass(frozen=True)
class SmrBillingPromotions:
    active: list[SmrBillingPromotionCampaign]
    claimable: list[SmrBillingPromotionCampaign]
    expired: list[SmrBillingPromotionCampaign]
    grants: list[SmrBillingPromotionGrant]

    @classmethod
    def from_wire(cls, payload: object | None) -> SmrBillingPromotions:
        if payload is None:
            return cls(active=[], claimable=[], expired=[], grants=[])
        mapping = _require_mapping(payload, label="SMR billing promotions")
        return cls(
            active=[
                SmrBillingPromotionCampaign.from_wire(item)
                for item in _optional_array(mapping, "active")
            ],
            claimable=[
                SmrBillingPromotionCampaign.from_wire(item)
                for item in _optional_array(mapping, "claimable")
            ],
            expired=[
                SmrBillingPromotionCampaign.from_wire(item)
                for item in _optional_array(mapping, "expired")
            ],
            grants=[
                SmrBillingPromotionGrant.from_wire(item)
                for item in _optional_array(mapping, "grants")
            ],
        )


@dataclass(frozen=True)
class SmrBillingUsageTotals:
    event_count: int
    nominal_microcents: int
    billed_microcents: int
    internal_cost_microcents: int

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingUsageTotals:
        mapping = _require_mapping(payload, label="SMR billing usage totals")
        return cls(
            event_count=_int_value(mapping, "event_count"),
            nominal_microcents=_int_value(mapping, "nominal_microcents"),
            billed_microcents=_int_value(mapping, "billed_microcents"),
            internal_cost_microcents=_int_value(mapping, "internal_cost_microcents"),
        )

    @classmethod
    def zero(cls) -> SmrBillingUsageTotals:
        return cls(
            event_count=0,
            nominal_microcents=0,
            billed_microcents=0,
            internal_cost_microcents=0,
        )


@dataclass(frozen=True)
class SmrBillingUsageBreakdown:
    key: str
    event_count: int
    nominal_microcents: int
    billed_microcents: int
    internal_cost_microcents: int

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingUsageBreakdown:
        mapping = _require_mapping(payload, label="SMR billing usage breakdown")
        return cls(
            key=_require_string(mapping, "key", label="SMR billing usage breakdown.key"),
            event_count=_int_value(mapping, "event_count"),
            nominal_microcents=_int_value(mapping, "nominal_microcents"),
            billed_microcents=_int_value(mapping, "billed_microcents"),
            internal_cost_microcents=_int_value(mapping, "internal_cost_microcents"),
        )


@dataclass(frozen=True)
class SmrBillingUsageSummary:
    today: SmrBillingUsageTotals
    seven_days: SmrBillingUsageTotals
    thirty_days: SmrBillingUsageTotals
    by_surface: list[SmrBillingUsageBreakdown]
    by_project: list[SmrBillingUsageBreakdown]
    by_factory: list[SmrBillingUsageBreakdown]
    by_actor: list[SmrBillingUsageBreakdown]

    @classmethod
    def from_wire(cls, payload: object | None) -> SmrBillingUsageSummary:
        if payload is None:
            return cls.empty()
        mapping = _require_mapping(payload, label="SMR billing usage summary")
        return cls(
            today=SmrBillingUsageTotals.from_wire(mapping["today"]),
            seven_days=SmrBillingUsageTotals.from_wire(mapping["seven_days"]),
            thirty_days=SmrBillingUsageTotals.from_wire(mapping["thirty_days"]),
            by_surface=[
                SmrBillingUsageBreakdown.from_wire(item)
                for item in _optional_array(mapping, "by_surface")
            ],
            by_project=[
                SmrBillingUsageBreakdown.from_wire(item)
                for item in _optional_array(mapping, "by_project")
            ],
            by_factory=[
                SmrBillingUsageBreakdown.from_wire(item)
                for item in _optional_array(mapping, "by_factory")
            ],
            by_actor=[
                SmrBillingUsageBreakdown.from_wire(item)
                for item in _optional_array(mapping, "by_actor")
            ],
        )

    @classmethod
    def empty(cls) -> SmrBillingUsageSummary:
        return cls(
            today=SmrBillingUsageTotals.zero(),
            seven_days=SmrBillingUsageTotals.zero(),
            thirty_days=SmrBillingUsageTotals.zero(),
            by_surface=[],
            by_project=[],
            by_factory=[],
            by_actor=[],
        )


@dataclass(frozen=True)
class SmrBillingRecoveryAction:
    action_id: str
    label: str
    surface: str
    requires_admin: bool
    preview_endpoint: str | None
    execute_endpoint: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingRecoveryAction:
        mapping = _require_mapping(payload, label="SMR billing recovery action")
        return cls(
            action_id=_require_string(
                mapping,
                "action_id",
                label="SMR billing recovery action.action_id",
            ),
            label=_require_string(
                mapping,
                "label",
                label="SMR billing recovery action.label",
            ),
            surface=_require_string(
                mapping,
                "surface",
                label="SMR billing recovery action.surface",
            ),
            requires_admin=_required_bool(
                mapping,
                "requires_admin",
                label="SMR billing recovery action.requires_admin",
            ),
            preview_endpoint=_optional_string(mapping, "preview_endpoint"),
            execute_endpoint=_optional_string(mapping, "execute_endpoint"),
        )


@dataclass(frozen=True)
class SmrBillingBlockedDetail:
    is_blocked: bool
    reason_code: str | None
    human_message: str | None
    recovery_actions: list[SmrBillingRecoveryAction]

    @classmethod
    def from_wire(
        cls,
        payload: object | None,
        *,
        blocked: bool,
        blocked_reason: str | None,
    ) -> SmrBillingBlockedDetail:
        if payload is None:
            return cls(
                is_blocked=blocked,
                reason_code=blocked_reason,
                human_message=blocked_reason,
                recovery_actions=[],
            )
        mapping = _require_mapping(payload, label="SMR billing blocked detail")
        return cls(
            is_blocked=_required_bool(
                mapping,
                "is_blocked",
                label="SMR billing blocked detail.is_blocked",
            ),
            reason_code=_optional_string(mapping, "reason_code"),
            human_message=_optional_string(mapping, "human_message"),
            recovery_actions=[
                SmrBillingRecoveryAction.from_wire(item)
                for item in _optional_array(mapping, "recovery_actions")
            ],
        )


@dataclass(frozen=True)
class SmrBillingCatalogAllowance:
    model_class: str
    five_hour_microcents: int
    weekly_microcents: int

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingCatalogAllowance:
        mapping = _require_mapping(payload, label="SMR billing catalog allowance")
        return cls(
            model_class=_require_string(
                mapping,
                "model_class",
                label="SMR billing catalog allowance.model_class",
            ),
            five_hour_microcents=_int_value(mapping, "five_hour_microcents"),
            weekly_microcents=_int_value(mapping, "weekly_microcents"),
        )


@dataclass(frozen=True)
class SmrBillingCatalogPlan:
    plan_tier: str
    display_name: str
    monthly_price_microcents: int
    allowances: list[SmrBillingCatalogAllowance]

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingCatalogPlan:
        mapping = _require_mapping(payload, label="SMR billing catalog plan")
        return cls(
            plan_tier=_require_string(
                mapping,
                "plan_tier",
                label="SMR billing catalog plan.plan_tier",
            ),
            display_name=_require_string(
                mapping,
                "display_name",
                label="SMR billing catalog plan.display_name",
            ),
            monthly_price_microcents=_int_value(
                mapping,
                "monthly_price_microcents",
            ),
            allowances=[
                SmrBillingCatalogAllowance.from_wire(item)
                for item in _require_array(
                    mapping,
                    "allowances",
                    label="SMR billing catalog plan.allowances",
                )
            ],
        )


@dataclass(frozen=True)
class SmrBillingCatalog:
    billing_mode: str
    reset_policy: str
    debit_pool_order: list[str]
    plans: list[SmrBillingCatalogPlan]
    generated_at: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingCatalog:
        mapping = _require_mapping(payload, label="SMR billing catalog")
        return cls(
            billing_mode=_require_string(
                mapping,
                "billing_mode",
                label="SMR billing catalog.billing_mode",
            ),
            reset_policy=_require_string(
                mapping,
                "reset_policy",
                label="SMR billing catalog.reset_policy",
            ),
            debit_pool_order=_required_string_list(
                mapping,
                "debit_pool_order",
                label="SMR billing catalog.debit_pool_order",
            ),
            plans=[
                SmrBillingCatalogPlan.from_wire(item)
                for item in _require_array(
                    mapping,
                    "plans",
                    label="SMR billing catalog.plans",
                )
            ],
            generated_at=_require_string(
                mapping,
                "generated_at",
                label="SMR billing catalog.generated_at",
            ),
        )


@dataclass(frozen=True)
class SmrBillingPlanSnapshot:
    schema_version: str
    org_id: str
    plan_tier: str
    legacy_plan: str
    billing_mode: str
    plan: SmrBillingPlanInfo
    allowance_windows: list[SmrBillingAllowanceWindow]
    wallet: SmrBillingWallet
    reset_bank: SmrBillingResetBank
    promotions: SmrBillingPromotions
    usage_summary: SmrBillingUsageSummary
    blocked: bool
    blocked_reason: str | None
    blocked_detail: SmrBillingBlockedDetail
    next_actions: list[SmrBillingRecoveryAction]
    generated_at: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingPlanSnapshot:
        mapping = _require_mapping(payload, label="SMR billing plan snapshot")
        plan_tier = _require_string(
            mapping,
            "plan_tier",
            label="SMR billing plan snapshot.plan_tier",
        )
        legacy_plan = _require_string(
            mapping,
            "legacy_plan",
            label="SMR billing plan snapshot.legacy_plan",
        )
        billing_mode = _require_string(
            mapping,
            "billing_mode",
            label="SMR billing plan snapshot.billing_mode",
        )
        blocked = _required_bool(
            mapping,
            "blocked",
            label="SMR billing plan snapshot.blocked",
        )
        blocked_reason = _optional_string(mapping, "blocked_reason")
        plan_payload = _optional_wire_value(mapping, "plan")
        return cls(
            schema_version=_optional_string(mapping, "schema_version")
            or "smr_billing_plan.v0",
            org_id=_require_string(
                mapping,
                "org_id",
                label="SMR billing plan snapshot.org_id",
            ),
            plan_tier=plan_tier,
            legacy_plan=legacy_plan,
            billing_mode=billing_mode,
            plan=SmrBillingPlanInfo.from_wire(plan_payload)
            if plan_payload is not None
            else SmrBillingPlanInfo.fallback(
                plan_tier=plan_tier,
                legacy_plan=legacy_plan,
                billing_mode=billing_mode,
            ),
            allowance_windows=[
                SmrBillingAllowanceWindow.from_wire(item)
                for item in _optional_array(mapping, "allowance_windows")
            ],
            wallet=SmrBillingWallet.from_wire(mapping["wallet"]),
            reset_bank=SmrBillingResetBank.from_wire(
                _optional_wire_value(mapping, "reset_bank")
            ),
            promotions=SmrBillingPromotions.from_wire(
                _optional_wire_value(mapping, "promotions")
            ),
            usage_summary=SmrBillingUsageSummary.from_wire(
                _optional_wire_value(mapping, "usage_summary")
            ),
            blocked=blocked,
            blocked_reason=blocked_reason,
            blocked_detail=SmrBillingBlockedDetail.from_wire(
                _optional_wire_value(mapping, "blocked_detail"),
                blocked=blocked,
                blocked_reason=blocked_reason,
            ),
            next_actions=[
                SmrBillingRecoveryAction.from_wire(item)
                for item in _optional_array(mapping, "next_actions")
            ],
            generated_at=_require_string(
                mapping,
                "generated_at",
                label="SMR billing plan snapshot.generated_at",
            ),
        )


@dataclass(frozen=True)
class SmrManualBillingGrantPreview:
    target_org_id: str
    grant_kind: str
    wallet_credit_microcents: int
    created_by: str
    will_create_reset_grant: bool
    will_create_wallet_ledger_entry: bool
    will_update_allowance_windows: bool
    execute_endpoint: str
    effect_summary: list[str]

    @classmethod
    def from_wire(cls, payload: object) -> SmrManualBillingGrantPreview:
        mapping = _require_mapping(payload, label="SMR manual billing grant preview")
        return cls(
            target_org_id=_require_string(
                mapping,
                "target_org_id",
                label="SMR manual billing grant preview.target_org_id",
            ),
            grant_kind=_require_string(
                mapping,
                "grant_kind",
                label="SMR manual billing grant preview.grant_kind",
            ),
            wallet_credit_microcents=_int_value(
                mapping,
                "wallet_credit_microcents",
            ),
            created_by=_require_string(
                mapping,
                "created_by",
                label="SMR manual billing grant preview.created_by",
            ),
            will_create_reset_grant=_required_bool(
                mapping,
                "will_create_reset_grant",
                label="SMR manual billing grant preview.will_create_reset_grant",
            ),
            will_create_wallet_ledger_entry=_required_bool(
                mapping,
                "will_create_wallet_ledger_entry",
                label="SMR manual billing grant preview.will_create_wallet_ledger_entry",
            ),
            will_update_allowance_windows=_required_bool(
                mapping,
                "will_update_allowance_windows",
                label="SMR manual billing grant preview.will_update_allowance_windows",
            ),
            execute_endpoint=_require_string(
                mapping,
                "execute_endpoint",
                label="SMR manual billing grant preview.execute_endpoint",
            ),
            effect_summary=_required_string_list(
                mapping,
                "effect_summary",
                label="SMR manual billing grant preview.effect_summary",
            ),
        )


@dataclass(frozen=True)
class SmrManualBillingGrantPreviewRequest:
    target_org_id: str
    grant_kind: str
    reason: str
    scope: str = "org"
    model_class: str | None = None
    window_kind: str | None = None
    amount_microcents: int | None = None
    resets_at: str | None = None
    expires_at: str | None = None
    campaign_id: str | None = None
    wallet_credit_microcents: int | None = None
    metadata: Mapping[str, object] | None = None

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "target_org_id": self.target_org_id,
            "grant_kind": self.grant_kind,
            "reason": self.reason,
            "scope": self.scope,
            "metadata": dict(self.metadata or {}),
        }
        if self.model_class is not None:
            payload["model_class"] = self.model_class
        if self.window_kind is not None:
            payload["window_kind"] = self.window_kind
        if self.amount_microcents is not None:
            payload["amount_microcents"] = int(self.amount_microcents)
        if self.resets_at is not None:
            payload["resets_at"] = self.resets_at
        if self.expires_at is not None:
            payload["expires_at"] = self.expires_at
        if self.campaign_id is not None:
            payload["campaign_id"] = self.campaign_id
        if self.wallet_credit_microcents is not None:
            payload["wallet_credit_microcents"] = int(
                self.wallet_credit_microcents
            )
        return payload


@dataclass(frozen=True)
class SmrBillingDebit:
    billing_debit_id: str
    surface: str
    model_class: str
    debit_pool: str
    customer_debit_microcents: int
    provider: str | None
    model: str | None
    meter_kind: str | None
    cost_lane: str | None
    usage_fact_id: str | None
    wallet_ledger_id: str | None
    breach_id: str | None
    created_at: str
    metadata: dict[str, object]

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingDebit:
        mapping = _require_mapping(payload, label="SMR billing debit")
        return cls(
            billing_debit_id=_require_string(
                mapping,
                "billing_debit_id",
                label="SMR billing debit.billing_debit_id",
            ),
            surface=_require_string(mapping, "surface", label="SMR billing debit.surface"),
            model_class=_require_string(
                mapping,
                "model_class",
                label="SMR billing debit.model_class",
            ),
            debit_pool=_require_string(
                mapping,
                "debit_pool",
                label="SMR billing debit.debit_pool",
            ),
            customer_debit_microcents=_int_value(mapping, "customer_debit_microcents"),
            provider=_optional_string(mapping, "provider"),
            model=_optional_string(mapping, "model"),
            meter_kind=_optional_string(mapping, "meter_kind"),
            cost_lane=_optional_string(mapping, "cost_lane"),
            usage_fact_id=_optional_string(mapping, "usage_fact_id"),
            wallet_ledger_id=_optional_string(mapping, "wallet_ledger_id"),
            breach_id=_optional_string(mapping, "breach_id"),
            created_at=_require_string(mapping, "created_at", label="SMR billing debit.created_at"),
            metadata=_optional_object_dict(_optional_wire_value(mapping, "metadata")),
        )


@dataclass(frozen=True)
class SmrBillingDebitPoolSummary:
    debit_pool: str
    debit_count: int
    customer_debit_microcents: int

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingDebitPoolSummary:
        mapping = _require_mapping(payload, label="SMR billing debit pool summary")
        return cls(
            debit_pool=_require_string(
                mapping,
                "debit_pool",
                label="SMR billing debit pool summary.debit_pool",
            ),
            debit_count=_int_value(mapping, "debit_count"),
            customer_debit_microcents=_int_value(
                mapping,
                "customer_debit_microcents",
            ),
        )


@dataclass(frozen=True)
class SmrBillingDrawdown:
    org_id: str
    run_id: str | None
    factory_effort_id: str | None
    surface: str
    billing_plan: SmrBillingPlanSnapshot
    debits: list[SmrBillingDebit]
    debit_pool_summaries: list[SmrBillingDebitPoolSummary]
    total_customer_debit_microcents: int
    blocked: bool
    blocked_reason: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingDrawdown:
        mapping = _require_mapping(payload, label="SMR billing drawdown")
        return cls(
            org_id=_require_string(mapping, "org_id", label="SMR billing drawdown.org_id"),
            run_id=_optional_string(mapping, "run_id"),
            factory_effort_id=_optional_string(mapping, "factory_effort_id"),
            surface=_require_string(mapping, "surface", label="SMR billing drawdown.surface"),
            billing_plan=SmrBillingPlanSnapshot.from_wire(mapping["billing_plan"]),
            debits=[SmrBillingDebit.from_wire(item) for item in _optional_array(mapping, "debits")],
            debit_pool_summaries=[
                SmrBillingDebitPoolSummary.from_wire(item)
                for item in _optional_array(mapping, "debit_pool_summaries")
            ],
            total_customer_debit_microcents=_int_value(
                mapping,
                "total_customer_debit_microcents",
            ),
            blocked=_required_bool(
                mapping,
                "blocked",
                label="SMR billing drawdown.blocked",
            ),
            blocked_reason=_optional_string(mapping, "blocked_reason"),
        )


@dataclass(frozen=True)
class SmrBillingPreflight:
    org_id: str
    surface: str
    project_id: str | None
    run_id: str | None
    factory_effort_id: str | None
    model_class: str
    estimated_customer_debit_microcents: int
    allowed: bool
    blocked_reason: str | None
    debit_pool_order: list[str]
    selected_debit_pool: str | None
    available_microcents: int
    allowance_windows: list[SmrBillingAllowanceWindow]
    wallet_balance_microcents: int
    generated_at: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingPreflight:
        mapping = _require_mapping(payload, label="SMR billing preflight")
        return cls(
            org_id=_require_string(mapping, "org_id", label="SMR billing preflight.org_id"),
            surface=_require_string(mapping, "surface", label="SMR billing preflight.surface"),
            project_id=_optional_string(mapping, "project_id"),
            run_id=_optional_string(mapping, "run_id"),
            factory_effort_id=_optional_string(mapping, "factory_effort_id"),
            model_class=_require_string(
                mapping,
                "model_class",
                label="SMR billing preflight.model_class",
            ),
            estimated_customer_debit_microcents=_int_value(
                mapping,
                "estimated_customer_debit_microcents",
            ),
            allowed=_required_bool(mapping, "allowed", label="SMR billing preflight.allowed"),
            blocked_reason=_optional_string(mapping, "blocked_reason"),
            debit_pool_order=_string_list(
                _optional_wire_value(mapping, "debit_pool_order"),
                label="SMR billing preflight.debit_pool_order",
            ),
            selected_debit_pool=_optional_string(mapping, "selected_debit_pool"),
            available_microcents=_int_value(mapping, "available_microcents"),
            allowance_windows=[
                SmrBillingAllowanceWindow.from_wire(item)
                for item in _optional_array(mapping, "allowance_windows")
            ],
            wallet_balance_microcents=_int_value(mapping, "wallet_balance_microcents"),
            generated_at=_require_string(
                mapping,
                "generated_at",
                label="SMR billing preflight.generated_at",
            ),
        )


@dataclass(frozen=True)
class SmrBillingPreflightRequest:
    model_class: str
    estimated_customer_debit_microcents: int = 0
    project_id: str | None = None

    def to_wire(self) -> dict[str, object]:
        return {
            "model_class": self.model_class,
            "estimated_customer_debit_microcents": self.estimated_customer_debit_microcents,
            "project_id": self.project_id,
        }


@dataclass(frozen=True)
class SmrFactoryEffortBillingPreflightRequest:
    model_class: str
    estimated_customer_debit_microcents: int = 0

    def to_wire(self) -> dict[str, object]:
        return {
            "model_class": self.model_class,
            "estimated_customer_debit_microcents": self.estimated_customer_debit_microcents,
        }


__all__ = [
    "SmrBillingAllowanceWindow",
    "SmrBillingBlockedDetail",
    "SmrBillingCatalog",
    "SmrBillingCatalogAllowance",
    "SmrBillingCatalogPlan",
    "SmrBillingDebit",
    "SmrBillingDebitPoolSummary",
    "SmrBillingDrawdown",
    "SmrManualBillingGrantPreview",
    "SmrManualBillingGrantPreviewRequest",
    "SmrBillingPlanInfo",
    "SmrBillingPlanSnapshot",
    "SmrBillingPreflight",
    "SmrBillingPreflightRequest",
    "SmrBillingPromotionCampaign",
    "SmrBillingPromotionGrant",
    "SmrBillingPromotions",
    "SmrBillingRecoveryAction",
    "SmrBillingResetBank",
    "SmrBillingResetGrant",
    "SmrBillingUsageBreakdown",
    "SmrBillingUsageSummary",
    "SmrBillingUsageTotals",
    "SmrBillingWallet",
    "SmrBillingWalletSource",
    "SmrFactoryEffortBillingPreflightRequest",
]
