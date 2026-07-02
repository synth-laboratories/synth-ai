"""SMR billing plan, preflight, and drawdown models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from synth_ai.managed_research.models.types import (
    _int_value,
    _optional_array,
    _optional_bool,
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
            source=_require_string(mapping, "source", label="SMR billing allowance window.source"),
            reset_anchor_id=_optional_string(mapping, "reset_anchor_id"),
        )


@dataclass(frozen=True)
class SmrBillingWallet:
    balance_microcents: int
    expires_at: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingWallet:
        mapping = _require_mapping(payload, label="SMR billing wallet")
        return cls(
            balance_microcents=_int_value(mapping, "balance_microcents"),
            expires_at=_optional_string(mapping, "expires_at"),
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
    org_id: str
    plan_tier: str
    legacy_plan: str
    billing_mode: str
    allowance_windows: list[SmrBillingAllowanceWindow]
    wallet: SmrBillingWallet
    blocked: bool
    blocked_reason: str | None
    generated_at: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrBillingPlanSnapshot:
        mapping = _require_mapping(payload, label="SMR billing plan snapshot")
        return cls(
            org_id=_require_string(mapping, "org_id", label="SMR billing plan snapshot.org_id"),
            plan_tier=_require_string(
                mapping,
                "plan_tier",
                label="SMR billing plan snapshot.plan_tier",
            ),
            legacy_plan=_require_string(
                mapping,
                "legacy_plan",
                label="SMR billing plan snapshot.legacy_plan",
            ),
            billing_mode=_require_string(
                mapping,
                "billing_mode",
                label="SMR billing plan snapshot.billing_mode",
            ),
            allowance_windows=[
                SmrBillingAllowanceWindow.from_wire(item)
                for item in _optional_array(mapping, "allowance_windows")
            ],
            wallet=SmrBillingWallet.from_wire(mapping["wallet"]),
            blocked=_required_bool(
                mapping,
                "blocked",
                label="SMR billing plan snapshot.blocked",
            ),
            blocked_reason=_optional_string(mapping, "blocked_reason"),
            generated_at=_require_string(
                mapping,
                "generated_at",
                label="SMR billing plan snapshot.generated_at",
            ),
        )


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
    dev_environment_id: str | None
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
            dev_environment_id=_optional_string(mapping, "dev_environment_id"),
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
    dev_environment_id: str | None
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
            dev_environment_id=_optional_string(mapping, "dev_environment_id"),
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
    dev_environment_id: str | None
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
            dev_environment_id=_optional_string(mapping, "dev_environment_id"),
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
    "SmrBillingCatalog",
    "SmrBillingCatalogAllowance",
    "SmrBillingCatalogPlan",
    "SmrBillingDebit",
    "SmrBillingDebitPoolSummary",
    "SmrBillingDrawdown",
    "SmrBillingPlanSnapshot",
    "SmrBillingPreflight",
    "SmrBillingPreflightRequest",
    "SmrBillingWallet",
    "SmrFactoryEffortBillingPreflightRequest",
]
