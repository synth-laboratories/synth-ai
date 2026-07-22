"""Canonical Managed Research usage and entitlement models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from synth_ai.core.research._legacy.models.types import (
    _float_value,
    _int_value,
    _object_dict,
    _optional_array,
    _optional_string,
    _require_mapping,
    _require_string,
)


def _optional_int_value(mapping: dict[str, object], key: str) -> int | None:
    if mapping.get(key) is None:
        return None
    return _int_value(mapping, key)


def _optional_float_value(mapping: dict[str, object], key: str) -> float | None:
    if mapping.get(key) is None:
        return None
    return _float_value(mapping, key)


def _require_float_value(
    mapping: dict[str, object],
    key: str,
    *,
    label: str,
) -> float:
    if mapping.get(key) is None:
        raise ValueError(f"Missing required field {label}")
    return _float_value(mapping, key)


def _optional_object_dict(payload: object) -> dict[str, object]:
    if payload is None:
        return {}
    return _object_dict(payload)


@dataclass(frozen=True)
class BillingEntitlementProfile:
    code: str
    display_name: str
    source_product_ids: list[str]
    source_product_names: list[str]
    is_paid: bool

    @classmethod
    def from_wire(cls, payload: object) -> BillingEntitlementProfile:
        mapping = _require_mapping(payload, label="billing entitlement profile")
        return cls(
            code=_require_string(mapping, "code", label="billing entitlement profile.code"),
            display_name=_require_string(
                mapping,
                "display_name",
                label="billing entitlement profile.display_name",
            ),
            source_product_ids=[
                str(item).strip()
                for item in _optional_array(mapping, "source_product_ids")
                if str(item).strip()
            ],
            source_product_names=[
                str(item).strip()
                for item in _optional_array(mapping, "source_product_names")
                if str(item).strip()
            ],
            is_paid=bool(mapping.get("is_paid")),
        )


@dataclass(frozen=True)
class BillingEntitlementAsset:
    asset_id: str
    display_name: str
    kind: str
    included: bool
    enabled: bool
    source_feature_id: str | None
    balance_cents: int | None
    used_cents: int | None
    limit_cents: int | None
    remaining_cents: int | None
    limit_value: float | None
    used_value: float | None
    remaining_value: float | None
    quantity_unit: str | None
    next_reset_at: str | None

    @classmethod
    def from_wire(cls, payload: object) -> BillingEntitlementAsset:
        mapping = _require_mapping(payload, label="billing entitlement asset")
        return cls(
            asset_id=_require_string(
                mapping, "asset_id", label="billing entitlement asset.asset_id"
            ),
            display_name=_require_string(
                mapping,
                "display_name",
                label="billing entitlement asset.display_name",
            ),
            kind=_require_string(mapping, "kind", label="billing entitlement asset.kind"),
            included=bool(mapping.get("included")),
            enabled=bool(mapping.get("enabled")),
            source_feature_id=_optional_string(mapping, "source_feature_id"),
            balance_cents=_optional_int_value(mapping, "balance_cents"),
            used_cents=_optional_int_value(mapping, "used_cents"),
            limit_cents=_optional_int_value(mapping, "limit_cents"),
            remaining_cents=_optional_int_value(mapping, "remaining_cents"),
            limit_value=_optional_float_value(mapping, "limit_value"),
            used_value=_optional_float_value(mapping, "used_value"),
            remaining_value=_optional_float_value(mapping, "remaining_value"),
            quantity_unit=_optional_string(mapping, "quantity_unit"),
            next_reset_at=_optional_string(mapping, "next_reset_at"),
        )


@dataclass(frozen=True)
class BillingEntitlementSnapshot:
    org_id: str
    provider: str
    profile: BillingEntitlementProfile
    assets: list[BillingEntitlementAsset]
    fetched_at: str

    @classmethod
    def from_wire(cls, payload: object) -> BillingEntitlementSnapshot:
        mapping = _require_mapping(payload, label="billing entitlement snapshot")
        return cls(
            org_id=_require_string(mapping, "org_id", label="billing entitlement snapshot.org_id"),
            provider=_require_string(
                mapping, "provider", label="billing entitlement snapshot.provider"
            ),
            profile=BillingEntitlementProfile.from_wire(mapping.get("profile")),
            assets=[
                BillingEntitlementAsset.from_wire(item)
                for item in _optional_array(mapping, "assets")
            ],
            fetched_at=_require_string(
                mapping,
                "fetched_at",
                label="billing entitlement snapshot.fetched_at",
            ),
        )


@dataclass(frozen=True)
class SmrRunCostTotals:
    total_cents: int
    charged_cents: int
    internal_cost_cents: int
    total_pico_usd: int
    charged_pico_usd: int
    internal_cost_pico_usd: int
    total_usd: float
    charged_usd: float
    internal_cost_usd: float

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunCostTotals:
        mapping = _require_mapping(payload, label="run cost totals")
        return cls(
            total_cents=_int_value(mapping, "total_cents"),
            charged_cents=_int_value(mapping, "charged_cents"),
            internal_cost_cents=_int_value(mapping, "internal_cost_cents"),
            total_pico_usd=_optional_int_value(mapping, "total_pico_usd") or 0,
            charged_pico_usd=_optional_int_value(mapping, "charged_pico_usd") or 0,
            internal_cost_pico_usd=(_optional_int_value(mapping, "internal_cost_pico_usd") or 0),
            total_usd=_optional_float_value(mapping, "total_usd") or 0.0,
            charged_usd=_optional_float_value(mapping, "charged_usd") or 0.0,
            internal_cost_usd=_optional_float_value(mapping, "internal_cost_usd") or 0.0,
        )


@dataclass(frozen=True)
class SmrRunUsage:
    run_id: str
    project_id: str
    cost: SmrRunCostTotals
    totals: dict[str, int]
    tokens: dict[str, object]
    by_provider: dict[str, object]
    by_model: dict[str, object]
    by_actor: dict[str, object]
    breakdown: dict[str, object]
    entries: list[dict[str, object]]
    rows: list[dict[str, object]]
    total_cost_pico_usd: int
    total_charged_pico_usd: int
    total_internal_cost_pico_usd: int
    total_cost_usd: float
    total_charged_usd: float
    total_internal_cost_usd: float

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunUsage:
        mapping = _require_mapping(payload, label="run usage")
        totals_mapping = _object_dict(mapping.get("totals"))
        return cls(
            run_id=_require_string(mapping, "run_id", label="run usage.run_id"),
            project_id=_require_string(mapping, "project_id", label="run usage.project_id"),
            cost=SmrRunCostTotals.from_wire(mapping.get("cost")),
            totals={str(key): int(value) for key, value in totals_mapping.items()},
            tokens=_optional_object_dict(mapping.get("tokens")),
            by_provider=_optional_object_dict(mapping.get("by_provider")),
            by_model=_optional_object_dict(mapping.get("by_model")),
            by_actor=_optional_object_dict(mapping.get("by_actor")),
            breakdown=_optional_object_dict(mapping.get("breakdown")),
            entries=[_object_dict(item) for item in _optional_array(mapping, "entries")],
            rows=[_object_dict(item) for item in _optional_array(mapping, "rows")],
            total_cost_pico_usd=(_optional_int_value(mapping, "total_cost_pico_usd") or 0),
            total_charged_pico_usd=(_optional_int_value(mapping, "total_charged_pico_usd") or 0),
            total_internal_cost_pico_usd=(
                _optional_int_value(mapping, "total_internal_cost_pico_usd") or 0
            ),
            total_cost_usd=_optional_float_value(mapping, "total_cost_usd") or 0.0,
            total_charged_usd=(_optional_float_value(mapping, "total_charged_usd") or 0.0),
            total_internal_cost_usd=(
                _optional_float_value(mapping, "total_internal_cost_usd") or 0.0
            ),
        )


@dataclass(frozen=True)
class SmrProjectUsage:
    project_id: str
    month_to_date: dict[str, object]
    last_7_days: dict[str, object]
    per_run: list[dict[str, object]]
    budgets: dict[str, object]

    @classmethod
    def from_wire(cls, payload: object) -> SmrProjectUsage:
        mapping = _require_mapping(payload, label="project usage")
        return cls(
            project_id=_require_string(mapping, "project_id", label="project usage.project_id"),
            month_to_date=_optional_object_dict(mapping.get("month_to_date")),
            last_7_days=_optional_object_dict(mapping.get("last_7_days")),
            per_run=[_object_dict(item) for item in _optional_array(mapping, "per_run")],
            budgets=_optional_object_dict(mapping.get("budgets")),
        )


@dataclass(frozen=True)
class SmrProjectEntitlementOverlay:
    resolved_lane: str
    free_mode_enabled: bool
    free_tier_eligible: bool

    @classmethod
    def from_wire(cls, payload: object) -> SmrProjectEntitlementOverlay:
        mapping = _require_mapping(payload, label="project entitlement overlay")
        return cls(
            resolved_lane=_require_string(
                mapping,
                "resolved_lane",
                label="project entitlement overlay.resolved_lane",
            ),
            free_mode_enabled=bool(mapping.get("free_mode_enabled")),
            free_tier_eligible=bool(mapping.get("free_tier_eligible")),
        )


@dataclass(frozen=True)
class SmrProjectEconomics:
    project_id: str
    usage: SmrProjectUsage
    entitlements: BillingEntitlementSnapshot
    project_overlay: SmrProjectEntitlementOverlay
    budgets: dict[str, object]

    @classmethod
    def from_wire(cls, payload: object) -> SmrProjectEconomics:
        mapping = _require_mapping(payload, label="project economics")
        return cls(
            project_id=_require_string(
                mapping,
                "project_id",
                label="project economics.project_id",
            ),
            usage=SmrProjectUsage.from_wire(mapping.get("usage")),
            entitlements=BillingEntitlementSnapshot.from_wire(mapping.get("entitlements")),
            project_overlay=SmrProjectEntitlementOverlay.from_wire(mapping.get("project_overlay")),
            budgets=_optional_object_dict(mapping.get("budgets")),
        )


@dataclass(frozen=True)
class SmrResourceLimitSelector:
    kind: str
    capability: str | None
    provider: str | None
    model: str | None
    actor_type: str | None
    actor_id: str | None
    resource_id: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrResourceLimitSelector:
        mapping = _require_mapping(payload, label="resource limit selector")
        return cls(
            kind=_require_string(mapping, "kind", label="resource limit selector.kind"),
            capability=_optional_string(mapping, "capability"),
            provider=_optional_string(mapping, "provider"),
            model=_optional_string(mapping, "model"),
            actor_type=_optional_string(mapping, "actor_type"),
            actor_id=_optional_string(mapping, "actor_id"),
            resource_id=_optional_string(mapping, "resource_id"),
        )


@dataclass(frozen=True)
class SmrLimitQuantity:
    kind: str
    unit: str
    value: int

    @classmethod
    def from_wire(cls, payload: object) -> SmrLimitQuantity | None:
        if payload is None:
            return None
        mapping = _require_mapping(payload, label="limit quantity")
        return cls(
            kind=_require_string(mapping, "kind", label="limit quantity.kind"),
            unit=_require_string(mapping, "unit", label="limit quantity.unit"),
            value=_int_value(mapping, "value"),
        )


@dataclass(frozen=True)
class SmrResourceLimit:
    resource_limit_id: str
    scope: str
    org_id: str | None
    project_id: str | None
    run_id: str | None
    selector: SmrResourceLimitSelector
    metric: str
    limit_value: float | None
    limit_quantity: SmrLimitQuantity | None
    unit: str
    blocks_at_limit: bool
    warning_threshold_percent: float | None
    source: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrResourceLimit:
        mapping = _require_mapping(payload, label="resource limit")
        return cls(
            resource_limit_id=_require_string(
                mapping,
                "resource_limit_id",
                label="resource limit.resource_limit_id",
            ),
            scope=_require_string(mapping, "scope", label="resource limit.scope"),
            org_id=_optional_string(mapping, "org_id"),
            project_id=_optional_string(mapping, "project_id"),
            run_id=_optional_string(mapping, "run_id"),
            selector=SmrResourceLimitSelector.from_wire(mapping.get("selector")),
            metric=_require_string(mapping, "metric", label="resource limit.metric"),
            limit_value=_optional_float_value(mapping, "limit_value"),
            limit_quantity=SmrLimitQuantity.from_wire(mapping.get("limit_quantity")),
            unit=_require_string(mapping, "unit", label="resource limit.unit"),
            blocks_at_limit=bool(mapping.get("blocks_at_limit", True)),
            warning_threshold_percent=_optional_float_value(
                mapping,
                "warning_threshold_percent",
            ),
            source=_require_string(mapping, "source", label="resource limit.source"),
        )


@dataclass(frozen=True)
class SmrResourceLimitBlocker:
    resource_blocker_id: str | None
    target_kind: str | None
    target_id: str | None
    blocker_kind: str | None
    status: str | None
    required_action: str | None
    selector: SmrResourceLimitSelector | None
    metric: str | None
    summary: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrResourceLimitBlocker:
        mapping = _require_mapping(payload, label="resource limit blocker")
        selector_payload = mapping.get("selector")
        return cls(
            resource_blocker_id=_optional_string(mapping, "resource_blocker_id"),
            target_kind=_optional_string(mapping, "target_kind"),
            target_id=_optional_string(mapping, "target_id"),
            blocker_kind=_optional_string(mapping, "blocker_kind"),
            status=_optional_string(mapping, "status"),
            required_action=_optional_string(mapping, "required_action"),
            selector=(
                SmrResourceLimitSelector.from_wire(selector_payload)
                if selector_payload is not None
                else None
            ),
            metric=_optional_string(mapping, "metric"),
            summary=_optional_string(mapping, "summary"),
        )


@dataclass(frozen=True)
class SmrResourceLimitExtensionPolicy:
    can_request_extension: bool
    request_requires_human: bool
    reason: str | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrResourceLimitExtensionPolicy:
        mapping = _optional_object_dict(payload)
        return cls(
            can_request_extension=bool(mapping.get("can_request_extension", False)),
            request_requires_human=bool(mapping.get("request_requires_human", True)),
            reason=_optional_string(mapping, "reason"),
        )


@dataclass(frozen=True)
class SmrResourceLimitProgressItem:
    resource_limit_id: str
    scope: str
    org_id: str | None
    project_id: str | None
    run_id: str | None
    selector: SmrResourceLimitSelector
    metric: str
    limit_value: float | None
    current_value: float | None
    remaining_value: float | None
    limit_quantity: SmrLimitQuantity | None
    current_quantity: SmrLimitQuantity | None
    remaining_quantity: SmrLimitQuantity | None
    used_percent: float | None
    unit: str
    state: str
    blocking: bool
    blocks_at_limit: bool
    last_action: str | None
    last_action_outcome: str | None
    last_action_reason: str | None
    last_action_error: str | None
    last_pause_gate_id: str | None
    warning_threshold_percent: float | None
    source: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrResourceLimitProgressItem:
        mapping = _require_mapping(payload, label="resource limit progress item")
        return cls(
            resource_limit_id=_require_string(
                mapping,
                "resource_limit_id",
                label="resource limit progress item.resource_limit_id",
            ),
            scope=_require_string(
                mapping,
                "scope",
                label="resource limit progress item.scope",
            ),
            org_id=_optional_string(mapping, "org_id"),
            project_id=_optional_string(mapping, "project_id"),
            run_id=_optional_string(mapping, "run_id"),
            selector=SmrResourceLimitSelector.from_wire(mapping.get("selector")),
            metric=_require_string(
                mapping,
                "metric",
                label="resource limit progress item.metric",
            ),
            limit_value=_optional_float_value(mapping, "limit_value"),
            current_value=_optional_float_value(mapping, "current_value"),
            remaining_value=_optional_float_value(mapping, "remaining_value"),
            limit_quantity=SmrLimitQuantity.from_wire(mapping.get("limit_quantity")),
            current_quantity=SmrLimitQuantity.from_wire(mapping.get("current_quantity")),
            remaining_quantity=SmrLimitQuantity.from_wire(mapping.get("remaining_quantity")),
            used_percent=_optional_float_value(mapping, "used_percent"),
            unit=_require_string(mapping, "unit", label="resource limit progress item.unit"),
            state=_require_string(
                mapping,
                "state",
                label="resource limit progress item.state",
            ),
            blocking=bool(mapping.get("blocking", False)),
            blocks_at_limit=bool(mapping.get("blocks_at_limit", True)),
            last_action=_optional_string(mapping, "last_action"),
            last_action_outcome=_optional_string(mapping, "last_action_outcome"),
            last_action_reason=_optional_string(mapping, "last_action_reason"),
            last_action_error=_optional_string(mapping, "last_action_error"),
            last_pause_gate_id=_optional_string(mapping, "last_pause_gate_id"),
            warning_threshold_percent=_optional_float_value(
                mapping,
                "warning_threshold_percent",
            ),
            source=_require_string(
                mapping,
                "source",
                label="resource limit progress item.source",
            ),
        )

    def as_limit(self) -> SmrResourceLimit:
        return SmrResourceLimit(
            resource_limit_id=self.resource_limit_id,
            scope=self.scope,
            org_id=self.org_id,
            project_id=self.project_id,
            run_id=self.run_id,
            selector=self.selector,
            metric=self.metric,
            limit_value=self.limit_value,
            limit_quantity=self.limit_quantity,
            unit=self.unit,
            blocks_at_limit=self.blocks_at_limit,
            warning_threshold_percent=self.warning_threshold_percent,
            source=self.source,
        )


@dataclass(frozen=True)
class SmrResourceLimits:
    scope: str
    org_id: str | None
    project_id: str | None
    run_id: str | None
    items: list[SmrResourceLimit]

    @classmethod
    def from_wire(cls, payload: object) -> SmrResourceLimits:
        mapping = _require_mapping(payload, label="resource limits")
        return cls(
            scope=_require_string(mapping, "scope", label="resource limits.scope"),
            org_id=_optional_string(mapping, "org_id"),
            project_id=_optional_string(mapping, "project_id"),
            run_id=_optional_string(mapping, "run_id"),
            items=[SmrResourceLimit.from_wire(item) for item in _optional_array(mapping, "items")],
        )


@dataclass(frozen=True)
class SmrResourceLimitProgress:
    scope: str
    org_id: str | None
    project_id: str | None
    run_id: str | None
    state: str
    items: list[SmrResourceLimitProgressItem]
    active_blockers: list[SmrResourceLimitBlocker]
    extension_policy: SmrResourceLimitExtensionPolicy

    @classmethod
    def from_wire(cls, payload: object) -> SmrResourceLimitProgress:
        mapping = _require_mapping(payload, label="resource limit progress")
        return cls(
            scope=_require_string(mapping, "scope", label="resource limit progress.scope"),
            org_id=_optional_string(mapping, "org_id"),
            project_id=_optional_string(mapping, "project_id"),
            run_id=_optional_string(mapping, "run_id"),
            state=_optional_string(mapping, "state") or "unknown",
            items=[
                SmrResourceLimitProgressItem.from_wire(item)
                for item in _optional_array(mapping, "items")
            ],
            active_blockers=[
                SmrResourceLimitBlocker.from_wire(item)
                for item in _optional_array(mapping, "active_blockers")
            ],
            extension_policy=SmrResourceLimitExtensionPolicy.from_wire(
                mapping.get("extension_policy")
            ),
        )


@dataclass(frozen=True)
class SmrResourceLimitExtension:
    resource_limit_extension_id: str
    scope: str
    org_id: str | None
    project_id: str | None
    run_id: str | None
    selector: SmrResourceLimitSelector
    metric: str
    previous_limit_value: float | None
    new_limit_value: float
    unit: str
    source: str
    resolved_blocker_ids: list[str]
    resume_requested: bool
    resume_attempted: bool
    resumed: bool
    resume_error: dict[str, object] | None
    pause_gate_released: bool
    released_pause_gate_id: str | None
    progress: SmrResourceLimitProgress | None

    @classmethod
    def from_wire(cls, payload: object) -> SmrResourceLimitExtension:
        mapping = _require_mapping(payload, label="resource limit extension")
        progress_payload = mapping.get("progress")
        resume_error = mapping.get("resume_error")
        return cls(
            resource_limit_extension_id=_require_string(
                mapping,
                "resource_limit_extension_id",
                label="resource limit extension.resource_limit_extension_id",
            ),
            scope=_require_string(
                mapping,
                "scope",
                label="resource limit extension.scope",
            ),
            org_id=_optional_string(mapping, "org_id"),
            project_id=_optional_string(mapping, "project_id"),
            run_id=_optional_string(mapping, "run_id"),
            selector=SmrResourceLimitSelector.from_wire(mapping.get("selector")),
            metric=_require_string(
                mapping,
                "metric",
                label="resource limit extension.metric",
            ),
            previous_limit_value=_optional_float_value(
                mapping,
                "previous_limit_value",
            ),
            new_limit_value=_require_float_value(
                mapping,
                "new_limit_value",
                label="resource limit extension.new_limit_value",
            ),
            unit=_require_string(
                mapping,
                "unit",
                label="resource limit extension.unit",
            ),
            source=_require_string(
                mapping,
                "source",
                label="resource limit extension.source",
            ),
            resolved_blocker_ids=[
                str(item) for item in _optional_array(mapping, "resolved_blocker_ids")
            ],
            resume_requested=bool(mapping.get("resume_requested", False)),
            resume_attempted=bool(mapping.get("resume_attempted", False)),
            resumed=bool(mapping.get("resumed", False)),
            resume_error=(
                {str(key): value for key, value in resume_error.items()}
                if isinstance(resume_error, Mapping)
                else None
            ),
            pause_gate_released=bool(mapping.get("pause_gate_released", False)),
            released_pause_gate_id=_optional_string(
                mapping,
                "released_pause_gate_id",
            ),
            progress=(
                SmrResourceLimitProgress.from_wire(progress_payload)
                if progress_payload is not None
                else None
            ),
        )


@dataclass(frozen=True)
class OrgLimitItem:
    """One limit on a resource for a given window (e.g. daily cap in USD)."""

    metric: str
    window: str
    cap: float | None
    refresh: str
    unlimited: bool
    current_usage: float
    used_percent: float | None
    resets_at: str | None

    @classmethod
    def from_wire(cls, payload: object) -> OrgLimitItem:
        m = _require_mapping(payload, label="org limit item")
        cap_raw = m.get("cap")
        return cls(
            metric=_optional_string(m, "metric") or "spend_usd",
            window=_optional_string(m, "window") or "",
            cap=float(cap_raw) if cap_raw is not None else None,
            refresh=_optional_string(m, "refresh") or "calendar",
            unlimited=bool(m.get("unlimited", False)),
            current_usage=float(m.get("current_usage") or 0),
            used_percent=_optional_float_value(m, "used_percent"),
            resets_at=_optional_string(m, "resets_at"),
        )


@dataclass(frozen=True)
class OrgResourceUsage:
    """Usage and limits for one metered resource."""

    resource_id: str
    display_name: str
    description: str
    unit: str
    provider: str | None
    limits: list[OrgLimitItem]

    @property
    def daily(self) -> OrgLimitItem | None:
        return next((limit for limit in self.limits if limit.window == "daily"), None)

    @property
    def weekly(self) -> OrgLimitItem | None:
        return next((limit for limit in self.limits if limit.window == "weekly"), None)

    @property
    def monthly(self) -> OrgLimitItem | None:
        return next((limit for limit in self.limits if limit.window == "monthly"), None)

    @classmethod
    def from_wire(cls, payload: object) -> OrgResourceUsage:
        m = _require_mapping(payload, label="org resource usage")
        return cls(
            resource_id=_require_string(m, "resource_id", label="org resource usage"),
            display_name=_optional_string(m, "display_name") or "",
            description=_optional_string(m, "description") or "",
            unit=_optional_string(m, "unit") or "usd",
            provider=_optional_string(m, "provider"),
            limits=[OrgLimitItem.from_wire(item) for item in _optional_array(m, "limits")],
        )


@dataclass(frozen=True)
class OrgLimits:
    """Full resource limits snapshot for the authenticated org."""

    org_id: str
    plan: str
    resources: list[OrgResourceUsage]

    def resource(self, resource_id: str) -> OrgResourceUsage | None:
        return next((r for r in self.resources if r.resource_id == resource_id), None)

    @classmethod
    def from_wire(cls, payload: object) -> OrgLimits:
        m = _require_mapping(payload, label="org limits")
        return cls(
            org_id=_require_string(m, "org_id", label="org limits"),
            plan=_optional_string(m, "plan") or "unknown",
            resources=[
                OrgResourceUsage.from_wire(item) for item in _optional_array(m, "resources")
            ],
        )


__all__ = [
    "BillingEntitlementAsset",
    "BillingEntitlementProfile",
    "BillingEntitlementSnapshot",
    "OrgLimitItem",
    "OrgLimits",
    "OrgResourceUsage",
    "SmrLimitQuantity",
    "SmrProjectEconomics",
    "SmrProjectEntitlementOverlay",
    "SmrProjectUsage",
    "SmrResourceLimit",
    "SmrResourceLimitBlocker",
    "SmrResourceLimitExtension",
    "SmrResourceLimitExtensionPolicy",
    "SmrResourceLimitProgress",
    "SmrResourceLimitProgressItem",
    "SmrResourceLimitSelector",
    "SmrResourceLimits",
    "SmrRunCostTotals",
    "SmrRunUsage",
]
