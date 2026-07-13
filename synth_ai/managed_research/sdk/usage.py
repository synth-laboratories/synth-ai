"""Usage-oriented SDK namespace."""

from __future__ import annotations

from collections.abc import Mapping

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.models import (
    BillingEntitlementSnapshot,
    OrgLimits,
    SmrProjectUsage,
    SmrResourceLimitExtension,
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrResourceLimitSelector,
    SmrRunUsage,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


def _raise_on_error_payload(payload: object) -> object:
    if not isinstance(payload, Mapping):
        return payload
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return payload
    first = errors[0]
    if isinstance(first, Mapping):
        message = first.get("message")
        if isinstance(message, str) and message.strip():
            raise SmrApiError(message.strip())
    raise SmrApiError("Managed Research usage request failed")


def _selector_to_wire(
    selector: SmrResourceLimitSelector | Mapping[str, object] | None,
) -> dict[str, object] | None:
    if selector is None:
        return None
    if isinstance(selector, SmrResourceLimitSelector):
        return {
            "kind": selector.kind,
            "capability": selector.capability,
            "provider": selector.provider,
            "model": selector.model,
            "actor_type": selector.actor_type,
            "actor_id": selector.actor_id,
            "resource_id": selector.resource_id,
        }
    if isinstance(selector, Mapping):
        return dict(selector)
    raise TypeError("selector must be a SmrResourceLimitSelector or mapping")


def _limit_extension_payload(
    *,
    limit_value: float | None,
    additional_value: float | None,
    reason: str | None,
    selector: SmrResourceLimitSelector | Mapping[str, object] | None,
    resource_limit_id: str | None,
    metric: str | None,
    unit: str | None,
    resolve_blockers: bool,
    resume: bool,
    idempotency_key: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "resolve_blockers": bool(resolve_blockers),
        "resume": bool(resume),
    }
    if metric is not None:
        payload["metric"] = metric
    if unit is not None:
        payload["unit"] = unit
    selector_payload = _selector_to_wire(selector)
    if selector_payload is not None:
        payload["selector"] = selector_payload
    if resource_limit_id is not None:
        payload["resource_limit_id"] = resource_limit_id
    if limit_value is not None:
        payload["limit_value"] = float(limit_value)
    if additional_value is not None:
        payload["additional_value"] = float(additional_value)
    if reason is not None:
        payload["reason"] = reason
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    return payload


class UsageAPI(_ClientNamespace):
    """Canonical usage and entitlement helpers."""

    def get_limits(self) -> OrgLimits:
        """Return resource usage and caps for the authenticated org's current plan."""
        return OrgLimits.from_wire(
            _raise_on_error_payload(self._client._request_json("GET", "/smr/limits"))
        )

    def get_billing_entitlements(self) -> BillingEntitlementSnapshot:
        return BillingEntitlementSnapshot.from_wire(
            _raise_on_error_payload(self._client._request_json("GET", "/billing/entitlements"))
        )

    def get_run_usage(self, run_id: str) -> SmrRunUsage:
        return SmrRunUsage.from_wire(
            _raise_on_error_payload(self._client._request_json("GET", f"/smr/runs/{run_id}/usage"))
        )

    def get_run_resource_limits(self, run_id: str) -> SmrResourceLimits:
        return SmrResourceLimits.from_wire(
            _raise_on_error_payload(
                self._client._request_json("GET", f"/smr/runs/{run_id}/resource-limits")
            )
        )

    def get_run_progress_toward_resource_limits(
        self,
        run_id: str,
    ) -> SmrResourceLimitProgress:
        return SmrResourceLimitProgress.from_wire(
            _raise_on_error_payload(
                self._client._request_json(
                    "GET",
                    f"/smr/runs/{run_id}/progress-toward-resource-limits",
                )
            )
        )

    def extend_run_resource_limit(
        self,
        run_id: str,
        *,
        limit_value: float | None = None,
        additional_value: float | None = None,
        reason: str | None = None,
        selector: SmrResourceLimitSelector | Mapping[str, object] | None = None,
        resource_limit_id: str | None = None,
        metric: str | None = None,
        unit: str | None = None,
        resolve_blockers: bool = True,
        resume: bool = True,
        idempotency_key: str | None = None,
    ) -> SmrResourceLimitExtension:
        return SmrResourceLimitExtension.from_wire(
            _raise_on_error_payload(
                self._client._request_json(
                    "POST",
                    f"/smr/runs/{run_id}/resource-limit-extensions",
                    json_body=_limit_extension_payload(
                        limit_value=limit_value,
                        additional_value=additional_value,
                        reason=reason,
                        selector=selector,
                        resource_limit_id=resource_limit_id,
                        metric=metric,
                        unit=unit,
                        resolve_blockers=resolve_blockers,
                        resume=resume,
                        idempotency_key=idempotency_key,
                    ),
                )
            )
        )

    def get_project_run_resource_limits(
        self,
        project_id: str,
        run_id: str,
    ) -> SmrResourceLimits:
        return SmrResourceLimits.from_wire(
            _raise_on_error_payload(
                self._client._request_json(
                    "GET",
                    f"/smr/projects/{project_id}/runs/{run_id}/resource-limits",
                )
            )
        )

    def get_project_run_progress_toward_resource_limits(
        self,
        project_id: str,
        run_id: str,
    ) -> SmrResourceLimitProgress:
        return SmrResourceLimitProgress.from_wire(
            _raise_on_error_payload(
                self._client._request_json(
                    "GET",
                    f"/smr/projects/{project_id}/runs/{run_id}/progress-toward-resource-limits",
                )
            )
        )

    def extend_project_run_resource_limit(
        self,
        project_id: str,
        run_id: str,
        *,
        limit_value: float | None = None,
        additional_value: float | None = None,
        reason: str | None = None,
        selector: SmrResourceLimitSelector | Mapping[str, object] | None = None,
        resource_limit_id: str | None = None,
        metric: str | None = None,
        unit: str | None = None,
        resolve_blockers: bool = True,
        resume: bool = True,
        idempotency_key: str | None = None,
    ) -> SmrResourceLimitExtension:
        return SmrResourceLimitExtension.from_wire(
            _raise_on_error_payload(
                self._client._request_json(
                    "POST",
                    f"/smr/projects/{project_id}/runs/{run_id}/resource-limit-extensions",
                    json_body=_limit_extension_payload(
                        limit_value=limit_value,
                        additional_value=additional_value,
                        reason=reason,
                        selector=selector,
                        resource_limit_id=resource_limit_id,
                        metric=metric,
                        unit=unit,
                        resolve_blockers=resolve_blockers,
                        resume=resume,
                        idempotency_key=idempotency_key,
                    ),
                )
            )
        )

    def get_project_usage(self, project_id: str) -> SmrProjectUsage:
        return SmrProjectUsage.from_wire(
            _raise_on_error_payload(
                self._client._request_json("GET", f"/smr/projects/{project_id}/usage")
            )
        )


__all__ = ["UsageAPI"]
