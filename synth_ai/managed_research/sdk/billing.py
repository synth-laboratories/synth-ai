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


__all__ = ["BillingAPI"]
