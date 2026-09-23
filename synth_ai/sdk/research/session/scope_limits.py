"""Create, read, update and delete the per-scope limit rows of a run or objective.

A run that has not terminated refuses create/update/delete with
`active_run_limit_mutation_requires_extension`; raise a live run's caps with
`usage.extend_run_resource_limit` instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from synth_ai.sdk.research.contracts.scope_limits import LimitScope, ScopeLimit
from synth_ai.sdk.research.session._base import _ClientNamespace


def _path(project_id: str, scope: LimitScope, scope_id: str) -> str:
    return f"/smr/projects/{project_id}/{LimitScope(scope).value}/{scope_id}/limits"


class CapChange(Enum):
    """`update(cap_amount=KEEP_CAP)` leaves the cap as it is.

    The route distinguishes an omitted `cap_amount` (keep) from an explicit
    `None` (remove the cap), so the SDK cannot use `None` for "unchanged".
    """

    KEEP = "keep"


KEEP_CAP = CapChange.KEEP


def _body(cap_amount: float | None | CapChange, policy: Mapping[str, Any] | None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if cap_amount is not KEEP_CAP:
        body["cap_amount"] = cap_amount
    if policy is not None:
        body["policy"] = dict(policy)
    return body


class ScopeLimitsAPI(_ClientNamespace):
    """The `(scope, dimension)` caps of a run or objective."""

    def list(self, project_id: str, scope: LimitScope, scope_id: str) -> list[ScopeLimit]:
        payload = self._client._request_json("GET", _path(project_id, scope, scope_id))
        return [ScopeLimit.from_wire(item) for item in payload or []]

    def get(self, project_id: str, scope: LimitScope, scope_id: str, dimension: str) -> ScopeLimit:
        return ScopeLimit.from_wire(
            self._client._request_json("GET", f"{_path(project_id, scope, scope_id)}/{dimension}")
        )

    def create(
        self,
        project_id: str,
        scope: LimitScope,
        scope_id: str,
        *,
        dimension: str,
        cap_amount: float | None,
        policy: Mapping[str, Any] | None = None,
    ) -> ScopeLimit:
        body = {"dimension": dimension, **_body(cap_amount, policy)}
        return ScopeLimit.from_wire(
            self._client._request_json("POST", _path(project_id, scope, scope_id), json_body=body)
        )

    def update(
        self,
        project_id: str,
        scope: LimitScope,
        scope_id: str,
        dimension: str,
        *,
        cap_amount: float | None | CapChange = KEEP_CAP,
        policy: Mapping[str, Any] | None = None,
    ) -> ScopeLimit:
        """Change a cap and/or its policy; `cap_amount=None` removes the cap."""

        return ScopeLimit.from_wire(
            self._client._request_json(
                "PATCH",
                f"{_path(project_id, scope, scope_id)}/{dimension}",
                json_body=_body(cap_amount, policy),
            )
        )

    def delete(self, project_id: str, scope: LimitScope, scope_id: str, dimension: str) -> None:
        self._client._request_json("DELETE", f"{_path(project_id, scope, scope_id)}/{dimension}")


__all__ = ["KEEP_CAP", "CapChange", "ScopeLimitsAPI"]
