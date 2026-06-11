"""Approval-oriented SDK namespace."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk._base import _ClientNamespace


class ApprovalsAPI(_ClientNamespace):
    """Run approval helpers."""

    def list(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._client.list_run_approvals(
            run_id,
            project_id=project_id,
            status_filter=status_filter,
            limit=limit,
            cursor=cursor,
        )

    def approve(
        self,
        run_id: str,
        approval_id: str,
        *,
        project_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        return self._client.approve_run_approval(
            run_id,
            approval_id,
            project_id=project_id,
            comment=comment,
        )

    def deny(
        self,
        run_id: str,
        approval_id: str,
        *,
        project_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        return self._client.deny_run_approval(
            run_id,
            approval_id,
            project_id=project_id,
            comment=comment,
        )


__all__ = ["ApprovalsAPI"]
