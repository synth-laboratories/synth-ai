"""``client.research.hosted_artifacts`` — Open Research hosted artifact operator API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _artifact_items(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        return [dict(item) for item in artifacts if isinstance(item, Mapping)]
    return []


class ResearchHostedArtifactsAPI:
    """Operator CRUD access to SMR hosted artifacts (Open Research alpha).

    Hosted artifacts are HTML proof pages materialized by ``artifact_builder``
    workers during a run. Creation still happens in-run via the
    ``publish_hosted_artifact`` MCP tool; this namespace covers operator read,
    metadata patch, promote, review dispatch, and delete.

    | Operation | SDK | Backend |
    | --- | --- | --- |
    | **Create** | Worker MCP ``publish_hosted_artifact`` | In-run service write |
    | **Read** | ``list``, ``get``, ``get_for_run``, ``get_content`` | ``GET`` list/detail/receipt/content |
    | **Update** | ``update``, ``publish_public``, ``assign_reviewer`` | ``PATCH`` + promote/review routes |
    | **Delete** | ``delete`` | ``DELETE /smr/hosted-artifacts/{id}`` |

    Example:
        >>> artifacts = client.research.hosted_artifacts.list(project_id=project_id)
        >>> artifact = artifacts[0]
        >>> artifact["hosted_url"]
        >>> client.research.hosted_artifacts.update(
        ...     artifact["hosted_artifact_id"],
        ...     title="Revised title",
        ... )
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        *,
        project_id: str | None = None,
        limit: int = 100,
    ) -> List[dict[str, Any]]:
        """List hosted artifacts for the org or one project.

        Args:
            project_id: When set, restrict to artifacts built under that project.
            limit: Maximum rows to return (server capped at 250).

        Returns:
            Artifact receipts with ``project_id``, ``hosted_url``, ``public_url``,
            and ``slug`` when promoted.
        """
        if project_id:
            payload = self._session.list_project_hosted_artifacts(project_id, limit=limit)
        else:
            payload = self._session.list_hosted_artifacts(project_id=project_id, limit=limit)
        return _artifact_items(payload)

    def get(self, hosted_artifact_id: str) -> dict[str, Any]:
        """Read one hosted artifact receipt with URLs.

        Args:
            hosted_artifact_id: Primary key from ``list`` or ``get_for_run``.

        Returns:
            Receipt JSON from ``GET /smr/hosted-artifacts/{id}``.
        """
        return self._session.get_hosted_artifact(hosted_artifact_id)

    def get_for_run(self, run_id: str) -> dict[str, Any]:
        """Read hosted artifact receipt for a run.

        Args:
            run_id: SMR run id that built or owns the artifact.

        Returns:
            Status payload from ``GET /smr/runs/{run_id}/hosted-artifact``.
        """
        return self._session.get_run_hosted_artifact(run_id)

    def get_content(
        self,
        hosted_artifact_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        """Read hosted HTML body by artifact id."""
        payload = self._session.get_hosted_artifact_content(hosted_artifact_id)
        encoding = str(payload.get("encoding") or "utf-8")
        content = payload["content"]
        if encoding == "base64":
            import base64

            raw = base64.b64decode(str(content))
            return raw.decode("utf-8") if as_text else raw
        text = str(content)
        return text if as_text else text.encode("utf-8")

    def update(
        self,
        hosted_artifact_id: str,
        *,
        title: str | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
        theme: str | None = None,
        summary: str | None = None,
        kind: str | None = None,
        visibility: str | None = None,
    ) -> dict[str, Any]:
        """Patch hosted artifact metadata and optional public shell fields.

        Args:
            hosted_artifact_id: Artifact to update.
            title: Replace artifact title (and public title when promoted).
            metadata: Shallow-merge into artifact ``metadata``.
            theme: Public index theme when a publication exists.
            summary: Public summary when a publication exists.
            kind: Public kind when a publication exists.
            visibility: ``private``, ``org``, or ``public``. Demoting from
                ``public`` removes the public shell row.

        Returns:
            Updated receipt from ``PATCH /smr/hosted-artifacts/{id}``.
        """
        return self._session.patch_hosted_artifact(
            hosted_artifact_id,
            title=title,
            metadata=metadata,
            theme=theme,
            summary=summary,
            kind=kind,
            visibility=visibility,
        )

    def publish_public(
        self,
        hosted_artifact_id: str,
        slug: str,
        *,
        kind: str = "result",
        theme: str | None = None,
        summary: str | None = None,
        factory_id: str | None = None,
        effort_id: str | None = None,
    ) -> dict[str, Any]:
        """Promote a hosted artifact to the public Open Research index."""
        return self._session.publish_hosted_artifact_public(
            hosted_artifact_id,
            slug=slug,
            kind=kind,
            theme=theme,
            summary=summary,
            factory_id=factory_id,
            effort_id=effort_id,
        )

    def assign_reviewer(
        self,
        hosted_artifact_id: str,
        reason: str,
        *,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch an ``artifact_reviewer`` task for a hosted artifact."""
        return self._session.assign_hosted_artifact_reviewer(
            hosted_artifact_id,
            reason=reason,
            summary=summary,
        )

    def delete(self, hosted_artifact_id: str) -> dict[str, Any]:
        """Delete a hosted artifact and its stored HTML.

        Args:
            hosted_artifact_id: Artifact to delete.

        Returns:
            Deletion receipt with ``deleted`` and ``hosted_artifact_id``.
        """
        return self._session.delete_hosted_artifact(hosted_artifact_id)

    def list_public(self) -> List[dict[str, Any]]:
        """List public Open Research artifacts (unauthenticated index JSON)."""
        payload = self._session.list_public_hosted_artifacts()
        return _artifact_items(payload)

    def get_public(self, slug: str) -> dict[str, Any]:
        """Read one public artifact bundle by slug."""
        return self._session.get_public_hosted_artifact(slug)


__all__ = ["ResearchHostedArtifactsAPI"]
