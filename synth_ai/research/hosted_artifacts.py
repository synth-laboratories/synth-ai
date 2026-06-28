"""``client.research.hosted_artifacts`` — Open Research hosted artifact operator API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient


class ResearchHostedArtifactsAPI:
    """Operator CRUD-style access to SMR hosted artifacts (Open Research alpha).

    Hosted artifacts are HTML proof pages materialized by ``artifact_builder``
    workers during a run. Creation happens inside the worker via the
    ``publish_hosted_artifact`` MCP tool — there is no standalone SDK
    ``create()`` HTTP entrypoint yet.

    | Operation | SDK | Backend |
    | --- | --- | --- |
    | **Create** | Worker MCP ``publish_hosted_artifact`` | Internal service during run |
    | **Read** | ``get_for_run``, ``get_content``, ``list_public``, ``get_public`` | ``GET`` routes below |
    | **Update** | ``publish_public``, ``assign_reviewer`` | ``POST`` promote / review dispatch |
    | **Delete** | *Not implemented* | *No unpublish/delete route yet* |

    Example:
        >>> status = client.research.hosted_artifacts.get_for_run(run_id)
        >>> status["hosted_url"]
        >>> client.research.hosted_artifacts.publish_public(
        ...     status["hosted_artifact_id"],
        ...     slug="my-result-20260628",
        ... )
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get_for_run(self, run_id: str) -> dict[str, Any]:
        """Read hosted artifact receipt for a run.

        Args:
            run_id: SMR run id that built or owns the artifact.

        Returns:
            Status payload from ``GET /smr/runs/{run_id}/hosted-artifact`` including
            ``hosted_artifact_id``, ``hosted_url``, ``public_url``, ``slug``, and
            ``work_product_id`` when materialized.
        """
        return self._session.get_run_hosted_artifact(run_id)

    def get_content(
        self,
        hosted_artifact_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        """Read hosted HTML body by artifact id.

        Args:
            hosted_artifact_id: Primary key from ``get_for_run`` or publish receipt.
            as_text: When ``True``, return decoded UTF-8 text; otherwise raw bytes.

        Returns:
            HTML content from ``GET /smr/hosted-artifacts/{id}/content``.
        """
        payload = self._session.get_hosted_artifact_content(hosted_artifact_id)
        encoding = str(payload.get("encoding") or "utf-8")
        content = payload["content"]
        if encoding == "base64":
            import base64

            raw = base64.b64decode(str(content))
            return raw.decode("utf-8") if as_text else raw
        text = str(content)
        return text if as_text else text.encode("utf-8")

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
        """Promote a hosted artifact to the public Open Research index.

        Args:
            hosted_artifact_id: Artifact to publish.
            slug: Public slug (normalized server-side).
            kind: Publication kind: ``bloglet``, ``result``, ``analysis``, or ``blog``.
            theme: Optional theme label for the index card.
            summary: Optional public summary text.
            factory_id: Optional Factory lineage id.
            effort_id: Optional effort lineage id.

        Returns:
            Public bundle JSON from ``POST /smr/hosted-artifacts/{id}/publish-public``.
        """
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
        """Dispatch an ``artifact_reviewer`` task for a hosted artifact.

        Args:
            hosted_artifact_id: Artifact under review.
            reason: Operator reason shown to the reviewer dispatch.
            summary: Optional longer review brief.

        Returns:
            Reviewer dispatch payload from
            ``POST /smr/hosted-artifacts/{id}/assign-reviewer``.
        """
        return self._session.assign_hosted_artifact_reviewer(
            hosted_artifact_id,
            reason=reason,
            summary=summary,
        )

    def list_public(self) -> list[dict[str, Any]]:
        """List public Open Research artifacts (unauthenticated index JSON).

        Returns:
            Items from ``GET /api/open-research/v1/artifacts``.
        """
        payload = self._session.list_public_hosted_artifacts()
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, list):
            return [dict(item) for item in artifacts if isinstance(item, Mapping)]
        return []

    def get_public(self, slug: str) -> dict[str, Any]:
        """Read one public artifact bundle by slug.

        Args:
            slug: Public slug from the Open Research index.

        Returns:
            Bundle JSON from ``GET /api/open-research/v1/artifacts/{slug}``.
        """
        return self._session.get_public_hosted_artifact(slug)


__all__ = ["ResearchHostedArtifactsAPI"]
