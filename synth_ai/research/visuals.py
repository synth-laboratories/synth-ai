"""``client.research.visuals`` — first-class Synth Visual API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, List

from synth_ai.managed_research.sdk.client import ManagedResearchClient


class ResearchVisualsAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def publish(
        self,
        run_id: str,
        *,
        title: str,
        html: str | bytes | Path,
        visual_kind: str = "research_visual",
        visibility: str = "org",
        source_run_ids: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish self-contained HTML through the blob artifact authority."""
        content = html.read_bytes() if isinstance(html, Path) else html
        return self._session.publish_run_visual(
            run_id,
            title=title,
            html_content=content,
            visual_kind=visual_kind,
            visibility=visibility,
            source_run_ids=source_run_ids,
            metadata=metadata,
        )

    def list(self, *, project_id: str | None = None, limit: int = 100) -> List[dict[str, Any]]:
        payload = self._session.list_visuals(project_id=project_id, limit=limit)
        return [dict(item) for item in payload.get("visuals") or [] if isinstance(item, Mapping)]

    def get(self, visual_id: str) -> dict[str, Any]:
        return self._session.get_visual(visual_id)

    def get_content(self, visual_id: str, *, as_text: bool = True) -> str | bytes:
        payload = self._session.get_visual_content(visual_id)
        content = payload["content"]
        if payload.get("encoding") == "base64":
            import base64

            raw = base64.b64decode(str(content))
            return raw.decode("utf-8") if as_text else raw
        text = str(content)
        return text if as_text else text.encode("utf-8")


__all__ = ["ResearchVisualsAPI"]
