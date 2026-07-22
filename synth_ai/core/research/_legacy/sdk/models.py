"""Project model namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, List

from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class ModelsAPI(_ClientNamespace):
    def list(self, project_id: str) -> List[dict[str, Any]]:
        return self._client.list_project_models(project_id)

    def get(self, project_id: str, model_id: str) -> dict[str, Any]:
        return self._client.get_project_model(project_id, model_id)

    def download(
        self,
        project_id: str,
        model_id: str,
        *,
        to: str | Path | None = None,
    ) -> dict[str, Any]:
        payload = self._client.download_project_model(project_id, model_id)
        destination = Path(to) if to is not None else None
        if destination is not None:
            if payload.get("encoding") == "base64":
                destination.write_bytes(base64.b64decode(str(payload["content"])))
            else:
                destination.write_text(str(payload["content"]), encoding="utf-8")
        return payload

    def export(self, project_id: str, model_id: str) -> dict[str, Any]:
        return self._client.export_project_model(project_id, model_id)


__all__ = ["ModelsAPI"]
