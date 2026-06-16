"""Project dataset namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, List

from synth_ai.managed_research.sdk._base import _ClientNamespace


def _guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


class DatasetsAPI(_ClientNamespace):
    def list(self, project_id: str) -> List[dict[str, Any]]:
        return self._client.list_project_datasets(project_id)

    def upload(
        self,
        project_id: str,
        *,
        path: str | Path,
        name: str | None = None,
        format: str | None = None,
        row_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        file_path = Path(path)
        raw_bytes = file_path.read_bytes()
        return self._client.upload_project_dataset(
            project_id,
            {
                "name": name or file_path.name,
                "content": base64.b64encode(raw_bytes).decode("ascii"),
                "encoding": "base64",
                "content_type": _guess_content_type(file_path),
                "format": format,
                "row_count": row_count,
                "metadata": dict(metadata or {}),
            },
        )

    def download(
        self,
        project_id: str,
        dataset_id: str,
        *,
        to: str | Path | None = None,
    ) -> dict[str, Any]:
        payload = self._client.download_project_dataset(project_id, dataset_id)
        destination = Path(to) if to is not None else None
        if destination is not None:
            if payload.get("encoding") == "base64":
                destination.write_bytes(base64.b64decode(str(payload["content"])))
            else:
                destination.write_text(str(payload["content"]), encoding="utf-8")
        return payload


__all__ = ["DatasetsAPI"]
