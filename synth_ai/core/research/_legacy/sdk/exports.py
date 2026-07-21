"""Org-scoped export-target namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

from typing import Any

from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class ExportsAPI(_ClientNamespace):
    def list_targets(self) -> list[dict[str, Any]]:
        return self._client.list_export_targets()

    def create_target(
        self,
        *,
        label: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        prefix: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> dict[str, Any]:
        return self._client.create_export_target(
            {
                "kind": "s3",
                "label": label,
                "config": {
                    "bucket": bucket,
                    "prefix": prefix,
                    "region": region,
                    "endpoint_url": endpoint_url,
                },
                "access_key_id": access_key_id,
                "secret_access_key": secret_access_key,
            }
        )

    def update_target(self, target_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._client.patch_export_target(target_id, payload)

    def delete_target(self, target_id: str) -> dict[str, Any]:
        return self._client.delete_export_target(target_id)


__all__ = ["ExportsAPI"]
