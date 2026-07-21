"""Run WorkProduct SDK namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.core.research._legacy.models.factory_evidence import ArtifactBackedWorkProduct
from synth_ai.core.research._legacy.models.work_products import (
    ManagedResearchContainerEvalPackage,
    ManagedResearchRunWorkProduct,
    ManagedResearchWorkProductExport,
)
from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class WorkProductsAPI(_ClientNamespace):
    def list_for_run(
        self,
        project_id: str,
        run_id: str,
    ) -> list[ManagedResearchRunWorkProduct]:
        return [
            ManagedResearchRunWorkProduct.from_wire(item)
            for item in self._client.list_run_work_products(project_id, run_id)
        ]

    def get(self, work_product_id: str) -> ManagedResearchRunWorkProduct:
        return ManagedResearchRunWorkProduct.from_wire(
            self._client.get_run_work_product(work_product_id)
        )

    def get_artifact_backed(
        self,
        work_product_id: str,
        *,
        artifact_id: str | None = None,
    ) -> ArtifactBackedWorkProduct:
        """Resolve and verify the immutable artifact backing a WorkProduct."""

        work_product = self.get(work_product_id)
        resolved_artifact_id = artifact_id or work_product.artifact_id
        if resolved_artifact_id is None and work_product.artifact_links:
            resolved_artifact_id = work_product.artifact_links[0].artifact_id
        if resolved_artifact_id is None:
            raise ValueError(f"WorkProduct {work_product_id} has no linked artifact")
        return ArtifactBackedWorkProduct.from_records(
            work_product,
            self._client.get_artifact(resolved_artifact_id),
        )

    def list_artifact_backed_for_run(
        self,
        project_id: str,
        run_id: str,
    ) -> list[ArtifactBackedWorkProduct]:
        """Return only fully verified, artifact-backed WorkProducts for a run."""

        return [
            self.get_artifact_backed(work_product.work_product_id)
            for work_product in self.list_for_run(project_id, run_id)
        ]

    def content(self, work_product_id: str, *, as_text: bool = True) -> str | bytes:
        return self._client.get_run_work_product_content(
            work_product_id,
            as_text=as_text,
        )

    def export(
        self,
        work_product_id: str,
        *,
        destination: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> ManagedResearchWorkProductExport:
        return ManagedResearchWorkProductExport.from_wire(
            self._client.export_run_work_product(
                work_product_id,
                destination=destination,
                idempotency_key=idempotency_key,
            )
        )

    def get_export(self, export_id: str) -> ManagedResearchWorkProductExport:
        return ManagedResearchWorkProductExport.from_wire(
            self._client.get_work_product_export(export_id)
        )

    def explain_blocker(self, work_product_id: str) -> dict[str, Any]:
        return self._client.explain_work_product_blocker(work_product_id)

    def upload_container_eval_package(
        self,
        project_id: str,
        run_id: str,
        *,
        kind: str,
        name: str,
        version: str | None = None,
        artifact_id: str | None = None,
        storage_uri: str | None = None,
        archive_size_bytes: int | None = None,
        manifest: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedResearchContainerEvalPackage:
        return ManagedResearchContainerEvalPackage.from_wire(
            self._client.upload_container_eval_package(
                project_id,
                run_id,
                kind=kind,
                name=name,
                version=version,
                artifact_id=artifact_id,
                storage_uri=storage_uri,
                archive_size_bytes=archive_size_bytes,
                manifest=manifest,
                metadata=metadata,
            )
        )

    def list_container_eval_packages(
        self, project_id: str, run_id: str
    ) -> list[ManagedResearchContainerEvalPackage]:
        return [
            ManagedResearchContainerEvalPackage.from_wire(item)
            for item in self._client.list_run_container_eval_packages(
                project_id,
                run_id,
            )
        ]

    def validate_container_eval_package(
        self,
        package_id: str,
    ) -> ManagedResearchContainerEvalPackage:
        return ManagedResearchContainerEvalPackage.from_wire(
            self._client.validate_container_eval_package(package_id)
        )


__all__ = ["WorkProductsAPI"]
