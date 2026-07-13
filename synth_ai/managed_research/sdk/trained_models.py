"""Trained-model registry SDK namespace.

Wraps the ``/smr/trained_models`` and ``/smr/runs/{run_id}/trained_models``
routes. Used by agents to register a Tinker LoRA after training, publish it
as a model WorkProduct, update metrics once offline eval is done, queue exports
to Hugging Face or S3-compatible storage, and deliberately tear down temporary
adapters when they are not user-facing deliverables.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.work_products import (
    ManagedResearchTrainedModel,
    ManagedResearchTrainedModelAdapterUploadUrl,
    ManagedResearchTrainedModelExport,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


class TrainedModelsAPI(_ClientNamespace):
    def register(
        self,
        *,
        run_id: str,
        base_model: str,
        method: str,
        tinker_path: str,
        task_id: str | None = None,
        episode_id: str | None = None,
        lora_rank: int | None = None,
        base_metric: float | None = None,
        tuned_metric: float | None = None,
        uplift_abs: float | None = None,
        train_cost_usd: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedResearchTrainedModel:
        body = {
            "run_id": run_id,
            "base_model": base_model,
            "method": method,
            "tinker_path": tinker_path,
            "task_id": task_id,
            "episode_id": episode_id,
            "lora_rank": lora_rank,
            "metrics": {
                "base_metric": base_metric,
                "tuned_metric": tuned_metric,
                "uplift_abs": uplift_abs,
            },
            "train_cost_usd": train_cost_usd,
            "metadata": dict(metadata or {}),
        }
        return ManagedResearchTrainedModel.from_wire(
            self._client._request_json("POST", "/smr/trained_models", json_body=body)
        )

    def get(self, model_id: str) -> ManagedResearchTrainedModel:
        return ManagedResearchTrainedModel.from_wire(
            self._client._request_json("GET", f"/smr/trained_models/{model_id}")
        )

    def list_for_run(self, run_id: str) -> list[ManagedResearchTrainedModel]:
        result = self._client._request_json("GET", f"/smr/runs/{run_id}/trained_models")
        return [
            ManagedResearchTrainedModel.from_wire(item)
            for item in (list(result) if isinstance(result, list) else [])
            if isinstance(item, Mapping)
        ]

    def export(
        self,
        model_id: str,
        *,
        destination: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> ManagedResearchTrainedModelExport:
        body = {
            "destination": dict(destination),
            "idempotency_key": idempotency_key,
        }
        return ManagedResearchTrainedModelExport.from_wire(
            self._client._request_json(
                "POST", f"/smr/trained_models/{model_id}/exports", json_body=body
            )
        )

    def create_adapter_upload_url(
        self,
        model_id: str,
        *,
        expires_in: int = 3600,
        content_type: str = "application/gzip",
    ) -> ManagedResearchTrainedModelAdapterUploadUrl:
        body = {
            "expires_in": expires_in,
            "content_type": content_type,
        }
        return ManagedResearchTrainedModelAdapterUploadUrl.from_wire(
            self._client._request_json(
                "POST",
                f"/smr/trained_models/{model_id}/adapter_upload_url",
                json_body=body,
            )
        )

    def complete_adapter_upload(
        self,
        model_id: str,
        *,
        bucket: str,
        key: str,
        adapter_size_bytes: int,
        metadata_patch: Mapping[str, Any] | None = None,
    ) -> ManagedResearchTrainedModel:
        body = {
            "bucket": bucket,
            "key": key,
            "adapter_size_bytes": adapter_size_bytes,
        }
        if metadata_patch is not None:
            body["metadata_patch"] = dict(metadata_patch)
        return ManagedResearchTrainedModel.from_wire(
            self._client._request_json(
                "POST",
                f"/smr/trained_models/{model_id}/adapter_uploads/complete",
                json_body=body,
            )
        )

    def update(
        self,
        model_id: str,
        *,
        tuned_metric: float | None = None,
        uplift_abs: float | None = None,
        train_cost_usd: float | None = None,
        status: str | None = None,
        metadata_patch: Mapping[str, Any] | None = None,
    ) -> ManagedResearchTrainedModel:
        body: dict[str, Any] = {}
        if tuned_metric is not None:
            body["tuned_metric"] = tuned_metric
        if uplift_abs is not None:
            body["uplift_abs"] = uplift_abs
        if train_cost_usd is not None:
            body["train_cost_usd"] = train_cost_usd
        if status is not None:
            body["status"] = status
        if metadata_patch is not None:
            body["metadata_patch"] = dict(metadata_patch)
        return ManagedResearchTrainedModel.from_wire(
            self._client._request_json("PATCH", f"/smr/trained_models/{model_id}", json_body=body)
        )

    def delete(self, model_id: str) -> ManagedResearchTrainedModel:
        return ManagedResearchTrainedModel.from_wire(
            self._client._request_json("DELETE", f"/smr/trained_models/{model_id}")
        )


__all__ = ["TrainedModelsAPI"]
