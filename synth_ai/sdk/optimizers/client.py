"""Public sync and async clients for hosted optimizers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

from synth_ai.core.auth.credentials import resolve_api_credential
from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk.optimizers.contracts import (
    HostedTrainingModelCatalog,
    OptimizerRunOutputs,
    SavedLoraCheckpoint,
    SavedLoraCheckpointPage,
    SavedLoraRunPage,
)


def _object(payload: JsonValue) -> JsonObject:
    if not isinstance(payload, dict):
        raise ValueError("optimizer API response must be an object")
    return payload


def _params(**values: Any) -> dict[str, JsonValue]:
    return {key: value for key, value in values.items() if value is not None and value != ""}


class CheckpointsAPI:
    """List automatically persisted checkpoints and traverse their lineage."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(
        self,
        *,
        query: str | None = None,
        scope: str = "all",
        optimizer_algorithm: str | None = None,
        run_id: str | None = None,
        attempt_id: str | None = None,
        source_checkpoint_id: str | None = None,
        provider: str | None = None,
        checkpoint_kind: str | None = None,
        base_model: str | None = None,
        status: str | None = "ready",
        tags: Sequence[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SavedLoraCheckpointPage:
        payload = self._transport.request_json(
            "GET",
            "/api/v1/optimizers/checkpoints",
            params=_params(
                q=query,
                scope=scope,
                optimizer_algorithm=optimizer_algorithm,
                run_id=run_id,
                attempt_id=attempt_id,
                source_checkpoint_id=source_checkpoint_id,
                provider=provider,
                checkpoint_kind=checkpoint_kind,
                base_model=base_model,
                status=status,
                tags=tags,
                limit=max(1, min(100, limit)),
                offset=max(0, offset),
            ),
            operation_id="optimizers.checkpoints.list",
        )
        return SavedLoraCheckpointPage.model_validate(_object(payload))

    def get(self, checkpoint_id: str) -> SavedLoraCheckpoint:
        payload = self._transport.request_json(
            "GET",
            f"/api/v1/optimizers/checkpoints/{quote(checkpoint_id, safe='')}",
            operation_id="optimizers.checkpoints.get",
        )
        return SavedLoraCheckpoint.model_validate(_object(payload))

    def list_for_run(
        self,
        run_id: str,
        *,
        checkpoint_kind: str | None = None,
        status: str | None = "ready",
        limit: int = 100,
        offset: int = 0,
    ) -> SavedLoraRunPage:
        payload = self._transport.request_json(
            "GET",
            f"/api/v1/optimizers/runs/{quote(run_id, safe='')}/saved-checkpoints",
            params=_params(
                checkpoint_kind=checkpoint_kind,
                status=status,
                limit=max(1, min(100, limit)),
                offset=max(0, offset),
            ),
            operation_id="optimizers.checkpoints.list_for_run",
        )
        return SavedLoraRunPage.model_validate(_object(payload))


class ModelsAPI:
    """Discover models and algorithm support available for hosted training."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(
        self, *, algorithm: str | None = None, provider: str | None = None
    ) -> HostedTrainingModelCatalog:
        payload = self._transport.request_json(
            "GET",
            "/api/v1/optimizers/models/training",
            params=_params(algorithm=algorithm, provider=provider),
            operation_id="optimizers.models.list",
        )
        return HostedTrainingModelCatalog.model_validate(_object(payload))


class RunsAPI:
    """Inspect automatically persisted outputs for hosted optimizer runs."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def outputs(self, run_id: str) -> OptimizerRunOutputs:
        payload = self._transport.request_json(
            "GET",
            f"/api/v1/optimizers/runs/{quote(run_id, safe='')}/outputs",
            operation_id="optimizers.runs.outputs",
        )
        return OptimizerRunOutputs.model_validate(_object(payload))


class AsyncCheckpointsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(
        self,
        *,
        query: str | None = None,
        scope: str = "all",
        optimizer_algorithm: str | None = None,
        run_id: str | None = None,
        attempt_id: str | None = None,
        source_checkpoint_id: str | None = None,
        provider: str | None = None,
        checkpoint_kind: str | None = None,
        base_model: str | None = None,
        status: str | None = "ready",
        tags: Sequence[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SavedLoraCheckpointPage:
        payload = await self._transport.request_json(
            "GET",
            "/api/v1/optimizers/checkpoints",
            params=_params(
                q=query,
                scope=scope,
                optimizer_algorithm=optimizer_algorithm,
                run_id=run_id,
                attempt_id=attempt_id,
                source_checkpoint_id=source_checkpoint_id,
                provider=provider,
                checkpoint_kind=checkpoint_kind,
                base_model=base_model,
                status=status,
                tags=tags,
                limit=max(1, min(100, limit)),
                offset=max(0, offset),
            ),
            operation_id="optimizers.checkpoints.list",
        )
        return SavedLoraCheckpointPage.model_validate(_object(payload))

    async def get(self, checkpoint_id: str) -> SavedLoraCheckpoint:
        payload = await self._transport.request_json(
            "GET",
            f"/api/v1/optimizers/checkpoints/{quote(checkpoint_id, safe='')}",
            operation_id="optimizers.checkpoints.get",
        )
        return SavedLoraCheckpoint.model_validate(_object(payload))

    async def list_for_run(
        self,
        run_id: str,
        *,
        checkpoint_kind: str | None = None,
        status: str | None = "ready",
        limit: int = 100,
        offset: int = 0,
    ) -> SavedLoraRunPage:
        payload = await self._transport.request_json(
            "GET",
            f"/api/v1/optimizers/runs/{quote(run_id, safe='')}/saved-checkpoints",
            params=_params(
                checkpoint_kind=checkpoint_kind,
                status=status,
                limit=max(1, min(100, limit)),
                offset=max(0, offset),
            ),
            operation_id="optimizers.checkpoints.list_for_run",
        )
        return SavedLoraRunPage.model_validate(_object(payload))


class AsyncModelsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self, **filters: Any) -> HostedTrainingModelCatalog:
        payload = await self._transport.request_json(
            "GET",
            "/api/v1/optimizers/models/training",
            params=_params(**filters),
            operation_id="optimizers.models.list",
        )
        return HostedTrainingModelCatalog.model_validate(_object(payload))


class AsyncRunsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def outputs(self, run_id: str) -> OptimizerRunOutputs:
        payload = await self._transport.request_json(
            "GET",
            f"/api/v1/optimizers/runs/{quote(run_id, safe='')}/outputs",
            operation_id="optimizers.runs.outputs",
        )
        return OptimizerRunOutputs.model_validate(_object(payload))


class OptimizersClient:
    """Hosted optimizer namespace mounted at ``SynthClient.optimizers``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        credential = resolve_api_credential(api_key)
        self._transport = HttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers=credential.authorization_headers(),
            timeout_seconds=timeout_seconds,
        )
        self.checkpoints = CheckpointsAPI(self._transport)
        self.models = ModelsAPI(self._transport)
        self.runs = RunsAPI(self._transport)

    def close(self) -> None:
        self._transport.close()


class AsyncOptimizersClient:
    """Native asynchronous hosted optimizer namespace."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        credential = resolve_api_credential(api_key)
        self._transport = AsyncHttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers=credential.authorization_headers(),
            timeout_seconds=timeout_seconds,
        )
        self.checkpoints = AsyncCheckpointsAPI(self._transport)
        self.models = AsyncModelsAPI(self._transport)
        self.runs = AsyncRunsAPI(self._transport)

    async def close(self) -> None:
        await self._transport.close()


__all__ = ["AsyncOptimizersClient", "OptimizersClient"]
