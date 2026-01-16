"""Jobs API client for managing training jobs, files, and models.

This module provides a unified interface for interacting with Synth AI's job APIs,
including file uploads, SFT jobs, RL jobs, and model management.

Example:
    >>> from synth_ai.sdk.jobs import JobsClient
    >>>
    >>> async with JobsClient(
    ...     synth_base_url="https://api.usesynth.ai",
    ...     synth_user_key=os.environ["SYNTH_API_KEY"],
    ... ) as client:
    ...     # Upload training data
    ...     file_result = await client.files.upload(
    ...         filename="train.jsonl",
    ...         content=b"...",
    ...         purpose="training",
    ...     )
    ...
    ...     # Create SFT job
    ...     job = await client.sft.create(
    ...         training_file=file_result["id"],
    ...         model="Qwen/Qwen3-4B",
    ...     )
    ...
    ...     # Check job status
    ...     status = await client.sft.retrieve(job["id"])
"""

import importlib
from collections.abc import Callable
from typing import Any, cast

from synth_ai.core.urls import (
    synth_api_v1_base,
    synth_file_url,
    synth_files_url,
    synth_learning_job_cancel_url,
    synth_model_url,
    synth_models_url,
    synth_rl_job_cancel_url,
    synth_rl_job_events_url,
    synth_rl_job_metrics_url,
    synth_rl_job_url,
    synth_rl_jobs_url,
    synth_sft_job_events_url,
    synth_sft_job_url,
    synth_sft_jobs_url,
)

try:
    _supported_module = cast(Any, importlib.import_module("synth_ai.sdk.api.models.supported"))
    normalize_model_identifier = cast(
        Callable[[str], str], _supported_module.normalize_model_identifier
    )
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load supported model utilities") from exc

try:
    _http_module = cast(Any, importlib.import_module("synth_ai.core.http"))
    AsyncHttpClient = _http_module.AsyncHttpClient
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load HTTP client") from exc

try:
    _sft_config_module = cast(Any, importlib.import_module("synth_ai.sdk.learning.sft.config"))
    prepare_sft_job_payload = cast(
        Callable[..., dict[str, Any]], _sft_config_module.prepare_sft_job_payload
    )
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load SFT configuration helpers") from exc


class FilesApi:
    def __init__(self, http: AsyncHttpClient, synth_base_url: str | None) -> None:
        self._http = http
        self._synth_base_url = synth_base_url

    async def upload(
        self,
        *,
        filename: str,
        content: bytes,
        purpose: str,
        content_type: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        data = {"purpose": purpose}
        files = {"file": (filename, content, content_type)}
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http.post_multipart(
            synth_files_url(self._synth_base_url), data=data, files=files, headers=headers
        )

    async def list(
        self, *, purpose: str | None = None, after: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if purpose is not None:
            params["purpose"] = purpose
        if after is not None:
            params["after"] = after
        params["limit"] = limit
        return await self._http.get(synth_files_url(self._synth_base_url), params=params)

    async def retrieve(self, file_id: str) -> dict[str, Any]:
        return await self._http.get(synth_file_url(file_id, self._synth_base_url))

    async def delete(self, file_id: str) -> Any:
        return await self._http.delete(synth_file_url(file_id, self._synth_base_url))


class SftJobsApi:
    def __init__(self, http: AsyncHttpClient, synth_base_url: str | None) -> None:
        self._http = http
        self._synth_base_url = synth_base_url

    async def create(
        self,
        *,
        training_file: str,
        model: str,
        validation_file: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        integrations: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload = prepare_sft_job_payload(
            model=model,
            training_file=training_file,
            hyperparameters=hyperparameters,
            metadata=metadata,
            training_type=None,
            validation_file=validation_file,
            suffix=suffix,
            integrations=integrations,
            training_file_field="training_file",
            require_training_file=True,
        )
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http.post_json(
            synth_sft_jobs_url(self._synth_base_url), json=payload, headers=headers
        )

    async def list(
        self,
        *,
        status: str | None = None,
        model: str | None = None,
        file_id: str | None = None,
        created_after: int | None = None,
        created_before: int | None = None,
        after: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if model is not None:
            params["model"] = model
        if file_id is not None:
            params["file_id"] = file_id
        if created_after is not None:
            params["created_after"] = created_after
        if created_before is not None:
            params["created_before"] = created_before
        if after is not None:
            params["after"] = after
        return await self._http.get(synth_sft_jobs_url(self._synth_base_url), params=params)

    async def retrieve(self, job_id: str) -> dict[str, Any]:
        return await self._http.get(synth_sft_job_url(job_id, self._synth_base_url))

    async def cancel(self, job_id: str) -> dict[str, Any]:
        return await self._http.post_json(
            synth_learning_job_cancel_url(job_id, self._synth_base_url), json={}
        )

    async def list_events(
        self, job_id: str, *, since_seq: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        return await self._http.get(
            synth_sft_job_events_url(job_id, self._synth_base_url), params=params
        )


class RlJobsApi:
    def __init__(self, http: AsyncHttpClient, synth_base_url: str | None) -> None:
        self._http = http
        self._synth_base_url = synth_base_url

    async def create(
        self,
        *,
        model: str,
        endpoint_synth_base_url: str,
        trainer_id: str,
        trainer: dict[str, Any] | None = None,
        job_config_id: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": normalize_model_identifier(model),
            "endpoint_synth_base_url": endpoint_synth_base_url,
            "trainer_id": trainer_id,
        }
        if trainer is not None:
            payload["trainer"] = trainer
        if job_config_id is not None:
            payload["job_config_id"] = job_config_id
        if config is not None:
            payload["config"] = config
        if metadata is not None:
            payload["metadata"] = metadata
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http.post_json(
            synth_rl_jobs_url(self._synth_base_url), json=payload, headers=headers
        )

    async def list(
        self,
        *,
        status: str | None = None,
        model: str | None = None,
        created_after: int | None = None,
        created_before: int | None = None,
        after: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if model is not None:
            params["model"] = model
        if created_after is not None:
            params["created_after"] = created_after
        if created_before is not None:
            params["created_before"] = created_before
        if after is not None:
            params["after"] = after
        return await self._http.get(synth_rl_jobs_url(self._synth_base_url), params=params)

    async def retrieve(self, job_id: str) -> dict[str, Any]:
        return await self._http.get(synth_rl_job_url(job_id, self._synth_base_url))

    async def cancel(self, job_id: str) -> dict[str, Any]:
        return await self._http.post_json(
            synth_rl_job_cancel_url(job_id, self._synth_base_url), json={}
        )

    async def list_events(
        self, job_id: str, *, since_seq: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        return await self._http.get(
            synth_rl_job_events_url(job_id, self._synth_base_url), params=params
        )

    async def metrics(
        self, job_id: str, *, after_step: int = -1, limit: int = 200
    ) -> dict[str, Any]:
        params = {"after_step": after_step, "limit": limit}
        return await self._http.get(
            synth_rl_job_metrics_url(job_id, self._synth_base_url), params=params
        )


class ModelsApi:
    def __init__(self, http: AsyncHttpClient, synth_base_url: str | None) -> None:
        self._http = http
        self._synth_base_url = synth_base_url

    async def list(
        self,
        *,
        source: str | None = None,
        base_model: str | None = None,
        status: str | None = None,
        after: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if source is not None:
            params["source"] = source
        if base_model is not None:
            params["base_model"] = base_model
        if status is not None:
            params["status"] = status
        if after is not None:
            params["after"] = after
        return await self._http.get(synth_models_url(self._synth_base_url), params=params)

    async def retrieve(self, model_id: str) -> dict[str, Any]:
        return await self._http.get(synth_model_url(model_id, self._synth_base_url))

    async def delete(self, model_id: str) -> Any:
        return await self._http.delete(synth_model_url(model_id, self._synth_base_url))


class JobsClient:
    """High-level client for interacting with Synth AI job APIs.

    This client provides a unified interface for managing training jobs, files, and models
    across different job types (SFT, RL, prompt learning). It aggregates multiple API
    endpoints into a single, easy-to-use interface.

    The client is an async context manager, so it should be used with `async with`:

    Example:
        >>> from synth_ai.sdk.jobs import JobsClient
        >>>
    >>> async with JobsClient(
    ...     synth_base_url="https://api.usesynth.ai",
    ...     synth_user_key=os.environ["SYNTH_API_KEY"],
    ... ) as client:
        ...     # Upload a file
        ...     file_result = await client.files.upload(
        ...         filename="data.jsonl",
        ...         content=b"...",
        ...         purpose="training",
        ...     )
        ...
        ...     # Create an SFT job
        ...     job_result = await client.sft.create(
        ...         training_file=file_result["id"],
        ...         model="Qwen/Qwen3-4B",
        ...     )
        ...
        ...     # Check RL job status
        ...     rl_status = await client.rl.retrieve("rl_job_123")
        ...
        ...     # List models
        ...     models = await client.models.list()

    Attributes:
        files: FilesApi - File upload and management
        sft: SftJobsApi - Supervised fine-tuning jobs
        rl: RlJobsApi - Reinforcement learning jobs
        models: ModelsApi - Model management
    """

    def __init__(
        self,
        synth_user_key: str | None = None,
        timeout: float = 30.0,
        http: AsyncHttpClient | None = None,
        synth_base_url: str | None = None,
    ) -> None:
        """Initialize the Jobs API client.

        Args:
            synth_base_url: Base URL for the Synth AI API (e.g., "https://api.usesynth.ai")
            synth_user_key: Synth API key for authentication
            timeout: Request timeout in seconds (default: 30.0)
            http: Optional pre-configured HTTP client (for testing/customization)
        """
        self._synth_base_url = synth_base_url
        if synth_user_key is None:
            raise ValueError("synth_user_key is required")
        self._synth_user_key = synth_user_key
        self._timeout = timeout
        self._http = http or AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), synth_user_key, timeout=timeout
        )
        self.files = FilesApi(self._http, self._synth_base_url)
        self.sft = SftJobsApi(self._http, self._synth_base_url)
        self.rl = RlJobsApi(self._http, self._synth_base_url)
        self.models = ModelsApi(self._http, self._synth_base_url)

    async def __aenter__(self) -> "JobsClient":
        await self._http.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self._http.__aexit__(exc_type, exc, tb)
