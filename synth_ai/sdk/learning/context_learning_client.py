"""Client for interacting with context learning jobs.

Context Learning optimizes environment setup scripts (pre-flight/post-flight bash)
for terminal/coding agents. This is "prompt optimization" for terminal agents.

Example usage:
    from synth_ai.sdk.learning import ContextLearningClient, ContextLearningJobConfig

    client = ContextLearningClient("http://localhost:8000", "your-synth-user-key")

    # Create a job from config
    config = ContextLearningJobConfig.from_dict({
        "localapi_url": "http://localhost:8102",
        "evaluation_seeds": [0, 1, 2, 3, 4],
        "environment": {
            "preflight_script": "#!/bin/bash\\necho 'Setting up...'",
        },
        "algorithm_config": {
            "initial_population_size": 10,
            "num_generations": 3,
        },
    })

    job_id = await client.create_job(config)

    # Stream events
    async for event in client.stream_events(job_id):
        print(f"{event.event_type}: {event.message}")

    # Get best script when done
    result = await client.get_best_script(job_id)
    print(result.preflight_script)
"""

import asyncio
import contextlib
from typing import Any, AsyncIterator, Dict, List, Optional

from synth_ai.core.http import AsyncHttpClient
from synth_ai.core.urls import (
    synth_api_v1_base,
    synth_context_learning_best_script_url,
    synth_context_learning_cancel_url,
    synth_context_learning_events_url,
    synth_context_learning_job_url,
    synth_context_learning_jobs_url,
    synth_context_learning_metrics_url,
    synth_context_learning_start_url,
)

from .context_learning_types import (
    BestScriptResult,
    ContextLearningEvent,
    ContextLearningJobConfig,
    ContextLearningJobStatus,
    ContextLearningMetric,
    ContextLearningResults,
)


def _validate_job_id(job_id: str) -> None:
    """Validate that job_id has the expected context learning format.

    Args:
        job_id: Job ID to validate

    Raises:
        ValueError: If job_id doesn't start with 'cl_job_'
    """
    if not job_id.startswith("cl_job_"):
        raise ValueError(
            f"Invalid context learning job ID format: {job_id!r}. "
            f"Expected format: 'cl_job_<identifier>' (e.g., 'cl_job_abc123def456')"
        )


class ContextLearningClient:
    """Async client for interacting with context learning jobs.

    This client provides a Pythonic interface for:
    - Creating and managing context learning jobs
    - Streaming events and metrics
    - Retrieving best scripts and results

    Example:
        >>> client = ContextLearningClient("http://localhost:8000", "your-synth-user-key")
        >>> job_id = await client.create_job(config)
        >>> status = await client.get_job(job_id)
        >>> print(status.status)  # "running"
    """

    def __init__(
        self,
        synth_user_key: str | None = None,
        *,
        timeout: float = 60.0,
        synth_base_url: str | None = None,
    ) -> None:
        """Initialize the context learning client.

        Args:
            synth_user_key: API key for authentication
            timeout: Request timeout in seconds (default: 60s for long operations)
            synth_base_url: Backend URL override (defaults to SYNTH_BACKEND_URL or production)
        """
        self._synth_base_url = synth_base_url
        if synth_user_key is None:
            raise ValueError("synth_user_key is required")
        self._synth_user_key = synth_user_key
        self._timeout = timeout

    # -------------------------------------------------------------------------
    # Job Management
    # -------------------------------------------------------------------------

    async def create_job(
        self,
        config: ContextLearningJobConfig | Dict[str, Any],
    ) -> str:
        """Create a new context learning job.

        Args:
            config: Job configuration (ContextLearningJobConfig or dict)

        Returns:
            Job ID (e.g., "cl_job_abc123def456")

        Raises:
            RuntimeError: If job creation fails
        """
        if isinstance(config, dict):
            config = ContextLearningJobConfig.from_dict(config)

        payload = config.to_dict()

        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.post_json(
                synth_context_learning_jobs_url(self._synth_base_url), json=payload
            )

        job_id = result.get("job_id")
        if not job_id:
            raise RuntimeError(f"Job creation failed: response missing job_id. Response: {result}")

        return job_id

    async def get_job(self, job_id: str) -> ContextLearningJobStatus:
        """Get job details and status.

        Args:
            job_id: Job ID

        Returns:
            Job status with details

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.get(synth_context_learning_job_url(job_id, self._synth_base_url))
        return ContextLearningJobStatus.from_dict(result)

    async def list_jobs(self, *, limit: int = 50) -> List[ContextLearningJobStatus]:
        """List context learning jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of job statuses, sorted by created_at descending
        """
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.get(
                synth_context_learning_jobs_url(self._synth_base_url), params={"limit": limit}
            )

        if isinstance(result, list):
            return [ContextLearningJobStatus.from_dict(j) for j in result]
        return []

    async def start_job(self, job_id: str) -> ContextLearningJobStatus:
        """Start a pending job.

        Args:
            job_id: Job ID

        Returns:
            Updated job status

        Raises:
            ValueError: If job_id format is invalid
            RuntimeError: If job cannot be started
        """
        _validate_job_id(job_id)
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.post_json(
                synth_context_learning_start_url(job_id, self._synth_base_url), json={}
            )
        return ContextLearningJobStatus.from_dict(result)

    async def cancel_job(self, job_id: str) -> ContextLearningJobStatus:
        """Cancel a running job.

        Args:
            job_id: Job ID

        Returns:
            Updated job status

        Raises:
            ValueError: If job_id format is invalid
            RuntimeError: If job cannot be cancelled
        """
        _validate_job_id(job_id)
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.post_json(
                synth_context_learning_cancel_url(job_id, self._synth_base_url), json={}
            )
        return ContextLearningJobStatus.from_dict(result)

    # -------------------------------------------------------------------------
    # Events and Metrics
    # -------------------------------------------------------------------------

    async def get_events(self, job_id: str, *, limit: int = 100) -> List[ContextLearningEvent]:
        """Get events for a job.

        Args:
            job_id: Job ID
            limit: Maximum number of events to return

        Returns:
            List of events, most recent first

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.get(
                synth_context_learning_events_url(job_id, self._synth_base_url),
                params={"limit": limit},
            )

        if isinstance(result, list):
            return [ContextLearningEvent.from_dict(e) for e in result]
        return []

    async def get_metrics(
        self, job_id: str, *, name: Optional[str] = None, limit: int = 500
    ) -> List[ContextLearningMetric]:
        """Get metrics for a job.

        Args:
            job_id: Job ID
            name: Optional metric name filter
            limit: Maximum number of metrics to return

        Returns:
            List of metrics

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        params: Dict[str, Any] = {"limit": limit}
        if name:
            params["name"] = name

        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.get(
                synth_context_learning_metrics_url(job_id, self._synth_base_url),
                params=params,
            )

        points = result.get("points", []) if isinstance(result, dict) else []
        return [ContextLearningMetric.from_dict(p) for p in points]

    async def stream_events(
        self,
        job_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[ContextLearningEvent]:
        """Stream events from a job until it reaches a terminal state.

        This is a polling-based implementation that yields new events as they arrive.

        Args:
            job_id: Job ID
            poll_interval: Seconds between polls
            timeout: Maximum seconds to stream (None = no timeout)

        Yields:
            Events as they occur

        Raises:
            ValueError: If job_id format is invalid
            asyncio.TimeoutError: If timeout is reached
        """
        _validate_job_id(job_id)

        seen_seqs: set[int] = set()
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check timeout
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Stream timeout after {timeout}s")

            # Get job status
            status = await self.get_job(job_id)

            # Get events
            events = await self.get_events(job_id, limit=500)

            # Yield new events
            for event in events:
                seq = event.seq
                if seq is not None and seq not in seen_seqs:
                    seen_seqs.add(seq)
                    yield event

            # Check if done
            if status.is_terminal:
                break

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    async def get_best_script(self, job_id: str) -> BestScriptResult:
        """Get the best performing pre-flight script.

        Args:
            job_id: Job ID

        Returns:
            Best script result with score and metadata

        Raises:
            ValueError: If job_id format is invalid
            RuntimeError: If job is not completed or no best script available
        """
        _validate_job_id(job_id)
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.get(
                synth_context_learning_best_script_url(job_id, self._synth_base_url)
            )
        return BestScriptResult.from_dict(result)

    async def get_results(self, job_id: str) -> ContextLearningResults:
        """Get complete results from a job.

        Aggregates status, events, and best script into a single result object.

        Args:
            job_id: Job ID

        Returns:
            Complete results including status, events, and best script

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)

        status = await self.get_job(job_id)
        events = await self.get_events(job_id, limit=1000)

        best_script: Optional[BestScriptResult] = None
        if status.is_successful:
            with contextlib.suppress(Exception):
                best_script = await self.get_best_script(job_id)

        return ContextLearningResults.from_status_and_events(status, events, best_script)

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def wait_for_completion(
        self,
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        on_event: Optional[callable] = None,
    ) -> ContextLearningJobStatus:
        """Wait for a job to complete.

        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None = no timeout)
            on_event: Optional callback for each new event

        Returns:
            Final job status

        Raises:
            ValueError: If job_id format is invalid
            asyncio.TimeoutError: If timeout is reached
        """
        async for event in self.stream_events(job_id, poll_interval=poll_interval, timeout=timeout):
            if on_event:
                with contextlib.suppress(Exception):
                    on_event(event)

        return await self.get_job(job_id)

    async def run_job(
        self,
        config: ContextLearningJobConfig | Dict[str, Any],
        *,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        on_event: Optional[callable] = None,
    ) -> ContextLearningResults:
        """Create a job, wait for completion, and return results.

        This is the main entry point for running context learning jobs.

        Args:
            config: Job configuration
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None = no timeout)
            on_event: Optional callback for each new event

        Returns:
            Complete results including best script

        Example:
            >>> config = ContextLearningJobConfig.from_dict({
            ...     "localapi_url": "http://localhost:8102",
            ...     "evaluation_seeds": [0, 1, 2],
            ... })
            >>> results = await client.run_job(config)
            >>> print(results.best_script.preflight_script)
        """
        job_id = await self.create_job(config)

        await self.wait_for_completion(
            job_id,
            poll_interval=poll_interval,
            timeout=timeout,
            on_event=on_event,
        )

        return await self.get_results(job_id)


# -----------------------------------------------------------------------------
# Synchronous Wrappers
# -----------------------------------------------------------------------------


def create_job(
    config: ContextLearningJobConfig | Dict[str, Any],
    base_url: str | None = None,
    synth_user_key: str | None = None,
) -> str:
    """Synchronous wrapper to create a context learning job.

    Args:
        config: Job configuration
        base_url: Backend API base URL
        synth_user_key: API key for authentication

    Returns:
        Job ID
    """
    client = ContextLearningClient(base_url, synth_user_key)
    return asyncio.run(client.create_job(config))


def get_job_status(
    job_id: str,
    base_url: str | None = None,
    synth_user_key: str | None = None,
) -> ContextLearningJobStatus:
    """Synchronous wrapper to get job status.

    Args:
        job_id: Job ID
        base_url: Backend API base URL
        synth_user_key: API key for authentication

    Returns:
        Job status
    """
    client = ContextLearningClient(base_url, synth_user_key)
    return asyncio.run(client.get_job(job_id))


def get_best_script(
    job_id: str,
    base_url: str | None = None,
    synth_user_key: str | None = None,
) -> BestScriptResult:
    """Synchronous wrapper to get best script.

    Args:
        job_id: Job ID
        base_url: Backend API base URL
        synth_user_key: API key for authentication

    Returns:
        Best script result
    """
    client = ContextLearningClient(base_url, synth_user_key)
    return asyncio.run(client.get_best_script(job_id))


def run_job(
    config: ContextLearningJobConfig | Dict[str, Any],
    base_url: str | None = None,
    synth_user_key: str | None = None,
    *,
    poll_interval: float = 5.0,
    timeout: Optional[float] = None,
    on_event: Optional[callable] = None,
) -> ContextLearningResults:
    """Synchronous wrapper to run a complete job.

    Args:
        config: Job configuration
        base_url: Backend API base URL
        synth_user_key: API key for authentication
        poll_interval: Seconds between status checks
        timeout: Maximum seconds to wait
        on_event: Optional callback for each event

    Returns:
        Complete results
    """
    client = ContextLearningClient(base_url, synth_user_key)
    return asyncio.run(
        client.run_job(
            config,
            poll_interval=poll_interval,
            timeout=timeout,
            on_event=on_event,
        )
    )
