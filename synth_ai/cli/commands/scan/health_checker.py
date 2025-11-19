"""Health check utilities for scan command."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx


async def check_app_health(
    url: str,
    api_key: str | None,
    timeout: float = 2.0,
) -> tuple[Literal["healthy", "unhealthy", "unknown"], dict[str, Any]]:
    """Check health and fetch metadata from a task app.

    Performs HTTP health checks on a task app by:
    1. Checking the /health endpoint for health status
    2. Fetching the /info endpoint for metadata (app_id, version, etc.)

    Health Status Determination:
        - "healthy": HTTP 200 with valid JSON response containing "status": "healthy"
            or "healthy": true, or HTTP 200 with any valid JSON
        - "unhealthy": HTTP error status (4xx, 5xx), Cloudflare tunnel errors (530, 502),
            or HTML responses (Cloudflare error pages)
        - "unknown": Request timeout, connection errors, or other exceptions

    Args:
        url: Base URL of the task app (e.g., "http://localhost:8000" or
            "https://abc123.trycloudflare.com"). Trailing slashes are handled.
        api_key: API key for authentication via X-API-Key header. If None, requests
            are made without authentication (may fail for apps requiring auth).
        timeout: Request timeout in seconds for both /health and /info endpoints.
            Default is 2.0 seconds.

    Returns:
        Tuple of (health_status, metadata_dict) where:
            - health_status: One of "healthy", "unhealthy", or "unknown"
            - metadata_dict: Dictionary containing:
                - Full /info endpoint response (if available)
                - Error information (if health check failed)
                - HTTP status codes and error messages

    Examples:
        >>> import asyncio
        >>> status, metadata = asyncio.run(check_app_health("http://localhost:8000", "key123"))
        >>> print(status)  # "healthy", "unhealthy", or "unknown"
        >>> print(metadata.get("service", {}).get("task", {}).get("id"))  # app_id

    Raises:
        No exceptions are raised - all errors are captured and returned as "unknown"
        status with error details in metadata.
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    metadata: dict[str, Any] = {}

    # Check /health endpoint
    health_status: Literal["healthy", "unhealthy", "unknown"] = "unknown"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            health_url = f"{url.rstrip('/')}/health"
            health_resp = await client.get(health_url, headers=headers)
            
            # Check for Cloudflare tunnel errors (530, 502, etc.) or HTML responses
            if health_resp.status_code in (530, 502, 503, 504):
                # Cloudflare error - tunnel pointing to dead server
                health_status = "unhealthy"
                metadata["tunnel_error"] = f"HTTP {health_resp.status_code}"
            elif "text/html" in health_resp.headers.get("content-type", "").lower():
                # HTML response (Cloudflare error page) - tunnel not working
                health_status = "unhealthy"
                metadata["tunnel_error"] = "HTML response (tunnel error)"
            elif health_resp.status_code == 200:
                try:
                    health_data = health_resp.json()
                    if isinstance(health_data, dict):
                        # Check for "status": "healthy" or "healthy": true
                        status = health_data.get("status", "").lower()
                        healthy_flag = health_data.get("healthy")
                        if status == "healthy" or healthy_flag is True:
                            health_status = "healthy"
                        elif status == "unhealthy" or healthy_flag is False:
                            health_status = "unhealthy"
                        else:
                            # If status is 200 but no clear health indicator, consider healthy
                            health_status = "healthy"
                    else:
                        # If JSON parsing fails but status is 200, consider it healthy
                        health_status = "healthy"
                except Exception:
                    # If JSON parsing fails but status is 200, consider it healthy
                    health_status = "healthy"
            else:
                health_status = "unhealthy"
                metadata["http_status"] = health_resp.status_code
    except httpx.TimeoutException:
        health_status = "unknown"
        metadata["error"] = "timeout"
    except Exception as exc:
        health_status = "unknown"
        metadata["error"] = str(exc)

    # Fetch /info endpoint for metadata
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            info_url = f"{url.rstrip('/')}/info"
            info_resp = await client.get(info_url, headers=headers)
            if info_resp.status_code == 200:
                try:
                    info_data = info_resp.json()
                    if isinstance(info_data, dict):
                        metadata.update(info_data)
                except Exception:
                    pass
    except Exception:
        pass

    return health_status, metadata


def extract_app_info(metadata: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract app information from /info endpoint metadata.

    Parses the metadata dictionary (typically from /info endpoint response) to
    extract key application identifiers and version information. Handles missing
    or malformed data gracefully by returning None for unavailable fields.

    Metadata Structure Expected:
        {
            "service": {
                "task": {
                    "id": "app_id",
                    "name": "Task Name",
                    "version": "1.0.0"
                }
            },
            "dataset": {
                "id": "dataset_id"
            }
        }

    Args:
        metadata: Dictionary containing /info endpoint response or service records.
            May contain nested "service" and "dataset" dictionaries.

    Returns:
        Tuple of (app_id, task_name, dataset_id, version) where:
            - app_id: Task app identifier from service.task.id
            - task_name: Human-readable task name from service.task.name
            - dataset_id: Dataset identifier from dataset.id (if available)
            - version: App version from service.task.version

    Examples:
        >>> metadata = {
        ...     "service": {"task": {"id": "banking77", "name": "Banking77", "version": "1.0.0"}},
        ...     "dataset": {"id": "banking77_dataset"}
        ... }
        >>> app_id, task_name, dataset_id, version = extract_app_info(metadata)
        >>> print(app_id)  # "banking77"
        >>> print(version)  # "1.0.0"
    """
    app_id: str | None = None
    task_name: str | None = None
    dataset_id: str | None = None
    version: str | None = None

    try:
        service = metadata.get("service", {})
        if isinstance(service, dict):
            task = service.get("task", {})
            if isinstance(task, dict):
                app_id = task.get("id")
                task_name = task.get("name")
                version = task.get("version")

        dataset = metadata.get("dataset", {})
        if isinstance(dataset, dict):
            dataset_id = dataset.get("id")
    except Exception:
        pass

    return app_id, task_name, dataset_id, version


async def check_multiple_apps_health(
    urls: list[str],
    api_key: str | None,
    timeout: float = 2.0,
    max_concurrent: int = 10,
) -> dict[str, tuple[Literal["healthy", "unhealthy", "unknown"], dict[str, Any]]]:
    """Check health for multiple apps concurrently.

    Performs health checks on multiple task apps in parallel using asyncio
    semaphores to limit concurrent requests. This is more efficient than
    checking apps sequentially, especially when scanning many ports or services.

    Concurrency Control:
        Uses asyncio.Semaphore to limit the number of concurrent HTTP requests,
        preventing resource exhaustion when checking many apps simultaneously.

    Args:
        urls: List of app URLs to check (e.g., ["http://localhost:8000", ...]).
            Each URL will be checked independently.
        api_key: API key for authentication via X-API-Key header. Applied to
            all requests. If None, requests are made without authentication.
        timeout: Request timeout in seconds per app. Default is 2.0 seconds.
        max_concurrent: Maximum number of concurrent HTTP requests. Default is 10.
            Increase for faster scanning, decrease to reduce resource usage.

    Returns:
        Dictionary mapping URL -> (health_status, metadata) where:
            - Keys are the input URLs
            - Values are tuples of (health_status, metadata_dict) as returned
              by check_app_health()

    Examples:
        >>> import asyncio
        >>> urls = ["http://localhost:8000", "http://localhost:8001"]
        >>> results = asyncio.run(check_multiple_apps_health(urls, "key123"))
        >>> for url, (status, metadata) in results.items():
        ...     print(f"{url}: {status}")

    Note:
        Exceptions during individual health checks are caught and result in
        "unknown" status. The function never raises exceptions.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, tuple[Literal["healthy", "unhealthy", "unknown"], dict[str, Any]]] = {}

    async def check_one(url: str) -> None:
        async with semaphore:
            status, metadata = await check_app_health(url, api_key, timeout)
            results[url] = (status, metadata)

    await asyncio.gather(*[check_one(url) for url in urls], return_exceptions=True)

    return results

