"""Health check utilities for scanning."""

import asyncio
from typing import Any, Literal

import httpx


async def check_app_health(
    url: str,
    api_key: str | None,
    timeout: float = 2.0,
) -> tuple[Literal["healthy", "unhealthy", "unknown"], dict[str, Any]]:
    """Check health and fetch metadata from a task app.

    Args:
        url: Base URL of the task app
        api_key: API key for authentication via X-API-Key header
        timeout: Request timeout in seconds

    Returns:
        Tuple of (health_status, metadata_dict)
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    metadata: dict[str, Any] = {}

    health_status: Literal["healthy", "unhealthy", "unknown"] = "unknown"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            health_url = f"{url.rstrip('/')}/health"
            health_resp = await client.get(health_url, headers=headers)

            if health_resp.status_code in (530, 502, 503, 504):
                health_status = "unhealthy"
                metadata["tunnel_error"] = f"HTTP {health_resp.status_code}"
            elif "text/html" in health_resp.headers.get("content-type", "").lower():
                health_status = "unhealthy"
                metadata["tunnel_error"] = "HTML response (tunnel error)"
            elif health_resp.status_code == 200:
                try:
                    health_data = health_resp.json()
                    if isinstance(health_data, dict):
                        status = health_data.get("status", "").lower()
                        healthy_flag = health_data.get("healthy")
                        if status == "healthy" or healthy_flag is True:
                            health_status = "healthy"
                        elif status == "unhealthy" or healthy_flag is False:
                            health_status = "unhealthy"
                        else:
                            health_status = "healthy"
                    else:
                        health_status = "healthy"
                except Exception:
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


def extract_app_info(
    metadata: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract app information from /info endpoint metadata.

    Args:
        metadata: Dictionary containing /info endpoint response

    Returns:
        Tuple of (app_id, task_name, dataset_id, version)
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

    Args:
        urls: List of app URLs to check
        api_key: API key for authentication
        timeout: Request timeout in seconds per app
        max_concurrent: Maximum number of concurrent HTTP requests

    Returns:
        Dictionary mapping URL -> (health_status, metadata)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, tuple[Literal["healthy", "unhealthy", "unknown"], dict[str, Any]]] = {}

    async def check_one(url: str) -> None:
        async with semaphore:
            status, metadata = await check_app_health(url, api_key, timeout)
            results[url] = (status, metadata)

    await asyncio.gather(*[check_one(url) for url in urls], return_exceptions=True)

    return results
