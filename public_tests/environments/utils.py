"""Utility functions for environment tests."""

import httpx
import pytest


async def check_service_running(port: int = 8901) -> None:
    """Check if the environment service is running and raise helpful error if not."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:{port}/health", timeout=2.0)
            if response.status_code != 200:
                raise RuntimeError(f"Service returned status {response.status_code}")
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.fail(  # type: ignore[no-untyped-call]
            f"\n\nEnvironment service is not running on port {port}!\n"
            f"Please start the service with:\n"
            f"  uvicorn synth_ai.environments.service.app:app --port {port}\n"
            f"You should see: INFO:     Uvicorn running on http://0.0.0.0:{port} (Press CTRL+C to quit)\n"
        )
    except Exception as e:
        pytest.fail(f"Failed to connect to service on port {port}: {e}")  # type: ignore[no-untyped-call]


def require_service(port: int = 8901):
    """Decorator to skip tests if service is not running."""

    def decorator(func):
        return pytest.mark.asyncio(
            pytest.mark.skipif(
                not is_service_running(port),
                reason=f"Environment service not running on port {port}",
            )(func)
        )

    return decorator


def is_service_running(port: int = 8901) -> bool:
    """Synchronously check if service is running."""
    try:
        import requests

        response = requests.get(f"http://localhost:{port}/health", timeout=1.0)
        return response.status_code == 200
    except:
        return False
