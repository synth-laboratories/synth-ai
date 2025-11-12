"""Shared HTTP client for task app communication."""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx


class TaskAppClient:
    """Client for communicating with task apps via HTTP."""

    def __init__(self, task_app_url: str, api_key: str | None = None):
        """Initialize task app client.

        Args:
            task_app_url: Base URL of the task app (e.g., "http://127.0.0.1:8115")
            api_key: API key for authentication (defaults to ENVIRONMENT_API_KEY env var)
        """
        self.task_app_url = task_app_url.rstrip("/")
        self.api_key = api_key or os.getenv("ENVIRONMENT_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=300.0)

    async def evaluate_prompt(
        self,
        prompt_messages: list[dict[str, Any]],
        seed: int,
        task_app_id: str,
        model: str = "openai/gpt-oss-120b",
        provider: str = "groq",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a prompt on a task app seed.

        Args:
            prompt_messages: List of message dicts (role, pattern/content)
            seed: Seed ID (maps to dataset example)
            task_app_id: Task app identifier (e.g., "iris")
            model: Model identifier (default: gpt-oss-120b)
            provider: Provider name (default: groq)
            run_id: Optional run ID (generated if not provided)

        Returns:
            Rollout response dictionary with metrics, trajectory, etc.
        """
        if run_id is None:
            run_id = f"eval_{seed}_{uuid.uuid4().hex[:8]}"

        # Determine inference URL based on provider
        if provider == "groq":
            inference_url = "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "openai":
            inference_url = "https://api.openai.com/v1/chat/completions"
        else:
            inference_url = f"https://api.{provider}.com/v1/chat/completions"

        rollout_request = {
            "run_id": run_id,
            "env": {
                "env_name": task_app_id,
                "seed": seed,
                "config": {},
            },
            "policy": {
                "policy_name": "optimizer",
                "config": {
                    "model": model,
                    "provider": provider,
                    "inference_url": inference_url,
                    "messages": prompt_messages,
                    "temperature": 1.0,
                    "max_completion_tokens": 512,
                },
            },
            "ops": ["policy"],
            "mode": "eval",
        }

        response = await self.client.post(
            f"{self.task_app_url}/rollout",
            json=rollout_request,
            headers={"X-API-Key": self.api_key} if self.api_key else {},
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> dict[str, Any]:
        """Check task app health.

        Returns:
            Health check response
        """
        response = await self.client.get(f"{self.task_app_url}/health")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> TaskAppClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

