from __future__ import annotations

import asyncio
import random
from typing import ClassVar, Dict, Optional

import httpx

from .config import TracingConfig


class ClientManager:
    """Singleton manager for HTTP clients with both sync and async support"""

    _instance: ClassVar[Optional[ClientManager]] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._config: Optional[TracingConfig] = None
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._credentials_cache: Dict[str, str] = {}

    @classmethod
    async def get_instance(cls) -> ClientManager:
        """Get or create the singleton instance asynchronously"""
        if not cls._instance:
            async with cls._lock:
                if not cls._instance:
                    cls._instance = ClientManager()
        return cls._instance

    @classmethod
    def initialize(cls, config: TracingConfig) -> ClientManager:
        """Initialize or return the singleton instance synchronously"""
        if cls._instance is None:
            cls._instance = cls()
        cls._instance.configure(config)
        return cls._instance

    def configure(self, config: TracingConfig) -> None:
        """Configure the client manager with new settings"""
        self._config = config
        self._credentials_cache = {
            "api_key": config.api_key,
            "api_secret": getattr(config, "api_secret", None),
        }
        self._close_clients()

    def get_sync_client(self) -> httpx.Client:
        """Get or create synchronized HTTP client with connection pooling"""
        if not self._sync_client and self._config:
            limits = httpx.Limits(
                max_connections=self._config.max_connections,
                max_keepalive_connections=self._config.max_connections,
            )
            timeout = httpx.Timeout(timeout=self._config.timeout)
            self._sync_client = httpx.Client(
                base_url=self._config.base_url,
                timeout=timeout,
                limits=limits,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
            )
        return self._sync_client

    async def get_async_client(self) -> httpx.AsyncClient:
        """Get or create asynchronous HTTP client with connection pooling"""
        if not self._async_client and self._config:
            limits = httpx.Limits(
                max_connections=self._config.max_connections,
                max_keepalive_connections=self._config.max_connections,
            )
            timeout = httpx.Timeout(timeout=self._config.timeout)
            self._async_client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=timeout,
                limits=limits,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
            )
        return self._async_client

    def calculate_backoff(self, retry_number: int) -> float:
        """Calculate backoff time with exponential backoff and jitter"""
        if not self._config:
            raise RuntimeError("ClientManager not configured")

        if retry_number <= 0:
            return 0

        if not getattr(self._config, "retry_exponential_backoff", True):
            return self._config.retry_backoff

        # Calculate exponential backoff
        delay = min(
            self._config.retry_backoff * (2 ** (retry_number - 1)),
            60.0,  # Cap at 60 seconds
        )

        # Add jitter (±25% of delay)
        jitter = delay * 0.25
        delay = random.uniform(delay - jitter, delay + jitter)

        return max(0.0, delay)

    def close(self) -> None:
        """Close the sync client"""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Close the async client"""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def _close_clients(self) -> None:
        """Close both sync and async clients"""
        self.close()
        if self._async_client:
            asyncio.create_task(self.aclose())

    @property
    def config(self) -> Optional[TracingConfig]:
        """Get the current configuration"""
        return self._config

    @config.setter
    def config(self, value: TracingConfig) -> None:
        """Set the configuration and reset clients"""
        self.configure(value)
