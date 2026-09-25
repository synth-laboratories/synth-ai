"""Credential-free clients for browsing published Synth Index Contributions.

Search requires an authenticated funding identity and uses ``SynthClient().index``.
This boundary accepts no credential, private collection, or write method.
"""

from __future__ import annotations

from types import TracebackType

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base

from .client import AsyncPublicIndexAPI, PublicIndexAPI
from .timeouts import INDEX_TRANSPORT_TIMEOUT_SECONDS


class PublicIndexClient(PublicIndexAPI):
    """Owned synchronous client for anonymous public Index browse."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = INDEX_TRANSPORT_TIMEOUT_SECONDS,
    ) -> None:
        transport = HttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers={},
            timeout_seconds=timeout_seconds,
        )
        super().__init__(transport)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> PublicIndexClient:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AsyncPublicIndexClient(AsyncPublicIndexAPI):
    """Owned asynchronous client for anonymous public Index browse."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = INDEX_TRANSPORT_TIMEOUT_SECONDS,
    ) -> None:
        transport = AsyncHttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers={},
            timeout_seconds=timeout_seconds,
        )
        super().__init__(transport)

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncPublicIndexClient:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
