"""Synchronous facade over one dedicated async client loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._blocking import Runner
from ._sync import (
    AccountClient,
    ArtifactoryClient,
    ArtifactsClient,
    BillingClient,
    ExperimentsClient,
    NotebooksClient,
    ResearchLogsClient,
    SwarmsClient,
)
from .aclient import AsyncSynthClient

if TYPE_CHECKING:
    from types import TracebackType

    import httpx

    from ._generated.models import ApiVersionInfo
    from ._transport import RetryPolicy
    from .diagnostics import DiagnosticHook
    from .environments import Environment

__all__ = ["SynthClient"]


class SynthClient:
    """Access Swarms, account and billing through the public API.

    Use as a context manager to close client resources. API keys and base URLs
    can be passed explicitly or resolved by the documented authentication flow.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        environment: Environment | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30,
        retries: int = 2,
        retry_policy: RetryPolicy | None = None,
        diagnostic_hook: DiagnosticHook | None = None,
    ) -> None:
        """Create a client with a request timeout in seconds and bounded retries."""
        self._async = AsyncSynthClient(
            api_key,
            base_url=base_url,
            environment=environment,
            http_client=http_client,
            timeout=timeout,
            retries=retries,
            retry_policy=retry_policy,
            diagnostic_hook=diagnostic_hook,
        )
        self._runner = Runner()
        self.swarms = SwarmsClient(self._runner, self._async.swarms)
        self.account = AccountClient(self._runner, self._async.account)
        self.billing = BillingClient(self._runner, self._async.billing)
        self.artifacts = ArtifactsClient(self._runner, self._async.artifacts)
        self.experiments = ExperimentsClient(self._runner, self._async.experiments)
        self.research_logs = ResearchLogsClient(self._runner, self._async.research_logs)
        self.notebooks = NotebooksClient(self._runner, self._async.notebooks)
        self.artifactory = ArtifactoryClient(self._runner, self._async.artifactory)

    def version(self) -> ApiVersionInfo:
        """Return the server's SemVer, public API contract version, and compatibility bounds."""
        return self._runner.call(self._async.version())

    def close(self) -> None:
        """Close the client transport and owned network resources."""
        if self._runner.closed:
            return
        try:
            self._runner.call(self._async.close())
        finally:
            self._runner.close()

    def __enter__(self) -> SynthClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
