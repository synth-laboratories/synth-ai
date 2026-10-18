"""Organization retention policy resource. See SDK index_retention.md."""

from __future__ import annotations

from synth_ai.contracts import operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.options import MutationOptions, ReadOptions

from .types.retention import RetentionMode
from .types.retention_policy import RetentionPolicy, RetentionPolicyUpdate


def _update(
    default_mode: RetentionMode,
    allowed_modes: tuple[RetentionMode, ...],
    enforced: bool | None,
    expected_revision: int | None,
) -> RetentionPolicyUpdate:
    return RetentionPolicyUpdate(
        default_mode=default_mode,
        allowed_modes=allowed_modes,
        enforced=enforced,
        expected_revision=expected_revision,
    )


class Retention:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def get(self, *, request: ReadOptions | None = None) -> RetentionPolicy:
        """Read the organization's current retention policy with fresh server state."""
        return self._transport.request(operations.GET_RETENTION, RetentionPolicy, options=request)

    def update(
        self,
        *,
        default_mode: RetentionMode,
        allowed_modes: tuple[RetentionMode, ...],
        enforced: bool | None = None,
        expected_revision: int | None = None,
        mutation: MutationOptions | None = None,
    ) -> RetentionPolicy:
        """Replace the organization policy (org admins only); keyed and idempotent."""
        return self._transport.request(
            operations.PUT_RETENTION,
            RetentionPolicy,
            body=_update(default_mode, allowed_modes, enforced, expected_revision),
            options=mutation,
        )


class AsyncRetention:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def get(self, *, request: ReadOptions | None = None) -> RetentionPolicy:
        """Read the organization's current retention policy with fresh server state."""
        return await self._transport.request(
            operations.GET_RETENTION, RetentionPolicy, options=request
        )

    async def update(
        self,
        *,
        default_mode: RetentionMode,
        allowed_modes: tuple[RetentionMode, ...],
        enforced: bool | None = None,
        expected_revision: int | None = None,
        mutation: MutationOptions | None = None,
    ) -> RetentionPolicy:
        """Replace the organization policy (org admins only); keyed and idempotent."""
        return await self._transport.request(
            operations.PUT_RETENTION,
            RetentionPolicy,
            body=_update(default_mode, allowed_modes, enforced, expected_revision),
            options=mutation,
        )
