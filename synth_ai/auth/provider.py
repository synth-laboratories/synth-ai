"""Lazy CLI credential adoption for SDK/MCP. See backend auth-023.md.

Reload under the same cross-process lock as login/logout. Never cache a positive
credential between operations, and never replay a product request after refresh.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

import httpx

from synth_ai.core.http.blocking import CancellationScope, DeadlineTransport
from synth_ai.errors import AuthenticationError, RequestTimeoutError

from .oauth import AuthRefusalError, stored_access_token
from .storage import default_store


@dataclass(eq=False)
class _Invocation:
    scope: CancellationScope = field(default_factory=CancellationScope)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self.cancelled.set()
        self.scope.cancel()


class StoredCredentialProvider:
    """Construction performs no filesystem, Keychain or provider IO."""

    def __init__(self, origin: str):
        self.origin = origin
        self._condition = threading.Condition()
        self._active: set[_Invocation] = set()
        self._closed = False

    def resolve(self, deadline: float) -> str:
        return self._resolve(deadline, _Invocation())

    def _resolve(self, deadline: float, invocation: _Invocation) -> str:
        with self._condition:
            if self._closed or invocation.cancelled.is_set():
                raise AuthenticationError("Credential lookup cancelled", code="login_cancelled")
            self._active.add(invocation)
        try:

            def remaining() -> float:
                if invocation.cancelled.is_set():
                    raise AuthenticationError("Credential lookup cancelled", code="login_cancelled")
                seconds = deadline - time.monotonic()
                if seconds <= 0:
                    raise RequestTimeoutError("Credential lookup exceeded the request deadline")
                return seconds

            def bound(request: httpx.Request) -> None:
                seconds = remaining()
                request.extensions["timeout"] = httpx.Timeout(
                    seconds, connect=min(seconds, 5)
                ).as_dict()

            store = default_store()
            with store.locked(timeout_seconds=remaining()):
                remaining()
                with (
                    invocation.scope.activate(),
                    httpx.Client(
                        transport=DeadlineTransport(1),
                        # Proxies and CA selection belong to the inner pool.
                        trust_env=False,
                        auth=None,
                        follow_redirects=False,
                        event_hooks={"request": [bound]},
                    ) as http,
                ):
                    token = stored_access_token(store, api_origin=self.origin, http=http)
                remaining()
                if not token:
                    raise AuthenticationError(
                        "Run synth-ai login", code="login_required", status=401
                    )
                return token
        except AuthRefusalError as error:
            raise AuthenticationError(
                "Stored login is unavailable", code=error.code, status=error.status
            ) from None
        except (httpx.TimeoutException, TimeoutError) as error:
            raise RequestTimeoutError("Credential lookup exceeded the request deadline") from error
        except (OSError, ValueError) as error:
            raise AuthenticationError(
                "Stored login is unavailable", code="credential_storage_unavailable", status=401
            ) from error
        finally:
            with self._condition:
                self._active.discard(invocation)
                self._condition.notify_all()

    async def aresolve(self, deadline: float) -> str:
        invocation = _Invocation()
        task = asyncio.create_task(asyncio.to_thread(self._resolve, deadline, invocation))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            invocation.cancel()
            # Keep ownership until both the bounded file lock and network unwind.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    invocation.cancel()
                except Exception:
                    break
            if task.done() and not task.cancelled():
                task.exception()
            raise

    def close(self) -> None:
        with self._condition:
            self._closed = True
            active = tuple(self._active)
        for invocation in active:
            invocation.cancel()
        with self._condition:
            if not self._condition.wait_for(lambda: not self._active, timeout=10):
                raise RuntimeError("Credential lookup cleanup did not complete")
