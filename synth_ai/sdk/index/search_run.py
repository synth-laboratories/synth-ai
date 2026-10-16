"""Keyed durable Search handles and bounded observers. See SDK index_contract.md."""

from __future__ import annotations

import asyncio
import time
from typing import NoReturn

from synth_ai.contracts import operations as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.observation import AsyncEventStream, EventStream
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.errors import ContractError, OperationFailedError, RequestTimeoutError
from synth_ai.options import MutationOptions, ObservationOptions, ReadOptions
from synth_ai.types.base import EventCursor, Page
from synth_ai.types.receipts import ControlReceipt
from synth_ai.types.usage import UsageOperation, UsageSummary

from .search import validate_search_retention
from .types.history import SearchHistoryDeletion, SearchHistoryDeletionReceipt
from .types.retention import RetentionMode
from .types.search import SearchEvent, SearchResponse, SearchSnapshot


_TERMINAL_WITHOUT_RESULT = frozenset({"failed", "cancelled"})


def _raise_terminal(search_id: str, snapshot: SearchSnapshot, result: SearchResponse) -> NoReturn:
    """Fail early and hard on a terminal Search with no deliverable result."""
    if result.status != snapshot.state:
        raise ContractError(
            f"Search {search_id} reported state {snapshot.state!r} but its result is "
            f"{result.status!r}",
            code="search_terminal_state_mismatch",
            request_id=result.metadata.request_id,
        )
    failure = result.failure
    code = failure.code if failure is not None else "search_cancelled"
    retryable = failure.retryable if failure is not None else False
    raise OperationFailedError(
        f"Search {search_id} {result.status}: {code} (retryable={retryable})",
        code=code,
        request_id=result.metadata.request_id,
    )


class SearchRun:
    """Durable identity; each method reads fresh authorized state."""

    def __init__(
        self,
        transport: HttpTransport,
        search_id: str,
        *,
        expected_retention: RetentionMode | None = None,
    ) -> None:
        self._transport = transport
        self.search_id = identifier(search_id)
        self._expected_retention = expected_retention

    def status(self, *, request: ReadOptions | None = None) -> SearchSnapshot:
        """Read fresh state; this does not cancel, restart or finalize work.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.GET_SEARCH,
            SearchSnapshot,
            path=f"/index/searches/{self.search_id}",
            options=request,
            validate=lambda received: verified(
                received,
                received.search_id == self.search_id,
                "Response resource identity or bounds mismatch",
            ),
        )
        self._validate_retention(result)
        return result

    def result(self, *, request: ReadOptions | None = None) -> SearchResponse:
        """Read the saved canonical outcome, including partial/failure/cancellation.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.SEARCH_RESULT,
            SearchResponse,
            path=f"/index/searches/{self.search_id}/result",
            options=request,
            validate=lambda received: verified(
                received,
                received.search_id == self.search_id,
                "Response resource identity or bounds mismatch",
            ),
        )
        self._validate_retention(result)
        return result

    def _validate_retention(self, result: SearchResponse | SearchSnapshot) -> None:
        validate_search_retention(result, self._expected_retention)

    def delete_history(
        self,
        *,
        expected_archive_digest_sha256: str,
        mutation: MutationOptions | None = None,
    ) -> SearchHistoryDeletionReceipt:
        """Delete the exact archived payload without cancelling or rerunning the search.

        See: backend/notes/specifications/sdk/index_contract.md."""
        body = SearchHistoryDeletion(expected_archive_digest_sha256=expected_archive_digest_sha256)
        return self._transport.request(
            operations.DELETE_SEARCH_HISTORY,
            SearchHistoryDeletionReceipt,
            path=f"/index/searches/{self.search_id}/history/delete",
            body=body,
            options=mutation,
            validate=lambda receipt: verified(
                receipt,
                receipt.search_id == self.search_id
                and receipt.archive_digest_sha256 == body.expected_archive_digest_sha256,
                "History deletion returned a different resource or archived payload",
            ),
        )

    def event_page(
        self, *, limit: int, cursor: EventCursor | None = None, request: ReadOptions | None = None
    ) -> Page[SearchEvent, EventCursor]:
        """Read a bounded cursor page tied to this search and current grants.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.SEARCH_EVENTS,
            Page[SearchEvent, EventCursor],
            path=f"/index/searches/{self.search_id}/events",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    len(received.items) > limit
                    or any(event.search_id != self.search_id for event in received.items)
                ),
                "Response resource identity or bounds mismatch",
            ),
        )
        return result

    def usage(self, *, request: ReadOptions | None = None) -> UsageSummary:
        """Read authoritative usage, cost and customer charge with independent finality.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.SEARCH_USAGE,
            UsageSummary,
            path=f"/index/searches/{self.search_id}/usage",
            options=request,
        )
        return result

    def usage_operations(
        self, *, limit: int, cursor: EventCursor | None = None, request: ReadOptions | None = None
    ) -> Page[UsageOperation]:
        """Read one bounded page of attributable operation/attempt observations.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.SEARCH_USAGE_ITEMS,
            Page[UsageOperation],
            path=f"/index/searches/{self.search_id}/usage/operations",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not len(received.items) > limit,
                "Response resource identity or bounds mismatch",
            ),
        )
        return result

    def cancel(self, *, mutation: MutationOptions | None = None) -> ControlReceipt:
        """Request explicit remote cancellation and return an accepted/applied/refused receipt.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.CANCEL_SEARCH,
            ControlReceipt,
            path=f"/index/searches/{self.search_id}/cancel",
            options=mutation,
            validate=lambda received: verified(
                received,
                received.resource_id == self.search_id,
                "Cancel receipt identifies a different search",
            ),
        )
        return result

    def control(self, operation_id: str, *, request: ReadOptions | None = None) -> ControlReceipt:
        """Read the current control receipt without reissuing the control operation.

        See: backend/notes/specifications/sdk/index_contract.md."""
        operation_id = identifier(operation_id)
        result = self._transport.request(
            operations.CONTROL,
            ControlReceipt,
            path=f"/index/searches/{self.search_id}/controls/{operation_id}",
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    received.resource_id != self.search_id or received.operation_id != operation_id
                ),
                "Control receipt identity mismatch",
            ),
        )
        return result

    def events(
        self, *, observation: ObservationOptions, cursor: EventCursor | None = None
    ) -> EventStream[SearchEvent]:
        """Create a lazy managed observer; local close never cancels remote work.

        See: backend/notes/specifications/sdk/index_contract.md."""
        return EventStream(self.event_page, cursor, observation, owner=self._transport)

    def wait(self, *, observation: ObservationOptions) -> SearchResponse:
        """Observe canonical result availability within a local deadline; never cancel or resubmit.

        See: backend/notes/specifications/sdk/index_contract.md."""
        deadline = time.monotonic() + observation.timeout_seconds
        polls = min(10000, int(observation.timeout_seconds / observation.poll_interval_seconds) + 2)
        for _ in range(polls):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RequestTimeoutError("Search wait expired; durable work continues")
            request = ReadOptions(
                timeout_seconds=min(remaining, observation.idle_timeout_seconds),
                read_retry_count=observation.retry_count_max,
            )
            snapshot = self.status(request=request)
            if snapshot.state in _TERMINAL_WITHOUT_RESULT and not snapshot.result_available:
                # A failed/cancelled Search never becomes result_available; fail
                # now with its typed cause instead of polling to the deadline.
                remaining = deadline - time.monotonic()
                _raise_terminal(
                    self.search_id,
                    snapshot,
                    self.result(
                        request=ReadOptions(
                            timeout_seconds=max(remaining, 1.0),
                            read_retry_count=observation.retry_count_max,
                        )
                    ),
                )
            if snapshot.result_available:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RequestTimeoutError("Search wait expired; durable work continues")
                return self.result(
                    request=ReadOptions(
                        timeout_seconds=remaining, read_retry_count=observation.retry_count_max
                    )
                )
            time.sleep(min(observation.poll_interval_seconds, max(0, deadline - time.monotonic())))
        raise RequestTimeoutError("Search wait expired; durable work continues")


class AsyncSearchRun:
    """Durable identity; each method reads fresh authorized state."""

    def __init__(
        self,
        transport: AsyncHttpTransport,
        search_id: str,
        *,
        expected_retention: RetentionMode | None = None,
    ) -> None:
        self._transport = transport
        self.search_id = identifier(search_id)
        self._expected_retention = expected_retention

    async def status(self, *, request: ReadOptions | None = None) -> SearchSnapshot:
        """Read fresh state; this does not cancel, restart or finalize work.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.GET_SEARCH,
            SearchSnapshot,
            path=f"/index/searches/{self.search_id}",
            options=request,
            validate=lambda received: verified(
                received,
                received.search_id == self.search_id,
                "Response resource identity or bounds mismatch",
            ),
        )
        self._validate_retention(result)
        return result

    async def result(self, *, request: ReadOptions | None = None) -> SearchResponse:
        """Read the saved canonical outcome, including partial/failure/cancellation.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.SEARCH_RESULT,
            SearchResponse,
            path=f"/index/searches/{self.search_id}/result",
            options=request,
            validate=lambda received: verified(
                received,
                received.search_id == self.search_id,
                "Response resource identity or bounds mismatch",
            ),
        )
        self._validate_retention(result)
        return result

    def _validate_retention(self, result: SearchResponse | SearchSnapshot) -> None:
        validate_search_retention(result, self._expected_retention)

    async def delete_history(
        self,
        *,
        expected_archive_digest_sha256: str,
        mutation: MutationOptions | None = None,
    ) -> SearchHistoryDeletionReceipt:
        """Delete the exact archived payload without cancelling or rerunning the search.

        See: backend/notes/specifications/sdk/index_contract.md."""
        body = SearchHistoryDeletion(expected_archive_digest_sha256=expected_archive_digest_sha256)
        return await self._transport.request(
            operations.DELETE_SEARCH_HISTORY,
            SearchHistoryDeletionReceipt,
            path=f"/index/searches/{self.search_id}/history/delete",
            body=body,
            options=mutation,
            validate=lambda receipt: verified(
                receipt,
                receipt.search_id == self.search_id
                and receipt.archive_digest_sha256 == body.expected_archive_digest_sha256,
                "History deletion returned a different resource or archived payload",
            ),
        )

    async def event_page(
        self, *, limit: int, cursor: EventCursor | None = None, request: ReadOptions | None = None
    ) -> Page[SearchEvent, EventCursor]:
        """Read a bounded cursor page tied to this search and current grants.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.SEARCH_EVENTS,
            Page[SearchEvent, EventCursor],
            path=f"/index/searches/{self.search_id}/events",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    len(received.items) > limit
                    or any(event.search_id != self.search_id for event in received.items)
                ),
                "Response resource identity or bounds mismatch",
            ),
        )
        return result

    async def usage(self, *, request: ReadOptions | None = None) -> UsageSummary:
        """Read authoritative usage, cost and customer charge with independent finality.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.SEARCH_USAGE,
            UsageSummary,
            path=f"/index/searches/{self.search_id}/usage",
            options=request,
        )
        return result

    async def usage_operations(
        self, *, limit: int, cursor: EventCursor | None = None, request: ReadOptions | None = None
    ) -> Page[UsageOperation]:
        """Read one bounded page of attributable operation/attempt observations.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.SEARCH_USAGE_ITEMS,
            Page[UsageOperation],
            path=f"/index/searches/{self.search_id}/usage/operations",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not len(received.items) > limit,
                "Response resource identity or bounds mismatch",
            ),
        )
        return result

    async def cancel(self, *, mutation: MutationOptions | None = None) -> ControlReceipt:
        """Request explicit remote cancellation and return an accepted/applied/refused receipt.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.CANCEL_SEARCH,
            ControlReceipt,
            path=f"/index/searches/{self.search_id}/cancel",
            options=mutation,
            validate=lambda received: verified(
                received,
                received.resource_id == self.search_id,
                "Cancel receipt identifies a different search",
            ),
        )
        return result

    async def control(
        self, operation_id: str, *, request: ReadOptions | None = None
    ) -> ControlReceipt:
        """Read the current control receipt without reissuing the control operation.

        See: backend/notes/specifications/sdk/index_contract.md."""
        operation_id = identifier(operation_id)
        result = await self._transport.request(
            operations.CONTROL,
            ControlReceipt,
            path=f"/index/searches/{self.search_id}/controls/{operation_id}",
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    received.resource_id != self.search_id or received.operation_id != operation_id
                ),
                "Control receipt identity mismatch",
            ),
        )
        return result

    def events(
        self, *, observation: ObservationOptions, cursor: EventCursor | None = None
    ) -> AsyncEventStream[SearchEvent]:
        """Create a lazy managed observer; local close never cancels remote work.

        See: backend/notes/specifications/sdk/index_contract.md."""
        return AsyncEventStream(self.event_page, cursor, observation, owner=self._transport)

    async def wait(self, *, observation: ObservationOptions) -> SearchResponse:
        """Observe canonical result availability within a local deadline; never cancel or resubmit.

        See: backend/notes/specifications/sdk/index_contract.md."""
        deadline = time.monotonic() + observation.timeout_seconds
        polls = min(10000, int(observation.timeout_seconds / observation.poll_interval_seconds) + 2)
        for _ in range(polls):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RequestTimeoutError("Search wait expired; durable work continues")
            request = ReadOptions(
                timeout_seconds=min(remaining, observation.idle_timeout_seconds),
                read_retry_count=observation.retry_count_max,
            )
            snapshot = await self.status(request=request)
            if snapshot.state in _TERMINAL_WITHOUT_RESULT and not snapshot.result_available:
                remaining = deadline - time.monotonic()
                _raise_terminal(
                    self.search_id,
                    snapshot,
                    await self.result(
                        request=ReadOptions(
                            timeout_seconds=max(remaining, 1.0),
                            read_retry_count=observation.retry_count_max,
                        )
                    ),
                )
            if snapshot.result_available:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RequestTimeoutError("Search wait expired; durable work continues")
                return await self.result(
                    request=ReadOptions(
                        timeout_seconds=remaining, read_retry_count=observation.retry_count_max
                    )
                )
            await asyncio.sleep(
                min(observation.poll_interval_seconds, max(0, deadline - time.monotonic()))
            )
        raise RequestTimeoutError("Search wait expired; durable work continues")
