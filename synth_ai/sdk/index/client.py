"""Small typed Index facade. See the public contract snapshots in contracts/"""

from __future__ import annotations

import time

from synth_ai.contracts.operations import PUBLIC_SEARCH, SEARCH
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.errors import OutcomeUncertainError, RequestTimeoutError
from synth_ai.options import MutationOptions, ObservationOptions

from .account import Account, AsyncAccount
from .collections import AsyncCollections, Collections
from .contents import AsyncContents, Contents
from .contributions import AsyncContributions, Contributions
from .retention import AsyncRetention, Retention
from .search import SearchOptions, search_request, validate_search_response
from .searches import AsyncSearches, Searches
from .types.search import KeyedSearchOptions, SearchEvent, SearchResponse
from .usage import AsyncUsage, Usage


def _deep_observation(
    options: KeyedSearchOptions,
    mutation: MutationOptions | None,
    started: float,
    search_id: str,
    client_timeout: float,
) -> ObservationOptions:
    # Preserve an explicit synchronous timeout, including time spent admitting.
    # Otherwise allow the admitted deadline plus time for reviewed delivery.
    budget = (
        mutation.timeout_seconds
        if mutation and mutation.timeout_seconds is not None
        else client_timeout
        if client_timeout != 30.0
        else (options.deadline_seconds or 180) + 90
    )
    remaining = budget - (time.monotonic() - started)
    if remaining <= 0:
        raise RequestTimeoutError(
            f"Search {search_id} observation expired after admission; current state is unknown",
            cause=f"search_id={search_id}",
        )
    return ObservationOptions(
        timeout_seconds=remaining,
        idle_timeout_seconds=min(75, remaining),
        reconnect_deadline_seconds=min(30, remaining),
        retry_count_max=2,
        maximum_events=10000,
        poll_interval_seconds=1,
    )


def _durable_deep(keyed: bool, options: SearchOptions, mutation: MutationOptions | None) -> bool:
    # An already-bound legacy mutation identity must keep its original route.
    return (
        keyed
        and isinstance(options, KeyedSearchOptions)
        and options.mode == "deep"
        and not (mutation and mutation.identity and mutation.identity.operation == SEARCH.id)
    )


class IndexClient:
    contents: Contents
    searches: Searches
    contributions: Contributions
    account: Account
    collections: Collections
    usage: Usage
    retention: Retention

    def __init__(self, transport: HttpTransport, keyed: bool) -> None:
        self._transport = transport
        self._keyed = keyed
        self.contents = Contents(transport, keyed)
        self.searches = Searches(transport)
        self.contributions = Contributions(transport, keyed)
        self.account = Account(transport)
        self.collections = Collections(transport)
        self.usage = Usage(transport)
        self.retention = Retention(transport)

    def search(
        self, query: str, *, options: SearchOptions, mutation: MutationOptions | None = None
    ) -> SearchResponse:
        """Return a terminal answer under explicit public/keyed authority and charge bounds.

        See: the public contract snapshots in contracts/"""
        body = search_request(query, options, self._keyed)
        if _durable_deep(self._keyed, options, mutation):
            assert isinstance(options, KeyedSearchOptions)
            started = time.monotonic()
            run = self.searches.create(query, options=options, mutation=mutation)
            observation = _deep_observation(
                options, mutation, started, run.search_id, self._transport._options.timeout_seconds
            )
            if run._expected_retention == "zdr_strict":
                with run.events(observation=observation) as events:
                    for event in events:
                        if isinstance(event, SearchEvent) and event.result is not None:
                            result = event.result
                            break
                    else:
                        raise OutcomeUncertainError(
                            f"Strict Search {run.search_id} ended without a terminal result",
                            code="strict_terminal_missing",
                        )
            else:
                try:
                    result = run.wait(observation=observation, raise_on_failure=False)
                except RequestTimeoutError as error:
                    raise RequestTimeoutError(
                        f"Search {run.search_id} observation timed out; current state is unknown",
                        code=error.code,
                        request_id=error.request_id,
                        operation=error.operation,
                        identity=error.identity,
                        reason=error.reason,
                        retryable=error.retryable,
                        cause=f"search_id={run.search_id}",
                    ) from error
            return validate_search_response(
                result,
                options,
                True,
                local_development_retention=self._transport.local_development_retention,
            )
        result = self._transport.request(
            SEARCH if self._keyed else PUBLIC_SEARCH,
            SearchResponse,
            body=body,
            options=mutation,
            validate=lambda response: validate_search_response(
                response,
                options,
                self._keyed,
                local_development_retention=self._transport.local_development_retention,
            ),
        )
        return result


class AsyncIndexClient:
    contents: AsyncContents
    searches: AsyncSearches
    contributions: AsyncContributions
    account: AsyncAccount
    collections: AsyncCollections
    usage: AsyncUsage
    retention: AsyncRetention

    def __init__(self, transport: AsyncHttpTransport, keyed: bool) -> None:
        self._transport = transport
        self._keyed = keyed
        self.contents = AsyncContents(transport, keyed)
        self.searches = AsyncSearches(transport)
        self.contributions = AsyncContributions(transport, keyed)
        self.account = AsyncAccount(transport)
        self.collections = AsyncCollections(transport)
        self.usage = AsyncUsage(transport)
        self.retention = AsyncRetention(transport)

    async def search(
        self, query: str, *, options: SearchOptions, mutation: MutationOptions | None = None
    ) -> SearchResponse:
        """Return a terminal answer under explicit public/keyed authority and charge bounds.

        See: the public contract snapshots in contracts/"""
        body = search_request(query, options, self._keyed)
        if _durable_deep(self._keyed, options, mutation):
            assert isinstance(options, KeyedSearchOptions)
            started = time.monotonic()
            run = await self.searches.create(query, options=options, mutation=mutation)
            observation = _deep_observation(
                options, mutation, started, run.search_id, self._transport._options.timeout_seconds
            )
            if run._expected_retention == "zdr_strict":
                async with run.events(observation=observation) as events:
                    async for event in events:
                        if isinstance(event, SearchEvent) and event.result is not None:
                            result = event.result
                            break
                    else:
                        raise OutcomeUncertainError(
                            f"Strict Search {run.search_id} ended without a terminal result",
                            code="strict_terminal_missing",
                        )
            else:
                try:
                    result = await run.wait(observation=observation, raise_on_failure=False)
                except RequestTimeoutError as error:
                    raise RequestTimeoutError(
                        f"Search {run.search_id} observation timed out; current state is unknown",
                        code=error.code,
                        request_id=error.request_id,
                        operation=error.operation,
                        identity=error.identity,
                        reason=error.reason,
                        retryable=error.retryable,
                        cause=f"search_id={run.search_id}",
                    ) from error
            return validate_search_response(
                result,
                options,
                True,
                local_development_retention=self._transport.local_development_retention,
            )
        result = await self._transport.request(
            SEARCH if self._keyed else PUBLIC_SEARCH,
            SearchResponse,
            body=body,
            options=mutation,
            validate=lambda response: validate_search_response(
                response,
                options,
                self._keyed,
                local_development_retention=self._transport.local_development_retention,
            ),
        )
        return result
