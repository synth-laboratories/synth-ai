"""Small typed Index facade. See backend/notes/specifications/sdk/index_contract.md."""

from __future__ import annotations

from synth_ai.contracts.operations import PUBLIC_SEARCH, SEARCH
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.options import MutationOptions

from .account import Account, AsyncAccount
from .collections import AsyncCollections, Collections
from .contents import AsyncContents, Contents
from .contributions import AsyncContributions, Contributions
from .search import SearchOptions, search_request, validate_search_response
from .searches import AsyncSearches, Searches
from .types.search import SearchResponse
from .usage import AsyncUsage, Usage


class IndexClient:
    contents: Contents
    searches: Searches
    contributions: Contributions
    account: Account
    collections: Collections
    usage: Usage

    def __init__(self, transport: HttpTransport, keyed: bool) -> None:
        self._transport = transport
        self._keyed = keyed
        self.contents = Contents(transport, keyed)
        self.searches = Searches(transport)
        self.contributions = Contributions(transport, keyed)
        self.account = Account(transport)
        self.collections = Collections(transport)
        self.usage = Usage(transport)

    def search(
        self, query: str, *, options: SearchOptions, mutation: MutationOptions | None = None
    ) -> SearchResponse:
        """Return a terminal answer under explicit public/keyed authority and charge bounds.

        See: backend/notes/specifications/sdk/index_contract.md."""
        body = search_request(query, options, self._keyed)
        result = self._transport.request(
            SEARCH if self._keyed else PUBLIC_SEARCH,
            SearchResponse,
            body=body,
            options=mutation,
            validate=lambda response: validate_search_response(response, options, self._keyed),
        )
        return result


class AsyncIndexClient:
    contents: AsyncContents
    searches: AsyncSearches
    contributions: AsyncContributions
    account: AsyncAccount
    collections: AsyncCollections
    usage: AsyncUsage

    def __init__(self, transport: AsyncHttpTransport, keyed: bool) -> None:
        self._transport = transport
        self._keyed = keyed
        self.contents = AsyncContents(transport, keyed)
        self.searches = AsyncSearches(transport)
        self.contributions = AsyncContributions(transport, keyed)
        self.account = AsyncAccount(transport)
        self.collections = AsyncCollections(transport)
        self.usage = AsyncUsage(transport)

    async def search(
        self, query: str, *, options: SearchOptions, mutation: MutationOptions | None = None
    ) -> SearchResponse:
        """Return a terminal answer under explicit public/keyed authority and charge bounds.

        See: backend/notes/specifications/sdk/index_contract.md."""
        body = search_request(query, options, self._keyed)
        result = await self._transport.request(
            SEARCH if self._keyed else PUBLIC_SEARCH,
            SearchResponse,
            body=body,
            options=mutation,
            validate=lambda response: validate_search_response(response, options, self._keyed),
        )
        return result
