"""Terminal search calls; never hide durable work. See SDK index_contract.md."""

from __future__ import annotations

from typing import TypeVar

from synth_ai.errors import AuthenticationError, ContractError
from synth_ai.types.base import Page

from .types.retention import RetentionMode
from .types.search import (
    KeyedSearchOptions,
    KeyedSearchRequest,
    PublicSearchOptions,
    PublicSearchRequest,
    SearchResponse,
    SearchSnapshot,
)

SearchOptions = PublicSearchOptions | KeyedSearchOptions
SearchRequest = PublicSearchRequest | KeyedSearchRequest
SearchObservation = TypeVar("SearchObservation", bound=SearchResponse | SearchSnapshot)


def validate_search_retention(
    result: SearchObservation,
    expected: RetentionMode | None = None,
    *,
    local_development_retention: bool = False,
) -> SearchObservation:
    """Require native policy acknowledgment for every customer search observation.

    See: backend/notes/specifications/sdk/customer_api_adoption.md. Default ZDR
    is an enforceable contract, including reattachment and MCP delivery.
    """
    if result.retention is None:
        raise ContractError("Search requires an explicit backend retention receipt")
    local_override = (
        local_development_retention and expected == "zdr" and result.retention.mode == "retained"
    )
    if (
        expected is not None
        and result.retention.mode is not None
        and result.retention.mode != expected
        and not local_override
    ):
        raise ContractError("Search changed the selected retention mode")
    return result


def validate_search_page(result: Page[SearchSnapshot], limit: int) -> Page[SearchSnapshot]:
    if len(result.items) > limit:
        raise ContractError("Search page exceeded its requested limit")
    for item in result.items:
        validate_search_retention(item)
    return result


def search_request(query: str, options: SearchOptions, keyed: bool) -> SearchRequest:
    if keyed:
        if not isinstance(options, KeyedSearchOptions):
            raise ValueError(
                "Keyed search requires KeyedSearchOptions and an explicit charge ceiling"
            )
        return KeyedSearchRequest(query=query, options=options)
    if isinstance(options, KeyedSearchOptions):
        raise AuthenticationError(
            "Anonymous search cannot select private collections or customer funding"
        )
    return PublicSearchRequest(query=query, options=options)


def validate_search_response(
    result: SearchResponse,
    options: SearchOptions,
    keyed: bool,
    *,
    local_development_retention: bool = False,
) -> SearchResponse:
    validate_search_retention(
        result,
        options.retention if isinstance(options, KeyedSearchOptions) else "zdr",
        local_development_retention=local_development_retention and keyed,
    )
    if result.mode != options.mode or len(result.citations) > options.max_results:
        raise ContractError("Search result violates requested mode or citation bounds")
    if keyed and result.search_id is None:
        raise ContractError("Keyed search requires a recoverable search identity")
    if not keyed and result.search_id is not None:
        raise ContractError("Anonymous search must not expose a durable search handle")
    if (
        not keyed
        and result.usage is not None
        and (result.usage.customer_charge.amount is not None)
        and (result.usage.customer_charge.amount.coefficient != 0)
    ):
        raise ContractError("Anonymous search cannot charge a customer")
    if keyed and result.usage is not None and isinstance(options, KeyedSearchOptions):
        charge = result.usage.customer_charge.amount
        ceiling = options.maximum_charge
        if (
            charge is not None
            and charge.coefficient * 10**ceiling.scale > ceiling.coefficient * 10**charge.scale
        ):
            raise ContractError("Backend customer charge exceeds the selected ceiling")
    return result


def validate_search_admission(
    result: SearchSnapshot,
    options: KeyedSearchOptions,
    *,
    local_development_retention: bool = False,
) -> SearchSnapshot:
    """Preserve selected intent; every search needs a native policy receipt.

    See: backend/notes/specifications/sdk/index_contract.md."""
    if result.mode != options.mode:
        raise ContractError("Search admission changed the requested mode")
    validate_search_retention(
        result, options.retention, local_development_retention=local_development_retention
    )
    return result
