"""Terminal search calls; never hide durable work. See SDK index_contract.md."""

from __future__ import annotations

from synth_ai.errors import AuthenticationError, ContractError

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
    result: SearchResponse, options: SearchOptions, keyed: bool
) -> SearchResponse:
    if result.retention is not None:
        expected_retention = options.retention if isinstance(options, KeyedSearchOptions) else "zdr"
        if result.retention.mode != expected_retention:
            raise ContractError("Search changed the selected retention mode")
    elif isinstance(options, KeyedSearchOptions) and options.retention == "retained":
        raise ContractError("Retained search requires an explicit backend retention receipt")
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
    result: SearchSnapshot, options: KeyedSearchOptions
) -> SearchSnapshot:
    """Preserve selected intent; retained history needs a native policy receipt.

    See: backend/notes/specifications/sdk/index_contract.md."""
    if result.mode != options.mode:
        raise ContractError("Search admission changed the requested mode")
    if result.retention is not None and result.retention.mode != options.retention:
        raise ContractError("Search admission changed the selected retention mode")
    if options.retention == "retained" and result.retention is None:
        raise ContractError("Retained search requires an explicit backend retention receipt")
    return result
