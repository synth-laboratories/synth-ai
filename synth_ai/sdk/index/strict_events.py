"""Typed request-owned strict DEEP stream, with no stored-result fallback."""

import json

from pydantic import ValidationError

from synth_ai.contracts.operations import DELIVER_SEARCH_EVENTS
from synth_ai.errors import ContractError, OutcomeUncertainError
from synth_ai.types.base import EventCursor, Page

from .types.search import SearchEvent, StrictSearchProgress


def _decode(frame, search_id, previous, identity):
    try:
        if frame.event == "progress":
            value = StrictSearchProgress.model_validate_json(frame.data)
        elif frame.event == "terminal":
            page = Page[SearchEvent, EventCursor].model_validate_json(frame.data)
            if not page.complete or page.next_cursor is not None or len(page.items) != 1:
                raise ValueError("terminal page shape")
            value = page.items[0]
            if value.result is None or value.result.metadata != page.metadata:
                raise ValueError("terminal result binding")
        else:
            error = json.loads(frame.data)
            if (
                not isinstance(error, dict)
                or set(error) != {"code", "outcome"}
                or error["outcome"] != "unknown"
                or not isinstance(error["code"], str)
            ):
                raise ValueError("error frame shape")
            raise OutcomeUncertainError(
                "Strict event delivery failed; only metadata is recoverable",
                code=error["code"],
                identity=identity,
            )
        if value.search_id != search_id or value.sequence <= previous:
            raise ValueError("event resource or sequence")
        return value
    except (ValidationError, ValueError, TypeError, KeyError) as error:
        raise ContractError(
            "Strict event response identity or bounds mismatch", identity=identity
        ) from error


class StrictSearchEvents:
    def __init__(self, owner, search_id, body, options):
        self.search_id = search_id
        self.stream = owner.mutation_events(
            DELIVER_SEARCH_EVENTS,
            path=f"/index/searches/{search_id}/events",
            body=body,
            options=options,
        )
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.stream.close()

    def __iter__(self):
        sequence = 0
        try:
            for frame in self.stream:
                value = _decode(frame, self.search_id, sequence, self.stream.identity)
                sequence = value.sequence
                if isinstance(value, SearchEvent):
                    self.result = value.result
                yield value
        finally:
            self.close()


class AsyncStrictSearchEvents:
    def __init__(self, owner, search_id, body, options):
        self.search_id = search_id
        self.stream = owner.mutation_events(
            DELIVER_SEARCH_EVENTS,
            path=f"/index/searches/{search_id}/events",
            body=body,
            options=options,
        )
        self.result = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        await self.stream.aclose()

    async def __aiter__(self):
        sequence = 0
        try:
            async for frame in self.stream:
                value = _decode(frame, self.search_id, sequence, self.stream.identity)
                sequence = value.sequence
                if isinstance(value, SearchEvent):
                    self.result = value.result
                yield value
        finally:
            await self.aclose()
