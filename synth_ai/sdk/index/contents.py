"""Exact currently authorized content. See SDK index_contract.md."""

from __future__ import annotations

from synth_ai.contracts.operations import CONTENTS, PUBLIC_CONTENTS
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.errors import ContractError
from synth_ai.options import ReadOptions

from .types.contents import ContentOptions, ContentRef, ContentRequest, ContentResponse


def validate_content(
    result: ContentResponse, reference: ContentRef, options: ContentOptions
) -> ContentResponse:
    if result.reference != reference:
        raise ContractError("Backend returned a different revision/asset/digest")
    if result.text is not None and len(result.text.encode("utf-8")) > options.maximum_bytes:
        raise ContractError("Backend content exceeded the requested byte ceiling")
    return result


class Contents:
    def __init__(self, transport: HttpTransport, keyed: bool) -> None:
        self._transport = transport
        self._operation = CONTENTS if keyed else PUBLIC_CONTENTS

    def get(
        self, reference: ContentRef, *, options: ContentOptions, request: ReadOptions | None = None
    ) -> ContentResponse:
        """Read or reattach the exact authorized resource with fresh server state.

        See: the public contract snapshots in contracts/"""
        result = self._transport.request(
            self._operation,
            ContentResponse,
            body=ContentRequest(reference=reference, options=options),
            options=request,
        )
        return validate_content(result, reference, options)


class AsyncContents:
    def __init__(self, transport: AsyncHttpTransport, keyed: bool) -> None:
        self._transport = transport
        self._operation = CONTENTS if keyed else PUBLIC_CONTENTS

    async def get(
        self, reference: ContentRef, *, options: ContentOptions, request: ReadOptions | None = None
    ) -> ContentResponse:
        """Read or reattach the exact authorized resource with fresh server state.

        See: the public contract snapshots in contracts/"""
        result = await self._transport.request(
            self._operation,
            ContentResponse,
            body=ContentRequest(reference=reference, options=options),
            options=request,
        )
        return validate_content(result, reference, options)
