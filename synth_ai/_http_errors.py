"""Bounded decoding of refused streaming HTTP responses."""

import httpx

from ._limits import ERROR_RESPONSE_MAX_BYTES
from .errors import ResponseContractError, local_problem, raise_for_response


async def raise_stream_error(response: httpx.Response) -> None:
    """Bound refused streaming responses before decoding their problem document."""

    def invalid() -> ResponseContractError:
        return ResponseContractError(
            local_problem(
                "error_response_bounds",
                "decode",
                status=response.status_code if 400 <= response.status_code <= 599 else 502,
                upstream_status=response.status_code,
            ),
            request_id=response.headers.get("x-request-id"),
            idempotency_key=response.request.headers.get("idempotency-key"),
        )

    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise invalid()
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(chunk) > ERROR_RESPONSE_MAX_BYTES - len(body):
            raise invalid()
        body.extend(chunk)
    raise_for_response(
        httpx.Response(
            response.status_code,
            headers=response.headers,
            content=bytes(body),
            request=response.request,
        )
    )
