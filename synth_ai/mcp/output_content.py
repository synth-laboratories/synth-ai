"""Exact-byte output envelope shared with the hosted Core MCP transport."""

from base64 import b64encode
from collections.abc import Awaitable

from .._generated.models import MCPOutputContent
from ..errors import DataIntegrityError, ResponseContractError, local_problem

# Both text and structured representations must fit the hosted reply bound.
OUTPUT_BYTES_MAX = 1 << 20


async def read_output_content(
    content: Awaitable[bytes], *, revision: str, digest: str
) -> MCPOutputContent:
    try:
        body = await content
    except DataIntegrityError as error:
        raise ResponseContractError(
            local_problem("mcp_output_content_invalid", "mcp_output_content", status=502)
        ) from error
    if len(body) > OUTPUT_BYTES_MAX:
        raise ResponseContractError(
            local_problem("mcp_response_too_large", "mcp_output_content", status=413)
        )
    return MCPOutputContent(
        encoding="base64",
        data=b64encode(body).decode("ascii"),
        content_type="application/octet-stream",
        revision_id=revision,
        digest_sha256=digest,
    )
