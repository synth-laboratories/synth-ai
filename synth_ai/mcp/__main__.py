"""Packaged stdio MCP, with credentials only in environment/configuration."""

from __future__ import annotations

import argparse
import asyncio

from synth_ai import AsyncSynthClient
from synth_ai.client import resolve_url
from synth_ai.mcp_entrypoint import require_mcp
from synth_ai.options import ClientOptions

from .catalog import TOOLS
from .runtime import SDKExecutor


async def serve(
    profile: str,
    base_url: str,
    candidate_manifest_sha256: str | None = None,
    *,
    local_development_retention: bool = False,
) -> None:
    """Own both clients for this one-principal process and close them on exit."""
    from mcp.server.stdio import stdio_server

    from .server import create_server

    if profile == "public" and local_development_retention:
        raise ValueError("Local development retention requires a keyed MCP profile")
    selected = frozenset(
        tool.name
        for tool in TOOLS
        if (
            not tool.keyed
            if profile == "public"
            else not tool.mutates and not tool.paid
            if profile == "read"
            else True
        )
    )
    public_options = ClientOptions(
        timeout_seconds=45, read_retry_count=0, candidate_manifest_sha256=candidate_manifest_sha256
    )
    # Anonymous tools keep their public retention contract in keyed profiles too.
    options = ClientOptions(
        timeout_seconds=45,
        read_retry_count=0,
        candidate_manifest_sha256=candidate_manifest_sha256,
        local_development_retention=local_development_retention,
    )
    async with AsyncSynthClient.anonymous(
        base_url=base_url, options=public_options
    ) as public_client:
        client = (
            public_client
            if profile == "public"
            else AsyncSynthClient(base_url=base_url, options=options)
        )
        try:
            server = create_server(SDKExecutor(client, public_client, selected))
            async with stdio_server() as (reader, writer):
                await server.run(reader, writer, server.create_initialization_options())
        finally:
            if client is not public_client:
                await client.aclose()


def main() -> None:
    require_mcp()
    parser = argparse.ArgumentParser(description="Synth Index and Swarms MCP")
    parser.add_argument("--profile", choices=("public", "read", "action"), default="read")
    parser.add_argument("--base-url", default=None, help="API origin (default: SYNTH_API_URL)")
    parser.add_argument("--candidate-manifest-sha256")
    parser.add_argument(
        "--local-development-retention",
        action="store_true",
        help="Accept retained debugging results from a loopback development API",
    )
    args = parser.parse_args()
    asyncio.run(
        serve(
            args.profile,
            resolve_url(args.base_url),
            args.candidate_manifest_sha256,
            local_development_retention=args.local_development_retention,
        )
    )


if __name__ == "__main__":
    main()
