"""Packaged stdio MCP, with credentials only in environment/configuration."""

from __future__ import annotations

import argparse
import asyncio
import os

from synth_ai import AsyncSynthClient
from synth_ai.mcp_entrypoint import require_mcp
from synth_ai.options import ClientOptions

from .catalog import TOOLS
from .runtime import SDKExecutor


async def serve(profile: str, base_url: str) -> None:
    """Own both clients for this one-principal process and close them on exit."""
    from mcp.server.stdio import stdio_server

    from .server import create_server

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
    options = ClientOptions(timeout_seconds=45, read_retry_count=0)
    async with AsyncSynthClient.anonymous(base_url=base_url, options=options) as public_client:
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
    parser.add_argument(
        "--base-url", default=os.environ.get("SYNTH_API_BASE_URL", "https://api.usesynth.ai")
    )
    args = parser.parse_args()
    asyncio.run(serve(args.profile, args.base_url))


if __name__ == "__main__":
    main()
