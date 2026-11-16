"""Bounded acceptance reads and transfers. See SYN-5692/SYN-5693."""

from __future__ import annotations

__all__ = [
    "COMMAND_SECONDS",
    "MCP_CALL_SECONDS",
    "MCP_UPLOAD_BYTES_MAX",
    "MCP_WAIT_CALLS_MAX",
    "MCP_WAIT_CALL_SECONDS",
    "OUTPUT_BYTES_MAX",
    "OUTPUT_PAGES_MAX",
    "PAGE_ITEMS",
    "WAIT_SECONDS",
]
COMMAND_SECONDS = 120
WAIT_SECONDS = 540
PAGE_ITEMS = 100
OUTPUT_PAGES_MAX = 50
OUTPUT_BYTES_MAX = 64 << 20
# swarms_run_wait timeout_seconds maximum in contracts/mcp-tools.json.
MCP_WAIT_CALL_SECONDS = 50
# Ceiling of WAIT_SECONDS / MCP_WAIT_CALL_SECONDS: the same total wait as SDK/CLI.
MCP_WAIT_CALLS_MAX = -(-WAIT_SECONDS // MCP_WAIT_CALL_SECONDS)
# swarms_upload_file body.content maxLength in contracts/mcp-tools.json.
MCP_UPLOAD_BYTES_MAX = 32768
# One tools/call round trip, including a full server wait.
MCP_CALL_SECONDS = MCP_WAIT_CALL_SECONDS + COMMAND_SECONDS
