"""Optional unified MCP adapter. Install synth-ai[mcp]. See SDK_0230_MCP.md."""

from synth_ai.mcp_entrypoint import mcp_available

__all__ = ["AuthorizationBackend", "VerifiedPrincipal", "create_http_app", "create_server"]

if mcp_available():
    from .hosted import AuthorizationBackend, VerifiedPrincipal, create_http_app
    from .server import create_server
