"""Optional MCP command without importing unavailable extras during startup."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def mcp_available() -> bool:
    try:
        version("mcp")
    except PackageNotFoundError:
        return False
    return True


def require_mcp() -> None:
    if not mcp_available():
        raise SystemExit("The MCP command requires the optional extra: install 'synth-ai[mcp]'.")


def main() -> None:
    require_mcp()
    from synth_ai.mcp.__main__ import main as run

    run()
