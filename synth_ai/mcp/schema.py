"""Contract export compatibility entrypoint; independent of MCP protocol IO."""

from synth_ai.contracts.mcp_schema import export_catalog, main

__all__ = ["export_catalog"]

if __name__ == "__main__":
    main()
