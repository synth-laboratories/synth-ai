"""Module entry point for `python -m synth_ai.mcp`."""

import asyncio

from synth_ai.mcp.main import main

if __name__ == "__main__":
    asyncio.run(main())
