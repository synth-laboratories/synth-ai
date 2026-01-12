---
name: synth-api
description: Use the Synth AI API (via Python SDK or HTTP) from inside OpenCode / the Synth-AI TUI
---

# Synth API (Python SDK + HTTP)

This skill helps you interact with Synth AI programmatically.

## Required env

- `SYNTH_API_KEY`: your API key
- `SYNTH_BACKEND_URL` (optional): backend base URL (defaults to the SDK default)

## Python SDK example

Use the official SDK clients when possible (typed, handles auth, etc.).

```python
import os
import asyncio

from synth_ai.sdk.jobs import JobsClient


async def main() -> None:
    async with JobsClient(
        base_url=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        api_key=os.environ["SYNTH_API_KEY"],
    ) as client:
        files = await client.files.list(limit=5)
        print(files)


if __name__ == "__main__":
    asyncio.run(main())
```

## HTTP example

If you need raw HTTP, keep it explicit and log-safe.

```python
import os
import requests

base = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
resp = requests.get(
    f"{base}/api/health",
    headers={"Authorization": f"Bearer {os.environ['SYNTH_API_KEY']}"},
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

