# Synth AI

<!-- CI release pins: PyPI-0.10.0-orange synth-ai==0.10.0 -->

[![image](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![image](https://img.shields.io/pypi/l/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![image](https://img.shields.io/pypi/pyversions/synth-ai.svg)](https://pypi.org/project/synth-ai/)

Python-only SDK and CLI for Synth's containers platform.

The stable surface is intentionally narrow:

- `synth_ai.sdk.containers`
- `synth_ai.sdk.tunnels`
- `synth_ai.sdk.pools`
- `synth_ai.client.SynthClient`

Legacy optimization, inference, graphs, verifiers, managed-research, and Rust-backed
modules have been archived under `../research/old/synth_ai` and are no longer part of the
supported import surface.

## Stable API

```python
from synth_ai import SynthClient

client = SynthClient(api_key="sk_...")
client.containers.list()
client.tunnels.list()
client.pools.list()
```

Canonical backend paths:

- `/v1/containers/*`
- `/v1/tunnels/*`
- `/v1/pools/*`
- `/v1/rollouts/*`

## Local development

```bash
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
# ty: use the same command as `.github/workflows/ci.yml` → job `type-check` (Lefthook runs it on staged files).
uv run ty check
```

Use **`uv run`** for Python tools (not bare **`python`** / **`python3`**). **Ruff** handles both formatting and linting for `synth_ai/`; **ty** type-checks `synth_ai/` (`[tool.ty.src]` in `pyproject.toml`). A plain `ty check` may be stricter than CI; match CI when debugging PR failures.

Optional: install [Lefthook](https://github.com/evilmartians/lefthook) and run **`lefthook install`** to run **`uv run ruff format`**, **`uv run ruff check`**, and **`uv run ty check`** on staged `.py` files (see `lefthook.yml`).
