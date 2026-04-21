# Synth AI

<!-- CI release pins: PyPI-0.11.0-orange synth-ai==0.11.0 -->

[![image](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![image](https://img.shields.io/pypi/l/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![image](https://img.shields.io/pypi/pyversions/synth-ai.svg)](https://pypi.org/project/synth-ai/)

Python-only SDK and CLI for Synth's infrastructure surfaces.

For this launch cycle, keep the product split explicit:

- **Managed Research** is the service/product surface.
- **Managed Agents** is **beta infrastructure** and is often used internally as
  verifier or backend machinery behind managed-research workflows.
- **Container pools** and related runtime surfaces are infrastructure that
  Managed Research builds on top of.

The stable surface is intentionally narrow:

- `synth_ai.sdk.containers`
- `synth_ai.sdk.tunnels`
- `synth_ai.sdk.pools`
- `synth_ai.sdk.horizons_private`
- `synth_ai.sdk.managed_agents_anthropic`
- `synth_ai.sdk.openai_agents_sdk`
- `synth_ai.client.SynthClient`

Legacy optimization, inference, graphs, verifiers, managed-research, and deprecated
modules have been archived under `../research/old/synth_ai` and are no longer part of the
supported import surface.

Launch-era verifier authoring assets for managed-research-facing workflows live
under `managed_agents/templates/` and `managed_agents/examples/`.

## Stable API

```python
from synth_ai import SynthClient

client = SynthClient(api_key="sk_...")
client.containers.list()
client.tunnels.list()
client.pools.list()
client.horizons_private.create_runtime({"name": "hp-runtime"})
client.managed_agents.health()
client.openai_agents_sdk.create_response(
    {
        "model": "gpt-4.1-mini",
        "input": [{"role": "user", "content": "Summarize this deployment note."}],
    }
)
```

Canonical backend paths:

- `/v1/containers/*`
- `/v1/tunnels/*`
- `/v1/pools/*`
- `/v1/rollouts/*`
- `/api/managed-agents/anthropic/v1/*`
- `/api/managed-agents/openai/v1/*` (backend BFF lane)
- `/openai/v1/*` (direct horizons-private lane)

OpenAI transport mode defaults to `auto`:

- try `/api/managed-agents/openai/v1/*` first
- fallback to `/openai/v1/*` only on `404`, `405`, or `501`
- preserve non-contract failures (auth/validation/runtime) without fallback

Transport mode can be selected explicitly:

```python
from synth_ai import SynthClient

client = SynthClient(
    api_key="sk_...",
    base_url="http://127.0.0.1:8000",
    openai_transport_mode="backend_bff",  # backend_bff | direct_hp | auto
)
```

## Product surfaces

Use this split to pick the right client quickly:

| Surface | Client | Backend path family | Use when |
| --- | --- | --- | --- |
| Container Pools | `client.pools` and `client.horizons_private` | `/v1/pools/*`, `/v1/rollouts/*` | You need repeatable pool/task/rollout execution and rollout artifacts/usage. |
| Managed Agents (Anthropic view, beta infra) | `client.managed_agents` | `/api/managed-agents/anthropic/v1/*` | You need live managed-agents session APIs as infrastructure, typically behind managed-research or verifier workflows. |
| OpenAI Agents SDK compat (phase1-core + phase1-adjacent) | `client.openai_agents_sdk` | `/api/managed-agents/openai/v1/*` with auto fallback to `/openai/v1/*` | You need OpenAI-compatible Responses/Conversations semantics without changing existing Synth auth/base-url posture. |

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
