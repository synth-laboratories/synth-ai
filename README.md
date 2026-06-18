# Synth AI SDK

<!-- CI release pins: PyPI-0.11.3-orange synth-ai==0.11.3 -->

[![PyPI version](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![License](https://img.shields.io/pypi/l/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/synth-ai.svg)](https://pypi.org/project/synth-ai/)

Python SDK and CLI for Synth Managed Research, Research Factory, GEPA/GELO
optimizer workflows, and the lower-level infrastructure surfaces those products
use.

**Documentation:** https://docs.usesynth.ai/sdk/overview

## Installation

```bash
uv add "synth-ai[research]"
```

Or install with pip:

```bash
pip install "synth-ai[research]"
```

## Authenticate

Set `SYNTH_API_KEY` before using the SDK or CLI:

```bash
export SYNTH_API_KEY="sk_..."
```

## Local Workspaces

For local multi-repo development, `synth-ai` treats workspace resolution as a
read-only overlay. `.env` and Synth home config may provide defaults and
secrets; selecting a worktree must not rewrite those defaults.

Use `SYNTH_WORKSPACE_MANIFEST` or `SYNTH_WORKSPACE_ROOT` for command-scoped
worktree resolution. The resolver in `synth_ai.core.utils.workspace` returns
repo paths and a scoped env mapping for subprocesses without mutating `.env`.

Pass `base_url` when you need to pin a production, local, staging, or private
backend explicitly:

```python
from synth_ai import SynthClient

client = SynthClient(base_url="http://127.0.0.1:8000")
```

The CLI also reads `SYNTH_BACKEND_URL` and accepts `--backend-url`.

## Quickstart

```python
from synth_ai import SynthClient

client = SynthClient()
research = client.research

run = research.runs.start(
    "Inspect this repository and leave a reviewable report.",
    work_mode="directed_effort",
    providers=[{"provider": "openrouter"}],
    runbook="lite",
)

print(run.run_id)
```

## CLI

```bash
synth-ai --help
```

## Public Surface

Use `SynthClient` as the front door:

| Surface | Client namespace | Use it for |
| --- | --- | --- |
| Managed Research | `client.research` | Hosted research workers, repo runs, evidence, checkpoints, MCP, usage, and reports. |
| Research Factory | `client.research` | Programmatic multi-run research campaigns on the same control plane. |
| Hosted Optimizers | `synth-optimizers` + `synth-ai` auth | GEPA and GELO hosted optimizer workflows. |
| Containers | `client.containers` | Lower-level hosted container records and lifecycle operations. |
| Tunnels | `client.tunnels` | Lower-level managed tunnel records, leases, health, and rotation. |
| Pools | `client.pools` | Lower-level container pools, tasks, rollouts, artifacts, usage, and events. |

For agent clients, connect to the hosted Managed Research MCP server:

```bash
codex mcp add synth-managed-research --url https://api.usesynth.ai/mcp
```

## Links

- [Install and authenticate](https://docs.usesynth.ai/sdk/install-and-auth)
- [Managed Research](https://docs.usesynth.ai/managed-research/intro)
- [Managed Research MCP quickstart](https://docs.usesynth.ai/managed-research/mcp-quickstart)
- [Research Factory](https://docs.usesynth.ai/managed-research/research-factory)
- [Hosted Optimizers](https://docs.usesynth.ai/sdk/hosted-optimizers)
- [SynthClient guide](https://docs.usesynth.ai/sdk/synth-client)
- [Tunnels](https://docs.usesynth.ai/sdk/tunnels)
- [Pools](https://docs.usesynth.ai/sdk/pools)
- [Containers](https://docs.usesynth.ai/sdk/containers)
- [SDK reference](https://docs.usesynth.ai/reference/sdk)
- [OpenAPI contracts](https://docs.usesynth.ai/reference/openapi)

## Local Development

Use `uv run` for Python tools:

```bash
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

Optional: install [Lefthook](https://github.com/evilmartians/lefthook) and run
`lefthook install` to run formatting, linting, and type checks on staged Python
files.
