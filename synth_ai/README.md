# `synth_ai` Package

Runtime package for the public Synth AI SDK and CLI.

The public first-mile surface is intentionally small:

- `SynthClient`
- `AsyncSynthClient`
- `client.research`
- `synth-ai` CLI

Infrastructure clients (containers, tunnels, pools) are archived under `old/`
for later restoration.

Public docs live at https://docs.usesynth.ai/sdk/overview.

## Package Structure

```text
synth_ai/
├── client.py       # SynthClient and AsyncSynthClient composition layer
├── sdk/            # Research + shared pagination plumbing
├── core/           # Shared runtime helpers and errors
├── cli/            # CLI commands for research (and local helpers)
└── __init__.py     # Package version and top-level exports
```

## Dependency Direction

```text
core/ -> sdk/ -> client.py -> cli/
```

- `core/` owns shared runtime plumbing such as errors, environment lookup, and URL normalization.
- `sdk/` owns Research clients/contracts plus shared pagination helpers.
- `client.py` composes Research behind `SynthClient` and `AsyncSynthClient`.
- `cli/` wraps the SDK for terminal use.

## Supported Imports

Prefer the front-door client:

```python
from synth_ai import SynthClient

research = SynthClient().research
catalog = research.advanced.economics.billing_catalog()
plan = research.advanced.economics.billing_plan()
```

## Layering

One DAG, enforced by `check_sdk_layering.py` in the sibling `testing` repo.
See `unify_sdk_layering.md` for the rationale.

```text
core/      plumbing only: auth, http, errors, utils, generic contracts
  |
  v
sdk/       Research + pagination (infra clients archived under old/)
  |
  v
client.py  SynthClient / AsyncSynthClient composition
  |
  +--> cli/            thin terminal adapter
  +--> mcp/research/   thin MCP adapter
```

| From \ To | `core` | `sdk` | `client` | `cli` | `mcp` |
|-----------|--------|-------|----------|-------|-------|
| `core` | yes | **no** | no | no | no |
| `sdk` | yes | yes | no | no | no |
| `client` | yes | yes | — | no | no |
| `cli` | yes | yes | yes | — | no |
| `mcp` | yes | yes | preferred yes | no | — |

`core/` must not import `sdk/`: plumbing cannot depend on the clients built on
top of it. Peer imports inside a layer are fine.

## Guidelines for New Code

1. Put shared errors, URL handling, and environment helpers in `core/`.
2. Put Research contracts, clients, and operator session surfaces in `sdk/research/`.
3. Put front-door composition in `client.py`.
4. Put terminal commands in `cli/`.
5. Keep unreleased or archived infra APIs out of public README examples.
