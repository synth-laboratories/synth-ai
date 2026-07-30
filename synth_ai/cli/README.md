# CLI Modules

Thin Click adapters over `SynthClient` / `sdk.research`.

`synth-ai research` is the whole CLI. Infrastructure commands (`containers`,
`tunnels`, `pools`) and the operator consoles (`dev-envs`, the
`synth-ai-research-factory-standup` script) were removed in 0.18.0; see
`CHANGELOG.md` for what replaced each one.

## Command Structure

```
synth-ai research --help
synth-ai research projects ...
synth-ai research swarms ...
```

## Import Rules

```python
# CLI importing from SDK (correct)
from synth_ai import SynthClient

# CLI importing from core (correct)
from synth_ai.core.utils.env import get_api_key

# SDK importing from CLI (wrong - never do this)
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| `core/` | `cli/` imports shared env/error helpers |
| `sdk/` / `client.py` | Business calls go through Research clients |
