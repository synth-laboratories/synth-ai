# CLI Modules

Thin Click adapters over `SynthClient`, `sdk.index` and `sdk.research`.

The CLI has two command families: `synth-ai index` (Synth Index) and
`synth-ai research` (Swarms and Research projects).

## Command Structure

```
synth-ai index --help
synth-ai index search QUERY --public         # free, anonymous; sends no API key
synth-ai index search QUERY --keyed ...      # paid keyed Search (SYNTH_API_KEY)
synth-ai index searches create|get|result|events|cancel ...   # keyed durable Search
synth-ai index research preview|submit ...   # authorized research intake

synth-ai research --help
synth-ai research projects ...
synth-ai research swarms ...
```

`index search` never picks a route on its own: with no key and no flag it
refuses; `--public` never sends an inherited `SYNTH_API_KEY`; `--keyed` (or a
keyed-only option) without a key is refused, not downgraded to public. An
inherited key with no flag uses the keyed route and prints a notice on stderr.

Version note: infrastructure commands (`containers`, `tunnels`, `pools`) and the
operator consoles (`dev-envs`, the `synth-ai-research-factory-standup` script)
were removed in 0.18.0; see `CHANGELOG.md` for what replaced each one.

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
| `sdk/` / `client.py` | Business calls go through `SynthClient` / `PublicIndexClient` (Index) and Research clients (Swarms) |
