# SDK Modules

Public HTTP clients and contracts. Right now that is Research plus shared
pagination helpers. Infrastructure clients are archived under `old/sdk/` and
will return later.

```
sdk/
├── pagination.py                                          shared list-page helpers
└── research/                                              Research implementation
```

Research callers should use `SynthClient().research`; `synth_ai.sdk.research.*`
is the supported module path when a specific surface is needed.
`synth_ai.core.research.*` still resolves through a deprecated alias.

Prefer the top-level client in user-facing examples:

```python
from synth_ai import SynthClient

research = SynthClient().research
```

## Supported Public Surfaces

- Research: projects, swarms, factories, billing/economics reads

## Ownership Rules

- Keep Research transport details in `sdk/research/`.
- Keep shared error and environment helpers in `core/`.
- Keep front-door composition in `client.py`.
- Keep public examples focused on `SynthClient().research`.
