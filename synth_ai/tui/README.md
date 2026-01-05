# TUI

This package contains the synth-ai TUI (Terminal User Interface).

## Structure

```
synth_ai/tui/
├── __init__.py      # exports run_prompt_learning_tui
├── launcher.py      # Python launcher (spawns bun)
├── README.md
└── app/             # OpenTUI JS app
    ├── src/         # TypeScript source
    ├── dist/        # Built bundle (index.mjs)
    ├── package.json
    └── tests/
```

## Build the JS bundle

```bash
cd synth_ai/tui/app
bun install
bun run build
```

## Entry point

```python
from synth_ai.tui import run_prompt_learning_tui
run_prompt_learning_tui()
```

## Keyboard shortcuts

- `r` refresh
- `c` cancel
- `a` artifacts
- `s` snapshot (opens modal)
- `q` quit
