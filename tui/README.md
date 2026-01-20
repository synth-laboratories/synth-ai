# TUI

This package contains the synth-ai TUI (Terminal User Interface).

## Structure

```
tui/
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
cd tui/app
bun install
bun run build
```

## Install via Homebrew

```bash
brew install --formula https://github.com/synth-laboratories/synth-ai/raw/main/Formula/synth-ai-tui.rb
```

Then run:

```bash
synth-ai-tui
```

## Entry point

```python
from tui import run_prompt_learning_tui
run_prompt_learning_tui()
```

## Keyboard shortcuts

- `r` refresh
- `c` cancel
- `a` artifacts
- `s` snapshot (opens modal)
- `q` quit
