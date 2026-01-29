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
brew tap synth-laboratories/tap
brew install synth-ai-tui
```

Or in one command:

```bash
brew install synth-laboratories/tap/synth-ai-tui
```

Then run:

```bash
synth-ai-tui
```

The first run will install JavaScript dependencies (~5 seconds).

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
