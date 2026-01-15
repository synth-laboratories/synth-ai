# TUI

This package contains the synth-ai TUI (Terminal User Interface).

## Structure

```
synth_ai/tui/
├── __init__.py      # exports run_tui
├── launcher.py      # Python launcher (spawns bun)
├── README.md
└── app/             # OpenTUI JS app
    ├── src/         # TypeScript source (runs directly via bun)
    ├── package.json
    └── tests/
```

## Setup

```bash
cd synth_ai/tui/app
bun install
bun run typecheck
```

## Entry point

```python
from synth_ai.tui import run_tui
run_tui()
```

## Keyboard shortcuts

- `Ctrl+C` force quit
- `Esc` back
- `q` quit
- `r` refresh
- `Tab` / `Shift+Tab` focus cycle
- `1` jobs, `2` agent, `3` logs
- `Enter` view/select, `↑/↓` or `k/j` navigate
- `/` event filter, `f` job filter
- `n` new job, `c` cancel job, `d` artifacts
- `g` snapshot, `i` config, `m` metrics, `t` traces, `v` candidates
- `u` usage, `a` task apps, `p` profile
