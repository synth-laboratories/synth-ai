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

- `Ctrl+C` force quit
- `Esc` back
- `q` quit
- `r` refresh
- `Tab` / `Shift+Tab` focus cycle (list/metrics/events)
- `1` jobs, `2` logs, `3` agent
- `Enter` view/select, `↑/↓` or `k/j` navigate
- `/` event filter, `f` job filter
- `n` new job, `c` cancel job, `d` artifacts
- `g` snapshot, `i` config, `m` metrics, `t` traces, `v` candidates
- `o` URLs, `s` settings, `u` usage, `a` task apps, `p` profile
