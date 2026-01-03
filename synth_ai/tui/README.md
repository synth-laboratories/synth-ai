# TUI

This package holds the CLI entrypoint that launches the JS OpenTUI app.

OpenTUI app layout:
- `synth_ai/_tui/src/index.ts` (source)
- `synth_ai/_tui/dist/index.mjs` (built bundle used at runtime)

Build the JS bundle:
1) `cd synth_ai/_tui`
2) `bun install`
3) `bun run build`

Entry points:
- `synth_ai.tui.prompt_learning.run_prompt_learning_tui`

Keys:
- `r` refresh
- `c` cancel
- `a` artifacts
- `s` snapshot (opens modal)
- `q` quit
