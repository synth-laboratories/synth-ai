CLI structure notes (for coding agents)

- Keep `synth_ai/cli/` flat: one file per command, filename == command name.
- Each command file holds Click wiring (options/args/validation/routing).
- Business logic lives in `synth_ai/sdk/` (reusable) or `synth_ai/core/` (internal).
- CLI-only helpers may live in `synth_ai/cli/lib/` or `synth_ai/cli/utils/` only.
- No shims/wrappers: do not duplicate modules across folders or re-export commands.
- Avoid heavy module-level imports in CLI; import inside functions for lazy load.
- Use Click conventions; no surprising CLI behavior.
