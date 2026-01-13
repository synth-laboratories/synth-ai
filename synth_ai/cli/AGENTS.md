CLI structure notes (for coding agents)

- Keep `synth_ai/cli/` flat: one file per command, filename == command name.
- Each command file holds Click wiring (options/args/validation/routing).
- Business logic lives in `synth_ai/sdk/` (reusable) or `synth_ai/core/` (internal).
- CLI-only helpers may live in `synth_ai/cli/lib/` or `synth_ai/cli/utils/` only.
- No shims/wrappers: do not duplicate modules across folders or re-export commands.
- Avoid heavy module-level imports in CLI; import inside functions for lazy load.
- Use Click conventions; no surprising CLI behavior.

Reward terminology (CRITICAL)
----------------------------
- **NEVER EVER USE THE TERM 'SCORE'. EVER.** Use reward-based language for everything.
- Use: `reward`, `verifier_reward`, `local_api_reward`, `fused_reward`, `outcome_reward`, etc.
- Never use: `score`, `verifier_score`, `scoring`, `scored`, `scores`, etc.
- This applies to variable names, function names, comments, documentation, API responses, and all code.
- Examples:
  - ✅ `verifier_reward`, `compute_reward()`, `reward_trajectory()`, `reward semantics`
  - ❌ `verifier_score`, `compute_score()`, `score_trajectory()`, `scoring semantics`
