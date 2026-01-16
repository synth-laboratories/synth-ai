# Demo Task Apps

This package contains tiny, ready-to-run demo Task Apps and helpers.

## Math Demo

- Config: `synth_ai/demos/demo_task_apps/math/config.toml`
- Local run (dev):

```
uvx python -c "from synth_ai.demos.demo_task_apps.math.app import run; run()"
```

- Expected envs:
  - `ENVIRONMENT_API_KEY` (for the task app to contact the environment service)
- `SYNTH_BACKEND_URL` and `SYNTH_API_KEY` for launching RL jobs from the CLI (if wired)

These demos are designed to pair with `uvx synth-ai demo.*` commands described in pip_install_rl.txt.
