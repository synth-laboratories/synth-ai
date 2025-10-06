# Math Demo Workflow

The math demo provides a curated end-to-end experience for new users. All commands proxy into `synth_ai.demos.core.cli`, which prints contextual next steps (`synth_ai/cli/root.py:75`).

## Steps at a glance

1. **Setup**: `uvx synth-ai demo setup`
   - Launches the browser handshake with the Synth dashboard.
   - Persists `SYNTH_API_KEY`, `ENVIRONMENT_API_KEY`, and math demo defaults to `.env`.
2. **Deploy**: `uvx synth-ai demo deploy --name synth-math-demo`
   - Uses Modal to host the math task app.
   - Accepts `--local` for local FastAPI testing or `--app` to point at a custom app module (`synth_ai/cli/rl_demo.py:26`).
3. **Run RL job**: `uvx synth-ai demo run --config demo_config.toml`
   - Executes a short FFT-first RL job tailored to the demo.
   - Supports overrides for batch size, group size, model, timeout, and dry-run preview.
4. **Top-level shortcut**: `uvx synth-ai run --config demo_config.toml`
   - Equivalent to `demo run`, exposed for convenience (`synth_ai/cli/rl_demo.py:181`).

## Files to inspect

- `demo_config.toml` – default RL configuration shipped with the repo.
- `synth_ai/demos/core/cli.py` – underlying implementation (polling, guidance, Modal utilities).

## Roadmap

The public RL example will migrate from Crafter to this math task app using FFT checkpoints. Once the new configs land under `examples/rl/`, this page will include cross-links and updated command snippets.

