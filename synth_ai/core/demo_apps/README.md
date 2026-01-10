# Demo Task Apps

This package contains tiny, ready-to-run demo Task Apps and helpers.

## Remote (Modal) Math Demo

- Template: select `math-modal` when prompted by `synth-ai demo init`
- Config: `synth_ai/demos/math/config.toml`
- Workflow:
  1. `uvx synth-ai demo init` (choose `math-modal`)
  2. `uvx synth-ai demo setup`
  3. `uvx synth-ai demo deploy`
  4. `uvx synth-ai demo run`
- Required env vars (persisted to `~/.synth-ai/user_config.json` during setup/deploy):
  - `SYNTH_API_KEY`
  - `ENVIRONMENT_API_KEY`
  - `TASK_APP_BASE_URL` (Modal HTTPS URL)

## Local Crafter Demo

- Template: select `crafter-local` when prompted by `synth-ai demo init`
- Config: `synth_ai/demos/crafter/configs/rl_from_base_qwen4b.toml`
- Workflow:
  1. `uvx synth-ai demo init` (choose `crafter-local`)
  2. `uvx synth-ai demo setup`
  3. `uvx synth-ai demo deploy` (starts FastAPI server on `http://127.0.0.1:8001`; leave terminal open)
  4. In another terminal:

     ```bash
     cd <path-to-your-demo>
     uvx python run_local_rollout_traced.py
    uvx python export_trace_sft.py --db traces/v3/task_app_traces_<timestamp>.db --output demo_sft.jsonl
     # Optional lighter run
     uvx python run_local_rollout.py
     ```
     (replace `<path-to-your-demo>` with the directory created during `demo init`, e.g. `cd crafter_demo`)

- Remote job submission (`uvx synth-ai demo run`) requires exposing the task app via an HTTPS endpoint.
- Required keys:
  - `ENVIRONMENT_API_KEY`
  - `OPENAI_API_KEY` (prompted if missing when using the OpenAI-backed rollout script)
