# Warming Up to RL (Crafter)

The Crafter example under `examples/warming_up_to_rl/` is the template for future Synth documentation. It includes a task app, evaluation harness, FFT training script, RL submission utilities, and tracing helpers.

## Directory structure

- `task_app/` – FastAPI wrapper delegating to `synth_ai.task.apps.grpo_crafter`.
- `configs/` – TOML files for eval, FFT, and RL runs.
- `run_eval.py` – baseline evaluation loop (Groq, Synth vLLM, FFT checkpoints).
- `run_fft_and_save.py` – submits an FFT job and stores the returned model ID.
- `run_rl_and_save.py` – creates RL jobs using TOML-only model selection.
- `export_trace_sft.py` – converts tracing v3 sqlite files into JSONL datasets.
- `run_local_rollout_traced.py` – local rollout helper that enables tracing and SFT export.

## Workflow summary

1. **Task app**
   - Local: `uvx synth-ai serve grpo-crafter --port 8001`
   - Modal: `uvx synth-ai deploy grpo-crafter --name grpo-crafter-task-app`
2. **Eval**
   - `TASK_APP_URL=… uv run python run_eval.py --toml configs/eval_groq_qwen32b.toml --use-rollout`
3. **FFT**
   - `uvx synth-ai train --type sft --config configs/crafter_fft.toml`
   - Legacy script available for automation.
4. **RL**
   - `uvx synth-ai train --type rl --config configs/rl_from_ft.toml`
   - Alternative: `rl_from_base_qwen4b.toml` to start from a base model.
5. **Tracing → Dataset**
   - Serve with tracing enabled or run traced rollouts.
   - Export with `export_trace_sft.py`, then feed the JSONL back into SFT.

## Roadmap for examples

- **Finetuning**: ship a QLoRA configuration for Crafter that mirrors the FFT scripts but uses low-rank adapters. Docs will reference the new TOML once merged.
- **RL**: pivot the public RL example to the Math task app using FFT checkpoints instead of Crafter. Evaluation configs will follow suit.
- **Examples organization**: align `examples/evals`, `examples/finetuning`, and `examples/rl` with the workflows documented in `docs/workflows/` so the README in each directory points to a single source of truth.

For a broader tracking list, see [Examples Refresh Roadmap](roadmap.md).
