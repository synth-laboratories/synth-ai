# Evaluation Workflow

The evaluation loop exercises a task app from an LLM policy (base, finetuned, or vendor-hosted). The reference script lives at `examples/warming_up_to_rl/run_eval.py`.

## Prerequisites

- Task app deployed or running locally (`uvx synth-ai serve grpo-crafter` or Modal deploy).
- `.env` with `ENVIRONMENT_API_KEY` and optional vendor keys (OpenAI, Groq).
- TOML config describing the evaluation run (see `examples/warming_up_to_rl/configs/eval_*.toml`).

## Running an Evaluation

```bash
TASK_APP_URL=https://<your-task-app>.modal.run \
uv run python examples/warming_up_to_rl/run_eval.py \
  --toml examples/warming_up_to_rl/configs/eval_groq_qwen32b.toml \
  --use-rollout
```

Key features implemented in `run_eval.py`:

- **TaskAppClient**: async HTTP shim handling `/initialize`, `/step`, `/terminate`, `/rollout`, and vendor proxies (`TaskAppClient.initialize/step/rollout`).
- **Model routing**: `_detect_provider` selects Groq vs vLLM, while `_rollout_inference_url_from_cfg` prefers the TOML `inference_url` or the task app’s advertised vLLM base URL.
- **Tool-calling policy**: `_build_messages_from_observation` encodes observation state into a chat prompt, and `_parse_tool_calls_from_openai_response` extracts `interact` actions from LLM responses.
- **Rollout mode**: `--use-rollout` drives the server-side `/rollout` endpoint, preserving metrics and final achievements when tracing is enabled.

## TOML Parameters

The Eval TOML schema includes:

- `task_app_url` – base URL (overridden by `TASK_APP_URL` env if placeholder).
- `model` – provider-specific identifier (`groq:qwen3-32b`, `Qwen/Qwen3-4B`).
- `inference_url` – optional explicit inference endpoint (vLLM or Synth backend).
- `num_episodes`, `max_turns`, `concurrency` – evaluation volume and parallelism.
- `difficulty` – forwarded to `initialize` via environment config.
- `max_tokens`, `temperature`, `thinking_mode`, `thinking_budget` – forwarded to the policy config when using `/rollout`.

## Output & Tracing

Evaluations print per-episode results and aggregate stats. When tracing is enabled on the task app (`TASKAPP_TRACING_ENABLED=1` during `uvx synth-ai serve`), each rollout populates tracing v3 sqlite files (`traces/v3/...`) with event and outcome rewards that can later be exported to SFT datasets.

## Next Steps

- Fine-tune the same policy with FFT (`workflows/sft.md`).
- Submit an RL job that uses the finetuned model as the starting policy (`workflows/rl.md`).
- Export traces to JSONL for behavioural cloning (`workflows/tracing.md`).

