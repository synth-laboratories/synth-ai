### Quickstart (Crafter) with config.toml

All defaults are in `examples/finetuning/synth_qwen/config.toml`. Place your API keys in `.env`.

1) Generate traces (v3 tracing)
```bash
set -a; source .env 2>/dev/null || true; set +a
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

2) Filter traces → SFT JSONL
```bash
uvpm examples.finetuning.synth_qwen.filter_traces_achievements
```

3) Finetune via learning service (SFT)
```bash
set -a; source .env 2>/dev/null || true; set +a
uvpm examples.finetuning.synth_qwen.sft_kickoff
```

4) Evaluate the fine-tuned model in Crafter
```bash
set -a; source .env 2>/dev/null || true; set +a
CRAFTER_MODEL="ft:...your-returned-id..." uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

Notes:
- If you see a 401, ensure your `.env` contains a valid production `SYNTH_API_KEY` or export it inline.
- Traces are stored in `traces/v3/synth_ai.db` (sqld); the filter derives the correct internal data file.


### Interactive Demo

Use the interactive script to walk through rollouts → filtering → SFT → optional rollout of the fine-tuned model.

```bash
examples/finetuning/synth_qwen/run_demo.sh
```

What it does:
- Prompts for rollout settings (model, episodes, max steps, difficulty, think).
- Prompts for filter settings (required achievements, model restriction, min reward, max cost/tokens, output path).
- Starts the SFT job and captures the returned fine-tuned model id.
- Asks you to confirm before rolling out the fine-tuned model.
- API key handling:
  - If a `SYNTH_API_KEY` is detected, you’re asked to confirm using it.
  - If not set, you can choose `SYNTH_API_KEY_PROD` (if present) or securely enter a key.
  - `OPENAI_API_KEY` is set to the same value if missing to prevent 401s.
