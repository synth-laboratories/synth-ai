# Crafter: From Rollouts to RL with the Synth AI CLI

This playbook mirrors the original “Warming Up to RL” walkthrough, but swaps the bespoke scripts for the first–class `uvx synth-ai` helpers. Every step—from deploying the task app to filtering rollouts, fine-tuning, and bootstrapping RL— now uses the same CLI you’d reach for in production.

All commands assume you are inside the repository root and have `uv`/`uvx` available.

---

## 0. Prerequisites

1. Install dependencies and authenticate once:
   ```bash
   uv pip install -e .
   uvx synth-ai setup
   ```
   The setup wizard writes the required `SYNTH_API_KEY`, `ENVIRONMENT_API_KEY`, and local `.env` helpers.

2. Copy the example secrets if you need a starter file:
   ```bash
   cp examples/warming_up_to_rl/.env.example .env
   ```

3. Export the path we use for trace capture (optional but keeps things tidy):
   ```bash
   export CRAFTER_TRACE_DB=traces/v3/crafter_blog.db
   ```

---

## 1. Ship the Crafter Task App

Deploy the hosted Crafter environment once. The Modal URL that prints at the end is reused by eval, SFT, and RL.

```bash
uvx synth-ai deploy grpo-crafter \
  --runtime modal \
  --modal-mode serve \
  --name crafter-blogpost \
  --env-file .env
```

For local testing you can run:

```bash
uvx synth-ai deploy grpo-crafter \
  --runtime uvicorn \
  --port 8001 \
  --trace traces/v3 \
  --env-file .env
```

Copy the Modal URL (e.g. `https://your-app.modal.run`) and replace the `task_app_url` placeholders inside every config under `examples/blog_posts/warming_up_to_rl/configs/`.

---

## 2. Collect High-Quality Rollouts

We lean on large teacher models to produce demonstrations. The configs in `configs/` already request full traces so we retain chain-of-thought.

 Groq Qwen3-32B (text-only prompt):
```bash
uvx synth-ai eval grpo-crafter \
  --config examples/blog_posts/warming_up_to_rl/configs/eval_groq_qwen32b.toml \
  --trace-db "${CRAFTER_TRACE_DB}"
```

 GPT-OSS-120B via Groq’s OpenAI-compatible endpoint (also text-only):
```bash
uvx synth-ai eval grpo-crafter \
  --config examples/blog_posts/warming_up_to_rl/configs/eval_openai_gpt_oss_120b.toml \
  --trace-db "${CRAFTER_TRACE_DB}"
```

 Both configs disable image attachments and rely on the textual observation renderer (`format_observation`) so Groq stays within its supported modalities. If you want to try other models, keep `use_vision = false` unless the provider explicitly supports image inputs.

---

## 3. Filter Into an SFT Dataset

Once traces are stored in `CRAFT_TRACE_DB`, trim to the crisp trajectories:

```bash
uvx synth-ai filter \
  --config examples/blog_posts/warming_up_to_rl/configs/filter_high_reward_dataset.toml
```

The output JSONL lands in `ft_data/crafter_blog_high_reward.jsonl`, ready for supervised fine-tuning.

---

## 4. Fine-Tune Qwen3-4B with `uvx synth-ai train`

Update the dataset path (and optionally hyperparameters) in `train_sft_qwen4b.toml`, then launch:

```bash
uvx synth-ai train \
  --type sft \
  --config examples/blog_posts/warming_up_to_rl/configs/train_sft_qwen4b.toml \
  --env-file .env \
  --poll
```

Capture the returned job id (it looks like `fft:Qwen/Qwen3-4B:job_xxxxx`). We reuse that identifier in the evaluation and RL configs.
At any time you can list recently minted checkpoints with:

```bash
uvx synth-ai status models
```

The output table shows the canonical model name/ID alongside the source job.

---

## 5. Evaluate the Fine-Tuned Checkpoint

Replace both `REPLACE-WITH-SFT-JOB-ID` strings inside `eval_ft_qwen4b.toml`, then run:

```bash
uvx synth-ai eval grpo-crafter \
  --config examples/blog_posts/warming_up_to_rl/configs/eval_ft_qwen4b.toml \
  --trace-db "${CRAFTER_TRACE_DB}"
```

This provides a clean, CLI-native comparison between the teacher rollouts and the fine-tuned model.

---

## 6. Kick Off RL from the Fine-Tuned Model

Point `train_rl_from_sft.toml` at the same Modal task app and set `model.source` to your SFT job id:

```bash
uvx synth-ai train \
  --type rl \
  --config examples/blog_posts/warming_up_to_rl/configs/train_rl_from_sft.toml \
  --env-file .env \
  --poll
```

The CLI streams rollout and judge metrics in real time. When the run finishes, you can re-use the Stage 5 config (substituting the RL job id) to quantify the uplift.
If you lose track of the produced RL label or want to confirm the latest status, run:

```bash
uvx synth-ai status jobs
uvx synth-ai status models
```

The first command shows job completion state; the second surfaces model IDs you can plug into new eval configs.

---

## 7. Where to Go Next

- The original `examples/warming_up_to_rl` folder still contains deeper experiments (auto-curricula, modal renderers, etc.).
- Add more `eval_*.toml` configs to compare alternative judges or reward shaping strategies.
- Plug the filtered dataset into `uvx synth-ai files upload` if you want to share it with a teammate without copying JSONL around.

This directory now holds everything a blog post needs: configs, output locations, and the CLI entrypoints to reproduce the Crafter SFT → RL pipeline end-to-end.
