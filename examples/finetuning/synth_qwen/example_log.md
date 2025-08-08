### Example execution log (exact commands used)

- All steps assume environment variables are loaded from `.temp` (copied from `env.example`).
- Commands are shown exactly as run, using `uvpm` for module execution.

### 1) Generate 10 trajectories (Qwen 4B base model)

```bash
set -a; source examples/finetuning/synth_qwen/.temp; set +a; \
CRAFTER_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
CRAFTER_EPISODES=10 \
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

Optional (to reduce service pressure):
```bash
CRAFTER_CONCURRENCY=1 \
CRAFTER_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

### 2) Filter traces into SFT JSONL (achievements: collect_wood)

```bash
set -a; source examples/finetuning/synth_qwen/.temp; set +a; \
CRAFTER_DB_URL="sqlite:///traces_v3_lm_synth/traces.db" \
OUTPUT_JSONL="ft_data/qwen4b_crafter_sft_collect_wood.jsonl" \
REQUIRED_ACHIEVEMENTS="collect_wood" \
uvpm examples.finetuning.synth_qwen.filter_traces_achievements
```

Baseline filter (reward/limits only):
```bash
set -a; source examples/finetuning/synth_qwen/.temp; set +a; \
CRAFTER_DB_URL="sqlite:///traces_v3_lm_synth/traces.db" \
OUTPUT_JSONL="ft_data/qwen4b_crafter_sft.jsonl" \
MIN_TOTAL_REWARD=1.0 MAX_COST=10.0 MAX_TOKENS=100000 \
MODELS="Qwen/Qwen3-4B-Instruct-2507" \
uvpm examples.finetuning.synth_qwen.filter_traces
```

### 3) Kick off SFT (Qwen 4B)

```bash
set -a; source examples/finetuning/synth_qwen/.temp; set +a; \
QWEN_TRAINING_JSONL="ft_data/qwen4b_crafter_sft.jsonl" \
QWEN_BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
uvpm examples.finetuning.synth_qwen.sft_kickoff
```

Note: Replace `QWEN_TRAINING_JSONL` with `ft_data/qwen4b_crafter_sft_collect_wood.jsonl` if using the achievements-filtered dataset.

### 4) Run evaluation on the fine-tuned model (10 episodes, concurrency 5)

```bash
set -a; source examples/finetuning/synth_qwen/.temp; set +a; \
CRAFTER_CONCURRENCY=5 \
CRAFTER_MODEL="ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-ab" \
CRAFTER_EPISODES=10 \
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

Where `ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-ab` is the fine-tuned model ID returned by the SFT job. Use your actual `ft:` ID.

