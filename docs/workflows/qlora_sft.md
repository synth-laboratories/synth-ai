### QLoRA SFT

Train larger models on constrained hardware by enabling QLoRA while keeping the same SFT flow and payload schema.

- Works via the standard SFT CLI: `uvx synth-ai train --type sft --config <path>`
- Toggle with `training.use_qlora = true` in your TOML
- Uses the same hyperparameter keys as FFT; the backend interprets QLoRA-appropriate settings

### Quickstart

```bash
uvx synth-ai train --type sft --config examples/warming_up_to_rl/configs/crafter_fft_4b.toml --dataset /abs/path/to/train.jsonl
```

### Minimal TOML (QLoRA enabled)

```toml
[job]
model = "Qwen/Qwen3-4B"
# Either set here or pass via --dataset
# data = "/abs/path/to/train.jsonl"

[compute]
gpu_type = "H100"       # required by backend
gpu_count = 1
nodes = 1

[data]
# Optional; forwarded into metadata.effective_config
topology = {}
# Optional local validation file; client uploads if present
# validation_path = "/abs/path/to/validation.jsonl"

[training]
mode = "sft_offline"
use_qlora = true         # QLoRA toggle

[training.validation]
enabled = true
evaluation_strategy = "steps"
eval_steps = 20
save_best_model_at_end = true
metric_for_best_model = "val.loss"
greater_is_better = false

[hyperparameters]
n_epochs = 1
per_device_batch = 1
gradient_accumulation_steps = 64
sequence_length = 4096
learning_rate = 5e-6
warmup_ratio = 0.03

# Optional parallelism block forwarded as-is
#[hyperparameters.parallelism]
# use_deepspeed = true
# deepspeed_stage = 2
# bf16 = true
```

### What the client validates and sends

- Validates dataset path existence (from `[job].data` or `--dataset`) and JSONL shape
- Uploads training (and optional validation) files to `/api/learning/files`
- Builds payload with:
  - `model` from `[job].model`
  - `training_type = "sft_offline"`
  - `hyperparameters` from `[hyperparameters]` (+ `[training.validation]` knobs)
  - `metadata.effective_config.compute` from `[compute]`
  - `metadata.effective_config.data.topology` from `[data.topology]`
  - `metadata.effective_config.training.{mode,use_qlora}` from `[training]`

### Multi‑GPU guidance

- Set `[compute].gpu_type`, `gpu_count`, and optionally `nodes`
- Use `[hyperparameters.parallelism]` for deepspeed/FSDP/precision/TP/PP knobs; forwarded verbatim
- Optionally add `[data.topology]` (e.g., `container_count`) for visibility; backend validates resource consistency

### Common issues

- HTTP 400 `missing_gpu_type`: set `[compute].gpu_type` (and typically `gpu_count`) so it appears under `metadata.effective_config.compute`
- Dataset not found: provide absolute path or use `--dataset`; the client resolves relative paths from the current working directory

### CLI flags that help

- `--dataset` to override `[job].data`
- `--examples N` to subset the first N rows for quick smoke tests
- `--dry-run` to preview payload without submitting


