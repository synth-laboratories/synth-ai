### Full Finetuning (FFT)

Run standard supervised finetuning using the SFT workflow with `training.use_qlora = false` (default) and typical FFT hyperparameters.

- Invoke via: `uvx synth-ai train --type sft --config <path>`
- Uses the same client/payload path as QLoRA; differs only in training mode/toggles and typical hyperparameters/parallelism

### Quickstart

```bash
uvx synth-ai train --type sft --config examples/warming_up_to_rl/configs/crafter_fft.toml --dataset /abs/path/to/train.jsonl
```

### Minimal TOML (FFT)

```toml
[job]
model = "Qwen/Qwen3-4B"
# data = "/abs/path/to/train.jsonl"  # or pass via --dataset

[compute]
gpu_type = "H100"       # required by backend
gpu_count = 4
nodes = 1

[data.topology]
container_count = 4      # optional; informational

[training]
mode = "full_finetune"  # documentation; backend infers
use_qlora = false

[training.validation]
enabled = true
evaluation_strategy = "steps"
eval_steps = 20
save_best_model_at_end = true
metric_for_best_model = "val.loss"
greater_is_better = false

[hyperparameters]
n_epochs = 2
sequence_length = 2048
per_device_batch = 2
gradient_accumulation_steps = 64
learning_rate = 8e-6
warmup_ratio = 0.03
train_kind = "fft"

[hyperparameters.parallelism]
use_deepspeed = true
deepspeed_stage = 3
fsdp = false
bf16 = true
fp16 = false
```

### Reference TOMLs

Below are battle-tested configurations you can mirror locally or reuse in automation. Each section maps directly to the payload the CLI assembles for `/api/learning/jobs`.

#### Qwen3‑32B full finetune on 4×H100 with FSDP

```toml
[job]
model = "Qwen/Qwen3-32B"
data = "/abs/path/to/train.jsonl"  # override at runtime with --dataset if preferred

[compute]
gpu_type = "H100"
gpu_count = 4
nodes = 1
variant = "H100-4x"
gpus_per_node = 4

[data.topology]
container_count = 4
gpus_per_node = 4
total_gpus = 4
nodes = 1
variant = "H100-4x"

[training]
mode = "full_finetune"
use_qlora = false

[training.validation]
enabled = false
evaluation_strategy = "steps"
eval_steps = 100
save_best_model_at_end = false
metric_for_best_model = "val.loss"
greater_is_better = false

[hyperparameters]
n_epochs = 1
train_kind = "fft"
per_device_batch = 1
gradient_accumulation_steps = 16
sequence_length = 4096
learning_rate = 5e-6
warmup_ratio = 0.03
global_batch = 64

[hyperparameters.parallelism]
fsdp = true
fsdp_sharding_strategy = "full_shard"
fsdp_auto_wrap_policy = "transformer_block"
fsdp_use_orig_params = true
activation_checkpointing = true
tensor_parallel_size = 1
pipeline_parallel_size = 1
bf16 = true
fp16 = false
use_deepspeed = false
```

Key points:

- `variant`, `gpus_per_node`, and the `data.topology` block keep the effective config aligned with the Hatchet resource planner. Omitting them can work, but setting them explicitly prevents mismatched container counts.
- The FSDP flags are forwarded verbatim. Remove or tweak them only if your trainer image applies different defaults.
- `global_batch` is optional; including it documents the intended throughput when you revisit traces and billing records.

#### Where these configs are exercised

- `tests/cli/test_cli_train_multi_gpu_dev.py` writes the 32B/FSDP TOML on the fly, calls `uvx synth-ai train`, and asserts on the `/api/learning/jobs/{id}` document. It runs in CI (see the "Integration (dev endpoints)" workflow) and doubles as a smoke test script.
- `examples/dev/qwen3_32b_qlora_4xh100.toml` remains the lightweight QLoRA sample. Use it when you prefer adapters over full finetuning.

### What the client validates and sends

- Validates dataset path existence and JSONL records
- Uploads files to `/api/learning/files`, then creates/starts job under `/api/learning/jobs`
- Payload mapping is identical to LoRA SFT: hyperparameters + `metadata.effective_config` (compute, data.topology, training)

### Multi‑GPU guidance (FFT)

- Use `[compute]` for cluster shape
- Prefer `[hyperparameters.parallelism]` for deepspeed stage, FSDP, precision, TP/PP sizes; forwarded verbatim
- `[data.topology]` is optional and informational; backend/trainer validates actual resource consistency

### Common issues

- HTTP 400 `missing_gpu_type`: add `[compute].gpu_type`
- Dataset not found: specify absolute path or use `--dataset` (paths resolved from current working directory)

### Helpful CLI flags

- `--examples N` to subset data for a quick smoke test
- `--dry-run` to preview payload before submitting

