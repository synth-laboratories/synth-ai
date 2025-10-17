### Multi-GPU SFT

Train large models (e.g., 32B parameters) efficiently across multiple GPUs using either QLoRA or Full Finetuning (FFT) with appropriate parallelism strategies.

- **QLoRA**: Memory-efficient training with adapters, suitable for 4+ GPUs
- **FFT**: Full model training with FSDP/DeepSpeed, requires 8+ GPUs for 32B models
- Both use the same CLI: `uvx synth-ai train --type sft --config <path>`

### Quickstart

```bash
# QLoRA on 4x H100
uvx synth-ai train --type sft --config examples/multi_gpu_qlora_4xh100.toml --dataset /abs/path/to/train.jsonl

# FFT on 8x H100  
uvx synth-ai train --type sft --config examples/multi_gpu_fft_8xh100.toml --dataset /abs/path/to/train.jsonl
```

### QLoRA Multi-GPU Configuration (4x H100)

```toml
[job]
model = "Qwen/Qwen3-32B"
data = "/abs/path/to/train.jsonl"

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
mode = "lora"
use_qlora = true

[training.validation]
enabled = false
evaluation_strategy = "steps"
eval_steps = 100
save_best_model_at_end = false
metric_for_best_model = "val.loss"
greater_is_better = false

[hyperparameters]
n_epochs = 1
train_kind = "peft"
per_device_batch = 1
gradient_accumulation_steps = 16
sequence_length = 4096
learning_rate = 5e-6
warmup_ratio = 0.03
global_batch = 64

[hyperparameters.parallelism]
fsdp = false
fsdp_sharding_strategy = "full_shard"
fsdp_auto_wrap_policy = "transformer_block"
fsdp_use_orig_params = true
tensor_parallel_size = 1
pipeline_parallel_size = 1
bf16 = true
fp16 = false
use_deepspeed = true
deepspeed_stage = 2
activation_checkpointing = true
```

### FFT Multi-GPU Configuration (8x H100)

```toml
[job]
model = "Qwen/Qwen3-32B"
data = "/abs/path/to/train.jsonl"

[compute]
gpu_type = "H100"
gpu_count = 8
nodes = 1
variant = "H100-8x"
gpus_per_node = 8

[data.topology]
container_count = 8
gpus_per_node = 8
total_gpus = 8
nodes = 1
variant = "H100-8x"

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

### Key Configuration Differences

#### QLoRA vs FFT

| Setting | QLoRA (4x H100) | FFT (8x H100) |
|---------|------------------|---------------|
| `training.use_qlora` | `true` | `false` |
| `training.mode` | `"lora"` | `"full_finetune"` |
| `hyperparameters.train_kind` | `"peft"` | `"fft"` |
| `hyperparameters.parallelism.use_deepspeed` | `true` (stage 2) | `false` |
| `hyperparameters.parallelism.fsdp` | `false` | `true` |
| Minimum GPUs | 4 | 8 |

#### Parallelism Strategies

**QLoRA (DeepSpeed ZeRO-2)**:
- Uses DeepSpeed for gradient/optimizer state sharding
- Lower memory footprint per GPU
- Suitable for 4-8 GPUs

**FFT (FSDP)**:
- Uses PyTorch FSDP for model parameter sharding
- Higher memory requirements but full model training
- Requires 8+ GPUs for 32B models

### Hardware Requirements

#### QLoRA (4x H100)
- **Memory**: ~80GB per GPU
- **Compute**: H100 (80GB) recommended
- **Network**: High-speed interconnect (NVLink/InfiniBand)
- **Storage**: Fast SSD for dataset loading

#### FFT (8x H100)
- **Memory**: ~80GB per GPU
- **Compute**: H100 (80GB) required
- **Network**: High-speed interconnect essential
- **Storage**: Fast SSD for dataset loading

### Performance Tuning

#### Batch Size Optimization
```toml
[hyperparameters]
# Start conservative, increase if stable
per_device_batch = 1
gradient_accumulation_steps = 16
global_batch = 64  # = per_device_batch * gpu_count * gradient_accumulation_steps
```

#### Learning Rate Scheduling
```toml
[hyperparameters]
learning_rate = 5e-6      # Conservative for large models
warmup_ratio = 0.03       # 3% warmup
```

#### Sequence Length
```toml
[hyperparameters]
sequence_length = 4096    # Balance memory vs context length
```

### Common Issues and Solutions

#### Out of Memory (OOM)
- **QLoRA**: Reduce `per_device_batch` or increase `gradient_accumulation_steps`
- **FFT**: Reduce `sequence_length` or increase `gpu_count`

#### Slow Training
- **Check**: `global_batch` size and learning rate scaling
- **Optimize**: Dataset loading and preprocessing
- **Verify**: Network interconnect performance

#### Convergence Issues
- **Adjust**: Learning rate and warmup ratio
- **Monitor**: Validation loss curves
- **Consider**: Different parallelism strategies

### Validation and Monitoring

```toml
[training.validation]
enabled = true
evaluation_strategy = "steps"
eval_steps = 100
save_best_model_at_end = true
metric_for_best_model = "val.loss"
greater_is_better = false
```

### CLI Flags for Multi-GPU

```bash
# Override dataset path
uvx synth-ai train --type sft --config config.toml --dataset /path/to/data.jsonl

# Quick smoke test with subset
uvx synth-ai train --type sft --config config.toml --examples 100

# Preview payload without submitting
uvx synth-ai train --type sft --config config.toml --dry-run

# Set GPU type for resource planning
SYNTH_GPU_TYPE=H100-4x uvx synth-ai train --type sft --config config.toml
```

### Testing Multi-GPU Configurations

The multi-GPU configurations are tested in CI:
- `tests/cli/test_cli_train_multi_gpu_dev.py::test_cli_train_multi_gpu_qlora` (4x H100)
- `tests/cli/test_cli_train_multi_gpu_dev.py::test_cli_train_multi_gpu_fft` (8x H100)

These tests validate:
- Job creation and submission
- Resource allocation
- Training completion
- Configuration consistency

### Best Practices

1. **Start with QLoRA** for initial experiments and fine-tuning
2. **Use FFT** for final model training when you need full parameter updates
3. **Monitor memory usage** and adjust batch sizes accordingly
4. **Validate configurations** with small datasets before full training
5. **Use appropriate parallelism** based on model size and hardware
