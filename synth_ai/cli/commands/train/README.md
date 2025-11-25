# Train Command TOML Validation

This module provides comprehensive TOML configuration validation for both SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) training jobs.

> Example configs moved to the Synth Cookbooks repo (`cookbooks/dev/...`). Replace any `examples/...` paths with the matching paths under https://github.com/synth-laboratories/cookbooks.

## Overview

The validation flow follows this pattern:
```
TOML file → load_toml() → validate_*_config() → Pydantic models → validated dict → API request
```

## Architecture

### Files

- **`validation.py`**: Core validation logic for SFT and RL configs
- **`errors.py`**: Structured exception types for validation failures
- **`__init__.py`**: Public API exports

### Validation Functions

#### SFT Validation

```python
from synth_ai.cli.commands.train import validate_sft_config, load_and_validate_sft

# Validate dict directly
config_dict = {...}
validated = validate_sft_config(config_dict)

# Load and validate from file
from pathlib import Path
validated = load_and_validate_sft(Path("config.toml"))
```

#### RL Validation

```python
from synth_ai.cli.commands.train import validate_rl_config, load_and_validate_rl

# Validate dict directly
config_dict = {...}
validated = validate_rl_config(config_dict)

# Load and validate from file
from pathlib import Path
validated = load_and_validate_rl(Path("config.toml"))
```

## SFT Configuration Requirements

### Required Sections

#### `[algorithm]`
```toml
[algorithm]
type = "offline"              # Required for SFT
method = "sft"                # or "supervised_finetune"
variety = "fft"               # or "lora", "qlora"
```

#### `[job]`
```toml
[job]
model = "Qwen/Qwen3-4B"       # Required: HuggingFace model identifier
data = "path/to/dataset.jsonl"  # OR data_path - at least one required
```

#### `[compute]`
```toml
[compute]
gpu_type = "H100"             # Required: H100, A100, etc.
gpu_count = 4                 # Required: number of GPUs
nodes = 1                     # Optional: number of nodes
```

### Optional Sections

#### `[training]`
```toml
[training]
mode = "full_finetune"        # or "lora", "sft_offline"
use_qlora = false

[training.validation]
enabled = true
evaluation_strategy = "steps"
eval_steps = 100
save_best_model_at_end = true
metric_for_best_model = "val.loss"
greater_is_better = false
```

#### `[hyperparameters]`
```toml
[hyperparameters]
n_epochs = 1
per_device_batch = 1
gradient_accumulation_steps = 1
sequence_length = 1024
learning_rate = 5e-6
warmup_ratio = 0.03
global_batch = 4
train_kind = "fft"            # or "peft"

[hyperparameters.parallelism]
fsdp = true
fsdp_sharding_strategy = "full_shard"
activation_checkpointing = true
tensor_parallel_size = 1
pipeline_parallel_size = 1
bf16 = true
fp16 = false
use_deepspeed = false
```

#### `[data]`
```toml
[data]
validation_path = "path/to/val.jsonl"  # Optional validation dataset

[data.topology]
container_count = 4
gpus_per_node = 4
total_gpus = 4
nodes = 1
```

## RL Configuration Requirements

### Required Sections

#### `[algorithm]`
```toml
[algorithm]
type = "online"               # Required for RL
method = "policy_gradient"    # or "ppo", "gspo"
variety = "gspo"              # Algorithm variety
```

#### `[model]`
```toml
[model]
base = "Qwen/Qwen3-1.7B"      # OR source - exactly one required
# source = "s3://bucket/checkpoint"  # Alternative to base
trainer_mode = "full"         # Required: "full" or "lora"
label = "my-model"            # Required: model label
```

#### `[compute]`
```toml
[compute]
gpu_type = "H100"             # Required: H100, A100, etc.
gpu_count = 2                 # Required: number of GPUs
```

#### `[topology]`
```toml
[topology]
type = "single_node_split"    # Topology type
gpus_for_vllm = 1             # GPUs for inference
gpus_for_training = 1         # GPUs for training
gpus_for_ref = 0              # GPUs for reference model
tensor_parallel = 1           # Tensor parallelism degree
```

#### `[rollout]`
```toml
[rollout]
env_name = "math"             # Required: environment name
policy_name = "math"          # Required: policy name
max_turns = 1                 # Required: max turns per episode
episodes_per_batch = 2        # Required: episodes per batch
max_concurrent_rollouts = 2   # Required: max concurrent rollouts
batches_per_step = 1          # Optional: batches per step
ops = ["agent", "env"]        # Optional: operation sequence

[rollout.policy_config]
max_llm_calls = 10
max_tokens = 512
temperature = 0.2
top_p = 0.95
```

### Optional Sections

#### `[lora]` (if trainer_mode = "lora")
```toml
[lora]
r = 16
alpha = 32
dropout = 0.05
target_modules = ["all-linear"]
```

#### `[vllm]`
```toml
[vllm]
tensor_parallel_size = 1
max_model_len = 4096
```

#### `[reference]`
```toml
[reference]
placement = "none"            # or "cpu", "gpu"
```

#### `[training]`
```toml
[training]
num_epochs = 1
iterations_per_epoch = 1
gradient_accumulation_steps = 1
max_accumulated_minibatch = 1
max_turns = 1
batch_size = 1
group_size = 2
learning_rate = 5e-6
log_interval = 1
weight_sync_interval = 1
weight_sync_verify_checksums = false

[training.weight_sync]
enable = true
targets = ["policy"]
mode = "full"
direct = true
verify_every_k = 0
chunk_bytes = 0
```

#### `[evaluation]`
```toml
[evaluation]
instances = 2
every_n_iters = 1
seeds = [0, 1]
```

#### `[judge]`
```toml
[judge]
type = "llm"
timeout_s = 30

[judge.options]
event = true
outcome = true
provider = "openai"
model = "gpt-4"
rubric_id = "custom-rubric"
max_concurrency = 10
```

#### `[tags]`
```toml
[tags]
experiment = "my-experiment"
version = "v1"
```

## Error Handling

### Exception Hierarchy

All validation errors inherit from `TrainCliError`:

```python
TrainCliError
├── TomlParseError              # TOML parsing failed
├── ConfigNotFoundError         # Config file not found
├── InvalidSFTConfigError       # SFT validation failed
├── InvalidRLConfigError        # RL validation failed
├── MissingAlgorithmError       # [algorithm] section missing
├── MissingModelError           # Model not specified
├── MissingDatasetError         # Dataset not specified (SFT)
├── MissingComputeError         # [compute] section incomplete
├── UnsupportedAlgorithmError   # Algorithm type not supported
├── InvalidHyperparametersError # Hyperparameters invalid
└── InvalidTopologyError        # Topology configuration invalid
```

### Example Error Handling

```python
from synth_ai.cli.commands.train import load_and_validate_sft
from synth_ai.cli.commands.train.errors import (
    InvalidSFTConfigError,
    MissingModelError,
    MissingDatasetError,
)

try:
    validated = load_and_validate_sft(Path("config.toml"))
except MissingModelError as e:
    print(f"Model missing: {e.detail}")
    if e.hint:
        print(f"Hint: {e.hint}")
except MissingDatasetError as e:
    print(f"Dataset missing: {e.detail}")
    if e.hint:
        print(f"Hint: {e.hint}")
except InvalidSFTConfigError as e:
    print(f"Invalid config: {e.detail}")
```

## Validation Rules

### SFT-Specific Rules

1. **Algorithm type** must be `"offline"` (or None)
2. **Algorithm method** must be `"sft"` or `"supervised_finetune"`
3. **Model** must be specified in `[job].model`
4. **Dataset** must be specified via `[job].data` or `[job].data_path`
5. **Compute** must specify `gpu_type` and `gpu_count`

### RL-Specific Rules

1. **Algorithm type** must be `"online"` (or None)
2. **Algorithm method** must be `"policy_gradient"`, `"ppo"`, or `"gspo"`
3. **Model** must specify exactly one of `[model].base` or `[model].source`
4. **Model** must specify `trainer_mode` (`"full"` or `"lora"`)
5. **Rollout** must specify `env_name`, `policy_name`, `max_turns`, `episodes_per_batch`, and `max_concurrent_rollouts`
6. **Topology** section is required
7. **Services** section is auto-injected if missing (task_url resolved at runtime)

## Integration with Existing Code

The validation logic integrates with the existing train CLI:

```python
# In synth_ai/api/train/builders.py
from synth_ai.cli.commands.train import load_and_validate_sft, load_and_validate_rl

def build_sft_payload(config_path: Path, ...) -> dict:
    # Load and validate TOML
    validated_config = load_and_validate_sft(config_path)
    
    # Apply overrides
    # ...
    
    # Build API payload
    return payload

def build_rl_payload(config_path: Path, ...) -> dict:
    # Load and validate TOML
    validated_config = load_and_validate_rl(config_path)
    
    # Inject task_url
    validated_config["services"]["task_url"] = task_url
    
    # Apply overrides
    # ...
    
    # Build API payload
    return payload
```

## Testing

Comprehensive unit tests are provided in `tests/unit/test_train_validation.py`:

```bash
# Run all validation tests
uv run pytest tests/unit/test_train_validation.py -v

# Run SFT tests only
uv run pytest tests/unit/test_train_validation.py::TestSFTValidation -v

# Run RL tests only
uv run pytest tests/unit/test_train_validation.py::TestRLValidation -v
```

### Test Coverage

- **SFT Tests (9 tests)**:
  - Valid FFT and LoRA configs
  - Missing required sections
  - Invalid algorithm types
  - Missing GPU configuration

- **RL Tests (11 tests)**:
  - Valid full and LoRA configs
  - Missing required sections
  - Invalid algorithm types
  - Model source validation
  - Auto-injection of services section

## Example Configs

> Full config examples now live in the Synth Cookbooks repo (`cookbooks/dev/...`).

### Minimal SFT Config

```toml
[algorithm]
type = "offline"
method = "sft"
variety = "fft"

[job]
model = "Qwen/Qwen3-4B"
data = "dataset.jsonl"

[compute]
gpu_type = "H100"
gpu_count = 1
nodes = 1

[hyperparameters]
n_epochs = 1
learning_rate = 5e-6
```

### Minimal RL Config

```toml
[algorithm]
type = "online"
method = "policy_gradient"
variety = "gspo"

[model]
base = "Qwen/Qwen3-1.7B"
trainer_mode = "full"
label = "test"

[compute]
gpu_type = "H100"
gpu_count = 2

[topology]
type = "single_node_split"
gpus_for_vllm = 1
gpus_for_training = 1

[rollout]
env_name = "math"
policy_name = "math"
max_turns = 1
episodes_per_batch = 2
max_concurrent_rollouts = 2

[training]
num_epochs = 1
iterations_per_epoch = 1
max_turns = 1
batch_size = 1
group_size = 2
learning_rate = 5e-6

[evaluation]
instances = 2
every_n_iters = 1
seeds = [0, 1]
```

## Future Enhancements

Potential improvements:

1. **Cross-field validation**: Validate relationships between fields (e.g., gpu_count matches topology)
2. **Model validation**: Check if model exists on HuggingFace
3. **Dataset validation**: Validate JSONL format and structure
4. **Resource validation**: Check if requested GPUs are available
5. **Schema versioning**: Support multiple config schema versions
6. **Config migration**: Auto-migrate old configs to new format
7. **Config templates**: Provide templates for common use cases

## Related Documentation

- [SFT Config Schema](/Users/joshpurtell/Documents/GitHub/synth-ai/synth_ai/api/train/configs/sft.py)
- [RL Config Schema](/Users/joshpurtell/Documents/GitHub/synth-ai/synth_ai/api/train/configs/rl.py)
- [Train CLI](/Users/joshpurtell/Documents/GitHub/synth-ai/synth_ai/api/train/cli.py)
- [Example Configs](https://github.com/synth-laboratories/cookbooks) — see `dev/`
