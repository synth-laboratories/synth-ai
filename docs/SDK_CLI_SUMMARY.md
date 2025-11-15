# SDK and CLI Support Summary

## Overview

Both Prompt Learning (MIPRO/GEPA) and SFT now have:
1. ✅ **First-class CLI support** via `uvx synth-ai train`
2. ✅ **High-level SDK classes** for programmatic usage
3. ✅ **Comprehensive unit and integration tests**

## Prompt Learning SDK

### Classes

- **`PromptLearningJob`**: Main SDK class
  - `from_config()`: Create from TOML config
  - `from_job_id()`: Resume existing job
  - `submit()`: Submit job to backend
  - `get_status()`: Get current status
  - `poll_until_complete()`: Poll until terminal state
  - `get_results()`: Get prompts, scores, candidates
  - `get_best_prompt_text()`: Get prompt text by rank

- **`PromptLearningJobConfig`**: Configuration dataclass
- **`PromptLearningJobPoller`**: Low-level poller

### Usage Example

```python
from synth_ai.api.train import PromptLearningJob

job = PromptLearningJob.from_config("my_config.toml")
job.submit()
result = job.poll_until_complete()
print(f"Best score: {result['best_score']}")
```

## SFT SDK

### Classes

- **`SFTJob`**: Main SDK class
  - `from_config()`: Create from TOML config
  - `from_job_id()`: Resume existing job
  - `submit()`: Submit job (handles file uploads)
  - `get_status()`: Get current status
  - `poll_until_complete()`: Poll until terminal state
  - `get_fine_tuned_model()`: Get fine-tuned model ID

- **`SFTJobConfig`**: Configuration dataclass

### Usage Example

```python
from synth_ai.api.train import SFTJob

job = SFTJob.from_config("my_sft_config.toml")
job.submit()
result = job.poll_until_complete()
print(f"Fine-tuned model: {result.get('fine_tuned_model')}")
```

## CLI Usage

Both work identically via CLI:

```bash
# Prompt Learning
uvx synth-ai train --type prompt_learning --config my_config.toml --poll

# SFT
uvx synth-ai train --type sft --config my_config.toml --poll
```

## Test Coverage

### Unit Tests (`tests/unit/api/train/`)
- ✅ `test_prompt_learning_sdk.py`: 14 tests covering config validation, job creation, method guards
- ✅ `test_sft_sdk.py`: 10 tests covering config validation, job creation, method guards

### Integration Tests (`tests/integration/api/train/`)
- ✅ `test_prompt_learning_sdk_cli.py`: 6 tests covering SDK usage and CLI validation
- ✅ `test_sft_sdk_cli.py`: 3 tests covering SDK usage and CLI validation

### Existing Integration Tests
- ✅ `test_cli_train_gepa_banking77.py`: Full GEPA CLI integration
- ✅ `test_cli_train_mipro_banking77.py`: Full MIPRO CLI integration
- ✅ `test_cli_train_sft_fft_and_qlora.py`: Full SFT CLI integration

## Files Created/Modified

### SDK Classes
- ✅ `synth_ai/api/train/prompt_learning.py` - Prompt Learning SDK
- ✅ `synth_ai/api/train/sft.py` - SFT SDK
- ✅ `synth_ai/api/train/pollers.py` - Added `PromptLearningJobPoller`
- ✅ `synth_ai/api/train/__init__.py` - Exports SDK classes

### Tests
- ✅ `tests/unit/api/train/test_prompt_learning_sdk.py` - Unit tests
- ✅ `tests/unit/api/train/test_sft_sdk.py` - Unit tests
- ✅ `tests/integration/api/train/test_prompt_learning_sdk_cli.py` - Integration tests
- ✅ `tests/integration/api/train/test_sft_sdk_cli.py` - Integration tests

### Documentation
- ✅ `docs/PROMPT_LEARNING_SDK_CLI.md` - Usage guide
- ✅ `docs/SDK_CLI_TESTING.md` - Test coverage summary
- ✅ `examples/sdk_prompt_learning_example.py` - Example script

## Running Tests

```bash
# All unit tests
uv run pytest tests/unit/api/train/ -v

# All integration tests
uv run pytest tests/integration/api/train/ -v

# Specific test file
uv run pytest tests/unit/api/train/test_prompt_learning_sdk.py -v
```

## Key Features

1. **Unified API**: Both SDKs follow the same pattern (`from_config()`, `submit()`, `poll_until_complete()`)
2. **Environment Resolution**: Automatically resolves backend URL and API keys from environment
3. **Error Handling**: Clear error messages for common mistakes
4. **Type Safety**: Full type hints and validation
5. **Test Coverage**: Comprehensive unit and integration tests


