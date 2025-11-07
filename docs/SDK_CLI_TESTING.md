# SDK and CLI Testing Summary

This document summarizes the test coverage for prompt learning and SFT SDK/CLI.

## Test Structure

### Unit Tests (`tests/unit/api/train/`)

#### Prompt Learning SDK (`test_prompt_learning_sdk.py`)
- ✅ `PromptLearningJobConfig` validation
  - Missing config file
  - Missing backend URL
  - Missing API key
  - Auto-resolve task app key from env
  - Explicit task app key
  
- ✅ `PromptLearningJob` class methods
  - `from_config()` with missing API key
  - `from_config()` resolves env vars
  - `from_job_id()` creates job for existing ID
  - `submit()` requires config for new jobs
  - `submit()` raises if already submitted
  - `get_status()` requires submission
  - `poll_until_complete()` requires submission
  - `get_results()` requires submission

- ✅ `PromptLearningJobPoller` basic functionality

#### SFT SDK (`test_sft_sdk.py`)
- ✅ `SFTJobConfig` validation
  - Missing config file
  - Missing backend URL
  - Missing API key
  
- ✅ `SFTJob` class methods
  - `from_config()` with missing API key
  - `from_config()` resolves env vars
  - `from_job_id()` creates job for existing ID
  - `submit()` requires config for new jobs
  - `submit()` raises if already submitted
  - `get_status()` requires submission
  - `poll_until_complete()` requires submission

### Integration Tests (`tests/integration/api/train/`)

#### Prompt Learning SDK/CLI (`test_prompt_learning_sdk_cli.py`)
- ✅ SDK: `from_config()` creates job instance
- ✅ SDK: `from_job_id()` resumes job
- ✅ SDK: `submit()` to real backend (with API keys)
- ✅ CLI: Help command works
- ✅ CLI: Invalid config validation
- ✅ CLI: Valid config dry run

#### SFT SDK/CLI (`test_sft_sdk_cli.py`)
- ✅ SDK: `from_config()` creates job instance
- ✅ SDK: `from_job_id()` resumes job
- ✅ CLI: Help command works

### Existing Integration Tests (`tests/integration/cli/`)

#### Prompt Learning CLI
- ✅ `test_cli_train_gepa_banking77.py` - GEPA CLI integration
- ✅ `test_cli_train_mipro_banking77.py` - MIPRO CLI integration
- ✅ `test_cli_prompt_learning_shell_scripts.py` - Shell script conversion tests

#### SFT CLI
- ✅ `test_cli_train_sft_fft_and_qlora.py` - SFT CLI integration

## Test Coverage Summary

### Prompt Learning
- ✅ Unit tests: Config validation, job creation, method guards
- ✅ Integration tests: SDK usage, CLI validation
- ✅ Existing: Full CLI integration tests

### SFT
- ✅ Unit tests: Config validation, job creation, method guards
- ✅ Integration tests: SDK usage, CLI validation
- ✅ Existing: Full CLI integration tests

## Running Tests

```bash
# Run all unit tests
uv run pytest tests/unit/api/train/ -v

# Run all integration tests
uv run pytest tests/integration/api/train/ -v

# Run specific test file
uv run pytest tests/unit/api/train/test_prompt_learning_sdk.py -v

# Run with coverage
uv run pytest tests/unit/api/train/ tests/integration/api/train/ --cov=synth_ai.api.train --cov-report=html
```

## Test Requirements

- Unit tests: No external dependencies (use mocks)
- Integration tests: May require API keys (skip if not available)
- CLI tests: May require backend connectivity (skip if not available)


