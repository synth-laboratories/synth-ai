# Synth AI SDK Refactor Implementation Plan

**Branch:** `sdk-refactor`  
**Base:** `synth-async-agent` (with `nightly` merged)  
**Spec:** See `refactor_spec.txt` for design decisions

---

## Overview

This plan breaks down the refactor into phases with concrete, atomic steps.
Each step should be a single commit that passes tests.

**Estimated Total Effort:** 2-3 weeks

---

## Pre-Refactor Checklist

- [x] Move `environments/` to synth-research (commit: `6927ab7`)
- [ ] Run baseline tests: `pytest tests/unit -q`
- [ ] Note any pre-existing test failures

---

## Phase 1: Create `data/` Layer (Days 1-2)

**Goal:** Establish pure data models with no IO dependencies.

### Step 1.1: Create `data/` directory structure

```bash
mkdir -p synth_ai/data
touch synth_ai/data/__init__.py
touch synth_ai/data/enums.py
touch synth_ai/data/jobs.py
touch synth_ai/data/traces.py
touch synth_ai/data/rewards.py
touch synth_ai/data/sft.py
touch synth_ai/data/judges.py
touch synth_ai/data/specs.py
```

**Commit:** `feat(data): create data layer directory structure`

### Step 1.2: Create `data/enums.py`

Define centralized enums:
- `JobType` (PROMPT_LEARNING, SFT, RL, GSPO, EVAL, RESEARCH_AGENT)
- `JobStatus` (PENDING, QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED)
- `PromptLearningMethod` (GEPA, MIPRO)
- `RLMethod` (PPO, GRPO, REINFORCE)
- `SFTMethod` (FULL, LORA, QLORA)
- `ResearchAgentAlgorithm` (SCAFFOLD_TUNING, EVALUATION, TRACE_ANALYSIS)
- `ContainerBackend` (DAYTONA, MODAL, DOCKER)

**Sources to reference:**
- `api/train/configs/prompt_learning.py` - existing method strings
- `api/research_agent/job.py` - AlgorithmType, BackendType

**Commit:** `feat(data): add centralized enums for job types and methods`

### Step 1.3: Create `data/jobs.py`

Define base job classes:
- `BaseJobConfig` (dataclass with model, timeout, metadata)
- `PromptLearningJobConfig(BaseJobConfig)`
- `SFTJobConfig(BaseJobConfig)`
- `RLJobConfig(BaseJobConfig)`
- `ResearchAgentJobConfig(BaseJobConfig)`
- `PollOutcome` (status, data, is_terminal, error)
- `JobInfo` (job_id, job_type, status, timestamps)
- `BaseJobResult` (job_id, status, error)

**Sources to reference:**
- `api/train/configs/prompt_learning.py`
- `api/train/configs/sft.py`
- `api/research_agent/job.py` - ResearchAgentJobConfig, PollOutcome
- `learning/jobs.py` - existing job types

**Commit:** `feat(data): add job config and result base classes`

### Step 1.4: Create `data/traces.py` (re-exports only)

```python
# data/traces.py
"""Re-export trace types from tracing_v3 (DO NOT modify tracing_v3)."""
from synth_ai.tracing_v3.abstractions import (
    SessionTrace,
    SessionTimeStep,
    BaseEvent,
    RuntimeEvent,
    EnvironmentEvent,
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    TimeRecord,
)

__all__ = [
    "SessionTrace", "SessionTimeStep", "BaseEvent", 
    "RuntimeEvent", "EnvironmentEvent", ...
]
```

**Commit:** `feat(data): add traces.py re-exports from tracing_v3`

### Step 1.5: Create `data/sft.py`

Move SFT dataclasses (pure data, no validation logic):
- `SFTMessage`
- `SFTToolCall`
- `SFTExample`

**Source:** `learning/sft/data.py` - extract dataclasses only

**Commit:** `feat(data): add SFT data schemas`

### Step 1.6: Create `data/judges.py`

Move judge schemas:
- `JudgeScoreRequest`
- `JudgeScoreResponse`
- `RewardJudgement`
- `Judgement`

**Sources:**
- `evals/types.py`
- `judge_schemas.py`

**Commit:** `feat(data): add judge schemas`

### Step 1.7: Create `data/specs.py`

Move spec dataclasses:

**Source:** `spec/dataclasses.py`

**Commit:** `feat(data): add spec dataclasses`

### Step 1.8: Update `data/__init__.py`

Export key types for convenience:
```python
from .enums import JobType, JobStatus, PromptLearningMethod, ...
from .jobs import BaseJobConfig, JobInfo, PollOutcome, ...
from .traces import SessionTrace, SessionTimeStep, ...
```

**Commit:** `feat(data): export all data types from data/__init__.py`

### Step 1.9: Run tests

```bash
pytest tests/unit -q
```

Fix any import errors introduced.

**Commit:** `fix(data): resolve import issues from data layer changes`

---

## Phase 2: Create `core/` Layer (Days 3-5)

**Goal:** Establish internal plumbing that SDK and CLI can share.

### Step 2.1: Create `core/` directory structure

```bash
mkdir -p synth_ai/core/config
mkdir -p synth_ai/core/storage
touch synth_ai/core/__init__.py
touch synth_ai/core/env.py
touch synth_ai/core/http.py
touch synth_ai/core/errors.py
touch synth_ai/core/logging.py
touch synth_ai/core/pricing.py
touch synth_ai/core/config/__init__.py
touch synth_ai/core/config/base.py
touch synth_ai/core/storage/__init__.py
```

**Commit:** `feat(core): create core layer directory structure`

### Step 2.2: Create `core/env.py`

Consolidate environment resolution:
- `resolve_api_key()` - get SYNTH_API_KEY
- `resolve_backend_url()` - get backend URL
- `resolve_env_key()` - get ENVIRONMENT_API_KEY
- `get_env_with_fallback(key, fallback_key, default)`

**Sources:**
- `utils/env.py`
- `utils/task_app_env.py`
- `api/train/env_resolver.py`

**Commit:** `feat(core): consolidate environment resolution in core/env.py`

### Step 2.3: Create `core/http.py`

Consolidate HTTP client:
- `SynthHTTPClient` - base client with auth, retries, timeout
- Headers, error handling

**Sources:**
- `learning/client.py`
- `utils/http.py`

**Commit:** `feat(core): consolidate HTTP client in core/http.py`

### Step 2.4: Create `core/errors.py`

Consolidate error types:
- `SynthError` (base)
- `ConfigError`
- `APIError`
- `ValidationError`
- `TaskAppError`

**Sources:**
- `task/errors.py`
- `utils/errors.py`

**Commit:** `feat(core): consolidate error types in core/errors.py`

### Step 2.5: Create `core/config/base.py`

TOML loading and config discovery:
- `load_toml_config(path)`
- `find_config_file(name, search_paths)`
- `validate_config_schema(config, schema)`

**Sources:**
- `utils/config.py`
- `utils/train_cfgs.py`

**Commit:** `feat(core): add config loading utilities`

### Step 2.6: Create `core/pricing.py`

Move pricing logic:

**Source:** `pricing/` directory

**Commit:** `feat(core): move pricing utilities to core`

### Step 2.7: Create `core/storage/`

Storage adapters:
- `BaseStorage` (abstract)
- `SQLiteStorage`
- `TursoStorage` (if needed)

**Sources:**
- `task/storage/`
- `utils/` storage utils

**Commit:** `feat(core): add storage adapters`

### Step 2.8: Run tests and fix imports

```bash
pytest tests/unit -q
```

**Commit:** `fix(core): resolve import issues from core layer changes`

---

## Phase 3: Create `sdk/` Layer (Days 6-10)

**Goal:** User-facing Python API.

### Step 3.1: Create `sdk/` directory structure

```bash
mkdir -p synth_ai/sdk/task_apps
mkdir -p synth_ai/sdk/training
mkdir -p synth_ai/sdk/tracing
mkdir -p synth_ai/sdk/judging
mkdir -p synth_ai/sdk/specs
mkdir -p synth_ai/sdk/research_agent
mkdir -p synth_ai/sdk/streaming
mkdir -p synth_ai/sdk/inference
mkdir -p synth_ai/sdk/data
touch synth_ai/sdk/__init__.py
```

**Commit:** `feat(sdk): create sdk layer directory structure`

### Step 3.2: Move task apps to `sdk/task_apps/`

Move and refactor:
- `task/in_process.py` → `sdk/task_apps/in_process.py`
- `task/server.py` → `sdk/task_apps/server.py`
- `task/client.py` → `sdk/task_apps/client.py`
- `modal.py` → `sdk/task_apps/modal.py`
- `cloudflare.py` → `sdk/task_apps/tunnels.py`
- `task/rubrics/` → `sdk/task_apps/rubrics/`

Update imports to use `core/` and `data/`.

**Commit:** `feat(sdk): move task apps to sdk/task_apps`

### Step 3.3: Create `task/__init__.py` compat layer

```python
# synth_ai/task/__init__.py
"""Backward compatibility - use synth_ai.sdk.task_apps instead."""
import warnings
from synth_ai.sdk.task_apps import *

def __getattr__(name):
    warnings.warn(
        f"synth_ai.task.{name} is deprecated, use synth_ai.sdk.task_apps.{name}",
        DeprecationWarning, stacklevel=2
    )
    from synth_ai.sdk import task_apps
    return getattr(task_apps, name)
```

**Commit:** `feat(sdk): add backward compat layer for synth_ai.task`

### Step 3.4: Move training to `sdk/training/`

Move and refactor:
- `api/train/prompt_learning.py` → `sdk/training/prompt_learning.py`
- `api/train/sft.py` → `sdk/training/sft.py`
- `learning/rl/client.py` → `sdk/training/rl.py`
- `api/train/pollers.py` → `sdk/training/_pollers.py`
- `api/train/builders.py` → `sdk/training/_builders.py`

Add result classes with methods:
- `PromptLearningResult.get_prompt_text(rank=1)`
- `SFTResult`
- `RLResult`

**Commit:** `feat(sdk): move training to sdk/training with result objects`

### Step 3.5: Create `api/train/__init__.py` compat layer

**Commit:** `feat(sdk): add backward compat layer for synth_ai.api.train`

### Step 3.6: Create `sdk/tracing/` (re-exports only)

```python
# sdk/tracing/__init__.py
"""Re-export from tracing_v3 - DO NOT modify tracing_v3."""
from synth_ai.tracing_v3 import *
```

**Commit:** `feat(sdk): add sdk/tracing re-exports`

### Step 3.7: Move judging to `sdk/judging/`

- `evals/client.py` → `sdk/judging/client.py`

**Commit:** `feat(sdk): move JudgeClient to sdk/judging`

### Step 3.8: Move specs to `sdk/specs/`

- `spec/loader.py` → `sdk/specs/loader.py`
- `spec/serializer.py` → `sdk/specs/serializer.py`
- `spec/validation.py` → `sdk/specs/validation.py`

**Commit:** `feat(sdk): move spec utilities to sdk/specs`

### Step 3.9: Move research agent to `sdk/research_agent/`

- `api/research_agent/job.py` → split:
  - Config → already in `data/jobs.py`
  - Job class → `sdk/research_agent/job.py`
  - Poller → `sdk/research_agent/poller.py`

**Commit:** `feat(sdk): move research agent to sdk/research_agent`

### Step 3.10: Move streaming to `sdk/streaming/`

- `streaming/` → `sdk/streaming/`

**Commit:** `feat(sdk): move streaming to sdk/streaming`

### Step 3.11: Move inference to `sdk/inference/`

- `inference/` → `sdk/inference/`

**Commit:** `feat(sdk): move inference client to sdk/inference`

### Step 3.12: Update `sdk/__init__.py`

Export main SDK classes:
```python
from .task_apps import InProcessTaskApp, TaskAppConfig, ...
from .training import PromptLearningJob, SFTJob, RLJob, ...
from .research_agent import ResearchAgentJob, ...
from .judging import JudgeClient
```

**Commit:** `feat(sdk): export main classes from sdk/__init__.py`

### Step 3.13: Update `synth_ai/__init__.py`

Re-export from sdk for top-level convenience:
```python
from .sdk import (
    InProcessTaskApp,
    PromptLearningJob,
    SFTJob,
    ...
)
```

**Commit:** `feat: update synth_ai/__init__.py to export from sdk`

### Step 3.14: Run tests and fix imports

```bash
pytest tests/unit -q
```

**Commit:** `fix(sdk): resolve import issues from sdk layer changes`

---

## Phase 4: Update CLI Layer (Days 11-13)

**Goal:** CLI uses SDK for business logic, keeps UX/local infra.

### Step 4.1: Update CLI imports

Update CLI commands to import from `sdk/` instead of old locations:
- `cli/commands/train.py` - use `sdk.training`
- `cli/commands/eval.py` - use `sdk.judging`
- `cli/commands/deploy.py` - use `sdk.task_apps`

**Commit:** `refactor(cli): update CLI to use sdk imports`

### Step 4.2: Move research agent CLI

- `api/research_agent/cli.py` → `cli/commands/agent.py`

**Commit:** `feat(cli): move research agent CLI to cli/commands/agent.py`

### Step 4.3: Organize CLI local infrastructure

Ensure clean separation:
- `cli/local/` - experiment queue, local DB
- `cli/session/` - session management

**Source:** `session/` → `cli/session/`

**Commit:** `refactor(cli): organize CLI local infrastructure`

### Step 4.4: Update CLI to use core/ for shared utilities

CLI should import from `core/` for:
- `core.env` - environment resolution
- `core.errors` - error types
- `core.logging` - logging setup
- `core.config` - config loading

**Commit:** `refactor(cli): use core utilities`

### Step 4.5: Run CLI smoke tests

```bash
uvx synth-ai --help
uvx synth-ai train --help
uvx synth-ai agent --help
```

**Commit:** `fix(cli): resolve CLI issues`

---

## Phase 5: Create Contracts Layer (Days 14-15)

**Goal:** Polyglot-friendly contracts at top level.

### Step 5.1: Create `contracts/` directory

```bash
mkdir -p synth_ai/contracts
touch synth_ai/contracts/__init__.py
touch synth_ai/contracts/task_app.py
touch synth_ai/contracts/rl.py
touch synth_ai/contracts/sft.py
```

**Commit:** `feat(contracts): create contracts layer`

### Step 5.2: Add task app contracts

Define request/response types for task app protocol.

**Commit:** `feat(contracts): add task app contracts`

### Step 5.3: Add RL contracts

Define RL training contracts.

**Commit:** `feat(contracts): add RL contracts`

### Step 5.4: Add SFT contracts

Define SFT training contracts.

**Commit:** `feat(contracts): add SFT contracts`

---

## Phase 6: Cleanup and Verification (Days 16-17)

### Step 6.1: Remove dead code

Identify and remove:
- Old module files that have been moved
- Unused imports
- Dead utilities

**Commit:** `chore: remove dead code and old module files`

### Step 6.2: Run full test suite

```bash
pytest tests/unit -q
pytest tests/ -q --ignore=tests/integration  # if integration tests exist
```

Fix any failures.

**Commit:** `fix: resolve remaining test failures`

### Step 6.3: Run linters

```bash
ruff check .
black . --check
```

Fix any issues.

**Commit:** `style: fix linting issues`

### Step 6.4: Update pyproject.toml

Update package metadata if needed:
- Entry points
- Package includes

**Commit:** `chore: update pyproject.toml for new structure`

### Step 6.5: Verify CLI still works

```bash
uvx synth-ai --version
uvx synth-ai demo --help
uvx synth-ai train --help
uvx synth-ai agent run --help
```

**Commit:** `fix: ensure CLI commands work`

---

## Phase 7: CLI Reorganization (Days 18-20)

**Goal:** Flatten 33 loose files in cli/ into logical subdirectories.

See `cli_reorg_proposal.md` for full details.

### Step 7.1: Create cli/agents/

Move agent integrations to shared directory with base class:
```bash
mkdir -p synth_ai/cli/agents
mv synth_ai/cli/claude.py synth_ai/cli/agents/
mv synth_ai/cli/codex.py synth_ai/cli/agents/
mv synth_ai/cli/opencode.py synth_ai/cli/agents/
# Create base.py with shared session logic
```

**Commit:** `refactor(cli): move agent integrations to cli/agents/`

### Step 7.2: Create cli/task_apps/

Consolidate 6 task app files:
```bash
mkdir -p synth_ai/cli/task_apps
mv synth_ai/cli/task_app.py synth_ai/cli/task_apps/
mv synth_ai/cli/task_app_serve.py synth_ai/cli/task_apps/serve.py
mv synth_ai/cli/task_app_deploy.py synth_ai/cli/task_apps/deploy.py
mv synth_ai/cli/task_app_list.py synth_ai/cli/task_apps/list.py
mv synth_ai/cli/task_app_modal_serve.py synth_ai/cli/task_apps/modal_serve.py
mv synth_ai/cli/task_apps.py synth_ai/cli/task_apps/commands.py
```

**Commit:** `refactor(cli): consolidate task app commands to cli/task_apps/`

### Step 7.3: Create cli/training/

Move training-related commands:
```bash
mkdir -p synth_ai/cli/training
mv synth_ai/cli/train.py synth_ai/cli/training/
mv synth_ai/cli/train_cfg.py synth_ai/cli/training/
mv synth_ai/cli/watch.py synth_ai/cli/training/
```

**Commit:** `refactor(cli): move training commands to cli/training/`

### Step 7.4: Create cli/infra/

Move infrastructure commands:
```bash
mkdir -p synth_ai/cli/infra
mv synth_ai/cli/balance.py synth_ai/cli/infra/
mv synth_ai/cli/setup.py synth_ai/cli/infra/
mv synth_ai/cli/status.py synth_ai/cli/infra/
mv synth_ai/cli/turso.py synth_ai/cli/infra/
mv synth_ai/cli/mcp.py synth_ai/cli/infra/
mv synth_ai/cli/modal_app.py synth_ai/cli/infra/
```

**Commit:** `refactor(cli): move infra commands to cli/infra/`

### Step 7.5: Create cli/demos/

Move demo commands:
```bash
mkdir -p synth_ai/cli/demos
mv synth_ai/cli/demo.py synth_ai/cli/demos/
mv synth_ai/cli/rl_demo.py synth_ai/cli/demos/
```

**Commit:** `refactor(cli): move demo commands to cli/demos/`

### Step 7.6: Create cli/_internal/

Move internal utilities:
```bash
mkdir -p synth_ai/cli/_internal
mv synth_ai/cli/_modal_wrapper.py synth_ai/cli/_internal/modal_wrapper.py
mv synth_ai/cli/_storage.py synth_ai/cli/_internal/storage.py
mv synth_ai/cli/_typer_patch.py synth_ai/cli/_internal/typer_patch.py
mv synth_ai/cli/_validate_task_app.py synth_ai/cli/_internal/validate_task_app.py
```

**Commit:** `refactor(cli): move internal utilities to cli/_internal/`

### Step 7.7: Create cli/utils/

Move CLI utility commands:
```bash
mkdir -p synth_ai/cli/utils
mv synth_ai/cli/recent.py synth_ai/cli/utils/
mv synth_ai/cli/traces.py synth_ai/cli/utils/
mv synth_ai/cli/queue.py synth_ai/cli/utils/
mv synth_ai/cli/experiments.py synth_ai/cli/utils/
```

**Commit:** `refactor(cli): move utility commands to cli/utils/`

### Step 7.8: Delete duplicate CLI directories

Remove duplicates, keep `commands/` versions:
```bash
rm -rf synth_ai/cli/eval/      # Use cli/commands/eval/
rm -rf synth_ai/cli/train/     # Use cli/commands/train/ (different from cli/training/)
rm -rf synth_ai/cli/filter/    # Use cli/commands/filter/
# Move serve/ and modal_serve/ into task_apps/
```

**Commit:** `refactor(cli): remove duplicate CLI directories`

### Step 7.9: Update cli/__init__.py and root.py

Update imports to use new locations. Add re-exports for backward compat.

**Commit:** `refactor(cli): update CLI imports for new structure`

### Step 7.10: Test CLI commands

```bash
uvx synth-ai --help
uvx synth-ai train --help
uvx synth-ai demo --help
uvx synth-ai claude --help
```

**Commit:** `fix(cli): ensure CLI commands work after reorg`

---

## Phase 8: Utils and Deprecation Cleanup (Days 21-22)

**Goal:** Clean up utils/ duplicates and add deprecation warnings.

### Step 8.1: Add deprecation warnings to utils/ duplicates

Files that duplicate core/:
- `utils/env.py` → deprecate, re-export from core/env.py
- `utils/http.py` → deprecate, re-export from core/http.py
- `utils/errors.py` → deprecate, re-export from core/errors.py
- `utils/logging.py` → deprecate, re-export from core/logging.py
- `utils/base_url.py` → already deprecated

**Commit:** `chore(utils): add deprecation warnings for utils duplicates`

### Step 8.2: Move contracts

Move contract definitions to contracts/:
- `task/contracts.py` → `contracts/task_app.py`
- `learning/rl/contracts.py` → `contracts/rl.py`

**Commit:** `feat(contracts): consolidate contracts from task/ and learning/`

### Step 8.3: Move jobs/client.py

Move to sdk/:
- `jobs/client.py` → `sdk/jobs/client.py`

**Commit:** `feat(sdk): move jobs client to sdk/jobs/`

### Step 8.4: Clean up dead code

Remove any remaining dead code identified in potpourri.

**Commit:** `chore: remove dead code`

---

## Phase 9: Documentation and Downstream Updates (Days 23-25)

### Step 9.1: Add deprecation warnings

Ensure all backward compat layers emit `DeprecationWarning`.

### Step 9.2: Update cookbooks

Update `cookbooks` repo to use new import paths where beneficial.
Fix broken imports tracked in `cookbooks/BROKEN_IMPORTS.md`.

### Step 9.3: Update synth-research

Update `synth-research` repo imports if needed.

### Step 9.4: Update README

Document new architecture and import paths.

### Step 9.5: Version bump

Update version to 0.3.0 in pyproject.toml.

---

## Final Directory Structure

```
synth_ai/
├── __init__.py              # Re-exports from sdk/
├── py.typed
│
├── data/                    # ✅ DONE - Pure data models
│   ├── __init__.py
│   ├── enums.py             # JobType, JobStatus, SynthModelName, etc.
│   ├── traces.py            # Re-exports from tracing_v3/
│   ├── rewards.py           # RewardRecord dataclasses
│   └── specs.py             # Re-exports from spec/
│
├── contracts/               # ✅ DONE - Polyglot contracts
│   ├── __init__.py
│   ├── README.md
│   ├── task_app.py          # TODO: move from task/contracts.py
│   ├── rl.py                # TODO: move from learning/rl/contracts.py
│   └── sft.py               # TODO: add
│
├── core/                    # ✅ DONE - Internal plumbing
│   ├── __init__.py
│   ├── env.py               # Environment resolution + get_backend_from_env
│   ├── http.py              # AsyncHttpClient
│   ├── errors.py            # Error types
│   ├── logging.py           # Logging setup
│   ├── pricing.py           # Model pricing
│   ├── config/base.py       # Config base classes
│   ├── storage/             # Storage interfaces
│   └── integrations/mcp/    # MCP placeholder
│
├── sdk/                     # ✅ DONE - User-facing API
│   ├── __init__.py
│   ├── task_apps/           # Re-exports from task/
│   ├── training/            # Re-exports from api/train/
│   ├── tracing/             # Re-exports from tracing_v3/
│   ├── judging/             # Re-exports from evals/
│   ├── specs/               # Re-exports from spec/
│   ├── research_agent/      # Re-exports from research_agent/
│   ├── streaming/           # Re-exports from streaming/
│   └── inference/           # Re-exports from inference/
│
├── cli/                     # 🔄 Phase 7 - Reorganize
│   ├── __init__.py
│   ├── root.py
│   ├── agents/              # NEW: claude, codex, opencode + base.py
│   ├── task_apps/           # NEW: serve, deploy, list, modal_serve
│   ├── training/            # NEW: train, train_cfg, watch
│   ├── demos/               # NEW: demo, rl_demo
│   ├── infra/               # NEW: balance, setup, status, turso, mcp, modal_app
│   ├── utils/               # NEW: recent, traces, queue, experiments
│   ├── _internal/           # NEW: modal_wrapper, storage, typer_patch
│   ├── commands/            # KEEP: artifacts/, baseline/, eval/, etc.
│   ├── lib/                 # KEEP: task_app_discovery, task_app_env
│   └── local/               # ✅ DONE: session/, experiment_queue/
│
├── tracing_v3/              # ✋ UNCHANGED - Don't touch!
├── task/                    # Compat layer → sdk/task_apps
├── learning/                # Compat layer → sdk/training
├── api/train/               # Compat layer → sdk/training
├── utils/                   # 📦 Deprecating duplicates
├── demos/                   # ✋ KEEP - Demo templates
├── baseline/                # ✋ KEEP - Baseline evaluation
└── ...                      # Other legacy modules
```

---

## Success Criteria

- [x] All unit tests pass (Phase 1-6)
- [x] CLI commands work (`synth-ai train`, `synth-ai agent`, etc.)
- [x] Old imports still work (with deprecation warnings)
- [x] New imports work: `from synth_ai.sdk.task_apps import InProcessTaskApp`
- [x] `tracing_v3/` is untouched
- [x] No circular imports
- [ ] CLI reorganized (Phase 7)
- [ ] Utils deprecated (Phase 8)
- [ ] Documentation updated (Phase 9)
- [ ] Version 0.3.0 released

---

## Progress Tracker

### ✅ Completed
- [x] Phase 1: data/ layer (enums, traces, rewards, specs)
- [x] Phase 2: core/ layer (env, http, errors, logging, pricing, config, storage)
- [x] Phase 3: sdk/ layer (training, task_apps, judging, inference, streaming, specs, research_agent, tracing)
- [x] Phase 4: cli/local/ (session/, experiment_queue/)
- [x] Phase 5: contracts/README.md
- [x] Phase 6: Tests pass, backward compat verified
- [x] Deleted types.py (replaced by data/enums.py)
- [x] Deleted calc.py (random calculator - wtf)
- [x] config/base_url.py → re-exports from core/env.py

### 🔄 In Progress
- [ ] Phase 7: CLI reorganization (agents/, task_apps/, training/, etc.)

### 📋 Pending
- [ ] Phase 8: Utils deprecation, contracts consolidation
- [ ] Phase 9: Documentation, downstream updates, version bump

---

## Rollback Plan

If issues arise:
1. All changes are in `sdk-refactor` branch
2. `nightly` branch is untouched
3. Can revert individual commits
4. Compat layers mean old code keeps working

---

## Notes

- **Don't be dogmatic** - if something doesn't fit cleanly, use pragmatic judgment
- **Atomic commits** - each step should be a working state
- **Test frequently** - run tests after each phase
- **Preserve tracing_v3** - DO NOT modify the tracing format

