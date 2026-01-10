# RolloutResponse Deprecated Fields Removal Plan

## Overview

This document outlines the plan to remove deprecated fields from `RolloutResponse` in synth-ai.
These fields are being removed as part of schema simplification (see monorepo/rollout.txt).

**Target Fields for Removal:**
- `pipeline_metadata` (dict)
- `branches` (dict)
- `aborted` (bool)

---

## 1. `pipeline_metadata` - CRITICAL (Many Consumers)

### Current Usage in synth-ai
- `synth_ai/sdk/task/contracts.py:260` - Field definition
- `synth_ai/cli/commands/smoke/core.py:924` - Reads `pipeline_metadata.inference_url`
- `synth_ai/sdk/task/trace_correlation_helpers.py:237-249` - Writes `trace_correlation_id` to `pipeline_metadata`
- `synth_ai/sdk/task/validators.py:63-80` - Validates `pipeline_metadata.inference_url`

### Downstream Consumers in Monorepo (CRITICAL)

#### GSPO Training (`clustered_trainer.py`)
| Line | Usage | Field Accessed |
|------|-------|----------------|
| 163 | `pipeline_meta = payload.get("pipeline_metadata")` | General access |
| 285 | `pipeline_meta = payload.get("pipeline_metadata")` | `inference_url` for hydration |
| 1936 | `pipeline_meta = rollout_data.get("pipeline_metadata")` | General access |
| 2830 | `pipeline_meta = response.get("pipeline_metadata")` | General access |
| 3368 | `pipeline_meta = batch_data.get("pipeline_metadata")` | General access |
| 4686-4694 | Sets `pipeline_metadata` on result | `trace_correlation_id` |
| 5764-5784 | Sets `pipeline_metadata` | `trace_correlation_id`, `inference_url` |

#### GSPO Pipeline (`coordinator.py`, `trace_hydrator.py`)
| File | Line | Usage |
|------|------|-------|
| coordinator.py | 909 | `meta = payload.setdefault("pipeline_metadata", {})` |
| trace_hydrator.py | 91 | `pipeline_meta = payload.setdefault("pipeline_metadata", {})` |

#### Prompt Learning - Trace Utils (`trace_utils.py`)
| Line | Usage | Field Accessed |
|------|-------|----------------|
| 123 | `pipeline_meta = response.get("pipeline_metadata")` | General |
| 160-167 | Fallback for `inference_url` extraction | `inference_url` |

#### Prompt Learning - Online Jobs (`online_jobs.py`)
| Line | Usage | Field Accessed |
|------|-------|----------------|
| 377-379 | Extract model info | `model` |

#### Prompt Learning - GEPA (`rollout_normalizer.py`, `optimizer.py`)
| File | Line | Field Accessed |
|------|------|----------------|
| rollout_normalizer.py | 989-992 | `output_mode` |
| optimizer.py | 7095-7097 | `trace_correlation_id` |
| optimizer.py | 10893-10898 | Logging `pipeline_metadata` |

#### Prompt Learning - MIPRO (`optimizer.py`)
| Line | Usage | Field Accessed |
|------|-------|----------------|
| 3024 | `pipeline_meta = resp.get("pipeline_metadata")` | `trace_correlation_id` |
| 4001 | `pipeline_meta = resp.get("pipeline_metadata")` | `trace_correlation_id` |
| 6918 | `pipeline_meta = response.get("pipeline_metadata", {})` | General |

#### Eval Service (`job_service.py`)
| Line | Usage | Field Accessed |
|------|-------|----------------|
| 186-190 | Extract model info | `model` |

#### Task Apps (Various)
- `backend/benchmarks/langprobe/task_apps/banking77/banking77_task_app.py` - Lines 179-181, 1236-1248
- `backend/app/routes/graphgen/task_app.py` - Line 812
- `backend/app/routes/context_learning/xbow_task_app.py` - Line 833
- `backend/app/routes/context_learning/swebench_task_app.py` - Line 1568
- `agora_ex/task_app.py` - Lines 442-456

### Fields Used from `pipeline_metadata`

| Field | Used By | Replacement |
|-------|---------|-------------|
| `trace_correlation_id` | GSPO, GEPA, MIPRO | Top-level `trace_correlation_id` (already exists) |
| `inference_url` | GSPO, trace_utils | Top-level `inference_url` (already exists) |
| `model` | online_jobs, job_service | Top-level `model` (NEW - needs to be added) |
| `output_mode` | rollout_normalizer | Top-level `output_mode` (NEW - needs to be added) |

### Migration Plan for `pipeline_metadata`

**Phase 1: Add top-level fields to synth-ai (if not present)**
- [ ] Ensure `trace_correlation_id` is top-level (DONE)
- [ ] Ensure `inference_url` is top-level (DONE)
- [ ] Add `model` as optional top-level field
- [ ] Add `output_mode` as optional top-level field

**Phase 2: Update monorepo to prefer top-level fields**
- [ ] Update `clustered_trainer.py` to check top-level first, fallback to `pipeline_metadata`
- [ ] Update `trace_utils.py` to check top-level first
- [ ] Update `online_jobs.py` to check top-level first
- [ ] Update `rollout_normalizer.py` to check top-level first
- [ ] Update `optimizer.py` (GEPA) to check top-level first
- [ ] Update `optimizer.py` (MIPRO) to check top-level first
- [ ] Update `job_service.py` to check top-level first

**Phase 3: Deprecate `pipeline_metadata` in synth-ai**
- [ ] Mark field as deprecated in docstring
- [ ] Keep populating for backward compatibility
- [ ] Add deprecation warning in validators

**Phase 4: Remove `pipeline_metadata` from synth-ai**
- [ ] Remove field from `RolloutResponse`
- [ ] Remove from `trace_correlation_helpers.py`
- [ ] Remove from `validators.py`
- [ ] Update tests

---

## 2. `aborted` - LOW IMPACT (Few Consumers)

### Current Usage in synth-ai
- `synth_ai/sdk/task/contracts.py:256-259` - Field definition (default=False)
- `synth_ai/sdk/localapi/rollouts.py:38,64` - Parameter in builder

### Downstream Consumers in Monorepo

| File | Line | Usage |
|------|------|-------|
| clustered_trainer.py | 3883-3884 | `if response.get("aborted"): outcome_meta["aborted"] = bool(...)` |
| trace_hydration.py | 114 | `if response.get("aborted"): # skip hydration` |
| test_exact_failing_case.py | 85 | Test fixture `"aborted": False` |
| test_rollout_normalizer_banking77.py | 81 | Test fixture `"aborted": False` |

### Replacement Strategy
- Move to `trace.metadata.rollout_status` with values: `"completed"`, `"aborted"`, `"error"`
- Or simply derive from trace metadata presence

### Migration Plan for `aborted`

**Phase 1: Update monorepo**
- [ ] Update `clustered_trainer.py:3883` to check `trace.metadata.rollout_status` first
- [ ] Update `trace_hydration.py:114` to check `trace.metadata.rollout_status` first
- [ ] Update test fixtures

**Phase 2: Remove from synth-ai**
- [ ] Remove field from `RolloutResponse`
- [ ] Remove from `RolloutResponseBuilder`
- [ ] Update tests

---

## 3. `branches` - MINIMAL IMPACT (Almost No Consumers)

### Current Usage in synth-ai
- `synth_ai/sdk/task/contracts.py:252-255` - Field definition (default_factory=dict)

### Downstream Consumers in Monorepo

| File | Line | Usage |
|------|------|-------|
| test_exact_failing_case.py | 73 | Test fixture `"branches": {}` |
| test_rollout_normalizer_banking77.py | 69 | Test fixture `"branches": {}` |

**Note:** The `github/routes.py:649` match is unrelated (GitHub branch caching).

### Migration Plan for `branches`

**Phase 1: Remove from synth-ai** (Low risk - only test fixtures use it)
- [ ] Remove field from `RolloutResponse`
- [ ] Update synth-ai tests

**Phase 2: Update monorepo test fixtures**
- [ ] Remove `"branches": {}` from test fixtures (optional - will be ignored)

---

## Recommended Execution Order

### Immediate (Can do now - low risk)
1. Remove `branches` from synth-ai (only test fixtures affected)

### Short-term (Requires monorepo coordination)
2. Remove `aborted` - update 2 files in monorepo first

### Medium-term (Requires significant monorepo refactor)
3. Deprecate then remove `pipeline_metadata`:
   - Add any missing top-level fields (`model`, `output_mode`)
   - Update ~10 monorepo files to prefer top-level
   - Keep `pipeline_metadata` for backward compat during transition
   - Remove after monorepo fully migrated

---

## Files to Modify (Summary)

### synth-ai
- `synth_ai/sdk/task/contracts.py` - Remove fields
- `synth_ai/sdk/localapi/rollouts.py` - Remove `aborted` param
- `synth_ai/sdk/task/trace_correlation_helpers.py` - Remove `pipeline_metadata` writes
- `synth_ai/sdk/task/validators.py` - Remove `pipeline_metadata` validation
- `synth_ai/cli/commands/smoke/core.py` - Use top-level `inference_url`
- Tests - Update fixtures

### monorepo (for `pipeline_metadata`)
- `backend/app/routes/clustered_training/core/algorithms/gspo/training/clustered_trainer.py`
- `backend/app/routes/clustered_training/core/algorithms/gspo/training/pipeline/coordinator.py`
- `backend/app/routes/clustered_training/core/algorithms/gspo/training/pipeline/trace_hydrator.py`
- `backend/app/routes/prompt_learning/core/trace_utils.py`
- `backend/app/routes/prompt_learning/online_jobs.py`
- `backend/app/routes/prompt_learning/algorithm/gepa/rollout_normalizer.py`
- `backend/app/routes/prompt_learning/algorithm/gepa/optimizer.py`
- `backend/app/routes/prompt_learning/algorithm/mipro/optimizer/optimizer.py`
- `backend/app/routes/eval/job_service.py`
- Various task apps

### monorepo (for `aborted`)
- `backend/app/routes/clustered_training/core/algorithms/gspo/training/clustered_trainer.py`
- `backend/app/routes/prompt_learning/core/trace_hydration.py`
- Test fixtures
