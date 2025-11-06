# SDK Multi-Stage GEPA Configuration - Complete ✅

## Summary

The SDK (`synth-ai`) now has full support for multi-stage GEPA module configuration, enabling type-safe configuration of pipeline-based prompt learning. The backend can now proceed with implementing the multi-stage GEPA optimizer.

---

## What Was Added

### 1. **`GEPAModuleConfig` Class**
Location: `synth_ai/api/train/configs/prompt_learning.py:167-187`

```python
class GEPAModuleConfig(ExtraModel):
    """Configuration for a single GEPA pipeline module/stage (instruction-only)."""
    module_id: str
    max_instruction_slots: int = 3
    allowed_tools: list[str] | None = None
    max_tokens: int | None = None
```

**Features:**
- ✅ `module_id` validation (non-empty, auto-stripped)
- ✅ `max_instruction_slots` validation (>= 1)
- ✅ Optional tool and token constraints per module
- ✅ Full Pydantic validation with helpful error messages

### 2. **Updated `GEPAConfig`**
Location: `synth_ai/api/train/configs/prompt_learning.py:190-207`

```python
class GEPAConfig(ExtraModel):
    # ... existing fields ...
    
    # Multi-stage pipeline support
    modules: list[GEPAModuleConfig] | None = None  # ✨ NEW
    
    # Nested subsections
    rollout: GEPARolloutConfig | None = None
    # ... rest ...
```

**Backwards Compatible:**
- `modules = None` → single-stage GEPA (existing behavior)
- `modules = [...]` → multi-stage GEPA (new behavior)

### 3. **Enhanced TOML Parsing**
Location: `synth_ai/api/train/configs/prompt_learning.py:389-426`

The `from_mapping` method now:
- ✅ Recognizes `modules` as nested config
- ✅ Validates each module config
- ✅ Preserves all other nested configs (rollout, evaluation, etc.)

### 4. **Comprehensive Tests**
Location: `tests/unit/api/train/configs/test_gepa_module_config.py`

**Test Coverage (10 tests, all passing):**
- ✅ Valid module configuration with all fields
- ✅ Module configuration with defaults
- ✅ Empty module_id validation
- ✅ Invalid max_instruction_slots validation
- ✅ Whitespace stripping
- ✅ Single-stage config (backwards compat)
- ✅ Multi-stage config with modules
- ✅ Loading from dict/TOML with modules
- ✅ Loading from dict/TOML without modules
- ✅ Modules recognized as nested data

### 5. **Example Configuration**
Location: `examples/gepa/multi_stage_gepa_example.toml`

A complete, working example of a 2-stage Banking77 pipeline configuration showing:
- Pipeline metadata with `pipeline_modules`
- Per-module configuration
- All standard GEPA sections (rollout, evaluation, mutation, etc.)

---

## TOML Structure

Users configure multi-stage GEPA like this:

```toml
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8000"

[prompt_learning.initial_prompt.metadata]
pipeline_modules = ["query_analyzer", "classifier"]

[prompt_learning.gepa]
env_name = "banking77_pipeline"

# Module-specific configuration (instruction-only)
[[prompt_learning.gepa.modules]]
module_id = "query_analyzer"
max_instruction_slots = 2
max_tokens = 512

[[prompt_learning.gepa.modules]]
module_id = "classifier"
max_instruction_slots = 3
max_tokens = 1024
allowed_tools = ["classify", "format_output"]

# Standard GEPA sections work as before
[prompt_learning.gepa.rollout]
budget = 1000
max_concurrent = 20

# ... other sections ...
```

---

## Backend Implementation Checklist

The backend can now proceed with these steps (from `multi_stage_gepa.md`):

### ✅ **Done (SDK Side)**
1. Add `GEPAModuleConfig` to SDK
2. Add `modules` field to `GEPAConfig`
3. Update TOML parsing (`from_mapping`)
4. Add validation logic
5. Create comprehensive tests
6. Provide example TOML

### 🔲 **TODO (Backend Side)**

#### 1. **Config Plumbing** (`backend/app/routes/prompt_learning/core/config.py`)
- [ ] Remove rejection at line ~735 for multi-module pipelines
- [ ] Add `GEPAModuleConfig` dataclass (mirror SDK structure)
- [ ] Assert every `pipeline_modules` entry has a corresponding `gepa.modules` entry
- [ ] Parse modules from TOML config

#### 2. **Optimizer Data Structures** (`backend/app/routes/prompt_learning/algorithm/gepa/optimizer.py`)
- [ ] Add `ModuleGene` dataclass:
  ```python
  @dataclass
  class ModuleGene:
      instruction_lines: list[str]
      rules: dict[str, Any] | None = None
      temperature: float | None = None
  ```
- [ ] Update `GEPAConfig` with `modules: dict[str, GEPAModuleConfig] | None`
- [ ] Update genome hashing for multi-module candidates
- [ ] Store `MultiStageCandidate = dict[str, ModuleGene]` in population

#### 3. **Program Assembly** (`backend/app/routes/prompt_learning/core/prompt_pipeline.py` - new file)
- [ ] Create `assemble_pipeline(genome) -> dict[str, PromptTemplate]`
- [ ] Create `build_stage_prompt_templates_dict(genome)`
- [ ] Reuse MiPRO's stage serialization helpers (lift to shared location)

#### 4. **Rollout Integration**
- [ ] Update `_execute_rollout_request` to accept `stage_prompt_templates`
- [ ] Add `policy_config["stage_prompt_templates"]` to payload
- [ ] Assert every stage has a template before evaluation

#### 5. **Genetic Operators** (module-aware)
- [ ] Update `_mutate_random_async` to select a module first
- [ ] Implement module-wise crossover
- [ ] Enforce per-module constraints (max_instruction_slots, allowed_tools, max_tokens)

#### 6. **Pattern Mode Gating**
- [ ] Reject `prompt_transformation` when `modules` is non-empty
- [ ] Add clear error message directing users to template mode

#### 7. **Testing**
- [ ] Unit tests: Config round-trip with modules
- [ ] Unit tests: Genome assembly produces correct stage templates
- [ ] Unit tests: Module-aware mutation targets correct stage
- [ ] Integration tests: End-to-end GEPA run with 2-stage pipeline

---

## Type Compatibility

The SDK `GEPAModuleConfig` exactly matches the backend spec:

| Field | Type | Default | Backend Match |
|-------|------|---------|---------------|
| `module_id` | `str` | (required) | ✅ |
| `max_instruction_slots` | `int` | `3` | ✅ |
| `allowed_tools` | `list[str] \| None` | `None` | ✅ |
| `max_tokens` | `int \| None` | `None` | ✅ |

**No Type Mismatches:** The backend can deserialize SDK configs directly.

---

## Validation Guarantees

The SDK validates these constraints **before** submission to the backend:

1. ✅ `module_id` is non-empty and trimmed
2. ✅ `max_instruction_slots >= 1`
3. ✅ All nested configs properly parsed
4. ✅ Pydantic type checking on all fields

This catches configuration errors early, at CLI submit time.

---

## Backwards Compatibility

**Single-stage GEPA still works:**
- If `modules` is omitted or `None`, GEPA behaves exactly as before
- Existing single-stage TOML configs continue to work
- No breaking changes to existing API

**Migration path:**
- Users can opt into multi-stage by adding `[[prompt_learning.gepa.modules]]` sections
- Backend can check `config.modules is None` to determine single vs. multi-stage

---

## Files Changed

### SDK (`synth-ai`)
1. **Modified:**
   - `synth_ai/api/train/configs/prompt_learning.py` (+31 lines)
     - Added `GEPAModuleConfig` class
     - Added `modules` field to `GEPAConfig`
     - Updated `from_mapping` to parse modules
     - Updated `__all__` exports

2. **Created:**
   - `examples/gepa/multi_stage_gepa_example.toml` (87 lines)
   - `tests/unit/api/train/configs/test_gepa_module_config.py` (123 lines)

### Backend (`monorepo`)
**Backend changes already complete:**
- ✅ Fixed `IndentationError` in MIPRO optimizer
- ✅ Fixed `SyntaxError` with mismatched else statement
- ✅ Pushed to `multistage` branch

---

## Git Status

### SDK (`synth-ai`)
- ✅ Committed: `a46444b` - "feat(gepa): add multi-stage module configuration support"
- ✅ Pushed to: `origin/multistage` (new branch)
- 📋 PR Ready: https://github.com/synth-laboratories/synth-ai/pull/new/multistage

### Backend (`monorepo`)
- ✅ Committed: `69c4dc4a` - "fix(mipro): fix syntax error with mismatched else statement"
- ✅ Pushed to: `origin/multistage`

---

## Next Steps for Backend Team

1. **Pull latest SDK**: The backend should pull the `multistage` branch from `synth-ai`
2. **Implement config parsing**: Start with `backend/app/routes/prompt_learning/core/config.py`
3. **Add module gene structures**: Implement `ModuleGene` dataclass in optimizer
4. **Follow the plan**: Use `multi_stage_gepa.md` as the implementation guide
5. **Test incrementally**: Run unit tests after each step

---

## Example Usage

Once backend implements multi-stage GEPA, users will submit like this:

```bash
# Submit multi-stage GEPA job
uv run synth-cli train prompt-learning \
  --config examples/gepa/multi_stage_gepa_example.toml \
  --job-name "banking77_pipeline_gepa_v1"
```

The SDK will:
1. ✅ Parse and validate the TOML config
2. ✅ Validate module configurations
3. ✅ Submit typed config to backend
4. ✅ Backend deserializes with full type safety

---

## Testing

Run SDK tests:

```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
uv run pytest tests/unit/api/train/configs/test_gepa_module_config.py -v
```

**Result:** 10/10 tests passing ✅

---

## Summary

✅ **SDK is ready** - Backend can now implement multi-stage GEPA with full type safety
✅ **Backwards compatible** - Single-stage GEPA unchanged
✅ **Well tested** - 10 comprehensive unit tests
✅ **Documented** - Example TOML and implementation plan
✅ **Pushed to remote** - Both repos synced on `multistage` branch

**Backend can proceed with implementation** following `multi_stage_gepa.md`.

