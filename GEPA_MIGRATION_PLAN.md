# GEPA Clean → Nightly Migration Plan

**Created:** 2025-11-04  
**Source Branch:** `gepa-clean` (9 commits ahead of main)  
**Target Branch:** `nightly` (extensively refactored)  
**Diff Files:** `gepa_clean_diff.txt` (51,080 lines), `gepa_clean_summary.txt`

## Summary
374 files changed: +41,726 insertions, -3,644 deletions

## Core Changes to Port Over

### 1. **Prompt Learning Infrastructure** ⭐ PRIMARY FEATURE
Files to port:
- `synth_ai/learning/prompt_learning_client.py` (276 lines) - Client for querying results
- `synth_ai/learning/prompt_learning_types.py` (184 lines) - Dataclasses for results/events
- `synth_ai/api/train/configs/prompt_learning.py` (442 lines) - Config classes (GEPA/MIPRO)
- `synth_ai/api/train/validators.py` (277 lines) - Config validation

### 2. **Streaming Improvements** ⭐ MAJOR ENHANCEMENT
Files to port (check for conflicts with nightly's streaming refactor):
- `synth_ai/streaming/__init__.py` (29 lines)
- `synth_ai/streaming/config.py` (94 lines)
- `synth_ai/streaming/handlers.py` (551 lines) - Includes metric display in CLIHandler
- `synth_ai/streaming/streamer.py` (320 lines) - StreamEndpoints with metrics support
- `synth_ai/streaming/types.py` (95 lines)

### 3. **CLI Enhancements**
**Train command:**
- `synth_ai/api/train/cli.py` - Added prompt learning support with:
  - Results file saving
  - Nested config support
  - Metrics streaming
  - Constants for event types
  - Helper functions (e.g., `_format_text_replacements`)

**Deploy command fixes:**
- `synth_ai/cli/deploy/__init__.py` - Added `deploy_cmd` export
- `synth_ai/cli/deploy.py` - Added docstring and `__all__`
- `synth_ai/cli/task_app_deploy.py` - Fixed import with `# type: ignore[attr-defined]`

**Baseline command:**
- `synth_ai/cli/commands/baseline/core.py` - Fixed Click callback and command resolution

**Filter/Status/Pricing:**
- Type checking fixes
- Import path corrections
- Error handling improvements

### 4. **Configuration Updates**
- `synth_ai/api/train/config_finder.py` - Added prompt learning detection
- `synth_ai/api/train/configs/__init__.py` - Exported PromptLearningConfig
- `synth_ai/api/train/configs/shared.py` - Removed `slots=True` for compatibility

### 5. **Type Checking Fixes** ⭐ QUALITY IMPROVEMENTS
Multiple files with fixes for:
- Circular imports
- Missing type ignores
- Click callback typing
- Optional attribute access

### 6. **Unit Tests** ⭐ COMPREHENSIVE COVERAGE
New test files to port:
- `tests/unit/learning/test_prompt_learning_config_nested.py` (314 lines) - 16 tests
- `tests/unit/learning/test_prompt_learning_types.py` (439 lines) - 50+ tests  
- `tests/unit/learning/test_prompt_learning_validation.py` (1,269 lines) - 60+ tests
- `tests/unit/cli/test_prompt_learning_cli_helpers.py` (159 lines) - 15+ tests
- `tests/unit/cli/test_prompt_learning_cli.py` (370 lines) - CLI integration tests
- `tests/unit/cli/test_prompt_learning_config_finder.py` (228 lines)
- `tests/unit/cli/test_prompt_learning_stream_config.py` (170 lines)
- `tests/unit/streaming/test_prompt_learning_endpoints.py` (120 lines)
- `tests/unit/streaming/test_prompt_learning_metrics.py` (266 lines)

### 7. **GEPA Examples and Documentation**
- `examples/blog_posts/gepa/` directory (entire structure)
- `examples/task_apps/banking77/` - Banking77 task app
- `examples/task_apps/gepa_benchmarks/` - HotpotQA, HoVer, IFBench, PUPA

### 8. **Baseline System** (if not in nightly)
- `synth_ai/baseline/` - Complete baseline infrastructure
- `examples/baseline/` - Baseline scripts
- `tests/baseline/` - Baseline tests

### 9. **Miscellaneous Quality Fixes**
- `.gitignore` - Added job artifacts, backup files, TOML exclusions
- Import fixes with `contextlib.suppress`
- Consistent exception handling
- Removed bare `except:` clauses

## Migration Strategy

### Phase 1: Core Infrastructure (Do First)
1. Create new branch `gepa-from-nightly` from `origin/nightly`
2. Port prompt learning types and client (no dependencies)
3. Port streaming enhancements (check for conflicts with nightly's streaming)
4. Port config classes and validators

### Phase 2: CLI Integration
1. Port train command enhancements for prompt learning
2. Port deploy command fixes
3. Port type checking fixes across CLI files

### Phase 3: Tests
1. Port all unit tests for prompt learning
2. Port streaming tests
3. Verify all tests pass

### Phase 4: Examples and Documentation
1. Port GEPA examples
2. Port Banking77 task app
3. Port benchmark task apps

### Phase 5: Quality Fixes
1. Port miscellaneous type checking fixes
2. Update .gitignore
3. Port any remaining improvements

## Conflict Resolution Guidelines

### Files Likely to Have Significant Conflicts:
1. **`synth_ai/streaming/handlers.py`** - Both branches likely modified
   - Solution: Manually merge, keeping both sets of improvements
   
2. **`synth_ai/api/train/cli.py`** - Core train command
   - Solution: Carefully merge prompt learning additions with nightly's structure
   
3. **`synth_ai/cli/__init__.py`** - CLI registration
   - Solution: Ensure all commands are registered from both branches

### Files to Take Entirely from gepa-clean:
- All `tests/unit/learning/test_prompt_learning_*.py`
- All `synth_ai/learning/prompt_learning_*.py`
- All `examples/blog_posts/gepa/`
- All `examples/task_apps/banking77/`
- All `examples/task_apps/gepa_benchmarks/`

### Files to Take Entirely from nightly:
- Any new CLI commands not in gepa-clean
- Any new streaming abstractions if significantly different
- Updated documentation files

## Verification Checklist

After migration:
- [ ] All prompt learning unit tests pass (135+ tests)
- [ ] Streaming tests pass
- [ ] Type checking passes (`uvx ty check`)
- [ ] Linting passes (`uvx ruff check`)
- [ ] Can run GEPA training job end-to-end
- [ ] Can query prompt learning results
- [ ] Metrics stream correctly in terminal
- [ ] Results files save correctly
- [ ] MIPRO shows "not implemented" error

## Notes

- **MIPRO Status:** Marked as not implemented in gepa-clean
- **Nested Config:** gepa-clean supports both nested and flat GEPA TOML structures
- **Backwards Compatibility:** Config parsing has fallback logic for old formats
- **Deploy Fix:** Fixed missing `deploy_cmd` export that was causing import errors

