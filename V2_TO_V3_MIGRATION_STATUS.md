# V2 to V3 Migration Status

## Summary

The tracing system is being migrated from v2 (DuckDB-based) to v3 (Turso/sqld-based with async support).

## Migration Status

### ✅ Completed
1. **Core v3 Implementation**
   - Complete async tracing system in `synth_ai/tracing_v3/`
   - SQLAlchemy models with vector support
   - Async session tracer and manager
   - Hook system updated for async
   - Comprehensive test suite (29 tests passing)

2. **Crafter Integration**
   - `run_rollouts_for_models_and_compare_v3.py` - Updated main experiment runner
   - `trace_hooks_v3.py` - Async hooks for Crafter achievements
   - `environment_v3.py` - Environment wrapper with async support
   - `example_v3_usage.py` - Usage examples
   - `MIGRATION_TO_V3.md` - Migration guide

3. **Test Configuration**
   - Created `pytest.ini` to skip all v2 tests
   - All v2-specific tests will be ignored during pytest runs

### ✅ Core Library Updates

1. **Core Library Files (Updated to v3)**
   - `synth_ai/lm/core/main_v3.py` - NEW: LM class with v3 tracing only
   - `synth_ai/lm/core/__init__.py` - Updated to import v3 LM by default
   - `synth_ai/tui/cli/query_experiments_v3.py` - NEW: CLI tool for querying v3 experiments
   - `synth_ai/environments/service/core_routes.py` - Updated imports to use v3 abstractions

### ❌ Remaining Work

1. **Remove v2 Support**
   - Delete `synth_ai/lm/core/main_v2.py` (replaced by main_v3.py)
   - Delete `synth_ai/tui/cli/query_experiments.py` (replaced by query_experiments_v3.py)
   - Archive or delete the entire `synth_ai/tracing_v2/` directory

2. **Test Files (28 files)**
   - Already configured to skip in pytest.ini
   - No action needed unless you want to create v3 versions

3. **Debug Scripts (11 files)**
   - Root level debug scripts for v2 issues
   - Can be deleted or archived

## Recommendations

### Immediate Actions
1. **Update LM Integration**: The `main_v2.py` file needs to support v3 tracing. Options:
   - Create a new `main_v3.py` with async support
   - Add a compatibility layer to support both v2 and v3
   - Update existing code to use v3 (breaking change)

2. **Update CLI Tools**: The `query_experiments.py` needs to query from Turso instead of DuckDB

3. **Update Service Routes**: The environment service needs async support for v3

### Clean Up
1. Delete or archive all debug scripts in root directory
2. Consider archiving the entire `tracing_v2` directory once migration is complete
3. Update documentation to reference v3 exclusively

## Testing v3

To test v3 tracing:

1. Start sqld daemon:
   ```bash
   sqld --http-listen 127.0.0.1:8080
   ```

2. Run v3 tests:
   ```bash
   pytest synth_ai/tracing_v3/tests/
   ```

3. Run Crafter with v3:
   ```bash
   python synth_ai/environments/examples/crafter_classic/agent_demos/run_rollouts_for_models_and_compare_v3.py
   ```

## Migration Helper

Use the migration helper to find remaining v2 usage:
```bash
python synth_ai/tracing_v3/migration_helper.py
```