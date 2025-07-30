# DuckDB Race Condition Tests

This directory contains tests that reproduce and verify the fix for the database race condition issue that occurs in concurrent environments.

## The Problem

When running the Crafter rollout script (`run_rollouts_for_models_and_compare.py`), multiple episodes execute concurrently, each creating their own `SessionTracer` instance. This leads to a race condition in the database insertion logic:

1. **Multiple concurrent episodes** try to insert session traces into DuckDB simultaneously
2. **Race condition occurs** in `DuckDBTraceManager.insert_session_trace()` between the "check if exists" SELECT and the INSERT
3. **Duplicate key constraint violations** result when multiple connections try to insert the same session_id

The error looks like:
```
Warning: Could not link session episode_1_gpt-4o-mini_1753810377286054_6ec7dd85-1d38-4935-b3b0-08c364def464 to experiment after 3 attempts: Constraint Error: Duplicate key "session_id: ..." violates primary key constraint.
```

## Test Files

### `test_duckdb_race_conditions.py`
Comprehensive pytest test suite with multiple race condition scenarios:

- **`test_concurrent_session_insertion_race_condition`**: Basic concurrent session insertion test
- **`test_async_concurrent_session_insertion_race_condition`**: Async version that simulates the actual rollout script
- **`test_experiment_linking_race_condition`**: Tests the experiment linking logic specifically  
- **`test_high_concurrency_stress_test`**: High-stress test with many concurrent operations

### `run_race_condition_test.py`
Simple standalone test runner that provides a clear demonstration:

- **Minimal reproduction** of the exact race condition
- **Clear output** showing success/failure
- **Both sync and async** versions
- **No pytest dependencies** - can be run directly

## Running the Tests

### Option 1: Run pytest test suite
```bash
cd synth-ai/public_tests
python -m pytest test_duckdb_race_conditions.py -v -s
```

### Option 2: Run standalone test 
```bash
cd synth-ai/public_tests  
python run_race_condition_test.py
```

### Option 3: Run specific test
```bash
cd synth-ai/public_tests
python -m pytest test_duckdb_race_conditions.py::TestDuckDBRaceConditions::test_concurrent_session_insertion_race_condition -v -s
```

## Expected Behavior

### Before Fix (Current State)
**Tests will FAIL** with output like:
```
🚨 RACE CONDITION DETECTED!
   3 workers failed due to duplicate key constraints
   This is the same error occurring in the rollout script.
```

### After Fix  
**Tests will PASS** with output like:
```
✅ NO RACE CONDITION DETECTED!
   All workers completed successfully.
   The race condition fix is working correctly.
```

## Recommended Fixes

### Option A: Database Write Semaphore (Simplest)
Add a semaphore to serialize database writes in the rollout script:

```python
# At module level
DB_WRITE_SEMAPHORE = asyncio.Semaphore(1)

# In run_episode_async function
async with DB_WRITE_SEMAPHORE:
    trace_path = tracer.end_session()  # The INSERT happens here
```

### Option B: ON CONFLICT DO NOTHING (Cleaner)
Modify the database manager to handle conflicts gracefully:

```python
# In DuckDBTraceManager.insert_session_trace()
self.conn.execute(
    """
    INSERT INTO session_traces (session_id, created_at, num_timesteps,
                                num_events, num_messages, metadata)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(session_id) DO NOTHING
    """,
    [...]
)
```

## Test Design

These tests are designed to:

1. **Reproduce the exact conditions** that cause the race condition in the rollout script
2. **Force collisions** by having multiple workers use the same session IDs
3. **Detect the specific error** (duplicate key constraint violations)
4. **Verify the fix works** by ensuring no race conditions occur after implementation
5. **Provide clear feedback** about what went wrong and how to fix it

The tests simulate the concurrent episode execution pattern from `run_rollouts_for_models_and_compare.py` where multiple async coroutines create `SessionTracer` instances simultaneously.

## Integration with CI/CD

These tests can be integrated into continuous integration to:

- **Catch regressions** if the race condition is reintroduced
- **Verify fixes** work correctly across different environments
- **Document the expected behavior** for future developers

Mark tests with `@pytest.mark.slow` since they involve database operations and concurrency testing. 