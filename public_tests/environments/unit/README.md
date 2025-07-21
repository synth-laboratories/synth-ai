# Unit Tests

This directory contains unit tests for the Environments service.

## Test Files

### `test_crafter_api_bug.py`

**Bug Fix Validation Test**

This test validates the fix for a critical bug in the Crafter environment API where JSON tool_calls were not properly converted to `EnvToolCall` objects.

**Original Bug:**
- Error: `"Processed call is not EnvToolCall: <class 'dict'>"`
- Cause: API received JSON dicts but environment expected `EnvToolCall` objects
- Status: ✅ **FIXED**

**Fix Applied:**
- Modified `core_routes.py` to convert dict tool_calls to `EnvToolCall` objects
- Added proper type conversion in the `/env/{env_name}/step` endpoint

**Test Coverage:**
- ✅ Basic action step with single tool_call
- ✅ Multiple tool_calls handling
- ✅ `EnvToolCall` object creation from dict
- ✅ Edge cases and validation

**How to Run:**
```bash
# Run just the Crafter API bug fix test
python tests/unit/test_crafter_api_bug.py

# Or with pytest
pytest tests/unit/test_crafter_api_bug.py -v

# Run all unit tests
pytest tests/unit/ -v
```

**Requirements:**
- Environment service must be running on `localhost:8001`
- CrafterClassic environment must be available
- Dependencies: `pytest`, `httpx`, `asyncio`

**Test Scenarios:**
1. **API Compatibility Test**: Verifies JSON tool_calls are accepted and processed correctly
2. **Response Validation**: Checks that responses contain expected Crafter observation data
3. **Edge Case Handling**: Tests multiple tool_calls and validation scenarios
4. **Unit Tests**: Direct testing of `EnvToolCall` object creation logic

This test ensures the environment API correctly handles JSON requests from external clients and converts them to the internal `EnvToolCall` format without errors. 