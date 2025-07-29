# Synth AI Test System

This repository has a comprehensive test system with speed-based and category-based filtering.

## Quick Start

```bash
# Run all tests
./run_tests.sh

# Run only fast tests (≤5 seconds each)
./run_tests.sh -s fast

# Run only public tests (no credentials required)
./run_tests.sh -c public

# Run fast public tests only
./run_tests.sh -s fast -c public

# List all tests without running them
./run_tests.sh --list-tests

# Run with coverage reporting
./run_tests.sh --coverage
```

## Test Categories

### By Speed
- **fast**: Tests that complete in 5 seconds or less
  - Unit tests
  - Simple functionality tests
  - Basic imports and configuration
- **slow**: Tests that take longer than 5 seconds
  - API calls to external services
  - Integration tests
  - Model inference tests

### By Access Level
- **public**: Tests that can run in any environment
  - No credentials required
  - Basic functionality tests
  - Unit tests
- **private**: Tests requiring special setup
  - May need API credentials
  - May require running services
  - Integration tests

## Test Markers

Tests are automatically categorized with pytest markers:

```python
import pytest

@pytest.mark.fast
def test_basic_functionality():
    """Quick unit test"""
    assert 2 + 2 == 4

@pytest.mark.slow  
def test_api_call():
    """Test requiring external API call"""
    lm = LM(model_name="gpt-4o-mini")
    response = lm.respond_sync("Hello")
    assert response is not None
```

## Manual Pytest Usage

You can also run pytest directly with markers:

```bash
# Fast tests only
pytest -m "fast"

# Slow tests only  
pytest -m "slow"

# Exclude slow tests (same as fast)
pytest -m "not slow"

# Public tests only
pytest public_tests/

# Private tests only
pytest private_tests/

# Specific test file
pytest public_tests/test_basic_functionality.py
```

## Test Structure

```
synth-ai/
├── public_tests/           # Tests that can run anywhere
│   ├── test_basic_functionality.py
│   ├── test_text.py
│   └── ...
├── private_tests/         # Tests requiring credentials/setup
│   ├── test_lm_implementation.py
│   ├── test_duckdb_manager.py
│   └── ...
├── run_tests.sh          # Test runner script
└── pyproject.toml        # Pytest configuration
```

## Pytest Configuration

The test system is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["public_tests", "private_tests"]
markers = [
    "fast: marks tests as fast (≤5 seconds)",
    "slow: marks tests as slow (>5 seconds)",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "public: marks tests that can run in any environment", 
    "private: marks tests that may require credentials",
]
timeout = 300  # 5 minute timeout for slow tests
```

## Adding New Tests

1. **Create test file** in appropriate directory:
   - `public_tests/` for general tests
   - `private_tests/` for tests requiring special setup

2. **Add pytest markers**:
   ```python
   import pytest

   @pytest.mark.fast  # or @pytest.mark.slow
   def test_my_feature():
       """Test description"""
       pass
   ```

3. **Run the test runner** - it will automatically detect and categorize your tests:
   ```bash
   ./run_tests.sh --list-tests  # See your test listed
   ./run_tests.sh -s fast       # Run if marked as fast
   ```

## CI/CD Integration

For continuous integration, run fast tests on every commit and slow tests nightly:

```yaml
# Fast tests for PRs
- name: Run Fast Tests
  run: ./run_tests.sh -s fast

# Full test suite for main branch
- name: Run All Tests  
  run: ./run_tests.sh --coverage
```

## Examples

### Fast Test Example
```python
@pytest.mark.fast
def test_string_operations():
    """Test string operations - should be very fast."""
    test_string = "Hello, World!"
    assert test_string.lower() == "hello, world!"
    assert len(test_string) == 13
```

### Slow Test Example  
```python
@pytest.mark.slow
def test_lm_api_call():
    """Test LM API call - slow due to network request."""
    lm = LM(model_name="gpt-4o-mini", temperature=0)
    response = lm.respond_sync(
        user_message="What is 2+2? Answer with just the number."
    )
    assert "4" in response.raw_response
```