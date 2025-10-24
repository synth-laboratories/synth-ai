# Fast and Slow Test Categorization

This guide explains how to categorize tests based on execution time and run them selectively.

## Overview

Tests are automatically marked as either `fast` or `slow` based on their execution time:
- **Fast tests**: Complete in < 5 seconds
- **Slow tests**: Take ≥ 5 seconds

This allows developers to:
- Run fast tests during development for quick feedback
- Run slow tests less frequently or only in CI
- Identify performance bottlenecks in the test suite

## Quick Start

### Running Tests by Category

Once tests are marked, you can run them selectively:

```bash
# Run only fast tests (great for development)
pytest -m fast

# Run only slow tests
pytest -m slow

# Run everything except slow tests
pytest -m "not slow"

# Combine with other markers
pytest -m "fast and unit"
pytest -m "fast and not integration"
```

## Categorizing Tests

### Method 1: Automatic Categorization (Recommended)

Run tests once to collect timing data, then automatically apply markers:

```bash
# Run tests and automatically apply markers
python scripts/categorize_tests.py --run-and-apply

# Or preview what would change without modifying files
python scripts/categorize_tests.py --run-and-apply --dry-run
```

### Method 2: Manual Analysis

If you already have test output:

```bash
# 1. Run tests and save output
pytest --durations=0 -v > test_output.txt 2>&1

# 2. Analyze and apply markers
python scripts/categorize_tests.py test_output.txt --apply

# Or just analyze without applying
python scripts/categorize_tests.py test_output.txt
```

### Custom Threshold

Change the threshold from the default 5 seconds:

```bash
# Mark tests > 10 seconds as slow
python scripts/categorize_tests.py --run-and-apply --threshold 10

# Mark tests > 2 seconds as slow (more aggressive)
python scripts/categorize_tests.py --run-and-apply --threshold 2
```

## What Gets Changed

The script automatically:

1. Adds `import pytest` if not present
2. Adds `@pytest.mark.fast` or `@pytest.mark.slow` decorators to test functions
3. Preserves existing decorators and formatting

### Example

Before:
```python
def test_quick_operation():
    assert 1 + 1 == 2

def test_slow_integration():
    # ... complex setup ...
    result = expensive_operation()
    assert result is not None
```

After:
```python
import pytest

@pytest.mark.fast
def test_quick_operation():
    assert 1 + 1 == 2

@pytest.mark.slow
def test_slow_integration():
    # ... complex setup ...
    result = expensive_operation()
    assert result is not None
```

## Best Practices

### For Developers

1. **Run fast tests frequently**: Use `pytest -m fast` during development
2. **Run all tests before pushing**: Ensure nothing breaks
3. **Keep tests fast**: If a test becomes slow, consider refactoring or mocking

### For CI/CD

```yaml
# Example GitHub Actions workflow
jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m fast
  
  slow-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m slow
    # Run less frequently or only on main branch
```

### Periodic Re-categorization

Tests may become faster or slower over time. Re-run categorization periodically:

```bash
# Every few weeks or after major refactoring
python scripts/categorize_tests.py --run-and-apply
```

## Troubleshooting

### No Duration Data Found

If the script can't find duration data:

```bash
# Make sure pytest runs with --durations=0
pytest --durations=0 -v > test_output.txt 2>&1

# Or use --run-and-apply to do it automatically
python scripts/categorize_tests.py --run-and-apply
```

### Markers Not Working

Ensure `pytest.ini` has the markers defined:

```ini
[pytest]
markers =
    fast: Fast running tests (< 5 seconds)
    slow: Slow running tests (>= 5 seconds)
```

### Tests Not Getting Marked

The script only marks test functions, not test classes. If you have:

```python
class TestSuite:
    def test_something(self):
        pass
```

Only `test_something` will get marked, not the class. This is by design.

## Advanced Usage

### Manual Marking

You can also manually mark tests:

```python
import pytest

@pytest.mark.slow
def test_complex_integration():
    """This test is inherently slow, mark it explicitly."""
    pass

@pytest.mark.fast
@pytest.mark.unit
def test_quick_unit():
    """Fast unit test with multiple markers."""
    pass
```

### Combining Markers

```bash
# Fast unit tests only
pytest -m "fast and unit"

# All integration tests that aren't slow
pytest -m "integration and not slow"

# Everything except slow integration tests
pytest -m "not (slow and integration)"
```

## Performance Tips

If you have many slow tests:

1. **Profile them**: Use `pytest --durations=10` to see the slowest tests
2. **Mock expensive operations**: Database calls, API requests, file I/O
3. **Use fixtures efficiently**: Leverage pytest's fixture scopes
4. **Parallel execution**: Use `pytest-xdist` for `pytest -n auto`
5. **Consider test architecture**: Unit tests should be fast, integration tests can be slower

## See Also

- [pytest markers documentation](https://docs.pytest.org/en/stable/how-to/mark.html)
- [pytest duration documentation](https://docs.pytest.org/en/stable/how-to/output.html#profiling-test-duration)




