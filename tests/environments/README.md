# Environments Test Suite

This directory contains comprehensive unit and integration tests for the Environments service.

## Test Structure

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_registry.py       # Tests for environment registry
│   ├── test_external_registry.py  # Tests for external env loading
│   ├── test_sokoban_qstar.py # Tests for Sokoban with Q* solver
│   ├── test_sokoban_failure.py # Existing Sokoban edge case test
│   └── test_math_environment.py # Tests for math environment
├── integration/               # Integration tests
│   ├── test_service_api.py    # Tests for new aligned API endpoints
│   └── test_sokoban_service.py # Sokoban-specific service tests
└── conftest.py               # Pytest configuration
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/test_sokoban_qstar.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run tests matching a pattern
```bash
pytest tests/ -k "sokoban"
```

## Test Categories

### Unit Tests

1. **Registry Tests** (`test_registry.py`)
   - Environment registration and retrieval
   - Registry error handling
   - Multiple environment support

2. **External Registry Tests** (`test_external_registry.py`)
   - Loading environments from external packages
   - Configuration-based loading
   - Error handling for missing modules

3. **Sokoban Q* Tests** (`test_sokoban_qstar.py`)
   - A* solver functionality
   - Various puzzle difficulties
   - Engine-level and environment-level APIs
   - Checkpointing and restoration

4. **Math Environment Tests** (`test_math_environment.py`)
   - Problem presentation
   - Answer submission
   - Correct/incorrect answer handling

### Integration Tests

1. **Service API Tests** (`test_service_api.py`)
   - New aligned endpoints (/initialize, /step, /terminate)
   - Request/response formats
   - Error handling
   - Multiple concurrent environments
   - Legacy endpoint deprecation

2. **Sokoban Service Tests** (`test_sokoban_service.py`)
   - Complete episode execution via API
   - Various puzzle types
   - Edge cases (max steps, invalid actions)
   - Concurrent Sokoban instances

## Key Test Features

### Sokoban Q* Solver Tests
The tests demonstrate using the A* algorithm to automatically solve Sokoban puzzles:
- Tests verify solver finds optimal solutions
- Covers trivial to medium complexity puzzles
- Validates both engine-level and environment-level solving

### Service API Alignment
Tests verify the new API structure:
- `/env/{env_name}/initialize` - Create and initialize environment
- `/env/{env_name}/step` - Execute actions
- `/env/{env_name}/terminate` - Clean up environment
- Backward compatibility with deprecated endpoints

### No Agent Demos
Following the requirements, tests focus on:
- Direct environment interaction
- Algorithmic solving (Q*/A*)
- API functionality
- State management
- Error handling

No tests use AI agents or language models.

## Adding New Tests

When adding tests for new environments:
1. Create unit tests for the environment logic
2. Add integration tests for service endpoints
3. Include tests for edge cases and error conditions
4. Use algorithmic approaches (like A*) rather than AI agents
5. Ensure proper cleanup in fixtures

## Test Configuration

The `conftest.py` file provides:
- Automatic cleanup of environment instances after each test
- Python path configuration for imports
- Option to disable external environment loading

## Dependencies

Tests require:
- pytest
- pytest-asyncio
- httpx (for async HTTP testing)
- fastapi[testclient]

Install with:
```bash
pip install pytest pytest-asyncio httpx fastapi[all]
```