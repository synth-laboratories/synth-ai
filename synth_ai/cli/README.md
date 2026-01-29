# Synth AI CLI

Command-line interface for Synth AI.

## Structure

```
cli/
├── __init__.py      # CLI package init
├── __main__.py      # Entry point for `python -m synth_ai.cli`
├── main.py          # Main CLI app setup
├── bin.py           # Binary/script entry points
├── prompts.py       # Interactive prompts and user input
├── setup.py         # CLI setup and initialization
├── commands/        # CLI command implementations
│   ├── auth.py      # Authentication commands (login, logout, whoami)
│   ├── jobs.py      # Job management commands (list, status, cancel)
│   ├── eval.py      # Evaluation commands
│   ├── optimize.py  # Optimization commands
│   └── ...
└── local/           # Local development utilities
    └── ...
```

## Guidelines for Additions

### DO add to `cli/` if:

1. **It's a CLI command** - User-invokable from terminal
2. **It's CLI-specific UI** - Prompts, formatters, progress bars
3. **It's CLI configuration** - Command-line argument parsing
4. **It's terminal output** - Tables, colors, spinners

### DO NOT add to `cli/` if:

1. **It's reusable business logic** - Put it in `sdk/` or `core/`
2. **It's a data type** - Put it in `data/`
3. **It's a client/API** - Put it in `sdk/`
4. **It's shared infrastructure** - Put it in `core/`

## Design Principles

### Thin CLI Layer

CLI commands should be thin wrappers around `sdk/` functionality:

```python
# Good - CLI delegates to SDK
@app.command()
def run_eval(task_app_url: str):
    job = EvalJob(task_app_url=task_app_url)
    result = job.run()
    print_result(result)

# Bad - CLI contains business logic
@app.command()
def run_eval(task_app_url: str):
    # Don't put evaluation logic here
    response = requests.post(task_app_url + "/rollout", ...)
    score = compute_score(response)
    ...
```

### User-Friendly Output

CLI should format output for human readability:

```python
from rich.console import Console
from rich.table import Table

def print_jobs(jobs: list[JobInfo]) -> None:
    table = Table(title="Jobs")
    table.add_column("ID")
    table.add_column("Status")
    for job in jobs:
        table.add_row(job.id, job.status)
    Console().print(table)
```

### Error Handling

Catch exceptions and display user-friendly messages:

```python
@app.command()
def my_command():
    try:
        result = sdk_function()
    except AuthenticationError:
        print("Not logged in. Run: synth login")
        raise SystemExit(1)
    except APIError as e:
        print(f"API error: {e.message}")
        raise SystemExit(1)
```

## Import Rules

- `cli/` can import from `sdk/`, `core/`, and `data/`
- `cli/` should NOT be imported by `sdk/` or `core/`

```python
# CLI importing from SDK (correct)
from synth_ai.sdk import EvalJob, PolicyOptimizationJob

# CLI importing from core (correct)
from synth_ai.core.auth import get_api_key

# SDK importing from CLI (wrong - never do this)
# from synth_ai.cli.prompts import ask_user  # Never
```

## Command Structure

Commands are organized by domain:

```
synth login          # auth commands
synth logout
synth whoami

synth jobs list      # job management
synth jobs status <id>
synth jobs cancel <id>

synth eval run       # evaluation
synth optimize run   # optimization
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| `data/` | `cli/` imports data types for display |
| `core/` | `cli/` imports infrastructure (auth, config) |
| `sdk/` | `cli/` imports and wraps SDK functionality |
