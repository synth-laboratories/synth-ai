# Synth AI CLI

Command-line interface for the live containers, tunnels, and pools SDK.

## Structure

```
cli/
├── __init__.py      # CLI package init
├── __main__.py      # Entry point for `python -m synth_ai.cli`
├── main.py          # Main CLI app setup
├── containers.py    # Container CRUD and status commands
├── pools.py         # Pool and rollout commands
├── tunnels.py       # Managed tunnel and SynthTunnel commands
└── README.md
```

## Guidelines for Additions

### DO add to `cli/` if:

1. **It's a CLI command** - User-invokable from terminal
2. **It's CLI-specific UI** - Prompts, formatters, progress bars
3. **It's CLI configuration** - Command-line argument parsing
4. **It's terminal output** - Tables, colors, spinners

### DO NOT add to `cli/` if:

1. **It's reusable business logic** - Put it in `sdk/` or `core/`
2. **It's a client/API** - Put it in `sdk/`
3. **It's shared infrastructure** - Put it in `core/`
4. **It's outside containers/tunnels/pools** - Archive it under `../research/old/synth_ai`

## Design Principles

### Thin CLI Layer

CLI commands should be thin wrappers around `sdk/` functionality:

```python
# Good - CLI delegates to SDK
@app.command()
def run_eval(container_url: str):
    # Delegate to the live SDK surface here.
    result = run_sdk_workflow(container_url=container_url)
    print_result(result)

# Bad - CLI contains business logic
@app.command()
def run_eval(container_url: str):
    # Don't put business logic here
    response = requests.post(container_url + "/rollout", ...)
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
from synth_ai import SynthClient

# CLI importing from core (correct)
from synth_ai.core.utils.env import get_api_key

# SDK importing from CLI (wrong - never do this)
# from synth_ai.cli.containers import list_containers  # Never
```

## Command Structure

Commands are organized by the three live domains:

```
synth-ai containers list
synth-ai containers create
synth-ai tunnels health
synth-ai tunnels lease
synth-ai pools list
synth-ai pools rollout-status <pool> <rollout>
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| `core/` | `cli/` imports shared env/error helpers |
| `sdk/` | `cli/` imports and wraps the live SDK functionality |
| `../research/old/synth_ai` | archived CLI commands live there for reference only |
