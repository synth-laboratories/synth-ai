# Contributing

Install Python 3.11 or newer and uv, then run `uv sync --group dev --extra mcp`.
Run the offline suite with `uv run pytest`.
Run `uv run ruff check synth_ai tests` and `uv run ruff format --check synth_ai tests`.
Run `uv run ty check --python .venv synth_ai`.
Release publication requires the qualified artifact workflow.
