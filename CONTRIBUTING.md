# Contributing

The core OpenAPI document is the API authority. Update its pinned copy and
`contracts/SOURCE.toml`, then run `uv run python scripts/gen_contract.py`.
After async method changes run `uv run python scripts/gen_sync.py`.

Run `uv sync --extra mcp`, `uv run ruff check synth_ai scripts tests`,
`uv run ruff format --check synth_ai scripts tests`, `uv run ty check`,
`uv run python scripts/lint_standards.py` and
`uv run pytest`. CI runs Python 3.11, 3.12 and 3.13.

Build with `uv build`; qualify with `uv run twine check dist/*` and
`uv run python scripts/check_package.py`. Never add unserved operations.

The SDK standards lint includes generated code, reports file/line diagnostics,
and never imports checked code. See [the rule catalog](docs/protocols.md#standards-lint)
when changing a boundary or adding a lint regression.
