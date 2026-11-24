# Contributing

Read the [SDK, CLI & MCP Style Guide](docs/sdk-style-guide.md) before making
changes. It covers public API ergonomics, sync/async parity, strict frozen
Pydantic boundaries, typed `SynthError` codes, CLI/MCP contracts, PyPI metadata
standards, and local Zed LSP setup (`.zed/settings.json`).

The core OpenAPI document is the API authority. Update its pinned copy and
`contracts/SOURCE.toml`, verify it with `uv run python scripts/check_cohort.py`,
then run `uv run python scripts/gen_contract.py`.
After async method changes run `uv run python scripts/gen_sync.py`.

Run `uv sync --locked --extra mcp`, `make check`, `make test`, and
`make test-e2e`. `make check` is the canonical static gate (also wired into
`lefthook.yml` and `scripts/qualify_release.py`) and runs:

- `uv run --locked --extra mcp ruff check synth_ai scripts tests examples typing_tests`
- `uv run --locked --extra mcp ruff format --check synth_ai scripts tests examples typing_tests`
- `uv run --locked --extra mcp ty check`
- `uv run --locked --extra mcp python scripts/lint_standards.py`
- `uv run --locked --extra mcp python scripts/check_type_contracts.py`

Install the local pre-push gate once per clone with `lefthook install`.
Supported Python versions are 3.11, 3.12 and 3.13.

Build and qualify packaging with `make package-check` (`uv build`,
`uv run twine check dist/*`, and `uv run python scripts/check_package.py`).
Never add unserved operations or relative markdown links to `README.md`.

The SDK standards lint includes generated code, reports file/line diagnostics,
and never imports checked code. See [the rule catalog](docs/protocols.md#standards-lint)
when changing a boundary or adding a lint regression.

`ty check` covers the SDK, all tooling scripts, examples, `tests/`, and accepted
consumer examples (`typing_tests/accepted.py`). The type-contract check
separately verifies every expected diagnostic in `typing_tests/rejected.py`;
those intentionally invalid examples are not part of the normal passing
type-check target. Tooling functions require annotations too.
