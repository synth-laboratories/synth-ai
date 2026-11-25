# synth-ai agent notes

Public Python SDK, CLI, and MCP client for Synth Swarms (`synth_ai/`).

## Layout

- `synth_ai/` — typed async/sync SDK clients, CLI (`cli.py`), MCP server (`mcp/server.py`), and generated OpenAPI contract/file-limit modules (`_generated/`).
- `contracts/` — pinned OpenAPI schema (`synth-v1.openapi.yaml`), `mcp-tools.json`, `file-limits.json`, and source pin manifests (`SOURCE.toml`, `FILE_LIMITS_SOURCE.toml`).
- `scripts/` — codegen and static contract parity/boundary gates (`gen_contract.py`, `gen_sync.py`, `gen_file_limits.py`, `check_surface_parity.py`, `lint_standards.py`, `check_package.py`).
- `tests/` & `typing_tests/` — hermetic unit/contract tests and `ty` acceptance tests.

## Developer Contract & Quality Gates

- `make fmt` — format Python sources (`ruff format`) across `synth_ai`, `scripts`, `tests`, `examples`, and `typing_tests`.
- `make check` — run staged secret hygiene (`gitleaks` when staged), `ruff check`, `ruff format --check`, Astral `ty check`, `scripts/lint_standards.py`, `scripts/check_type_contracts.py`, and `synth-landmines check` (<15s).
- `make test` — run hermetic unit tests (`pytest -q tests`).
- `make clean` — remove local `.pytest_cache`, `.pytest_tmp`, `.ruff_cache`, and `dist` scratch.
- Do not hand-edit `synth_ai/_generated/*` or `synth_ai/_sync.py`; regenerate via `scripts/gen_contract.py` / `scripts/gen_sync.py` and verify with `--check`.
- Commit subjects and pushes include `[skip ci]` except on release tag pushes (`v*`).
