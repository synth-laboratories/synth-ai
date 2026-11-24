.PHONY: check test test-e2e docs-check package-check

check:
	uv run --locked --extra mcp ruff check synth_ai scripts tests examples typing_tests
	uv run --locked --extra mcp ruff format --check synth_ai scripts tests examples typing_tests
	uv run --locked --extra mcp ty check
	uv run --locked --extra mcp python scripts/lint_standards.py
	uv run --locked --extra mcp python scripts/check_type_contracts.py

test:
	uv run --locked --extra mcp pytest -q tests

test-e2e:
	uv run --locked --extra mcp pytest -q tests/e2e

docs-check:
	uv run --locked --extra mcp pytest -q tests/test_doc_snippets.py tests/test_sync_docstrings.py

package-check:
	uv build
	uv run --locked --extra mcp twine check dist/*
	uv run --locked --extra mcp python scripts/check_package.py
