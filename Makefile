.PHONY: test docs-check

test:
	uv run --locked --extra mcp pytest -q tests

docs-check:
	uv run --locked --extra mcp pytest -q tests/test_doc_snippets.py tests/test_sync_docstrings.py
