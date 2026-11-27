# Synth AI SDK Makefile (SYN-5955)
# Uniform 4-command developer contract: fmt, check, test, clean.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

.PHONY: help fmt check gen-check test clean test-e2e docs-check package-check

help:
	@printf '%s\n' \
	  'Uniform Synth repo contract:' \
	  '  make fmt    Format Python sources mechanically with ruff format' \
	  '  make check  Run fast (<15s) ruff, ty, boundary checks, landmines, and secret hygiene' \
	  '  make test   Run hermetic unit tests' \
	  '  make clean  Remove disposable local build/test scratch'

fmt:
	uv run --locked --extra mcp ruff format synth_ai scripts tests examples typing_tests

gen-check:
	uv run --locked --extra mcp python scripts/gen_contract.py --check
	uv run --locked --extra mcp python scripts/gen_catalog.py --check
	uv run --locked --extra mcp python scripts/gen_sync.py --check
	uv run --locked --extra mcp python scripts/gen_file_limits.py --check

check: gen-check
	@if command -v gitleaks >/dev/null 2>&1 && [ -n "$$(git diff --cached --name-only)" ]; then \
		gitleaks protect --staged --no-banner --redact; \
	fi
	uv run --locked --extra mcp ruff check synth_ai scripts tests examples typing_tests
	uv run --locked --extra mcp ruff format --check synth_ai scripts tests examples typing_tests
	uv run --locked --extra mcp ty check
	uv run --locked --extra mcp python scripts/lint_standards.py
	uv run --locked --extra mcp python scripts/check_type_contracts.py
	@common_dir="$$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"; \
	ws_root=""; \
	if [ -n "$$common_dir" ]; then ws_root="$$(dirname "$$(dirname "$$common_dir")")"; fi; \
	tool=""; \
	for cand in \
		"./scripts/synth-landmines" \
		"../synth-dev/scripts/synth-landmines" \
		"../../synth-dev/scripts/synth-landmines" \
		"$${ws_root:+$$ws_root/synth-dev/scripts/synth-landmines}" \
		"$$HOME/.synth-dev/bin/synth-landmines" \
		"$$(command -v synth-landmines 2>/dev/null || true)"; do \
		if [ -n "$$cand" ] && [ -x "$$cand" ]; then tool="$$cand"; break; fi; \
	done; \
	[ -n "$$tool" ] || { echo "synth-landmines not found; gate failed closed" >&2; exit 1; }; \
	"$$tool" check --repo-root .

test:
	uv run --locked --extra mcp pytest -q tests

clean:
	rm -rf .pytest_cache .pytest_tmp .ruff_cache dist

test-e2e:
	uv run --locked --extra mcp pytest -q tests/e2e

docs-check:
	uv run --locked --extra mcp pytest -q tests/test_doc_snippets.py tests/test_sync_docstrings.py

package-check:
	uv build
	uv run --locked --extra mcp twine check dist/*
	uv run --locked --extra mcp python scripts/check_package.py
