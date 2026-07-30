.PHONY: test test-unit docs-gen docs-dev docs-check

# `specifications/` and the gates that read it live in ../testing. This repo keeps
# only the docs generator, which writes docs/ here and needs this venv for mdxify.
# SYNTH_AI_DIR points those gates back at this working tree rather than whichever
# synth-ai happens to sit beside ../testing.
TESTING ?= ../testing

docs-gen:
	uv sync --group dev
	TESTING_DIR=$(TESTING) uv run python scripts/generate_sdk_docs.py

docs-check: docs-gen
	@if [ -d $(TESTING)/specifications/sdk ]; then \
		cd $(TESTING) && SYNTH_AI_DIR=$(CURDIR) uv run python scripts/check_sdk_docstrings.py; \
	else \
		echo "Missing $(TESTING) checkout; clone synth-laboratories/testing beside synth-ai"; \
		exit 1; \
	fi

docs-dev:
	@echo "Local preview lives in the docs repo: cd ../docs/docs && npm run dev"
	@echo "This repo only generates reference pages: make docs-gen"

# SDK pytest suite lives in ../testing (see testing/backend/unit/synth_ai_sdk/README.md).
test test-unit:
	@if [ -d $(TESTING)/backend/unit/synth_ai_sdk ]; then \
		uv run python scripts/check_sdk_architecture.py && \
		uv run python scripts/check_research_openapi_contract.py && \
		cd $(TESTING) && export SYNTH_AI_DIR=$(CURDIR) && \
		uv run python scripts/check_research_migration_boundaries.py && \
		uv run python scripts/validate_synth_ai_contract.py && \
		uv run pytest --confcutdir=backend/unit/synth_ai_sdk backend/unit/synth_ai_sdk -v --maxfail=1; \
	else \
		echo "Missing $(TESTING) checkout; clone synth-laboratories/testing beside synth-ai"; \
		exit 1; \
	fi
