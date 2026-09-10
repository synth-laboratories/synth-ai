.PHONY: test test-unit docs-gen docs-dev docs-check

# This repo is the published package and nothing else. The gates, the specs, the
# guardrail manifest and the docs build moved out on 2026-07-29:
#
#   ../testing   SDK gates + pytest suite + specifications/ + guardrails/
#   ../docs      the Mintlify reference build
#
# These targets stay only as the entry point developers already type. Each one
# delegates and passes SYNTH_AI_DIR=$(CURDIR), so a worktree is checked instead
# of whatever synth-ai happens to sit beside the sibling repo.
TESTING ?= ../testing
DOCS ?= ../docs
# Consumer roots for the migration ratchet. Resolved before the recipe cds into
# $(TESTING), so they mean "beside synth-ai" regardless of where TESTING points.
# The gate skips a root that is not a directory, so a missing sibling degrades to
# a narrower check rather than an error.
BACKEND ?= $(abspath ../backend)
EVALS ?= $(abspath ../evals)

docs-gen:
	@if [ -f $(DOCS)/scripts/generate_sdk_reference.py ]; then \
		cd $(DOCS) && TESTING_DIR=$(abspath $(TESTING)) \
			uv run python scripts/generate_sdk_reference.py $(CURDIR); \
	else \
		echo "Missing $(DOCS) checkout; clone synth-laboratories/docs beside synth-ai"; \
		exit 1; \
	fi

docs-check: docs-gen
	@if [ -d $(TESTING)/specifications/sdk ]; then \
		cd $(TESTING) && SYNTH_AI_DIR=$(CURDIR) uv run python scripts/check_sdk_docstrings.py; \
	else \
		echo "Missing $(TESTING) checkout; clone synth-laboratories/testing beside synth-ai"; \
		exit 1; \
	fi

docs-dev:
	@echo "Reference build and preview both live in the docs repo: cd $(DOCS)/docs && npm run dev"
	@echo "Regenerate reference pages from this tree: make docs-gen"

# SDK gates and pytest suite live in ../testing (see testing/backend/unit/synth_ai_sdk/README.md).
test test-unit:
	@if [ -d $(TESTING)/backend/unit/synth_ai_sdk ]; then \
		cd $(TESTING) && export SYNTH_AI_DIR=$(CURDIR) BACKEND_DIR=$(BACKEND) && \
		uv run python scripts/check_sdk_layering.py && \
		uv run python scripts/check_sdk_architecture.py && \
		uv run python scripts/check_no_rust_sdk.py && \
		uv run python scripts/check_research_openapi_contract.py && \
		uv run python scripts/check_research_migration_boundaries.py \
			--backend-root $(BACKEND) --evals-root $(EVALS) && \
		uv run python scripts/validate_synth_ai_contract.py && \
		uv run pytest --confcutdir=backend/unit/synth_ai_sdk backend/unit/synth_ai_sdk -v --maxfail=1; \
	else \
		echo "Missing $(TESTING) checkout; clone synth-laboratories/testing beside synth-ai"; \
		exit 1; \
	fi
