.PHONY: test test-unit docs-gen docs-dev docs-check

docs-gen:
	uv sync --group dev
	uv run python scripts/generate_sdk_docs.py

docs-check: docs-gen
	uv run python scripts/check_sdk_docstrings.py

docs-dev:
	@test -f docs/docs.json || (echo "Run make docs-gen first" && exit 1)
	cd docs && mint dev

# SDK pytest suite lives in ../testing (see testing/backend/unit/synth_ai_sdk/README.md).
test test-unit:
	@if [ -d ../testing/backend/unit/synth_ai_sdk ]; then \
		uv run python scripts/check_sdk_architecture.py && \
		cd ../testing && uv run python scripts/validate_synth_ai_contract.py && \
		uv run pytest --confcutdir=backend/unit/synth_ai_sdk backend/unit/synth_ai_sdk -v --maxfail=1; \
	else \
		echo "Missing ../testing checkout; clone synth-laboratories/testing beside synth-ai"; \
		exit 1; \
	fi
