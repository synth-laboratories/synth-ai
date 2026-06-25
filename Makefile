.PHONY: test test-unit

# SDK pytest suite lives in ../testing (see testing/backend/unit/synth_ai_sdk/README.md).
test test-unit:
	@if [ -d ../testing/backend/unit/synth_ai_sdk ]; then \
		uv run python scripts/check_sdk_architecture.py && \
		cd ../testing && uv run python scripts/validate_synth_ai_contract.py && \
		uv run pytest backend/unit/synth_ai_sdk -m unit -v --maxfail=1; \
	else \
		echo "Missing ../testing checkout; clone synth-laboratories/testing beside synth-ai"; \
		exit 1; \
	fi
