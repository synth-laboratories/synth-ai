.PHONY: test test-unit

# SDK pytest suite lives in ../testing (see testing/backend/synth_ai_sdk/README.md).
test test-unit:
	@if [ -d ../testing/backend/unit/synth_ai_sdk ]; then \
		uv run --group dev pytest ../testing/backend/unit/synth_ai_sdk -m unit -v --maxfail=1; \
	else \
		echo "Missing ../testing checkout; clone synth-laboratories/testing beside synth-ai"; \
		exit 1; \
	fi
