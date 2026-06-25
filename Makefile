.PHONY: test test-unit test-integration test-fast test-slow

# In-repo fake-backend SDK tests: tests/sdk/
# Optional external suite: ../testing/synth_ai_sdk/sdk (testing repo checkout)
test-unit:
	@uv run --group dev pytest tests/sdk/ -v --maxfail=1
	@if [ -d ../testing/synth_ai_sdk/sdk ]; then uv run --group dev pytest ../testing/synth_ai_sdk/sdk -v --maxfail=1; fi

test-integration:
	@uv run --group dev pytest ../testing/synth_ai_sdk/sdk -v

test: test-unit

test-fast:
	@echo "Running fast tests (< 5 seconds)..."
	@uv run --group dev pytest -m fast -v

test-slow:
	@echo "Running slow tests (>= 5 seconds)..."
	@uv run --group dev pytest -m slow -v
