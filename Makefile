.PHONY: build build-debug test test-unit test-integration test-fast test-slow

SITE_PKG := $(shell .venv/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

build:
	@echo "Building synth_ai_py (release)..."
	@rm -rf "$(SITE_PKG)/synth_ai_py"
	@.venv/bin/maturin develop --release --uv
	@.venv/bin/python -c "import synth_ai_py; print('OK: synth_ai_py loaded (' + str(len(dir(synth_ai_py))) + ' symbols)')"

build-debug:
	@echo "Building synth_ai_py (debug)..."
	@rm -rf "$(SITE_PKG)/synth_ai_py"
	@.venv/bin/maturin develop --uv
	@.venv/bin/python -c "import synth_ai_py; print('OK: synth_ai_py loaded (' + str(len(dir(synth_ai_py))) + ' symbols)')"

# SDK pytest suite lives in ../testing/synth_ai_sdk/sdk (see testing repo README).
test-unit:
	@uv run --group dev pytest ../testing/synth_ai_sdk/sdk -v --maxfail=1

test-integration:
	@uv run --group dev pytest ../testing/synth_ai_sdk/sdk -v

test: test-unit

test-fast:
	@echo "Running fast tests (< 5 seconds)..."
	@pytest -m fast -v

test-slow:
	@echo "Running slow tests (>= 5 seconds)..."
	@pytest -m slow -v
