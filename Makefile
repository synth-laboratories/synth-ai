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

test-unit:
	@cd ../testing && bazel test //:synth_ai_unit_tests

test-integration:
	@cd ../testing && bazel test //:synth_ai_all_tests

test: test-unit

test-fast:
	@echo "Running fast tests (< 5 seconds)..."
	@pytest -m fast -v

test-slow:
	@echo "Running slow tests (>= 5 seconds)..."
	@pytest -m slow -v
