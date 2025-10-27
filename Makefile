.PHONY: test test-unit test-integration test-fast test-slow categorize-tests coverage

test-unit:
	@./scripts/test_unit.sh

test-integration:
	@./scripts/test_integration.sh

test: test-unit

test-fast:
	@echo "Running fast tests (< 5 seconds)..."
	@pytest -m fast -v

test-slow:
	@echo "Running slow tests (>= 5 seconds)..."
	@pytest -m slow -v

categorize-tests:
	@echo "Categorizing tests by speed..."
	@python scripts/categorize_tests.py --run-and-apply

categorize-tests-dry-run:
	@echo "Preview test categorization (dry run)..."
	@python scripts/categorize_tests.py --run-and-apply --dry-run

coverage:
	@python scripts/coverage_summary.py

coverage-ci:
	@python scripts/coverage_summary.py --no-readme

.PHONY: verify-trace-fixtures
verify-trace-fixtures:
	@python scripts/build_trace_fixtures.py --dest tests/artifacts/traces --overwrite
