.PHONY: test test-unit test-integration coverage

test-unit:
	@./scripts/test_unit.sh

test-integration:
	@./scripts/test_integration.sh

test: test-unit

coverage:
	@python scripts/coverage_summary.py

coverage-ci:
	@python scripts/coverage_summary.py --no-readme

