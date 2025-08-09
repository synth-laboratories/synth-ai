.PHONY: test test-unit test-integration

test-unit:
	@./scripts/test_unit.sh

test-integration:
	@./scripts/test_integration.sh

test: test-unit

