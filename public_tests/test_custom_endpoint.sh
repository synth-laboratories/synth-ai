#!/bin/bash

# Test script for custom endpoint support in synth-ai

echo "========================================="
echo "Testing Custom Endpoint Support"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

# Check if responses is installed
echo "Checking dependencies..."
python -c "import responses" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing responses package...${NC}"
    pip install responses
fi

# Run unit tests
echo ""
echo "Running unit tests..."
echo "----------------------------------------"
python -m pytest public_tests/test_custom_endpoint_integration.py::TestCustomEndpointIntegration -v -k "not modal_qwen_hello"
UNIT_TEST_RESULT=$?
print_status $UNIT_TEST_RESULT "Unit tests"

# Check if Modal test URL is set
echo ""
echo "Checking Modal integration test setup..."
echo "----------------------------------------"

if [ -z "$MODAL_TEST_URL" ]; then
    echo -e "${YELLOW}Modal integration test skipped - MODAL_TEST_URL not set${NC}"
    echo ""
    echo "To run Modal integration tests:"
    echo "1. Deploy the test Modal app:"
    echo "   modal deploy modal_test_qwen_app.py"
    echo ""
    echo "2. Set the environment variable with your deployed URL:"
    echo "   export MODAL_TEST_URL='your-org--qwen-test.modal.run'"
    echo ""
    echo "3. Run this script again"
else
    echo -e "${GREEN}Modal test URL found: $MODAL_TEST_URL${NC}"
    echo ""
    echo "Running Modal integration test..."
    echo "----------------------------------------"
    python -m pytest public_tests/test_custom_endpoint_integration.py::test_modal_qwen_hello -v
    MODAL_TEST_RESULT=$?
    print_status $MODAL_TEST_RESULT "Modal integration test"
fi

# Summary
echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="

if [ $UNIT_TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All unit tests passed${NC}"
else
    echo -e "${RED}✗ Some unit tests failed${NC}"
fi

if [ -n "$MODAL_TEST_URL" ]; then
    if [ $MODAL_TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Modal integration test passed${NC}"
    else
        echo -e "${RED}✗ Modal integration test failed${NC}"
    fi
fi

# Exit with appropriate code
if [ $UNIT_TEST_RESULT -ne 0 ]; then
    exit 1
fi

if [ -n "$MODAL_TEST_URL" ] && [ $MODAL_TEST_RESULT -ne 0 ]; then
    exit 1
fi

echo ""
echo "All tests completed successfully!"
exit 0