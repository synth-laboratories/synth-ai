#!/bin/bash

# Test Runner for Synth AI
# Supports fast/slow tests and public/private test categories

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
TEST_SPEED="all"     # all, fast, slow
TEST_CATEGORY="all"  # all, public, private
VERBOSE=false
COVERAGE=false
PARALLEL=false

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run tests for Synth AI with various filtering options.

OPTIONS:
    -s, --speed SPEED       Run tests by speed: fast, slow, all (default: all)
    -c, --category CAT      Run tests by category: public, private, all (default: all)
    -v, --verbose           Run tests with verbose output
    --coverage              Run tests with coverage reporting
    --parallel              Run tests in parallel (faster but less detailed output)
    --list-tests            List all available tests without running them
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Run all tests
    $0 -s fast             # Run only fast tests (≤5 seconds)
    $0 -c public           # Run only public tests
    $0 -s fast -c private  # Run fast private tests only
    $0 --coverage          # Run all tests with coverage
    $0 --list-tests        # List all tests without running

TEST CATEGORIES:
    - public: Tests that can be run in any environment
    - private: Tests that may require special setup or credentials
    
TEST SPEEDS:
    - fast: Tests that complete in 5 seconds or less
    - slow: Tests that take longer than 5 seconds
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--speed)
            TEST_SPEED="$2"
            shift 2
            ;;
        -c|--category)
            TEST_CATEGORY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --list-tests)
            LIST_TESTS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate speed option
if [[ "$TEST_SPEED" != "all" && "$TEST_SPEED" != "fast" && "$TEST_SPEED" != "slow" ]]; then
    echo -e "${RED}Error: Speed must be 'fast', 'slow', or 'all'${NC}"
    exit 1
fi

# Validate category option
if [[ "$TEST_CATEGORY" != "all" && "$TEST_CATEGORY" != "public" && "$TEST_CATEGORY" != "private" ]]; then
    echo -e "${RED}Error: Category must be 'public', 'private', or 'all'${NC}"
    exit 1
fi

# Function to check if pytest is available
check_pytest() {
    if ! command -v pytest &> /dev/null; then
        echo -e "${RED}Error: pytest is not installed. Install with 'pip install pytest'${NC}"
        exit 1
    fi
}

# Function to get test paths based on category
get_test_paths() {
    local category=$1
    local paths=""
    
    case $category in
        "public")
            paths="public_tests/"
            ;;
        "private")
            paths="private_tests/"
            ;;
        "all")
            paths="public_tests/ private_tests/"
            ;;
    esac
    
    echo $paths
}

# Function to build pytest marker expression
build_marker_expression() {
    local speed=$1
    local expr=""
    
    case $speed in
        "fast")
            expr="not slow"
            ;;
        "slow")
            expr="slow"
            ;;
        "all")
            expr=""
            ;;
    esac
    
    echo "$expr"
}

# Function to list tests
list_tests() {
    local paths=$(get_test_paths "$TEST_CATEGORY")
    local marker_expr=$(build_marker_expression "$TEST_SPEED")
    local cmd="pytest --collect-only -q"
    
    if [[ -n "$marker_expr" ]]; then
        cmd="$cmd -m \"$marker_expr\""
    fi
    
    cmd="$cmd $paths"
    
    echo -e "${BLUE}Test Discovery:${NC}"
    echo -e "${YELLOW}Category: $TEST_CATEGORY, Speed: $TEST_SPEED${NC}"
    echo ""
    
    eval $cmd
}

# Function to run tests
run_tests() {
    local paths=$(get_test_paths "$TEST_CATEGORY")
    local marker_expr=$(build_marker_expression "$TEST_SPEED")
    local cmd="pytest"
    
    # Add basic options
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd -v"
    else
        cmd="$cmd --tb=short"
    fi
    
    # Add marker expression
    if [[ -n "$marker_expr" ]]; then
        cmd="$cmd -m \"$marker_expr\""
    fi
    
    # Add coverage
    if [[ "$COVERAGE" == true ]]; then
        cmd="$cmd --cov=synth_ai --cov-report=term-missing --cov-report=html"
    fi
    
    # Add parallel execution
    if [[ "$PARALLEL" == true ]]; then
        # Check if pytest-xdist is available
        if python -c "import pytest_xdist" 2>/dev/null; then
            cmd="$cmd -n auto"
        else
            echo -e "${YELLOW}Warning: pytest-xdist not installed. Install with 'pip install pytest-xdist' for parallel execution${NC}"
        fi
    fi
    
    # Add test paths
    cmd="$cmd $paths"
    
    echo -e "${BLUE}Running Tests:${NC}"
    echo -e "${YELLOW}Category: $TEST_CATEGORY, Speed: $TEST_SPEED${NC}"
    echo -e "${YELLOW}Command: $cmd${NC}"
    echo ""
    
    # Execute the command
    if eval $cmd; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
        
        if [[ "$COVERAGE" == true ]]; then
            echo -e "${BLUE}📊 Coverage report generated in htmlcov/index.html${NC}"
        fi
    else
        echo -e "${RED}❌ Some tests failed!${NC}"
        exit 1
    fi
}

# Function to add pytest markers to test files
add_markers_to_tests() {
    echo -e "${YELLOW}Adding pytest markers to test files...${NC}"
    
    # Create a temporary Python script to add markers
    cat > /tmp/add_markers.py << 'EOF'
#!/usr/bin/env python3
import os
import re
import ast
import glob

def estimate_test_time(file_path):
    """Estimate if a test is fast or slow based on content analysis."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Indicators of slow tests
        slow_indicators = [
            'sleep', 'time.sleep', 'asyncio.sleep',
            'requests.get', 'requests.post', 'httpx',
            'subprocess', 'docker', 'container',
            'model', 'LM(', 'openai', 'anthropic',
            'large', 'integration', 'end_to_end',
            'database', 'db', 'sql',
            'crafter', 'gym', 'environment'
        ]
        
        # Count slow indicators
        slow_count = sum(1 for indicator in slow_indicators if indicator.lower() in content.lower())
        
        # Long files are often complex/slow
        line_count = len(content.split('\n'))
        
        # If many slow indicators or very long file, mark as slow
        return slow_count >= 3 or line_count > 200
    except:
        return False

def add_pytest_markers(file_path):
    """Add pytest markers to a test file if not already present."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if markers are already present (check for any pytest markers)
        if '@pytest.mark.fast' in content or '@pytest.mark.slow' in content:
            return False
        
        # Check if it's a test file
        if not ('def test_' in content or 'class Test' in content):
            return False
        
        # Determine if test should be marked as slow
        is_slow = estimate_test_time(file_path)
        
        lines = content.split('\n')
        modified = False
        result_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for test functions or classes
            if (line.strip().startswith('def test_') or 
                line.strip().startswith('class Test') or
                line.strip().startswith('async def test_')):
                
                # Add marker before the test
                marker = '@pytest.mark.slow' if is_slow else '@pytest.mark.fast'
                
                # Check indentation
                indent = len(line) - len(line.lstrip())
                marker_line = ' ' * indent + marker
                
                result_lines.append(marker_line)
                modified = True
            
            result_lines.append(line)
            i += 1
        
        if modified:
            # Make sure pytest is imported
            if 'import pytest' not in content and 'from pytest' not in content:
                # Find the first import and add pytest import after it
                import_inserted = False
                final_lines = []
                for i, line in enumerate(result_lines):
                    final_lines.append(line)
                    if (line.strip().startswith('import ') or line.strip().startswith('from ')) and not import_inserted:
                        final_lines.append('import pytest')
                        import_inserted = True
                
                # If no imports found, add at the top after any shebangs/docstrings
                if not import_inserted:
                    insert_index = 0
                    for i, line in enumerate(result_lines):
                        if not (line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''") or line.strip() == ''):
                            insert_index = i
                            break
                    final_lines = result_lines[:insert_index] + ['import pytest', ''] + result_lines[insert_index:]
                    
                result_lines = final_lines
            
            with open(file_path, 'w') as f:
                f.write('\n'.join(result_lines))
            
            return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    # Process public tests
    public_files = glob.glob('public_tests/test_*.py')
    private_files = glob.glob('private_tests/test_*.py')
    
    total_modified = 0
    
    for test_file in public_files + private_files:
        if add_pytest_markers(test_file):
            print(f"Added markers to {test_file}")
            total_modified += 1
    
    print(f"Modified {total_modified} test files")

if __name__ == '__main__':
    main()
EOF
    
    python /tmp/add_markers.py
    rm /tmp/add_markers.py
}

# Main execution
main() {
    echo -e "${BLUE}Synth AI Test Runner${NC}"
    echo "===================="
    
    # Check prerequisites
    check_pytest
    
    # Add markers to test files if they don't exist
    if [[ "$LIST_TESTS" != true ]]; then
        add_markers_to_tests
    fi
    
    # List tests or run them
    if [[ "$LIST_TESTS" == true ]]; then
        list_tests
    else
        run_tests
    fi
}

# Run main function
main "$@"