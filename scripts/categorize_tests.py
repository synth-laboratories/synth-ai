#!/usr/bin/env python3
"""
Automatically categorize tests as 'fast' or 'slow' based on execution time.

This script analyzes pytest's --durations output to categorize tests.

Usage:
    1. First, run your tests and save duration data:
       pytest --durations=0 -v > test_output.txt 2>&1
    
    2. Then analyze and optionally apply markers:
       python scripts/categorize_tests.py test_output.txt --apply
    
    Or do it all in one go:
       python scripts/categorize_tests.py --run-and-apply

After applying markers, you can run:
    pytest -m fast      # Run only fast tests
    pytest -m slow      # Run only slow tests
    pytest -m "not slow"  # Run all except slow tests
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).parent.parent
THRESHOLD_SECONDS = 5.0


def parse_pytest_durations(output_file: Path) -> Dict[str, float]:
    """Parse pytest output file to extract test durations."""
    durations = {}
    
    if not output_file.exists():
        print(f"❌ Output file not found: {output_file}")
        return {}
    
    content = output_file.read_text()
    
    # Parse the durations section
    # Format: "0.52s call     tests/test_file.py::TestClass::test_method"
    # or: "0.52s setup    tests/test_file.py::test_function"
    pattern = re.compile(r'([\d.]+)s\s+(setup|call|teardown)\s+(.+?)(?:\s|$)')
    
    # Track durations by test (sum setup + call + teardown)
    test_times = {}
    
    for line in content.split('\n'):
        match = pattern.match(line.strip())
        if match:
            duration_str, phase, test_id = match.groups()
            try:
                duration = float(duration_str)
                test_id = test_id.strip()
                
                if test_id not in test_times:
                    test_times[test_id] = 0.0
                test_times[test_id] += duration
                
            except ValueError:
                continue
    
    # Also try to parse from the summary at the end
    # Format: "10 passed in 5.23s"
    # But more importantly, look for individual test results
    
    # Try alternative format from verbose output
    # PASSED tests/test_file.py::test_name in 0.52s
    alt_pattern = re.compile(r'(PASSED|FAILED|SKIPPED|XFAIL|XPASS)\s+(.+?)\s+.*?in\s+([\d.]+)s')
    
    for line in content.split('\n'):
        match = alt_pattern.search(line)
        if match:
            status, test_id, duration_str = match.groups()
            try:
                duration = float(duration_str)
                test_id = test_id.strip()
                # Use this if we don't have it from durations section
                if test_id not in test_times:
                    test_times[test_id] = duration
            except ValueError:
                continue
    
    return test_times


def categorize_tests(durations: Dict[str, float], threshold: float = THRESHOLD_SECONDS) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Categorize tests by file.
    
    Returns:
        Tuple of (fast_by_file, slow_by_file) where keys are file paths
        and values are lists of test function names.
    """
    fast_by_file = {}
    slow_by_file = {}
    
    for test_id, duration in durations.items():
        # Parse test_id: "tests/test_file.py::TestClass::test_method" or "tests/test_file.py::test_function"
        if '::' not in test_id:
            continue
        
        parts = test_id.split('::')
        file_path = parts[0]
        test_name = parts[-1]  # Last part is always the test function name
        
        # Remove parametrize brackets if present
        test_name = re.sub(r'\[.*?\]$', '', test_name)
        
        if duration < threshold:
            if file_path not in fast_by_file:
                fast_by_file[file_path] = []
            if test_name not in fast_by_file[file_path]:
                fast_by_file[file_path].append(test_name)
        else:
            if file_path not in slow_by_file:
                slow_by_file[file_path] = []
            if test_name not in slow_by_file[file_path]:
                slow_by_file[file_path].append(test_name)
    
    return fast_by_file, slow_by_file


def print_summary(durations: Dict[str, float], threshold: float):
    """Print a summary of the test categorization."""
    if not durations:
        print("❌ No test duration data found")
        return
    
    fast_count = sum(1 for d in durations.values() if d < threshold)
    slow_count = len(durations) - fast_count
    total = len(durations)
    
    print("\n" + "="*70)
    print("📊 Test Duration Summary")
    print("="*70)
    print(f"Threshold: {threshold}s")
    print(f"Total tests: {total}")
    print(f"Fast tests (< {threshold}s): {fast_count} ({fast_count/total*100:.1f}%)")
    print(f"Slow tests (≥ {threshold}s): {slow_count} ({slow_count/total*100:.1f}%)")
    
    # Show slowest tests
    sorted_tests = sorted(durations.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🐌 Top 10 slowest tests:")
    for test_id, duration in sorted_tests[:10]:
        print(f"  {duration:>6.2f}s  {test_id}")
    
    print("="*70 + "\n")


def add_marker_to_function(source: str, function_name: str, marker: str) -> str:
    """Add a pytest marker to a function if it doesn't already have it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    
    lines = source.split('\n')
    modified = False
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                # Check if marker already exists
                has_marker = False
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        # @pytest.mark.fast
                        if (isinstance(decorator.value, ast.Attribute) and
                            decorator.value.attr == 'mark' and
                            decorator.attr == marker):
                            has_marker = True
                            break
                    elif isinstance(decorator, ast.Call):
                        # @pytest.mark.fast()
                        if isinstance(decorator.func, ast.Attribute):
                            if (isinstance(decorator.func.value, ast.Attribute) and
                                decorator.func.value.attr == 'mark' and
                                decorator.func.attr == marker):
                                has_marker = True
                                break
                
                if not has_marker:
                    # Find the line before the function definition
                    func_line = node.lineno - 1
                    
                    # Find the correct indentation
                    indent = len(lines[func_line]) - len(lines[func_line].lstrip())
                    marker_line = ' ' * indent + f'@pytest.mark.{marker}'
                    
                    # Insert the marker
                    lines.insert(func_line, marker_line)
                    modified = True
    
    return '\n'.join(lines) if modified else source


def apply_markers(fast_by_file: Dict[str, List[str]], slow_by_file: Dict[str, List[str]], dry_run: bool = False):
    """Apply markers to test files."""
    all_files = set(list(fast_by_file.keys()) + list(slow_by_file.keys()))
    modified_count = 0
    
    for file_path in sorted(all_files):
        test_file = REPO_ROOT / file_path
        
        if not test_file.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        try:
            content = test_file.read_text()
            original_content = content
            
            # Ensure pytest is imported
            if 'import pytest' not in content:
                # Add import at the top after any __future__ imports
                lines = content.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.startswith('from __future__'):
                        insert_pos = i + 1
                    elif line.strip() and not line.startswith('#'):
                        break
                lines.insert(insert_pos, 'import pytest')
                content = '\n'.join(lines)
            
            # Apply fast markers
            if file_path in fast_by_file:
                for test_name in fast_by_file[file_path]:
                    content = add_marker_to_function(content, test_name, 'fast')
            
            # Apply slow markers
            if file_path in slow_by_file:
                for test_name in slow_by_file[file_path]:
                    content = add_marker_to_function(content, test_name, 'slow')
            
            if content != original_content:
                if dry_run:
                    print(f"📝 Would modify: {file_path}")
                else:
                    test_file.write_text(content)
                    print(f"✅ Modified: {file_path}")
                modified_count += 1
        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    action = "Would modify" if dry_run else "Modified"
    print(f"\n{action} {modified_count} files")


def run_tests_and_collect_durations() -> Dict[str, float]:
    """Run pytest and collect durations."""
    print("🔍 Running tests to collect duration data...")
    print("   This will take some time...\n")
    
    output_file = REPO_ROOT / "test_durations_output.txt"
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--durations=0",
        "--tb=no",
        "-v"
    ]
    
    try:
        with open(output_file, 'w') as f:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=7200  # 2 hour timeout
            )
        
        print(f"\n✅ Test run complete (exit code: {result.returncode})")
        print(f"📄 Output saved to: {output_file}")
        
        return parse_pytest_durations(output_file)
        
    except subprocess.TimeoutExpired:
        print("\n❌ Test run timed out after 2 hours")
        return {}
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Categorize tests as fast or slow based on execution time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("output_file", nargs="?", type=Path, help="Path to pytest output file with --durations")
    parser.add_argument("--apply", action="store_true", help="Apply markers to test files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_SECONDS, 
                       help=f"Threshold in seconds (default: {THRESHOLD_SECONDS})")
    parser.add_argument("--run-and-apply", action="store_true", 
                       help="Run tests, collect durations, and apply markers in one step")
    
    args = parser.parse_args()
    
    threshold = args.threshold
    
    # Get durations
    if args.run_and_apply:
        durations = run_tests_and_collect_durations()
    elif args.output_file:
        durations = parse_pytest_durations(args.output_file)
    else:
        parser.print_help()
        print("\n❌ Error: Provide an output file or use --run-and-apply")
        return 1
    
    if not durations:
        print("❌ No test durations found. Make sure pytest was run with --durations=0")
        return 1
    
    # Analyze and print summary
    print_summary(durations, threshold)
    
    # Categorize
    fast_by_file, slow_by_file = categorize_tests(durations, threshold)
    
    # Apply markers if requested
    if args.apply or args.run_and_apply:
        apply_markers(fast_by_file, slow_by_file, dry_run=args.dry_run)
        
        if not args.dry_run:
            print("\n✨ Done! You can now run:")
            print("   pytest -m fast       # Run only fast tests")
            print("   pytest -m slow       # Run only slow tests")
            print("   pytest -m 'not slow' # Run all except slow tests")
    else:
        print("💡 Run with --apply to add markers to test files")
        print("   Or use --dry-run to see what would change")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




