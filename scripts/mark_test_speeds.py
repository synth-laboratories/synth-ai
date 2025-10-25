#!/usr/bin/env python3
"""
Script to automatically mark tests as 'fast' or 'slow' based on their execution time.

Usage:
    1. Run tests and collect duration data:
       pytest --durations=0 --durations-min=0.0 -v --tb=no -q > test_durations.txt
    
    2. Or use this script to do both:
       python scripts/mark_test_speeds.py --measure
    
    3. Apply markers to test files:
       python scripts/mark_test_speeds.py --apply
    
    4. Just analyze without applying:
       python scripts/mark_test_speeds.py --analyze
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).parent.parent
THRESHOLD_SECONDS = 5.0
DURATIONS_FILE = REPO_ROOT / "test_durations.json"


def run_tests_and_collect_durations(extra_args: List[str] = None) -> Dict[str, float]:
    """Run pytest with duration reporting and parse the results."""
    print("🔍 Running tests to measure durations...")
    print("   This may take a while...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--durations=0",
        "--durations-min=0.0",
        "-v",
        "--tb=no",
        "-q",
        "--co",  # collect-only for faster run, remove to get actual durations
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    # For actual measurement, we need to run tests, not just collect
    # Remove --co and add --json-report for better parsing
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--durations=0", "--durations-min=0.0", "-v", "--tb=line", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
    except subprocess.TimeoutExpired:
        print("❌ Test run timed out after 1 hour")
        return {}
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return {}
    
    # Parse durations from output
    durations = {}
    output = result.stdout + result.stderr
    
    # Look for duration lines like: "0.001s call     tests/test_example.py::test_something"
    duration_pattern = re.compile(r'([\d.]+)s\s+\w+\s+(.+?)(?:\[|::|\s|$)')
    
    for line in output.split('\n'):
        match = duration_pattern.search(line)
        if match:
            duration_str, test_id = match.groups()
            try:
                duration = float(duration_str)
                durations[test_id.strip()] = duration
            except ValueError:
                continue
    
    print(f"✅ Collected duration data for {len(durations)} tests")
    return durations


def analyze_test_durations(durations: Dict[str, float]) -> Tuple[List[str], List[str]]:
    """Categorize tests into fast and slow based on threshold."""
    fast_tests = []
    slow_tests = []
    
    for test_id, duration in durations.items():
        if duration < THRESHOLD_SECONDS:
            fast_tests.append(test_id)
        else:
            slow_tests.append(test_id)
    
    return fast_tests, slow_tests


def print_analysis(durations: Dict[str, float]):
    """Print analysis of test durations."""
    if not durations:
        print("❌ No duration data available")
        return
    
    fast_tests, slow_tests = analyze_test_durations(durations)
    
    print("\n" + "="*70)
    print("📊 Test Duration Analysis")
    print("="*70)
    print(f"Total tests: {len(durations)}")
    print(f"Fast tests (< {THRESHOLD_SECONDS}s): {len(fast_tests)} ({len(fast_tests)/len(durations)*100:.1f}%)")
    print(f"Slow tests (≥ {THRESHOLD_SECONDS}s): {len(slow_tests)} ({len(slow_tests)/len(durations)*100:.1f}%)")
    
    if slow_tests:
        print(f"\n🐌 Slowest 10 tests:")
        sorted_slow = sorted([(tid, durations[tid]) for tid in slow_tests], key=lambda x: x[1], reverse=True)
        for test_id, duration in sorted_slow[:10]:
            print(f"   {duration:>7.2f}s  {test_id}")
    
    # Group by file
    by_file = defaultdict(list)
    for test_id, duration in durations.items():
        if '::' in test_id:
            file_path = test_id.split('::')[0]
            by_file[file_path].append(duration)
    
    print(f"\n📁 Slowest test files (by average duration):")
    file_averages = [(f, sum(durations)/len(durations)) for f, durations in by_file.items() if durations]
    file_averages.sort(key=lambda x: x[1], reverse=True)
    for file_path, avg_duration in file_averages[:10]:
        test_count = len(by_file[file_path])
        print(f"   {avg_duration:>7.2f}s avg  ({test_count} tests)  {file_path}")
    
    print("="*70 + "\n")


def apply_markers_to_files(durations: Dict[str, float], dry_run: bool = False):
    """Apply pytest markers to test files based on durations."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Applying markers to test files...")
    
    # Group tests by file
    file_tests = defaultdict(lambda: {'fast': [], 'slow': []})
    
    for test_id, duration in durations.items():
        if '::' not in test_id:
            continue
        
        file_path = test_id.split('::')[0]
        test_name = test_id.split('::')[-1]
        
        category = 'fast' if duration < THRESHOLD_SECONDS else 'slow'
        file_tests[file_path][category].append(test_name)
    
    modified_count = 0
    
    for file_path, categories in file_tests.items():
        test_file = REPO_ROOT / file_path
        
        if not test_file.exists():
            continue
        
        try:
            content = test_file.read_text()
            original_content = content
            
            # Add markers to test functions
            for category in ['fast', 'slow']:
                for test_name in categories[category]:
                    # Look for the test function definition
                    pattern = rf'((?:@pytest\.mark\.\w+\s*\n)*)(def {re.escape(test_name)}\()'
                    
                    def add_marker(match):
                        existing_markers = match.group(1)
                        # Check if marker already exists
                        if f'@pytest.mark.{category}' in existing_markers:
                            return match.group(0)
                        return f'{existing_markers}@pytest.mark.{category}\n{match.group(2)}'
                    
                    content = re.sub(pattern, add_marker, content)
            
            if content != original_content:
                if not dry_run:
                    test_file.write_text(content)
                    print(f"✅ Updated {file_path}")
                else:
                    print(f"📝 Would update {file_path}")
                modified_count += 1
        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n{'Would modify' if dry_run else 'Modified'} {modified_count} test files")


def save_durations(durations: Dict[str, float]):
    """Save duration data to JSON file."""
    DURATIONS_FILE.write_text(json.dumps(durations, indent=2))
    print(f"💾 Saved duration data to {DURATIONS_FILE}")


def load_durations() -> Dict[str, float]:
    """Load duration data from JSON file."""
    if not DURATIONS_FILE.exists():
        print(f"❌ Duration data file not found: {DURATIONS_FILE}")
        print("   Run with --measure first to collect duration data")
        return {}
    
    try:
        durations = json.loads(DURATIONS_FILE.read_text())
        print(f"📂 Loaded duration data for {len(durations)} tests")
        return durations
    except Exception as e:
        print(f"❌ Error loading duration data: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Mark tests as fast or slow based on execution time")
    parser.add_argument("--measure", action="store_true", help="Run tests and measure durations")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing duration data")
    parser.add_argument("--apply", action="store_true", help="Apply markers to test files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_SECONDS, help=f"Threshold in seconds (default: {THRESHOLD_SECONDS})")
    parser.add_argument("pytest_args", nargs="*", help="Additional arguments to pass to pytest")
    
    args = parser.parse_args()
    
    global THRESHOLD_SECONDS
    THRESHOLD_SECONDS = args.threshold
    
    durations = {}
    
    if args.measure:
        durations = run_tests_and_collect_durations(args.pytest_args)
        if durations:
            save_durations(durations)
            print_analysis(durations)
    
    if args.analyze or (args.apply and not args.measure):
        durations = load_durations()
        if durations:
            print_analysis(durations)
    
    if args.apply:
        if not durations:
            durations = load_durations()
        if durations:
            apply_markers_to_files(durations, dry_run=args.dry_run)
        else:
            print("❌ No duration data available. Run with --measure first.")
            return 1
    
    if not (args.measure or args.analyze or args.apply):
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

