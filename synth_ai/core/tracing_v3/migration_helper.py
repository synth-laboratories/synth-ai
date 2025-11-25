#!/usr/bin/env python3
"""
Helper script to identify files that need migration from v2 to v3 tracing.
"""

import os
import re


def find_v2_imports(root_path: str = ".") -> list[tuple[str, list[str]]]:
    """Find all Python files importing from tracing_v2."""
    v2_files = []

    # Patterns to search for
    patterns = [
        re.compile(r"from\s+synth_ai\.tracing_v2"),
        re.compile(r"import\s+synth_ai\.tracing_v2"),
        re.compile(r"from\s+\.\.tracing_v2"),  # Relative imports
        re.compile(r"from\s+\.tracing_v2"),
    ]

    for root, dirs, files in os.walk(root_path):
        # Skip v2 implementation itself
        if "tracing_v2" in root:
            continue

        # Skip v3 implementation
        if "tracing_v3" in root:
            continue

        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path) as f:
                        content = f.read()

                    matches = []
                    for pattern in patterns:
                        for match in pattern.finditer(content):
                            matches.append(match.group(0))

                    if matches:
                        v2_files.append((file_path, matches))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return v2_files


def categorize_files(v2_files: list[tuple[str, list[str]]]) -> dict:
    """Categorize files by their type/location."""
    categories = {"tests": [], "core_library": [], "examples": [], "debug_scripts": [], "other": []}

    for file_path, imports in v2_files:
        if "test" in file_path or "Test" in file_path:
            categories["tests"].append((file_path, imports))
        elif (
            "debug" in file_path
            or file_path.startswith("./test_")
            or file_path.startswith("./isolate_")
        ):
            categories["debug_scripts"].append((file_path, imports))
        elif "example" in file_path:
            categories["examples"].append((file_path, imports))
        elif any(
            core in file_path
            for core in ["synth_ai/lm/", "synth_ai/environments/"]
        ):
            categories["core_library"].append((file_path, imports))
        else:
            categories["other"].append((file_path, imports))

    return categories


def print_migration_report():
    """Print a report of files needing migration."""
    print("=== Tracing v2 to v3 Migration Report ===\n")

    v2_files = find_v2_imports()
    categories = categorize_files(v2_files)

    total_files = len(v2_files)
    print(f"Total files using v2 tracing: {total_files}\n")

    for category, files in categories.items():
        if files:
            print(f"\n{category.upper().replace('_', ' ')} ({len(files)} files):")
            print("-" * 50)
            for file_path, imports in sorted(files):
                print(f"  {file_path}")
                for imp in imports[:3]:  # Show first 3 imports
                    print(f"    - {imp}")
                if len(imports) > 3:
                    print(f"    ... and {len(imports) - 3} more imports")

    print("\n\nRECOMMENDATIONS:")
    print("-" * 50)
    print("1. Test files: These are already configured to be skipped in pytest.ini")
    print("2. Debug scripts: Can be deleted or archived")
    print("3. Core library files: Need careful migration to v3")
    print("   - synth_ai/lm/core/main_v2.py")
    print("   - synth_ai/environments/service/core_routes.py")
    print("4. Examples: Should be updated to demonstrate v3 usage")

    print("\n\nNEXT STEPS:")
    print("-" * 50)
    print("1. Update core library files to support v3 tracing")
    print("2. Create v3 versions of essential tests")
    print("3. Archive or remove debug scripts")
    print("4. Update documentation to reference v3")


if __name__ == "__main__":
    print_migration_report()
