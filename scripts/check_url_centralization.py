#!/usr/bin/env python3
"""Check that usesynth.ai URLs are only defined in authorized files.

Ensures URL definitions are centralized in:
- synth_ai/core/urls.py (Python)
- synth_ai/tui/app/src/state/mode.ts (TypeScript)

Usage:
    python scripts/check_url_centralization.py              # Scan entire repo
    python scripts/check_url_centralization.py file1.py     # Scan specific files
"""

import re
import sys
from pathlib import Path

# Files allowed to define usesynth.ai URLs
ALLOWED_FILES = {
    "synth_ai/core/urls.py",
    "synth_ai/tui/app/src/state/mode.ts",
}

# File extensions to check
PROGRAM_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}

# Directories to skip entirely
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}

# Pattern to match usesynth.ai URLs (excluding docs.usesynth.ai)
URL_PATTERN = re.compile(r"(?<!docs\.)usesynth\.ai")

# Pattern to detect docs.usesynth.ai (to exclude)
DOCS_PATTERN = re.compile(r"docs\.usesynth\.ai")


def is_in_python_docstring(lines: list[str], line_idx: int, col: int) -> bool:
    """Check if position is inside a Python docstring."""
    # Simple heuristic: count triple quotes before this position
    text_before = "\n".join(lines[: line_idx + 1])
    text_before = (
        text_before[: text_before.rfind("\n") + 1 + col] if line_idx > 0 else lines[0][:col]
    )

    # Count unescaped triple quotes
    triple_double = len(re.findall(r'(?<!\\)"""', text_before))
    triple_single = len(re.findall(r"(?<!\\)'''", text_before))

    # If odd number of either, we're inside a docstring
    return (triple_double % 2 == 1) or (triple_single % 2 == 1)


def is_in_python_comment(line: str, col: int) -> bool:
    """Check if position is inside a Python comment."""
    # Find first # not inside a string
    in_single = False
    in_double = False
    for _, char in enumerate(line[:col]):
        if char == '"' and not in_single:
            in_double = not in_double
        elif char == "'" and not in_double:
            in_single = not in_single
        elif char == "#" and not in_single and not in_double:
            return True
    return False


def is_in_js_comment(lines: list[str], line_idx: int, col: int) -> bool:
    """Check if position is inside a JS/TS comment."""
    line = lines[line_idx]

    # Check for // line comment
    comment_start = line.find("//")
    if comment_start != -1 and comment_start < col:
        # Make sure // is not inside a string
        in_single = False
        in_double = False
        in_template = False
        for _, char in enumerate(line[:comment_start]):
            if char == '"' and not in_single and not in_template:
                in_double = not in_double
            elif char == "'" and not in_double and not in_template:
                in_single = not in_single
            elif char == "`" and not in_single and not in_double:
                in_template = not in_template
        if not in_single and not in_double and not in_template:
            return True

    # Check for /* */ block comment (simplified - check current line only)
    # This is a heuristic; full parsing would require tracking across lines
    block_start = line.rfind("/*", 0, col)
    block_end = line.rfind("*/", 0, col)
    return block_start != -1 and (block_end == -1 or block_end < block_start)


def check_file(filepath: Path, repo_root: Path) -> list[tuple[int, str]]:
    """Check a single file for URL violations.

    Returns list of (line_number, line_content) for violations.
    """
    relative_path = str(filepath.relative_to(repo_root))

    # Skip allowed files
    if relative_path in ALLOWED_FILES:
        return []

    try:
        content = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    lines = content.splitlines()
    violations = []
    is_python = filepath.suffix == ".py"

    for line_idx, line in enumerate(lines):
        # Find all usesynth.ai matches in this line
        for match in URL_PATTERN.finditer(line):
            col = match.start()

            # Skip if this is docs.usesynth.ai
            # Check if "docs." precedes this match
            if col >= 5 and line[col - 5 : col] == "docs.":
                continue

            # Skip if in comment or docstring
            if is_python:
                if is_in_python_comment(line, col):
                    continue
                if is_in_python_docstring(lines, line_idx, col):
                    continue
            else:
                if is_in_js_comment(lines, line_idx, col):
                    continue

            violations.append((line_idx + 1, line.strip()))

    return violations


def find_program_files(repo_root: Path) -> list[Path]:
    """Find all program files in the repository."""
    files = []

    for path in repo_root.rglob("*"):
        # Skip directories in SKIP_DIRS
        if any(skip_dir in path.parts for skip_dir in SKIP_DIRS):
            continue

        # Only include program files
        if path.is_file() and path.suffix in PROGRAM_EXTENSIONS:
            files.append(path)

    return files


def main() -> int:
    repo_root = Path(__file__).parent.parent.resolve()

    # Determine files to check
    if len(sys.argv) > 1:
        # Specific files provided (lefthook mode)
        files = []
        for arg in sys.argv[1:]:
            path = Path(arg)
            if not path.is_absolute():
                path = repo_root / path
            if path.exists() and path.suffix in PROGRAM_EXTENSIONS:
                files.append(path)
    else:
        # Scan entire repo
        files = find_program_files(repo_root)

    # Check all files
    all_violations: dict[str, list[tuple[int, str]]] = {}

    for filepath in files:
        violations = check_file(filepath, repo_root)
        if violations:
            relative_path = str(filepath.relative_to(repo_root))
            all_violations[relative_path] = violations

    # Report results
    if all_violations:
        print("URL Centralization Check FAILED")
        print("=" * 60)
        print("URLs must be defined only in:")
        print("  - synth_ai/core/urls.py")
        print("  - synth_ai/tui/app/src/state/mode.ts")
        print()
        print("Violations found:")
        print("-" * 60)

        for filepath, violations in sorted(all_violations.items()):
            for line_num, line_content in violations:
                print(f"{filepath}:{line_num}")
                print(f"    {line_content[:80]}")
            print()

        print(
            f"Total: {sum(len(v) for v in all_violations.values())} violations in {len(all_violations)} files"
        )
        return 1

    print("URL Centralization Check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
