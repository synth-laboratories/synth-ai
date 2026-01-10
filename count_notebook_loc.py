#!/usr/bin/env python3
"""Count lines of code in Jupyter notebooks."""

import json
from pathlib import Path


def count_notebook_loc(notebook_path: Path) -> int:
    """Count lines of code in a Jupyter notebook.

    Only counts lines in code cells, excluding markdown and other cell types.
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    total_lines = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                # Join list items first (handles both line-by-line and char-by-char formats)
                joined_source = "".join(source)
                # If the source appears to be stored character-by-character (each item ends with \n),
                # we need to reconstruct actual lines. Check if first few items are single chars.
                if len(source) > 0 and len(source[0].rstrip("\n")) <= 1:
                    # Character-by-character format: join everything, then split properly
                    # Remove trailing \n from each item before joining to reconstruct properly
                    cleaned = [item.rstrip("\n") for item in source]
                    joined_source = "".join(cleaned)
                # Count non-empty lines
                lines = [line for line in joined_source.split("\n") if line.strip()]
                total_lines += len(lines)
            elif isinstance(source, str):
                # Count non-empty lines
                lines = [line for line in source.split("\n") if line.strip()]
                total_lines += len(lines)

    return total_lines


def main():
    repo_root = Path(__file__).parent
    demos_dir = repo_root / "demos"

    notebooks = [
        (
            "Banking77 GEPA",
            demos_dir / "gepa_banking77" / "gepa_banking77_prompt_optimization.ipynb",
        ),
        (
            "GEPA Crafter VLM",
            demos_dir / "gepa_crafter_vlm" / "gepa_crafter_vlm_verifier_optimization.ipynb",
        ),
        (
            "Style Matching",
            demos_dir / "style_matching" / "style_matching_prompt_optimization.ipynb",
        ),
    ]

    print("=" * 60)
    print("Notebook Lines of Code Count")
    print("=" * 60)
    print()

    total = 0
    for name, path in notebooks:
        if not path.exists():
            print(f"{name}: NOT FOUND ({path})")
            continue

        loc = count_notebook_loc(path)
        total += loc
        print(f"{name}:")
        print(f"  Path: {path.relative_to(repo_root)}")
        print(f"  Lines of Code: {loc:,}")
        print()

    print("=" * 60)
    print(f"Total: {total:,} lines of code")
    print("=" * 60)


if __name__ == "__main__":
    main()
