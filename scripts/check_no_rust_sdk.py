#!/usr/bin/env python3
"""Fail if the Python SDK reintroduces Rust runtime/build dependencies."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATHS = (
    "Cargo.toml",
    "Cargo.lock",
    "synth_ai_core",
    "synth_ai_core_types",
    "synth_ai_py",
    "synth_ai_rs",
)
FORBIDDEN_TEXT = (
    "synth_ai_py",
    "synth_ai_core",
    "synth_ai_rs",
    "maturin",
    "pyo3",
    "Cargo",
)
SCAN_GLOBS = (
    "synth_ai/**/*.py",
    "pyproject.toml",
    "MANIFEST.in",
    ".github/**/*.yml",
    ".github/**/*.yaml",
)


def main() -> int:
    failures: list[str] = []
    for rel in FORBIDDEN_PATHS:
        if (ROOT / rel).exists():
            failures.append(f"forbidden path exists: {rel}")
    for pattern in SCAN_GLOBS:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in FORBIDDEN_TEXT:
                if needle in text:
                    failures.append(f"{path.relative_to(ROOT)} contains {needle!r}")
    if failures:
        print("Rust SDK dependency guard failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Rust SDK dependency guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
