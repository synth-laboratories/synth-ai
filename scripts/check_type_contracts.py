#!/usr/bin/env python3
"""Verify that ty accepts valid SDK consumers and rejects every invalid fixture."""

from __future__ import annotations

import io
import re
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = re.compile(r"^(.+):(\d+):\d+: (?:error|warning)\[([^]]+)\]")
EXPECTATION = re.compile(r"# type-error: ([a-z][a-z-]+)$")


@dataclass(frozen=True, order=True)
class TypeErrorLocation:
    file: Path
    line: int
    rule: str


def expectations(path: Path) -> Counter[TypeErrorLocation]:
    expected: Counter[TypeErrorLocation] = Counter()
    for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if token.type != tokenize.COMMENT:
            continue
        if match := EXPECTATION.fullmatch(token.string):
            expected[TypeErrorLocation(path.resolve(), token.start[0], match[1])] += 1
        elif "type-error:" in token.string:
            raise ValueError(f"malformed type expectation: {path}:{token.start[0]}")
    if not expected:
        raise ValueError(f"negative type fixture has no expectations: {path}")
    return expected


def diagnostics(output: str) -> Counter[TypeErrorLocation]:
    found: Counter[TypeErrorLocation] = Counter()
    for line in output.splitlines():
        if match := DIAGNOSTIC.match(line):
            path = Path(match[1])
            if not path.is_absolute():
                path = ROOT / path
            found[TypeErrorLocation(path.resolve(), int(match[2]), match[3])] += 1
    return found


def check(path: Path, *, rejected: bool) -> None:
    result = subprocess.run(
        [
            "ty",
            "check",
            "--project",
            str(ROOT),
            "--python",
            sys.executable,
            "--output-format",
            "concise",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if not rejected:
        if result.returncode != 0:
            raise RuntimeError(f"valid consumer failed type checking:\n{output}")
        return
    expected = expectations(path)
    actual = diagnostics(output)
    if result.returncode != 1 or actual != expected:
        missing = sorted((expected - actual).elements())
        unexpected = sorted((actual - expected).elements())
        raise RuntimeError(
            f"negative type contract changed: missing={missing}, unexpected={unexpected}\n{output}"
        )


def main() -> None:
    check(ROOT / "typing_tests/accepted.py", rejected=False)
    check(ROOT / "typing_tests/rejected.py", rejected=True)
    print("SDK type contracts passed (accepted and rejected consumers)")


if __name__ == "__main__":
    main()
