"""The guide must cover each Problem example in the pinned public contract."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def problem_examples(value):
    if isinstance(value, dict):
        if isinstance(value.get("code"), str) and isinstance(value.get("status"), int):
            yield value["code"], value["status"]
        for child in value.values():
            yield from problem_examples(child)
    elif isinstance(value, list):
        for child in value:
            yield from problem_examples(child)


def test_public_problem_examples_are_in_error_guide():
    contract = yaml.safe_load((ROOT / "contracts/synth-v1.openapi.yaml").read_text())
    guide = (ROOT / "docs/errors.md").read_text()
    examples = set(problem_examples(contract))
    assert examples, "the public contract should contain Problem examples"
    missing = sorted(code for code, _ in examples if f"`{code}`" not in guide)
    assert not missing, f"Problem examples missing from error guide: {missing}"
    mismatches = sorted(
        (code, status)
        for code, status in examples
        if not re.search(rf"`{re.escape(code)}`[^\n]*\b{status}\b", guide)
    )
    assert not mismatches, f"Problem examples missing their documented HTTP status: {mismatches}"
