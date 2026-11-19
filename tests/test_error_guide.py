"""The guide must cover each Problem example in the pinned public contract."""

import re
from pathlib import Path

import pytest
import yaml

from synth_ai.errors import _BY_CODE

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


@pytest.mark.parametrize(
    "code",
    [
        "optimizer_topology_not_allowed",
        "optimizer_topology_unavailable",
        "optimizer_version_unavailable",
        "launch_request_too_large",
        "route_not_found",
        "billing_admin_required",
    ],
)
def test_journey_refusal_has_a_guide_entry(code):
    guide = (Path(__file__).parents[1] / "docs/errors.md").read_text()
    assert f"- `{code}`" in guide


CODE = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`")
STATUS = re.compile(r"^\((\d{3})\b")


def refusal_clauses(text):
    """Yield each top-level `Refusals:` list in one operation description."""
    text = " ".join(text.split())
    for start in [m.end() for m in re.finditer(r"\bRefusals:", text)]:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            depth += char == "("
            depth -= char == ")"
            end_of_sentence = text.startswith(". ", index) and text[index + 2 : index + 3].isupper()
            if depth == 0 and (end_of_sentence or index == len(text) - 1):
                yield text[start : index + 1]
                break


def listed_refusals(clause):
    """Return (code, status) for each top-level code; a status covers the codes before it."""
    pending = []
    refusals = []
    depth = 0
    index = 0
    while index < len(clause):
        char = clause[index]
        if char == "(":
            if depth == 0 and pending:
                status = STATUS.match(clause[index:])
                if status:
                    refusals.extend((code, int(status.group(1))) for code in pending)
                    pending = []
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "`" and depth == 0:
            code = CODE.match(clause, index)
            if code:
                pending.append(code.group(1))
                index = code.end()
                continue
        index += 1
    assert not pending, f"refusal codes without an HTTP status in {clause!r}: {pending}"
    return refusals


def operation_refusals(contract):
    for path, item in contract["paths"].items():
        for method, operation in item.items():
            if not isinstance(operation, dict):
                continue
            for clause in refusal_clauses(operation.get("description", "")):
                for code, status in listed_refusals(clause):
                    yield f"{method.upper()} {path}", code, status


def test_listed_operation_refusals_are_in_error_guide():
    contract = yaml.safe_load((ROOT / "contracts/synth-v1.openapi.yaml").read_text())
    guide = (ROOT / "docs/errors.md").read_text()
    refusals = sorted(set(operation_refusals(contract)))
    assert len(refusals) >= 40, f"too few listed refusals parsed: {len(refusals)}"
    missing = sorted(
        (operation, code, status)
        for operation, code, status in refusals
        if not re.search(rf"`{re.escape(code)}`[^\n]*\b{status}\b", guide)
    )
    assert not missing, f"listed refusals missing from the error guide with their status: {missing}"


def test_typed_refusals_document_the_recovery_action():
    guide = (ROOT / "docs/errors.md").read_text()
    for code, error_class in _BY_CODE.items():
        if code == "client_upgrade_required":
            assert re.search(r"`client_upgrade_required`.*UpgradeRequiredError", guide)
            assert "Run `pip install -U synth-ai`" in guide
            continue
        row = re.search(rf"^\| `{code}` \|.+`{error_class.__name__}` \| ([^|]+)\|$", guide, re.M)
        assert row is not None, f"Missing next action for {code} ({error_class.__name__})"
        assert row[1].strip().endswith("."), f"Missing next action for {code}"
