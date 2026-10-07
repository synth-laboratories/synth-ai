"""SYN-4008: every unused metadata entry is reviewed and resolves to its producer.

Metadata-only operations are not presented as executed SDK methods. New orphan
entries fail this local gate rather than silently extending an audit allowlist.
"""

import ast
import json
import re
from pathlib import Path

from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS

ROOT = Path(__file__).resolve().parents[2]


def test_metadata_dispositions_match_exact_unused_inventory__SYN4008():
    strings = set()
    for path in (ROOT / "synth_ai").rglob("*.py"):
        if path.name == "operations.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.add(node.value)
    unused = set(RESEARCH_OPERATIONS) - strings
    entries = json.loads((ROOT / "openapi/operation_metadata_dispositions.json").read_text())
    assert len({item["operation_id"] for item in entries}) == len(entries)
    assert {item["operation_id"] for item in entries} == unused
    specification = json.loads((ROOT / "openapi/research-v1.json").read_text())

    def normalize(path):
        return re.sub(r"\{[^}]+\}", "{}", path)

    for item in entries:
        assert item["disposition"] in {"legacy_direct_call", "backend_metadata_catalogue"}
        assert item["reason"]
        metadata = RESEARCH_OPERATIONS[item["operation_id"]]
        assert metadata.method.value == item["method"]
        assert metadata.path_template == item["path"]
        operation = next(
            (
                operation
                for path, methods in specification["paths"].items()
                for method, operation in methods.items()
                if method.upper() == item["method"] and normalize(path) == normalize(item["path"])
            ),
            None,
        )
        assert operation is not None, item["operation_id"]
        assert operation["operationId"] == item["operation_id"]
        if item["disposition"] == "legacy_direct_call":
            assert item["call_sites"]
        else:
            assert item["call_sites"] == []
