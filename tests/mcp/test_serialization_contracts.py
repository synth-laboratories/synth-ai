"""Public MCP JSON contract. See backend scientific delivery spec and MX-10."""

import json
from pathlib import Path

import pytest
from offline_backend import assert_serializable_public_tool

CASES = json.loads(Path(__file__).with_name("serialization_contract_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["tool"])
def test_public_call_tool_returns_json(monkeypatch, case):
    assert_serializable_public_tool(monkeypatch, case)
