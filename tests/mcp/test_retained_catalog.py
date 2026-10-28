"""The packaged MCP reference must agree with the runtime catalog."""

import json
from pathlib import Path

import pydantic
import pytest
from synth_ai.contracts.fingerprint import QUALIFICATION_PYDANTIC_VERSION
from synth_ai.contracts.mcp_schema import export_catalog


@pytest.mark.skipif(
    pydantic.__version__ != QUALIFICATION_PYDANTIC_VERSION,
    reason="Catalog generation requires the qualified pydantic serializer",
)
def test_retained_mcp_catalog_matches_the_runtime_export():
    root = Path(__file__).resolve().parents[2]
    assert json.loads((root / "contracts/mcp-v1.json").read_text()) == json.loads(export_catalog())
