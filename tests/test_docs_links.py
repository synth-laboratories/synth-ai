"""Public contributor links resolve without private sibling repositories."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_document_relative_links_exist():
    for name in ("README.md", "RELEASING.md", "CONTRIBUTING.md"):
        document = ROOT / name
        if not document.exists():
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            assert (document.parent / target.split("#", 1)[0]).exists(), (name, target)


def test_package_docstrings_have_public_contract_references():
    for path in (ROOT / "synth_ai").rglob("*.py"):
        text = path.read_text()
        assert all(
            term not in text for term in ("backend/notes", "specifications/tanha", "gpt-5.6-luna")
        ), path
