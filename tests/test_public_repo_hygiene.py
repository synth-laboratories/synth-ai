"""Published source contains product contracts rather than internal run receipts."""

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    re.compile("/" + r"Users/"),
    re.compile("/" + r"home/[^/\s]+/"),
    re.compile(r"\bslot[0-9]+\b"),
    re.compile("spend" + r"/ledger(?:\.md)?"),
)


def test_public_source_has_no_internal_workspace_receipts():
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        pytest.skip("source distribution has no Git metadata")
    findings = []
    for name in result.stdout.split("\0"):
        path = ROOT / name
        if not name or not path.is_file():
            continue
        if path.suffix == ".pyc" or (
            path.parent == ROOT and path.name.startswith(("HANDOFF_", "SDK_0230_"))
        ):
            findings.append(name)
            continue
        text = path.read_text(errors="replace")
        if any(pattern.search(text) for pattern in PATTERNS):
            findings.append(name)
    assert not findings, f"Internal workspace artifacts remain: {findings}"
