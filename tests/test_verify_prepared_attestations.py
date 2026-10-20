"""verify_prepared after the PyPI publish action adds PEP 740 attestations (SYN-4373)."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.release

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_qualified_release", ROOT / ".github/verify_qualified_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
NAME = "synth_ai-0.23.0-py3-none-any.whl"
WHEEL = b"wheel bytes"
SHA = hashlib.sha256(WHEEL).hexdigest()


def attestation(name: str = NAME, sha: str = SHA) -> bytes:
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": name, "digest": {"sha256": sha}}],
    }
    envelope = {"statement": base64.b64encode(json.dumps(statement).encode()).decode()}
    return json.dumps({"version": 1, "envelope": envelope}).encode()


@pytest.fixture
def prepared(tmp_path):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / NAME).write_bytes(WHEEL)
    qualification = b'{"schema": "fixture"}'
    (tmp_path / "qualification.json").write_bytes(qualification)
    receipt = json.dumps(
        {
            "qualification_sha256": hashlib.sha256(qualification).hexdigest(),
            "distributions": {NAME: SHA},
        }
    ).encode()
    (tmp_path / "publish-receipt.json").write_bytes(receipt)
    return tmp_path, hashlib.sha256(receipt).hexdigest()


def test_unpublished_prepared_directory_still_verifies(prepared):
    output, receipt_sha = prepared
    assert release.verify_prepared(output, receipt_sha)["distributions"] == {NAME: SHA}


def test_attestation_written_by_the_publish_action_is_accepted(prepared):
    output, receipt_sha = prepared
    (output / "dist" / (NAME + ".publish.attestation")).write_bytes(attestation())
    assert release.verify_prepared(output, receipt_sha)["distributions"] == {NAME: SHA}


def test_attestation_for_other_bytes_is_refused(prepared):
    output, receipt_sha = prepared
    (output / "dist" / (NAME + ".publish.attestation")).write_bytes(attestation(sha="0" * 64))
    with pytest.raises(ValueError, match="does not attest"):
        release.verify_prepared(output, receipt_sha)


def test_malformed_attestation_is_refused(prepared):
    output, receipt_sha = prepared
    (output / "dist" / (NAME + ".publish.attestation")).write_bytes(b"{}")
    with pytest.raises(ValueError, match="not a valid PEP 740"):
        release.verify_prepared(output, receipt_sha)


@pytest.mark.parametrize("extra", ["other.whl", "other.whl.publish.attestation", "notes.txt"])
def test_any_other_extra_file_is_still_refused(prepared, extra):
    output, receipt_sha = prepared
    (output / "dist" / extra).write_bytes(attestation())
    with pytest.raises(ValueError, match="distribution set changed"):
        release.verify_prepared(output, receipt_sha)


def test_missing_or_changed_distribution_is_still_refused(prepared):
    output, receipt_sha = prepared
    (output / "dist" / NAME).write_bytes(b"changed")
    with pytest.raises(ValueError, match="distribution changed"):
        release.verify_prepared(output, receipt_sha)
    (output / "dist" / NAME).unlink()
    with pytest.raises(ValueError, match="distribution set changed"):
        release.verify_prepared(output, receipt_sha)
