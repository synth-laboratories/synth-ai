"""Owner-waived qualification checks (SYN-4862): typed, policy-bound, never read as passed."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.release

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


producer = load("build_qualification", ROOT / "scripts/release/build_qualification.py")
verifier = load("verify_qualified_release", ROOT / ".github/verify_qualified_release.py")
SOURCE = "d" * 40
VERSION = "0.23.5"
REASON = "no live receipt for this wheel; E2E acceptance skipped by owner directive"
POLICY = json.loads((ROOT / ".github/release-policy.json").read_text())
AUTHORITY = POLICY["releases"][VERSION]["owner_waivers"]


def wheel_for(tmp_path: Path, version: str) -> Path:
    wheel = tmp_path / f"synth_ai-{version}-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            f"synth_ai-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: synth-ai\nVersion: {version}\n",
        )
    wheel.write_bytes(buffer.getvalue())
    return wheel


def build(tmp_path: Path, waived: set[str], *, version: str = VERSION, authority=AUTHORITY):
    evidence = tmp_path / "receipt.json"
    evidence.write_text('{"fixture": "format test only"}')
    checks = {
        name: (
            producer.CheckInput(producer.CheckStatus.OWNER_WAIVED, None, reason=REASON)
            if name.value in waived
            else producer.CheckInput(producer.CheckStatus.PASSED, evidence)
        )
        for name in producer.CheckName
    }
    output = tmp_path / "q.zip"
    producer.build_bundle(
        checks=checks,
        wheel=wheel_for(tmp_path, version),
        source_sha=SOURCE,
        version=version,
        channel=producer.Channel.STABLE,
        output=output,
        owner_waivers=authority,
    )
    return output


def rewrite(archive: Path, edit) -> Path:
    """Copy the bundle with an edited manifest (and re-hashed evidence where edit says so)."""
    with zipfile.ZipFile(archive) as source:
        members = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(members["qualification.json"])
    edit(manifest, members)
    for check in manifest["checks"]:
        check["evidence_sha256"] = hashlib.sha256(members[check["evidence_path"]]).hexdigest()
    members["qualification.json"] = json.dumps(manifest).encode()
    tampered = archive.with_name("tampered.zip")
    with zipfile.ZipFile(tampered, "w") as target:
        for name, raw in members.items():
            target.writestr(name, raw)
    return tampered


def verify(tmp_path: Path, archive: Path, version: str = VERSION):
    return verifier.verify_bundle(
        archive,
        tmp_path / "verified",
        source=SOURCE,
        version=version,
        channel="stable",
        archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


def test_policy_waives_only_the_e2e_checks_under_the_recorded_directive():
    assert POLICY["releases"][VERSION]["deferrable"] == []
    assert AUTHORITY["checks"] == [
        "native-index-fast",
        "native-index-deep",
        "native-monitor",
        "native-swarms",
    ]
    assert "E2E stays skipped" in AUTHORITY["directive"]
    assert AUTHORITY["ticket"] == "SYN-4862"
    assert verifier.owner_waiver_authority(VERSION) == AUTHORITY
    assert verifier.owner_waiver_authority("0.23.1") == {}


def test_waived_checks_are_recorded_as_waived_never_as_passed(tmp_path):
    waived = {"native-monitor", "native-swarms"}
    archive = build(tmp_path, waived)
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("qualification.json"))
    assert manifest["status"] == "passed_with_owner_waivers"
    statuses = {check["name"]: check["status"] for check in manifest["checks"]}
    assert {name for name, status in statuses.items() if status == "owner_waived"} == waived

    receipt = verify(tmp_path, archive)
    assert set(receipt["owner_waived"]) == waived
    assert receipt["owner_waived"]["native-swarms"] == {
        "reason": REASON,
        **{key: AUTHORITY[key] for key in ("directive", "approved_by", "approved_at", "ticket")},
    }
    assert set(receipt["passed"]) == verifier.CHECKS - waived
    assert receipt["deferred"] == {}


def test_fully_passed_receipt_keeps_its_shape(tmp_path):
    receipt = verify(tmp_path, build(tmp_path, set()))
    assert "owner_waived" not in receipt and "passed" not in receipt


def test_waived_check_under_a_plain_passed_manifest_is_refused(tmp_path):
    archive = build(tmp_path, {"native-swarms"})
    tampered = rewrite(archive, lambda manifest, _: manifest.update(status="passed"))
    with pytest.raises(ValueError, match="declare owner waivers"):
        verify(tmp_path, tampered)
    assert not (tmp_path / "verified/dist").exists()


def test_waiver_status_without_a_waived_check_is_refused(tmp_path):
    archive = build(tmp_path, set())
    tampered = rewrite(
        archive, lambda manifest, _: manifest.update(status="passed_with_owner_waivers")
    )
    with pytest.raises(ValueError, match="declare owner waivers"):
        verify(tmp_path, tampered)


@pytest.mark.parametrize("name", ["sdk-contracts", "fresh-install", "staging"])
def test_checks_outside_the_directive_cannot_be_waived(tmp_path, name):
    widened = {**AUTHORITY, "checks": [*AUTHORITY["checks"], name]}
    archive = build(tmp_path, {name}, authority=widened)
    with pytest.raises(ValueError, match="not owner-waivable"):
        verify(tmp_path, archive)


@pytest.mark.parametrize("name", ["sdk-contracts", "fresh-install", "staging"])
def test_producer_refuses_to_waive_checks_outside_the_directive(tmp_path, name):
    with pytest.raises(producer.QualificationError, match="waiver_invalid"):
        build(tmp_path, {name})
    assert not (tmp_path / "q.zip").exists()


def test_version_without_an_owner_directive_cannot_waive(tmp_path):
    archive = build(tmp_path, {"native-swarms"}, version="0.23.1")
    with pytest.raises(ValueError, match="not owner-waivable"):
        verify(tmp_path, archive, version="0.23.1")


def test_waiver_record_must_match_the_policy_directive(tmp_path):
    def forge(manifest, members):
        path = next(c["evidence_path"] for c in manifest["checks"] if c["name"] == "native-swarms")
        record = json.loads(members[path])
        record["directive"] = "owner said ship it"
        members[path] = json.dumps(record).encode()

    tampered = rewrite(build(tmp_path, {"native-swarms"}), forge)
    with pytest.raises(ValueError, match="policy authority"):
        verify(tmp_path, tampered)


@pytest.mark.parametrize("reason", ["", "   "])
def test_waiver_requires_a_reason(tmp_path, reason):
    def blank(manifest, members):
        for check in manifest["checks"]:
            if check["name"] == "native-swarms":
                check["reason"] = reason
                record = json.loads(members[check["evidence_path"]])
                record["reason"] = reason
                members[check["evidence_path"]] = json.dumps(record).encode()

    tampered = rewrite(build(tmp_path, {"native-swarms"}), blank)
    with pytest.raises(ValueError, match="reason"):
        verify(tmp_path, tampered)
