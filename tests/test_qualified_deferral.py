"""Typed, policy-bounded deferral of a qualification check (SYN-4373); synthetic archives only."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qualified_release", ROOT / ".github/verify_qualified_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
SOURCE = "a" * 40
VERSION = "0.23.0"
REASON = "Swarms not admitted in v0.23 production admission; qualified separately"
TICKET = "SYN-4509"


def build(tmp_path, deferred, *, reason=REASON, ticket=TICKET, record=None):
    """A stable 0.23.0 bundle in which the named checks are deferred, the rest passed."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        wheel.writestr(
            "synth_ai-0.23.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: synth-ai\nVersion: {VERSION}\n",
        )
    raw = buffer.getvalue()
    filename = f"synth_ai-{VERSION}-py3-none-any.whl"
    files = {filename: hashlib.sha256(raw).hexdigest()}
    checks, evidence = [], {}
    for name in sorted(release.CHECKS):
        if name in deferred:
            body = json.dumps(
                record if record is not None else {"check": name, "reason": reason, "ticket": ticket}
            ).encode()
        else:
            body = b'{"fixture": "provenance validator only"}'
        path = f"evidence/{name}.json"
        evidence[path] = body
        check = dict(
            name=name,
            status="deferred" if name in deferred else "passed",
            source_sha=SOURCE,
            distributions=files,
            evidence_path=path,
            evidence_sha256=hashlib.sha256(body).hexdigest(),
        )
        if name in deferred:
            check.update(reason=reason, ticket=ticket)
        checks.append(check)
    manifest = dict(
        schema=release.SCHEMA,
        status="passed",
        source_sha=SOURCE,
        version=VERSION,
        channel="stable",
        distributions=[dict(filename=filename, sha256=files[filename])],
        checks=checks,
    )
    archive = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(filename, raw)
        zipped.writestr("qualification.json", json.dumps(manifest))
        for path, body in evidence.items():
            zipped.writestr(path, body)
    return archive


def verify(tmp_path, archive):
    return release.verify_bundle(
        archive,
        tmp_path / "verified",
        source=SOURCE,
        version=VERSION,
        channel="stable",
        archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


def test_declared_swarms_deferral_is_accepted_and_recorded(tmp_path):
    receipt = verify(tmp_path, build(tmp_path, {"native-swarms"}))
    assert receipt["deferred"] == {"native-swarms": {"reason": REASON, "ticket": TICKET}}
    assert receipt["checks"] == sorted(release.CHECKS)


@pytest.mark.parametrize("name", ["native-monitor", "native-index-deep", "staging", "sdk-contracts"])
def test_undeclared_deferral_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match="not deferrable"):
        verify(tmp_path, build(tmp_path, {name}))
    assert not (tmp_path / "verified/dist").exists()


def test_deferring_swarms_alongside_another_check_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not deferrable"):
        verify(tmp_path, build(tmp_path, {"native-swarms", "native-monitor"}))


@pytest.mark.parametrize("reason", ["", "   ", None])
def test_missing_reason_is_refused(tmp_path, reason):
    with pytest.raises(ValueError, match="reason"):
        verify(tmp_path, build(tmp_path, {"native-swarms"}, reason=reason))


@pytest.mark.parametrize("ticket", ["", None, "swarms", "SYN-", "4509"])
def test_missing_or_malformed_ticket_is_refused(tmp_path, ticket):
    with pytest.raises(ValueError, match="ticket"):
        verify(tmp_path, build(tmp_path, {"native-swarms"}, ticket=ticket))


def test_deferral_record_must_match_the_declared_reason_and_ticket(tmp_path):
    record = {"check": "native-swarms", "reason": "something else", "ticket": TICKET}
    with pytest.raises(ValueError, match="deferral record"):
        verify(tmp_path, build(tmp_path, {"native-swarms"}, record=record))


def test_version_without_a_policy_defers_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "POLICY_PATH", tmp_path / "missing-policy.json")
    with pytest.raises(ValueError, match="not deferrable"):
        verify(tmp_path, build(tmp_path, {"native-swarms"}))


def test_policy_file_declares_only_swarms_for_v023():
    policy = json.loads((ROOT / ".github/release-policy.json").read_text())
    assert policy["schema"] == "synth.sdk.release-policy.v1"
    assert policy["releases"]["0.23.0"]["deferrable"] == ["native-swarms"]
