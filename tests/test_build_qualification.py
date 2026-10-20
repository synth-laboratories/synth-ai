"""The release producer and the publisher's verifier agree (SYN-4373); synthetic inputs only."""

from __future__ import annotations

import hashlib
import importlib.util
import io
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
SOURCE = "c" * 40
VERSION = "0.23.0"
REASON = "Swarms not admitted in v0.23 production admission; qualified separately (SYN-4509)"
POLICY = ROOT / ".github/release-policy.json"


@pytest.fixture
def inputs(tmp_path):
    wheel = tmp_path / f"synth_ai-{VERSION}-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            f"synth_ai-{VERSION}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: synth-ai\nVersion: {VERSION}\n",
        )
    wheel.write_bytes(buffer.getvalue())
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"fixture": "format test only"}')
    checks = {
        name: producer.CheckInput(producer.CheckStatus.PASSED, evidence)
        for name in producer.CheckName
    }
    return wheel, checks


def build(tmp_path, wheel, checks, **extra):
    output = tmp_path / "q.zip"
    producer.build_bundle(
        checks=checks,
        wheel=wheel,
        source_sha=SOURCE,
        version=VERSION,
        channel=producer.Channel.STABLE,
        output=output,
        **extra,
    )
    return output


def verify(tmp_path, archive):
    return verifier.verify_bundle(
        archive,
        tmp_path / "verified",
        source=SOURCE,
        version=VERSION,
        channel="stable",
        archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


def test_check_names_match_the_verifier():
    assert {item.value for item in producer.CheckName} == verifier.CHECKS


def test_all_passed_bundle_is_accepted_by_the_verifier(tmp_path, inputs):
    wheel, checks = inputs
    receipt = verify(tmp_path, build(tmp_path, wheel, checks))
    assert receipt["deferred"] == {}


def test_pending_check_is_refused_by_the_producer(tmp_path, inputs):
    wheel, checks = inputs
    checks[producer.CheckName.NATIVE_MONITOR] = producer.CheckInput(
        producer.CheckStatus.PENDING, None
    )
    with pytest.raises(producer.QualificationError) as error:
        build(tmp_path, wheel, checks)
    assert error.value.code is producer.QualificationErrorCode.CHECK_NOT_PASSED
    assert not (tmp_path / "q.zip").exists()


def swarms_deferred(reason=REASON, ticket="SYN-4509"):
    return producer.CheckInput(producer.CheckStatus.DEFERRED, None, reason, ticket)


def test_declared_deferral_round_trips_through_the_verifier(tmp_path, inputs):
    wheel, checks = inputs
    checks[producer.CheckName.NATIVE_SWARMS] = swarms_deferred()
    archive = build(
        tmp_path, wheel, checks, deferrable=frozenset({producer.CheckName.NATIVE_SWARMS})
    )
    assert verify(tmp_path, archive)["deferred"] == {
        "native-swarms": {"reason": REASON, "ticket": "SYN-4509"}
    }


def test_undeclared_deferral_is_refused_by_the_producer(tmp_path, inputs):
    wheel, checks = inputs
    checks[producer.CheckName.NATIVE_SWARMS] = swarms_deferred()
    with pytest.raises(producer.QualificationError) as error:
        build(tmp_path, wheel, checks)
    assert error.value.code is producer.QualificationErrorCode.DEFERRAL_INVALID


@pytest.mark.parametrize(("reason", "ticket"), [("", "SYN-4509"), ("why", ""), ("why", "nope")])
def test_incomplete_deferral_is_refused_by_the_producer(tmp_path, inputs, reason, ticket):
    wheel, checks = inputs
    checks[producer.CheckName.NATIVE_SWARMS] = swarms_deferred(reason, ticket)
    with pytest.raises(producer.QualificationError) as error:
        build(tmp_path, wheel, checks, deferrable=frozenset({producer.CheckName.NATIVE_SWARMS}))
    assert error.value.code is producer.QualificationErrorCode.DEFERRAL_INVALID


def test_repo_policy_makes_only_swarms_deferrable():
    import json

    policy = json.loads(POLICY.read_text())
    assert policy["releases"][VERSION]["deferrable"] == ["native-swarms"]
