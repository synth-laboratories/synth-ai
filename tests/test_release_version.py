"""Tests for tag-to-PEP440 conversion, version stamping, and PyPI-only workflow hygiene."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "release_version", ROOT / "scripts" / "release_version.py"
)
assert _SPEC is not None and _SPEC.loader is not None
release_version = importlib.util.module_from_spec(_SPEC)
sys.modules["release_version"] = release_version
_SPEC.loader.exec_module(release_version)


@pytest.mark.parametrize(
    ("tag", "expected_pep440", "expected_semver", "expected_channel"),
    [
        ("v0.42.0", "0.42.0", "0.42.0", "stable"),
        ("v1.0.0", "1.0.0", "1.0.0", "stable"),
        ("v0.43.0-nightly.20261128", "0.43.0.dev20261128", "0.43.0-nightly.20261128", "nightly"),
        ("v0.43.0.dev20261128", "0.43.0.dev20261128", "0.43.0-nightly.20261128", "nightly"),
        ("v0.43.0-rc.1", "0.43.0rc1", "0.43.0-rc.1", "rc"),
        ("v0.43.0rc12", "0.43.0rc12", "0.43.0-rc.12", "rc"),
    ],
)
def test_tag_to_pep440_supported_formats(
    tag: str, expected_pep440: str, expected_semver: str, expected_channel: str
) -> None:
    parsed = release_version.tag_to_pep440(tag)
    assert parsed.pep440_version == expected_pep440
    assert parsed.semver_version == expected_semver
    assert parsed.channel.value == expected_channel


@pytest.mark.parametrize(
    "bad_tag",
    [
        "0.42.0",
        "v0.42",
        "v0.43.0-nightly.20260230",
        "v0.43.0.dev20261301",
        "v0.43.0-rc.0",
        "v0.43.0rc0",
        "v0.43.0-alpha.1",
        "v0.43.0+local",
        "nightly-20261128",
        "",
    ],
)
def test_tag_to_pep440_rejects_invalid_tags(bad_tag: str) -> None:
    with pytest.raises(ValueError):
        release_version.tag_to_pep440(bad_tag)


def test_verify_tag_version_alignment_accepts_nightly_semver_tag_for_pep440_package() -> None:
    parsed = release_version.verify_tag_version_alignment(
        "v0.43.0-nightly.20261128",
        "0.43.0.dev20261128",
        "0.43.0.dev20261128",
    )
    assert parsed.pep440_version == "0.43.0.dev20261128"
    assert parsed.channel is release_version.ReleaseChannel.NIGHTLY


def test_verify_tag_version_alignment_rejects_mismatches() -> None:
    with pytest.raises(ValueError, match=r"pyproject\.toml declares"):
        release_version.verify_tag_version_alignment(
            "v0.43.0-nightly.20261128",
            "0.42.0",
            "0.43.0.dev20261128",
        )
    with pytest.raises(ValueError, match=r"_version\.py declares"):
        release_version.verify_tag_version_alignment(
            "v0.43.0-nightly.20261128",
            "0.43.0.dev20261128",
            "0.42.0",
        )


def test_stamp_version_and_cli_verify_round_trip(tmp_path: Path, capsys) -> None:
    (tmp_path / "synth_ai").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "synth-ai"\nversion = "0.42.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "synth_ai" / "_version.py").write_text(
        '# Generated from pyproject.toml; do not edit.\n__version__ = "0.42.0"\n',
        encoding="utf-8",
    )

    code = release_version.main(
        ["--repo-root", str(tmp_path), "stamp", "--tag", "v0.43.0-nightly.20261128"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pep440_version"] == "0.43.0.dev20261128"
    assert payload["channel"] == "nightly"

    code = release_version.main(
        ["--repo-root", str(tmp_path), "verify", "--tag", "v0.43.0-nightly.20261128"]
    )
    assert code == 0
    capsys.readouterr()

    # Mismatched stable tag fails closed.
    bad_code = release_version.main(["--repo-root", str(tmp_path), "verify", "--tag", "v0.43.0"])
    assert bad_code == 1
    assert "error:" in capsys.readouterr().err


def test_github_workflows_allow_only_pypi_publishing() -> None:
    workflows_dir = ROOT / ".github" / "workflows"
    workflow_files = sorted(p.name for p in workflows_dir.glob("*.y*ml"))
    assert workflow_files == ["publish-pypi.yml"]
    content = (workflows_dir / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "tags: ['v*']" in content
    assert "scripts/release_version.py verify" in content
    assert "pypa/gh-action-pypi-publish@release/v1" in content
