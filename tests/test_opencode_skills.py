from __future__ import annotations

from pathlib import Path

import pytest


def test_packaged_opencode_skill_is_listed() -> None:
    from synth_ai.sdk.opencode_skills import list_packaged_opencode_skill_names

    assert "synth-api" in list_packaged_opencode_skill_names()


def test_packaged_opencode_skill_can_be_read() -> None:
    from synth_ai.sdk.opencode_skills import read_packaged_opencode_skill_markdown

    text = read_packaged_opencode_skill_markdown("synth-api")
    assert "name: synth-api" in text


def test_install_packaged_opencode_skill(tmp_path: Path) -> None:
    from synth_ai.sdk.opencode_skills import install_packaged_opencode_skill

    out = install_packaged_opencode_skill(
        skill_name="synth-api", dest_skills_dir=tmp_path, force=False
    )
    assert out.exists()
    assert out.name == "SKILL.md"
    assert out.parent.name == "synth-api"


def test_install_existing_requires_force(tmp_path: Path) -> None:
    from synth_ai.sdk.opencode_skills import install_packaged_opencode_skill

    install_packaged_opencode_skill(skill_name="synth-api", dest_skills_dir=tmp_path, force=False)
    with pytest.raises(FileExistsError):
        install_packaged_opencode_skill(
            skill_name="synth-api", dest_skills_dir=tmp_path, force=False
        )
