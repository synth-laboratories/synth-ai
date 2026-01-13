"""OpenCode skill helpers.

This module ships a small set of OpenCode-compatible skills inside the synth-ai wheel
and provides utilities to:
- list skills bundled with the SDK
- install them into a user's OpenCode local skills directory
- materialize an OpenCode config directory for the Synth TUI (writable path)
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Iterable

_SYNTH_PACKAGE = "synth_ai"


def _xdg_config_home() -> Path:
    xdg = (os.environ.get("XDG_CONFIG_HOME") or "").strip()
    return Path(xdg).expanduser() if xdg else (Path.home() / ".config")


def default_opencode_global_skills_dir() -> Path:
    """Default OpenCode global skills directory.

    OpenCode convention:
    - project: .opencode/skill/<name>/SKILL.md
    - global:  ~/.config/opencode/skill/<name>/SKILL.md

    Allow overriding via OPENCODE_SKILLS_DIR.
    """

    override = (os.environ.get("OPENCODE_SKILLS_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (_xdg_config_home() / "opencode" / "skill").resolve()


def _packaged_opencode_skill_root():
    return files(_SYNTH_PACKAGE).joinpath("skills", "opencode", "skill")


def list_packaged_opencode_skill_names() -> list[str]:
    """List OpenCode skill names bundled inside the synth-ai wheel."""

    root = _packaged_opencode_skill_root()
    if not root.is_dir():
        return []

    names: list[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        skill_md = child.joinpath("SKILL.md")
        if skill_md.is_file():
            names.append(child.name)
    return sorted(names)


def read_packaged_opencode_skill_markdown(skill_name: str) -> str:
    """Read the packaged SKILL.md content for a given skill."""

    src = _packaged_opencode_skill_root().joinpath(skill_name, "SKILL.md")
    if not src.is_file():
        available = ", ".join(list_packaged_opencode_skill_names())
        raise FileNotFoundError(f"Unknown skill '{skill_name}'. Available: {available or '(none)'}")
    return src.read_text(encoding="utf-8")


def install_packaged_opencode_skill(
    *,
    skill_name: str,
    dest_skills_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Install a packaged skill into a local OpenCode skills directory.

    Writes:
      <dest_skills_dir>/<skill_name>/SKILL.md
    """

    if dest_skills_dir is None:
        dest_skills_dir = default_opencode_global_skills_dir()

    dest_skills_dir = dest_skills_dir.expanduser().resolve()
    out = dest_skills_dir / skill_name / "SKILL.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        raise FileExistsError(f"Skill already exists at {out} (use force=True to overwrite)")

    out.write_text(read_packaged_opencode_skill_markdown(skill_name), encoding="utf-8")
    return out


def install_all_packaged_opencode_skills(
    *,
    dest_skills_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for name in list_packaged_opencode_skill_names():
        paths.append(
            install_packaged_opencode_skill(
                skill_name=name, dest_skills_dir=dest_skills_dir, force=force
            )
        )
    return paths


def _copy_resource_tree(*, src, dest: Path) -> None:
    # `src` is an importlib.resources Traversable. We avoid importing the protocol for 3.11 simplicity.
    if src.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _copy_resource_tree(src=child, dest=dest / child.name)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def materialize_tui_opencode_config_dir(
    *,
    dest_dir: Path | None = None,
    force: bool = True,
    include_packaged_skills: Iterable[str] | None = None,
) -> Path:
    """Create a writable OPENCODE_CONFIG_DIR for the Synth TUI.

    Why: the package directory inside site-packages can be read-only, but OpenCode config
    often needs to be a normal filesystem directory that we can extend (e.g., include skills).

    This function copies:
    - Synth TUI's bundled OpenCode config: synth_ai/tui/opencode_config/**
    - Selected packaged skills from synth_ai/skills/opencode/skill/** (default: all)
    """

    if dest_dir is None:
        dest_dir = (Path.home() / ".synth-ai" / "tui" / "opencode_config").resolve()
    else:
        dest_dir = dest_dir.expanduser().resolve()

    if dest_dir.exists() and force:
        # Best-effort cleanup of known files only; don't delete arbitrary user files.
        # We overwrite on copy anyway, so this is mostly to ensure directories exist.
        pass

    # 1) Copy base TUI OpenCode config
    base_src = files(_SYNTH_PACKAGE).joinpath("tui", "opencode_config")
    if base_src.is_dir():
        _copy_resource_tree(src=base_src, dest=dest_dir)

    # 2) Copy packaged skills into <dest>/skill/<name>/SKILL.md (additive)
    skill_root = _packaged_opencode_skill_root()
    if skill_root.is_dir():
        names = list_packaged_opencode_skill_names()
        if include_packaged_skills is not None:
            allow = set(include_packaged_skills)
            names = [n for n in names if n in allow]
        for name in names:
            src = skill_root.joinpath(name)
            if src.is_dir():
                _copy_resource_tree(src=src, dest=dest_dir / "skill" / name)

    return dest_dir
