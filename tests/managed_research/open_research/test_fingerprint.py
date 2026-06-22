"""Anonymous fingerprint persistence tests."""

from __future__ import annotations

from pathlib import Path

from synth_ai.managed_research.open_research.fingerprint import load_or_create_fingerprint


def test_load_or_create_mints_and_persists(tmp_path: Path) -> None:
    target = tmp_path / "or_fingerprint"
    fp = load_or_create_fingerprint(target)
    assert target.exists()
    assert target.read_text(encoding="utf-8").strip() == fp
    assert len(fp) == 32  # uuid4 hex


def test_load_or_create_is_stable_for_existing_value(tmp_path: Path) -> None:
    target = tmp_path / "or_fingerprint"
    first = load_or_create_fingerprint(target)
    second = load_or_create_fingerprint(target)
    assert first == second


def test_env_override_path_is_honored(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "override" / "fp"
    monkeypatch.setenv("MANAGED_RESEARCH_OR_FINGERPRINT_PATH", str(override))
    # Call without arg so the default resolver picks up the override.
    fp = load_or_create_fingerprint()
    assert override.exists()
    assert override.read_text(encoding="utf-8").strip() == fp
