from __future__ import annotations

from pathlib import Path

import pytest

from synth_ai.utils.agents import (
    SYNTH_DIV_END,
    SYNTH_DIV_START,
    AGENTS_TEXT,
    write_agents_md,
)


@pytest.fixture()
def agents_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path / "AGENTS.md"


def test_write_agents_md_handles_missing_file(agents_path: Path) -> None:
    write_agents_md()

    content = agents_path.read_text(encoding="utf-8")
    expected = f"{SYNTH_DIV_START}\n{AGENTS_TEXT}\n{SYNTH_DIV_END}\n"
    assert content == expected


def test_write_agents_md_removes_orphan_end(agents_path: Path) -> None:
    agents_path.write_text(
        "    who\n\n### ---- SYNTH SECTION END ----\n",
        encoding="utf-8",
    )

    write_agents_md()

    content = agents_path.read_text(encoding="utf-8")
    expected = f"{SYNTH_DIV_START}\n{AGENTS_TEXT}\n{SYNTH_DIV_END}\n"
    assert content == expected


def test_write_agents_md_updates_existing_block(agents_path: Path) -> None:
    agents_path.write_text(
        f"header\n\n{SYNTH_DIV_START}\nold\ncontent\n{SYNTH_DIV_END}\n\nfooter\n",
        encoding="utf-8",
    )

    write_agents_md()

    content = agents_path.read_text(encoding="utf-8")
    expected = (
        "header\n\n"
        f"{SYNTH_DIV_START}\n{AGENTS_TEXT}\n{SYNTH_DIV_END}\n"
        "\nfooter\n"
    )
    assert content == expected


def test_write_agents_md_consolidates_multiple_sections(agents_path: Path) -> None:
    agents_path.write_text(
        "who\n\n"
        "### ---- SYNTH SECTION END ----\n\n"
        f"{SYNTH_DIV_START}\n{AGENTS_TEXT}\n{SYNTH_DIV_END}\n\n"
        f"{SYNTH_DIV_START}\n{AGENTS_TEXT}\n{SYNTH_DIV_END}\n",
        encoding="utf-8",
    )

    write_agents_md()

    content = agents_path.read_text(encoding="utf-8")
    expected = (
        "who\n\n"
        f"{SYNTH_DIV_START}\n{AGENTS_TEXT}\n{SYNTH_DIV_END}\n"
    )
    assert content == expected
