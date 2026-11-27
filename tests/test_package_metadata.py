"""PyPI metadata and README link checks (SYN-5917)."""

from __future__ import annotations

import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.check_package import (
    REQUIRED_PROJECT_URLS,
    check_project_urls,
    check_readme_links,
    check_wheel_metadata,
    find_relative_markdown_links,
)

ROOT = Path(__file__).resolve().parents[1]


def test_check_project_urls_rejects_missing_table_and_keys() -> None:
    with pytest.raises(ValueError, match=r"missing \[project\.urls\]"):
        check_project_urls({"project": {"name": "synth-ai"}})

    with pytest.raises(ValueError, match="missing required keys"):
        check_project_urls({"project": {"urls": {"Homepage": "https://www.usesynth.ai"}}})

    with pytest.raises(ValueError, match="must use https://"):
        check_project_urls(
            {
                "project": {
                    "urls": {
                        "Homepage": "http://www.usesynth.ai",
                        "Documentation": "https://www.usesynth.ai/docs",
                        "Repository": "https://github.com/synth-laboratories/synth-ai",
                        "Changelog": "https://github.com/synth-laboratories/synth-ai/blob/main/CHANGELOG.md",
                        "Issues": "https://github.com/synth-laboratories/synth-ai/issues",
                    }
                }
            }
        )


def test_check_readme_links_rejects_relative_markdown_links() -> None:
    sample = "See [quickstart](docs/quickstart.md) and [pin](contracts/SOURCE.toml)."
    assert find_relative_markdown_links(sample) == ["docs/quickstart.md", "contracts/SOURCE.toml"]
    with pytest.raises(ValueError, match="relative markdown links"):
        check_readme_links(sample)


def test_check_readme_links_allows_https_anchors_and_code_blocks() -> None:
    sample = "\n".join(
        [
            "See [docs](https://www.usesynth.ai/docs) and [top](#top).",
            "Literal `[example](docs/local.md)` in code span.",
            "```markdown",
            "[fenced](docs/fenced.md)",
            "```",
        ]
    )
    check_readme_links(sample)


def test_check_wheel_metadata_fails_closed_on_missing_urls_or_relative_links(
    tmp_path: Path,
) -> None:
    bad_wheel = tmp_path / "synth_ai-0.41.0-py3-none-any.whl"
    with zipfile.ZipFile(bad_wheel, "w") as archive:
        archive.writestr(
            "synth_ai-0.41.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: synth-ai\nVersion: 0.41.0\n\nSee [q](docs/quickstart.md).\n",
        )
    with pytest.raises(ValueError, match="missing Project-URL"):
        check_wheel_metadata(bad_wheel)

    rel_wheel = tmp_path / "synth_ai-0.41.0-rel-py3-none-any.whl"
    headers = "\n".join(
        f"Project-URL: {key}, https://example.test/{key.lower()}" for key in REQUIRED_PROJECT_URLS
    )
    with zipfile.ZipFile(rel_wheel, "w") as archive:
        archive.writestr(
            "synth_ai-0.41.0.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: synth-ai\nVersion: 0.41.0\n{headers}\n\nSee [q](docs/quickstart.md).\n",
        )
    with pytest.raises(ValueError, match="relative markdown links"):
        check_wheel_metadata(rel_wheel)


def test_checkout_pyproject_has_required_project_urls() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    check_project_urls(pyproject)


def test_checkout_readme_has_no_relative_markdown_links() -> None:
    check_readme_links((ROOT / "README.md").read_text())


def test_checkout_pyproject_mcp_extra_includes_jsonschema() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mcp_deps = pyproject["project"]["optional-dependencies"]["mcp"]
    assert any(dep.startswith("jsonschema") for dep in mcp_deps), mcp_deps
