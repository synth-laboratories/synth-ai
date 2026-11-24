"""Validate the built wheel, PyPI metadata, source budget and isolated installation."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from email.parser import Parser as EmailParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PROJECT_URLS: tuple[str, ...] = (
    "Homepage",
    "Documentation",
    "Repository",
    "Changelog",
    "Issues",
)
# Match markdown links `[text](target)` while skipping image/code fences below.
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_FENCE = re.compile(r"^\s*```")


def require(condition: bool, message: str) -> None:
    """Fail closed with ValueError when a package acceptance invariant does not hold."""
    if not condition:
        raise ValueError(message)


def check_project_urls(pyproject: Mapping[str, object]) -> None:
    """Require complete HTTPS `[project.urls]` metadata for PyPI."""
    project = pyproject.get("project")
    require(isinstance(project, Mapping), "pyproject.toml is missing [project]")
    assert isinstance(project, Mapping)
    urls = project.get("urls")
    require(isinstance(urls, Mapping), "pyproject.toml is missing [project.urls]")
    assert isinstance(urls, Mapping)
    missing = [key for key in REQUIRED_PROJECT_URLS if key not in urls]
    require(not missing, f"[project.urls] missing required keys: {', '.join(missing)}")
    invalid = [
        f"{key}={urls[key]!r}"
        for key in REQUIRED_PROJECT_URLS
        if not isinstance(urls[key], str) or not str(urls[key]).startswith("https://")
    ]
    require(not invalid, f"[project.urls] must use https:// URLs: {', '.join(invalid)}")


def find_relative_markdown_links(markdown: str) -> list[str]:
    """Return every relative markdown link target outside fenced code blocks."""
    relative: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Strip inline code spans so literal markdown syntax examples do not false-positive.
        stripped = re.sub(r"`[^`]*`", "", line)
        for match in _MARKDOWN_LINK.finditer(stripped):
            target = match.group(1)
            if not target.startswith(("https://", "http://", "#", "mailto:")):
                relative.append(target)
    return relative


def check_readme_links(markdown: str) -> None:
    """Reject relative markdown links that break when rendered on PyPI."""
    relative = find_relative_markdown_links(markdown)
    require(
        not relative,
        f"README.md contains relative markdown links (broken on PyPI): {', '.join(relative)}",
    )


def check_wheel_metadata(wheel: Path) -> None:
    """Verify the wheel's dist-info METADATA ships required Project-URLs and absolute README links."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        require(len(metadata_names) == 1, "wheel must contain exactly one METADATA file")
        raw = archive.read(metadata_names[0]).decode("utf-8")
    message = EmailParser().parsestr(raw)
    raw_urls = message.get_all("Project-URL") or []
    parsed_urls: dict[str, str] = {}
    for entry in raw_urls:
        label, sep, url = entry.partition(",")
        if sep:
            parsed_urls[label.strip()] = url.strip()
    missing = [key for key in REQUIRED_PROJECT_URLS if key not in parsed_urls]
    require(not missing, f"wheel METADATA missing Project-URL entries: {', '.join(missing)}")
    invalid = [
        f"{key}={parsed_urls[key]!r}"
        for key in REQUIRED_PROJECT_URLS
        if not parsed_urls[key].startswith("https://")
    ]
    require(not invalid, f"wheel METADATA Project-URL must use https://: {', '.join(invalid)}")
    payload = message.get_payload()
    require(
        isinstance(payload, str) and bool(payload.strip()), "wheel METADATA missing README body"
    )
    assert isinstance(payload, str)
    check_readme_links(payload)


def main(argv: Sequence[str] | None = None) -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    version = pyproject["project"]["version"]
    parser = argparse.ArgumentParser(description=__doc__)
    # An exact candidate (scripts/qualify_release.py); the default stays the checkout's dist wheel.
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.metadata_only:
        check_project_urls(pyproject)
        check_readme_links((ROOT / "README.md").read_text())
        print({"pypi_metadata": "passed"})
        return 0
    wheel = arguments.wheel or ROOT / "dist" / f"synth_ai-{version}-py3-none-any.whl"
    if arguments.wheel is None:
        check_project_urls(pyproject)
        check_readme_links((ROOT / "README.md").read_text())
        source_bytes = sum(
            p.stat().st_size
            for p in (ROOT / "synth_ai").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    else:
        # Measure the candidate's own package source, not whatever the checkout holds now.
        with zipfile.ZipFile(wheel) as candidate:
            source_bytes = sum(
                member.file_size
                for member in candidate.infolist()
                if member.filename.startswith("synth_ai/") and not member.is_dir()
            )
    check_wheel_metadata(wheel)
    require(wheel.stat().st_size < 150 * 1024, "package acceptance invariant failed")
    # 0.32 added the task-verb CLI, run results and stream recovery (about 274 KiB).
    # Contract 1.25/1.26 (rollout views, per-role launch choices, allocation) brought
    # it to about 414 KiB: two operations' generated models, MCP tools and CLI views.
    # 0.41 public source docstrings and their generated sync copies add 44 KiB
    # (469,361 total source bytes). Track B (SYN-5973, SYN-5974) adds typed
    # Interactive Artifacts, Experiments, and Research Logs clients + MCP tool
    # schemas (~24 KiB). Keep a 540 KiB source cap and the unchanged 150 KiB
    # wheel cap; neither permits an accidental vendored dependency tree.
    source_limit = 540 * 1024
    require(
        source_bytes < source_limit,
        f"SDK source size {source_bytes} exceeds budget {source_limit} bytes",
    )
    with zipfile.ZipFile(wheel) as archive:
        require("synth_ai/py.typed" in archive.namelist(), "wheel must ship PEP 561 types")
        require(
            all(
                name.startswith(("synth_ai/", f"synth_ai-{version}.dist-info/"))
                for name in archive.namelist()
            ),
            "package acceptance invariant failed",
        )
        require(
            not any("/public/" in name or "/sdk/" in name for name in archive.namelist()),
            "package acceptance invariant failed",
        )
    # Under the checkout so workspace file boundaries also apply to local validation.
    with tempfile.TemporaryDirectory(prefix=".wheel-smoke-", dir=ROOT) as directory:
        environment = Path(directory) / "venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
        python = environment / "bin/python"
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-cache", str(wheel)], check=True
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                'import sys; from synth_ai import SynthClient, AsyncSynthClient; from synth_ai.errors import SwarmsAPIError, ConflictError; from synth_ai.models import Money, RunEvent; result = RunEvent.model_fields["observed_at"].annotation == str | None and Money(amount="0.10",currency="USD").amount == "0.10"; result or sys.exit("installed model acceptance failed")',
            ],
            check=True,
            cwd=directory,
        )
        subprocess.run(
            [str(environment / "bin/synth"), "--help"],
            check=True,
            cwd=directory,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [str(python), "-I", str(ROOT / "scripts/check_installed.py")],
            check=True,
            cwd=directory,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-cache",
                str(wheel) + "[mcp]",
                "pytest",
                "pytest-asyncio",
                "pyyaml",
            ],
            check=True,
        )
        subprocess.run(
            [str(python), "-I", str(ROOT / "scripts/check_installed.py"), "--mcp"],
            check=True,
            cwd=directory,
        )

        # Execute the published documentation against site-packages. Only fixtures,
        # tests and customer docs are copied into the isolated runner directory.
        for name in (
            "docs/quickstart.md",
            "docs/errors.md",
            "docs/cli-migration-0.31-0.32.md",
            "contracts/synth-v1.openapi.yaml",
            "tests/conftest.py",
            "tests/test_journey.py",
            "tests/test_messages.py",
            "tests/quickstart_examples.py",
            "tests/test_error_guide.py",
        ):
            destination = Path(directory) / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pytest",
                "-q",
                "--asyncio-mode=auto",
                "--basetemp",
                str(Path(directory) / "pytest-tmp"),
                "tests/quickstart_examples.py",
                "tests/test_error_guide.py",
                "tests/test_journey.py",
            ],
            check=True,
            cwd=directory,
            env=dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1"),
            timeout=120,
        )

    print(
        {
            "wheel_bytes": wheel.stat().st_size,
            "package_source_bytes": source_bytes,
            "project_urls": "passed",
            "readme_links": "passed",
            "isolated_install": "passed",
            "installed_quickstarts_and_error_guide": "passed",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
