"""Shared MDX post-processing for mdxify-generated SDK reference pages.

See: specifications/sdk/docstrings.md
"""

from __future__ import annotations

import re
from pathlib import Path

# Mintlify sidebar titles for generated pages (module slug -> display name).
TITLE_MAPPINGS: dict[str, str] = {
    "synth_ai-research-client.mdx": "ResearchClient",
    "synth_ai-research-factories.mdx": "Factories",
    "synth_ai-research-limits.mdx": "Limits",
    "synth_ai-research-secrets.mdx": "Secrets",
    "synth_ai-research-projects.mdx": "Projects",
    "synth_ai-research-project_namespaces.mdx": "Project namespaces",
    "synth_ai-research-runs.mdx": "Runs",
    "synth_ai-research-run_readouts.mdx": "Run readouts",
    "synth_ai-research-models.mdx": "Models",
    "synth_ai-research-enums.mdx": "Enums",
    "synth_ai-research-errors.mdx": "Errors",
    "synth_ai-research-async_client.mdx": "AsyncResearchClient",
    "synth_ai-client.mdx": "SynthClient",
    "synth-client.mdx": "SynthClient",
    "synth_ai-sdk-containers.mdx": "ContainersClient",
    "containers-client.mdx": "ContainersClient",
    "synth_ai-sdk-tunnels.mdx": "TunnelsClient",
    "tunnels-client.mdx": "TunnelsClient",
    "synth_ai-sdk-pools.mdx": "ContainerPoolsClient",
    "pools-client.mdx": "ContainerPoolsClient",
}

STATUS_CONFIG: dict[str, tuple[str, str, str]] = {
    "experimental": ("ALPHA", "yellow", "triangle-exclamation"),
    "alpha": ("ALPHA", "yellow", "triangle-exclamation"),
    "beta": ("BETA", "blue", "circle-check"),
}

SIGNATURE_REPLACEMENTS: dict[str, dict[str, str]] = {
    "containers-client.mdx": {
        "create(self, spec: ContainerSpec) -> Container": (
            "create(self, spec: ContainerSpec, *, "
            "timeout_seconds: float | None = None) -> Container"
        ),
        "get(self, container_id: str) -> Container": (
            "get(self, container_id: str, *, "
            "timeout_seconds: float | None = None) -> Container"
        ),
        "list(self) -> builtins.list[Container]": (
            "list(self, *, timeout_seconds: float | None = None) "
            "-> builtins.list[Container]"
        ),
        "delete(self, container_id: str) -> None": (
            "delete(self, container_id: str, *, "
            "timeout_seconds: float | None = None) -> None"
        ),
        "wait_ready(self, container_id: str) -> Container": (
            "wait_ready(self, container_id: str, *, "
            "timeout_seconds: float = 300.0, "
            "poll_interval_seconds: float = 2.0, "
            "timeout: float | None = None, "
            "poll_interval: float | None = None) -> Container"
        ),
    },
    "pools-client.mdx": {
        "list_pools(self) -> dict[str, Any]": (
            "list_pools(self, *, state: str | None = None, limit: int = 100, "
            "cursor: str | None = None) -> dict[str, Any]"
        ),
        "list(self) -> dict[str, Any]": (
            "list(self, *, state: str | None = None, limit: int = 100, "
            "cursor: str | None = None) -> dict[str, Any]"
        ),
        "list_rollouts(self, pool_id: str) -> dict[str, Any]": (
            "list_rollouts(self, pool_id: str, *, state: str | None = None, "
            "limit: int = 100, cursor: str | None = None) -> dict[str, Any]"
        ),
        (
            "stream_rollout_events(self, pool_id: str, rollout_id: str) "
            "-> Iterator[dict[str, Any]]"
        ): (
            "stream_rollout_events(self, pool_id: str, rollout_id: str, *, "
            "cursor: str | None = None) -> Iterator[dict[str, Any]]"
        ),
        "list_global_rollouts(self) -> dict[str, Any]": (
            "list_global_rollouts(self, *, state: str | None = None, "
            "limit: int = 100, cursor: str | None = None) -> dict[str, Any]"
        ),
        (
            "stream_global_rollout_events(self, rollout_id: str) "
            "-> Iterator[dict[str, Any]]"
        ): (
            "stream_global_rollout_events(self, rollout_id: str, *, "
            "cursor: str | None = None) -> Iterator[dict[str, Any]]"
        ),
    },
}

def wrap_examples_in_code_blocks(content: str) -> str:
    """Wrap indented Example sections in fenced ```python blocks."""
    lines = content.split("\n")
    result: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if re.match(r"^Example[^:]*:?\s*$", line.strip()):
            result.append(line)
            index += 1

            while index < len(lines) and lines[index].strip() == "":
                result.append(lines[index])
                index += 1
            if not result or result[-1].strip():
                result.append("")

            code_lines: list[str] = []
            while index < len(lines) and (
                lines[index].startswith("    ") or lines[index].strip() == ""
            ):
                if lines[index].strip():
                    current = lines[index]
                    trimmed = current[4:] if current.startswith("    ") else current
                    code_lines.append(trimmed)
                else:
                    code_lines.append("")
                index += 1

            if code_lines:
                while code_lines and code_lines[-1].strip() == "":
                    code_lines.pop()

                first_non_empty = next((ln for ln in code_lines if ln.strip()), "")
                if first_non_empty.strip().startswith("```"):
                    result.extend(code_lines)
                else:
                    result.append("```python")
                    result.extend(code_lines)
                    result.append("```")
                result.append("")

            continue

        result.append(line)
        index += 1

    return "\n".join(result)


def normalize_labeled_prose(content: str) -> str:
    """Render module-level prose labels as readable Markdown, not code blocks."""
    labels = {"Availability", "Contract", "Errors", "Expected output", "Pagination"}
    lines = content.split("\n")
    result: list[str] = []
    index = 0

    while index < len(lines):
        label = lines[index].strip().removesuffix(":")
        if label not in labels:
            result.append(lines[index])
            index += 1
            continue

        result.extend([f"**{label}:**", ""])
        index += 1
        while index < len(lines) and (
            lines[index].startswith("    ") or not lines[index].strip()
        ):
            result.append(
                lines[index][4:]
                if lines[index].startswith("    ")
                else lines[index]
            )
            index += 1

    return "\n".join(result)


def escape_remaining_curly_braces(content: str) -> str:
    """Escape ``{`` and ``}`` outside fenced code blocks (Mintlify JSX safety)."""
    lines = content.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        if stripped.startswith("<") and ">" in stripped:
            result.append(line)
            continue

        if "{'{" in line or stripped.startswith("#"):
            result.append(line)
            continue

        needs_escape = "{" in line or "}" in line
        operator_pattern = r"<(?![A-Za-z/!])|(?<![A-Za-z\"'/])>"
        if re.search(operator_pattern, line):
            needs_escape = True

        if needs_escape:
            escaped = line.replace("{", "&#123;").replace("}", "&#125;")
            escaped = re.sub(r"<(?![A-Za-z/!])", "&lt;", escaped)
            escaped = re.sub(r"(?<![A-Za-z\"'/])>", "&gt;", escaped)
            result.append(escaped)
        else:
            result.append(line)

    return "\n".join(result)


def fix_see_also_sections(content: str) -> str:
    """Remove indentation from See Also sections so Mintlify renders them."""
    lines = content.split("\n")
    result: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.strip() == "See Also:":
            result.append("See Also:")
            index += 1

            while index < len(lines) and (
                lines[index].startswith("    ") or lines[index].strip() == ""
            ):
                if lines[index].strip():
                    result.append(lines[index].strip())
                else:
                    result.append("")
                index += 1
            continue

        result.append(line)
        index += 1

    return "\n".join(result)


def normalize_rst_inline_markup(content: str) -> str:
    """Convert common reStructuredText inline roles to Markdown code spans."""
    lines = content.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue
        if in_code_block:
            result.append(line)
            continue

        line = re.sub(
            r":(?:attr|class|data|func|meth):`~?([^`]+)`",
            r"`\1`",
            line,
        )
        line = re.sub(r"``([^`\n]+)``", r"`\1`", line)
        result.append(line)

    return "\n".join(result)


def indent_wrapped_list_continuations(content: str) -> str:
    """Keep mdxify-wrapped list descriptions attached to their bullets."""
    lines = content.split("\n")
    result: list[str] = []
    in_code_block = False
    in_list_item = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            in_list_item = False
            result.append(line)
            continue
        if in_code_block:
            result.append(line)
            continue

        if line.startswith("- "):
            in_list_item = True
            result.append(line)
            continue
        if not stripped or stripped.startswith("#") or stripped.startswith("**"):
            in_list_item = False
            result.append(line)
            continue
        if in_list_item and not line.startswith(("  ", "\t")):
            result.append(f"  {line}")
            continue

        result.append(line)

    return "\n".join(result)


def restore_keyword_only_signatures(content: str, filename: str) -> str:
    """Repair signatures whose keyword-only parameters mdxify omits."""
    for generated, actual in SIGNATURE_REPLACEMENTS.get(filename, {}).items():
        content = content.replace(generated, actual)
    return content


def apply_title_mappings(content: str, filename: str) -> str:
    """Replace mdxify module titles with customer-facing sidebar names."""
    if filename not in TITLE_MAPPINGS:
        return content

    title = TITLE_MAPPINGS[filename]
    content = re.sub(
        r"^title: .+$",
        f"title: {title}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^sidebarTitle: .+$",
        f"sidebarTitle: {title}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    return content


def extract_and_add_status_tags(content: str) -> str:
    """Promote ``**Status:** alpha|beta`` in module docstrings to Mintlify badges."""
    status_match = re.search(r"\*\*Status:\*\*\s*(\w+)", content, re.IGNORECASE)
    if not status_match:
        return content

    status_value = status_match.group(1).lower()
    if status_value not in STATUS_CONFIG:
        return content

    tag, color, icon = STATUS_CONFIG[status_value]
    content = re.sub(r"\*\*Status:\*\*\s*\w+\s*\n?", "", content)

    if f'tag: "{tag}"' not in content:
        content = re.sub(
            r"^(sidebarTitle: .+)$",
            rf'\1\ntag: "{tag}"',
            content,
            count=1,
            flags=re.MULTILINE,
        )

    badge = f'<Badge color="{color}" icon="{icon}">{tag.title()}</Badge>\n\n'

    def add_badge(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n\n{badge}{match.group(2)}"

    return re.sub(
        r"^(# `.+`)(\n+)",
        add_badge,
        content,
        count=1,
        flags=re.MULTILINE,
    )


def postprocess_mdx_file(content: str, filename: str) -> str:
    """Run the full postprocess pipeline on one MDX file."""
    content = wrap_examples_in_code_blocks(content)
    content = normalize_labeled_prose(content)
    content = fix_see_also_sections(content)
    content = normalize_rst_inline_markup(content)
    content = indent_wrapped_list_continuations(content)
    content = restore_keyword_only_signatures(content, filename)
    content = escape_remaining_curly_braces(content)
    content = apply_title_mappings(content, filename)
    content = extract_and_add_status_tags(content)
    return content


def postprocess_mdx_files(output_dir: Path) -> int:
    """Post-process all generated MDX under ``output_dir``. Returns modified count."""
    modified = 0

    for mdx_file in output_dir.rglob("*.mdx"):
        original = mdx_file.read_text(encoding="utf-8")
        updated = postprocess_mdx_file(original, mdx_file.name)
        if updated != original:
            mdx_file.write_text(updated, encoding="utf-8")
            modified += 1

    return modified
