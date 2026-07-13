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
    content = fix_see_also_sections(content)
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
