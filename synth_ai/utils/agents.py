from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SYNTH_DIV_START = "### --- SYNTH SECTION START ---"
SYNTH_DIV_END = "### ---- SYNTH SECTION END ----"


@dataclass(frozen=True)
class AgentGuide:
    """Structured description of an external IDE/CLI integration."""

    id: str
    name: str
    summary: str
    homepage: str
    binary: str
    cli_command: str
    install_options: Sequence[str]
    quickstart_steps: Sequence[str]
    environment: Sequence[str]
    notes: Sequence[str] = ()
    categories: Sequence[str] = ()

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"#### {self.name}",
            "",
            self.summary,
        ]
        if self.homepage:
            lines.append(f"Homepage: {self.homepage}")
        if self.cli_command:
            lines.append(f"Launch: `{self.cli_command}`")

        if self.install_options:
            lines.append("")
            lines.append("**Install options**")
            for option in self.install_options:
                lines.append(f"- {option}")

        if self.quickstart_steps:
            lines.append("")
            lines.append("**Quickstart**")
            for step in self.quickstart_steps:
                lines.append(f"- {step}")

        if self.environment:
            lines.append("")
            lines.append("**Environment**")
            for item in self.environment:
                lines.append(f"- {item}")

        if self.notes:
            lines.append("")
            lines.append("**Notes**")
            for note in self.notes:
                lines.append(f"- {note}")

        return "\n".join(lines)

    def to_resource(self) -> dict[str, Any]:
        data = asdict(self)
        data["markdown"] = self.to_markdown()
        return data


_AGENT_GUIDES: tuple[AgentGuide, ...] = (
    AgentGuide(
        id="claude",
        name="Claude Code",
        summary=(
            "Launch Anthropic's Claude Code desktop/CLI client with Synth Research's "
            "Anthropic-compatible backend."
        ),
        homepage="https://claude.com/claude-code",
        binary="claude",
        cli_command="synth-ai claude --model <model>",
        install_options=(
            "curl -fsSL https://claude.ai/install.sh | bash",
        ),
        quickstart_steps=(
            "Run `synth-ai claude --model <model>` to select a Synth-managed Anthropic model.",
            "Provide your Synth API key when prompted; the CLI stores it for future sessions.",
            "Use the `--force` flag if you want to re-enter cached credentials.",
        ),
        environment=(
            "`SYNTH_API_KEY` — requested interactively if missing.",
            "`ANTHROPIC_BASE_URL` & `ANTHROPIC_AUTH_TOKEN` — derived automatically when a model is provided.",
        ),
        notes=(
            "Supports any model slug from `synth_ai.types.MODEL_NAMES`.",
            "Use `--url` to supply a custom Anthropic-compatible base URL.",
        ),
        categories=("ide", "anthropic"),
    ),
    AgentGuide(
        id="codex",
        name="OpenAI Codex CLI",
        summary=(
            "Run OpenAI's Codex CLI configured to send traffic through the Synth Research "
            "OpenAI-compatible endpoint."
        ),
        homepage="https://developers.openai.com/codex/cli/",
        binary="codex",
        cli_command="synth-ai codex --model <model>",
        install_options=(
            "brew install codex",
            "npm install -g @openai/codex",
        ),
        quickstart_steps=(
            "Run `synth-ai codex --model <model>` to pin the Synth-hosted OpenAI-compatible model.",
            "Pick an install option if the binary is missing; Synth will walk you through installation.",
            "Use `--force` to prompt for a new API key even if it is cached.",
        ),
        environment=(
            "`SYNTH_API_KEY` — mirrored to `OPENAI_API_KEY` for Codex CLI.",
            "`model_providers.synth` overrides — injected via `codex -c` flags when a model is selected.",
        ),
        notes=(
            "Set `--url` to override the base URL instead of using the Synth Research default.",
            "You can invoke `codex` without a model, but Synth routing is only applied when a model is supplied.",
        ),
        categories=("ide", "openai-compatible"),
    ),
    AgentGuide(
        id="opencode",
        name="OpenCode",
        summary=(
            "Configure the OpenCode IDE to talk to Synth Research's OpenAI-compatible API."
        ),
        homepage="https://opencode.ai",
        binary="opencode",
        cli_command="synth-ai opencode --model <model>",
        install_options=(
            "brew install opencode",
            "bun add -g opencode-ai",
            "curl -fsSL https://opencode.ai/install | bash",
            "npm i -g opencode-ai",
            "paru -S opencode",
        ),
        quickstart_steps=(
            "Run `synth-ai opencode --model <model>` to validate the binary and update OpenCode's config.",
            "When prompted, paste your Synth API key; it is stored in OpenCode's auth.json.",
            "Override the base URL with `--url` if you need a different endpoint.",
        ),
        environment=(
            "`SYNTH_API_KEY` — stored under the `synth` provider in OpenCode's auth.json.",
            "`~/.config/opencode/opencode.json` — Synth provider + model are added automatically.",
        ),
        notes=(
            "Subsequent invocations reuse stored credentials; pass `--force` to capture a new key.",
            "The CLI ensures the Synth provider block remains idempotent across runs.",
        ),
        categories=("ide", "openai-compatible"),
    ),
)


def get_agent_guides() -> tuple[AgentGuide, ...]:
    return _AGENT_GUIDES


def render_agents_markdown(guides: Iterable[AgentGuide] | None = None) -> str:
    guides = tuple(guides or _AGENT_GUIDES)
    lines: list[str] = ["### Synth Agents", ""]
    for index, guide in enumerate(guides):
        lines.append(guide.to_markdown())
        if index != len(guides) - 1:
            lines.append("")
    return "\n".join(lines)


def agent_catalog() -> dict[str, Any]:
    return {"agents": [guide.to_resource() for guide in _AGENT_GUIDES]}


def _render_block() -> str:
    return f"{SYNTH_DIV_START}\n{render_agents_markdown()}\n{SYNTH_DIV_END}"


def _append_block(prefix: str) -> str:
    prefix = prefix.rstrip()
    block = _render_block()
    if prefix:
        return f"{prefix}\n\n{block}\n"
    return f"{block}\n"


def write_agents_md() -> None:
    path = Path.cwd() / "AGENTS.md"
    if not path.exists():
        path.write_text(_append_block(""), encoding="utf-8")
        return

    file_text = path.read_text(encoding="utf-8")

    div_start_i = file_text.find(SYNTH_DIV_START)
    div_end_i = file_text.find(SYNTH_DIV_END)

    if div_start_i != -1 and div_end_i != -1 and div_end_i > div_start_i:
        before = file_text[:div_start_i].rstrip()
        after = file_text[div_end_i + len(SYNTH_DIV_END):].lstrip()

        parts: list[str] = []
        if before:
            parts.append(before)
        parts.append(_render_block())
        if after:
            parts.append(after)

        new_text = "\n\n".join(parts)
        if not new_text.endswith("\n"):
            new_text += "\n"
        path.write_text(new_text, encoding="utf-8")
        return

    cleaned = file_text
    for token in (
        _render_block(),
        SYNTH_DIV_START,
        SYNTH_DIV_END,
        render_agents_markdown(),
    ):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    if cleaned:
        cleaned += "\n\n"
    path.write_text(f"{cleaned}{_render_block()}\n", encoding="utf-8")
