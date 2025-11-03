from pathlib import Path


AGENTS_TEXT = """
sinf
"""
SYNTH_DIV_START = "### --- SYNTH SECTION START ---"
SYNTH_DIV_END = "### ---- SYNTH SECTION END ----"


def _render_block() -> str:
    return f"{SYNTH_DIV_START}\n{AGENTS_TEXT}\n{SYNTH_DIV_END}"


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
        AGENTS_TEXT,
        AGENTS_TEXT.lstrip("\n"),
        AGENTS_TEXT.rstrip("\n"),
        AGENTS_TEXT.strip("\n"),
    ):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    if cleaned:
        cleaned += "\n\n"
    path.write_text(f"{cleaned}{_render_block()}\n", encoding="utf-8")
