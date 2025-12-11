from pathlib import Path

AGENTS_TEXT = """
sinf

NEVER USE timeout. Do not add timeout parameters, timeout arguments, or timeout configurations to any code, API calls, or operations.
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

    # Remove orphan end markers first (end markers without a preceding start marker)
    cleaned = file_text
    end_pos = cleaned.find(SYNTH_DIV_END)
    start_pos = cleaned.find(SYNTH_DIV_START)
    
    # If there's an end marker before any start marker
    if end_pos != -1 and (start_pos == -1 or end_pos < start_pos):
        if start_pos == -1:
            # No start markers at all - remove everything including content before orphan
            cleaned = cleaned[end_pos + len(SYNTH_DIV_END):].lstrip()
        else:
            # There are start markers after the orphan - preserve content before orphan
            before_orphan = cleaned[:end_pos].rstrip()
            after_orphan = cleaned[end_pos + len(SYNTH_DIV_END):].lstrip()
            cleaned = "\n\n".join(filter(None, [before_orphan, after_orphan]))
    
    # Find the first start and last end marker to consolidate multiple sections
    first_start = cleaned.find(SYNTH_DIV_START)
    last_end = cleaned.rfind(SYNTH_DIV_END)
    
    if first_start != -1 and last_end != -1 and last_end > first_start:
        # We have at least one valid section, consolidate all into one
        before = cleaned[:first_start].rstrip()
        after = cleaned[last_end + len(SYNTH_DIV_END):].lstrip()
        
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
    
    # No valid sections found, remove any remaining orphan markers
    cleaned = cleaned.replace(SYNTH_DIV_END, "")
    cleaned = cleaned.replace(AGENTS_TEXT, "")
    cleaned = cleaned.strip()
    if cleaned:
        cleaned += "\n\n"
    path.write_text(f"{cleaned}{_render_block()}\n", encoding="utf-8")
