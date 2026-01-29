from pathlib import Path

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for utils.json.") from exc


def strip_json_comments(raw: str) -> str:
    """Remove // and /* */ comments from JSONC text."""
    return synth_ai_py.strip_json_comments(raw)


def create_and_write_json(path: Path, content: dict) -> None:
    synth_ai_py.create_and_write_json(str(path), content)


def load_json_to_dict(path: Path) -> dict:
    return synth_ai_py.load_json_to_dict(str(path))
