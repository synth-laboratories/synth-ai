import json
from pathlib import Path


def strip_json_comments(raw: str) -> str:
    """Remove // and /* */ comments from JSONC text."""
    result: list[str] = []
    in_string = False
    in_line_comment = False
    in_block_comment = False
    escape = False
    i = 0
    length = len(raw)
    while i < length:
        char = raw[i]
        next_char = raw[i + 1] if i + 1 < length else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                result.append(char)
            i += 1
            continue

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            result.append(char)
            if char == "\"" and not escape:
                in_string = False
            escape = (char == "\\") and not escape
            i += 1
            continue

        if char == "/" and next_char == "/":
            in_line_comment = True
            i += 2
            continue

        if char == "/" and next_char == "*":
            in_block_comment = True
            i += 2
            continue

        if char == "\"":
            in_string = True
            escape = False

        result.append(char)
        i += 1

    return "".join(result)


def create_and_write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")


def load_json_to_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(strip_json_comments(path.read_text()))
    except (json.JSONDecodeError, OSError):
        return {}
