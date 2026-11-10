"""Pure task verifier, shared by Harbor and the streaming evaluator adapter."""

import json
from pathlib import Path


def score(inputs: dict[str, list[int]], output: object) -> float:
    expected = sorted(set(inputs["values"]))
    values = output.get("values") if isinstance(output, dict) else None
    return float(
        isinstance(values, list) and all(type(v) is int for v in values) and values == expected
    )


if __name__ == "__main__":
    inputs = {"values": [5, -2, 5, 0, 9, -2]}
    try:
        output = json.loads(Path("/app/result.json").read_text())
    except (OSError, ValueError):
        output = None
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.txt").write_text(str(score(inputs, output)))
