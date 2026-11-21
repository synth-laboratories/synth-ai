"""The quickstart's Python examples must type-check against this SDK checkout."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def test_quickstart_python_examples_typecheck(tmp_path):
    root = Path(__file__).parents[1]
    examples = re.findall(r"```python\n(.*?)```", (root / "docs/quickstart.md").read_text(), re.S)
    assert len(examples) == 2
    for index, source in enumerate(examples):
        target = tmp_path / f"quickstart_{index}.py"
        target.write_text(source)
        result = subprocess.run(
            [
                str(Path(sys.executable).parent / "ty"),
                "check",
                "--project",
                str(root),
                "--python",
                sys.executable,
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
