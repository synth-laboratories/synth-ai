import ast
from pathlib import Path


def test_runtime_strategies_do_not_import_public_facades():
    root = Path(__file__).parents[1] / "synth_ai"
    for name in [
        "_operations",
        "_observer",
        "_file_source",
        "_transfers",
        "_pagination",
        "_requests",
    ]:
        tree = ast.parse((root / (name + ".py")).read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not imports & {
            "swarms",
            "account",
            "billing",
            "aclient",
            "client",
            "_sync",
            "_transport",
        }, name
