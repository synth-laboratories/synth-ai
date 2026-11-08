"""The lint rules must reject bypasses, while allowing typed protocol boundaries."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.lint_standards import check_source

ROOT = Path(__file__).parents[1]


def codes(source, filename="api.py"):
    return {diagnostic.code for diagnostic in check_source(source, Path(filename))}


@pytest.mark.parametrize(
    "source",
    [
        "from typing import Any as Blob\ndef f(value: Blob) -> None: ...",
        "import typing as t\nData = dict[str, t.Any]",
        "from typing_extensions import Any as Blob",
        "import typing as t\ndef f(value: 'list[t.Any]') -> None: ...",
        "import typing as t\nBlob = 'dict[str, t.Any]'\ndef f() -> Blob: ...",
        "def f(value: 'Any') -> None: ...",
        "from typing import *",
        "from typing import TypeAlias\nimport typing as t\nData: TypeAlias = 't.Any'",
    ],
)
def test_any_cannot_hide_behind_aliases_or_forward_annotations(source):
    assert "SYN001" in codes(source)


@pytest.mark.parametrize(
    "source",
    [
        "class Client:\n def __init__(self, **options: str) -> None: ...",
        "class Client:\n async def launch(self, **kwargs: str) -> None: ...",
        "def create(**kwargs: str) -> None: ...",
    ],
)
def test_public_keyword_bags_are_refused_even_if_annotated(source):
    assert "SYN002" in codes(source)


@pytest.mark.parametrize(
    "annotation",
    [
        "dict",
        "list",
        "list[dict]",
        "dict[str, object]",
        "dict[str, list[object]]",
        "'dict[str, object]'",
    ],
)
def test_unstructured_public_boundaries_are_refused(annotation):
    assert "SYN003" in codes(f"def read() -> {annotation}: ...")


def test_collection_aliases_are_checked():
    assert "SYN003" in codes(
        "from collections.abc import Mapping as Map\nData = Map[str, object]\ndef read() -> Data: ..."
    )
    assert "SYN003" in codes(
        "from pydantic import BaseModel\nclass Record(BaseModel):\n data: dict[str, object]"
    )


@pytest.mark.parametrize(
    "config",
    [
        "",
        "model_config = ConfigDict(strict=True)",
        "model_config = ConfigDict(strict=False, frozen=True)",
        "model_config = ConfigDict(strict=True, frozen=False)",
        "model_config = ConfigDict(**settings)",
        "model_config = settings",
    ],
)
def test_record_configuration_must_be_provably_strict_and_frozen(config):
    assert "SYN004" in codes(
        f"from pydantic import BaseModel, ConfigDict\nclass Record(BaseModel):\n {config or 'pass'}"
    )


def test_local_inheritance_preserves_or_explicitly_invalidates_record_configuration():
    good = "from pydantic import BaseModel, ConfigDict\nclass Base(BaseModel):\n model_config = ConfigDict(strict=True, frozen=True)\nclass Child(Base):\n value: str\n"
    assert not codes(good)
    assert "SYN004" in codes(good + " model_config = ConfigDict(strict=False)\n")
    assert "SYN004" in codes(good + "class Mutable(Base, frozen=False): pass\n")


@pytest.mark.parametrize(
    "call",
    [
        "Record.model_construct(value=1)",
        "Record.construct(value=1)",
        "record.model_copy(update={'value': 1})",
        "record.model_copy(**options)",
        "Record.model_validate(data, strict=False)",
        "Record.model_validate_json(data, strict=flag)",
        "adapter.validate_python(data, strict=False)",
    ],
)
def test_validation_bypasses_are_refused(call):
    assert "SYN005" in codes(call)


@pytest.mark.parametrize(
    "source,filename",
    [
        ("from pydantic import JsonValue\ndef read() -> dict[str, JsonValue]: ...", "api.py"),
        ("class Client:\n def _private(self, **kwargs: str) -> None: ...", "api.py"),
        ("def outer() -> None:\n def callback(**kwargs: str) -> None: ...", "api.py"),
        ("def decode(value: dict[str, object]) -> None: ...", "_wire.py"),
        ("class State:\n count: int\n def update(self) -> None: ...", "api.py"),
        (
            "from pydantic import BaseModel as Model, ConfigDict as Config\nclass Record(Model):\n model_config = Config(strict=True, frozen=True)\n value: str",
            "api.py",
        ),
        ("from ._wire import WireModel as Wire\nclass Record(Wire):\n value: str", "models.py"),
        (
            "record.model_copy()\nRecord.model_validate(data)\nadapter.validate_python(data, strict=True)",
            "api.py",
        ),
        ('text = "typing.Any and dict[str, object] are not annotations"', "api.py"),
    ],
)
def test_allowed_patterns_do_not_trigger_rules(source, filename):
    assert not codes(source, filename)


def test_diagnostics_and_cli_failure_are_actionable_and_do_not_import_code(tmp_path):
    source = "raise RuntimeError('must never execute')\nclass Client:\n def launch(self, **kwargs: str) -> None: ...\n"
    path = tmp_path / "bad.py"
    path.write_text(source)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lint_standards.py"), str(path)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert f"{path}:3:21: SYN002" in result.stdout
    assert "Traceback" not in result.stderr


def test_syntax_errors_and_missing_inputs_fail_closed(tmp_path):
    assert codes("def broken(") == {"SYN000"}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lint_standards.py"), str(tmp_path / "missing")],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "path does not exist" in result.stderr


def test_production_sdk_obeys_the_standards():
    diagnostics = [
        diagnostic
        for path in (ROOT / "synth_ai").rglob("*.py")
        for diagnostic in check_source(path.read_text(), path)
    ]
    assert not diagnostics, "\n".join(map(str, diagnostics))


@pytest.mark.parametrize(
    "annotation", ['Literal["dict", "Any", "typing.Any"]', 'Annotated[str, "dict[str, object]"]']
)
def test_annotation_metadata_and_literal_values_are_not_forward_types(annotation):
    assert not codes(f"from typing import Annotated, Literal\ndef read() -> {annotation}: ...")


@pytest.mark.parametrize(
    "source,rule",
    [
        ("def f(value): ...", "ANN001"),
        ("def f(value: str): ...", "ANN201"),
        ("from typing import Any\ndef f(value: Any) -> None: ...", "ANN401"),
        ("value = missing  # noqa", "PGH004"),
        ("raise Exception('generic failure')", "TRY002"),
    ],
)
def test_ruff_policy_catches_missing_types_and_blanket_suppression(source, rule):
    result = subprocess.run(
        [
            "ruff",
            "check",
            "--config",
            str(ROOT / "pyproject.toml"),
            "--stdin-filename",
            "synth_ai/probe.py",
            "-",
        ],
        input=source,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert rule in result.stdout
