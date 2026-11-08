"""Schema pruning follows the supported SDK surface, not its own generated registry."""

import ast
from pathlib import Path

import pytest

from scripts.gen_contract import JSONObject, sdk_model_roots, select_sdk_schemas
from synth_ai._generated.operations import OPERATIONS

ROOT = Path(__file__).parents[1]


def test_schema_selection_preserves_transitive_models_and_public_enum_owners():
    schemas = {
        "Request": JSONObject.parse(
            {"properties": {"input": {"$ref": "#/components/schemas/Input"}}}
        ),
        "Input": JSONObject.parse({"properties": {"kind": {"enum": ["text", "file"]}}}),
        "Public": JSONObject.parse({"type": "string"}),
        "Unused": JSONObject.parse({"$ref": "#/components/schemas/UnusedCycle"}),
        "UnusedCycle": JSONObject.parse({"$ref": "#/components/schemas/Unused"}),
    }
    assert set(select_sdk_schemas(schemas, {"Request", "Public"})) == {"Request", "Input", "Public"}
    assert set(select_sdk_schemas(schemas, {"InputKind"})) == {"Input"}


@pytest.mark.parametrize(
    "schemas,roots",
    [
        ({"Known": JSONObject()}, {"Unknown"}),
        ({"Known": JSONObject({"$ref": "#/components/schemas/Missing"})}, {"Known"}),
    ],
)
def test_missing_export_or_transitive_schema_refuses_generation(schemas, roots):
    with pytest.raises(ValueError, match="unknown"):
        select_sdk_schemas(schemas, roots)


def test_generated_files_and_registry_cannot_keep_unused_models_alive(tmp_path, monkeypatch):
    from scripts import gen_contract

    source = tmp_path / "synth_ai"
    (source / "_generated").mkdir(parents=True)
    (source / "models.py").write_text("from ._generated.models import Public as PublicExport\n")
    (source / "transport.py").write_text("from ._generated.models import MODELS\n")
    (source / "_sync.py").write_text("from ._generated.models import OldSyncType\n")
    (source / "_generated/models.py").write_text(
        "from synth_ai._generated.models import OldGeneratedType\n"
    )
    monkeypatch.setattr(gen_contract, "ROOT", tmp_path)
    operations = {"get": JSONObject({"request_model": "Request", "response_model": "Response"})}
    assert sdk_model_roots(operations) == {"Public", "Request", "Response"}
    (source / "bad.py").write_text("from ._generated.models import *\n")
    with pytest.raises(ValueError, match="explicit model imports"):
        sdk_model_roots(operations)


def unreachable_python_definitions(source, roots):
    definitions = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            definitions[node.name] = node
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id != "MODELS"
        ):
            definitions[node.target.id] = node
    reached = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        assert name in definitions, f"missing generated API type: {name}"
        reached.add(name)
        pending.extend(
            node.id
            for node in ast.walk(definitions[name])
            if isinstance(node, ast.Name) and node.id in definitions and node.id not in reached
        )
    return definitions.keys() - reached


def test_generated_python_has_no_orphan_definitions():
    # Verify emitted Python independently of the producer's JSON-schema traversal.
    operations = {name: JSONObject.parse(operation) for name, operation in OPERATIONS.items()}
    roots = sdk_model_roots(operations)
    source = (ROOT / "synth_ai/_generated/models.py").read_text()
    assert not unreachable_python_definitions(source, roots)
    assert unreachable_python_definitions(source + "\nclass UnusedSchema: pass\n", roots) == {
        "UnusedSchema"
    }
