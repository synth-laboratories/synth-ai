import pytest

from scripts.check_type_contracts import ROOT, check, diagnostics, expectations
from scripts.gen_contract import JSONObject
from scripts.smoke import ObservedRun, SmokeReceipt


def test_sdk_consumer_type_contracts():
    check(ROOT / "typing_tests/accepted.py", rejected=False)
    check(ROOT / "typing_tests/rejected.py", rejected=True)


@pytest.mark.parametrize(
    "source",
    [
        'from synth_ai import SynthClient\ndef f(client: SynthClient) -> None:\n client.swarms.launch("task", budget="1")  # type-error: invalid-argument-type\n',
        'from synth_ai import SynthClient\ndef f(client: SynthClient) -> None:\n client.swarms.launch("task")  # type-error: missing-argument\n client.missing()\n',
    ],
)
def test_contract_gate_rejects_missing_or_unexpected_diagnostics(tmp_path, source):
    path = tmp_path / "consumer.py"
    path.write_text(source)
    with pytest.raises(RuntimeError, match="negative type contract changed"):
        check(path, rejected=True)


def test_expectation_parser_rejects_typos_and_ignores_string_contents(tmp_path):
    path = tmp_path / "consumer.py"
    for source in ['x = "# type-error: missing-argument"', "# type-error: invalid_rule"]:
        path.write_text(source)
        with pytest.raises(ValueError):
            expectations(path)


def test_diagnostic_parser_counts_duplicate_errors():
    line = "typing_tests/rejected.py:7:5: error[missing-argument] missing budget\n"
    found = diagnostics(line + line)
    assert sum(found.values()) == 2
    assert next(iter(found)).file == ROOT / "typing_tests/rejected.py"


@pytest.mark.parametrize(
    "value,method",
    [
        ({"value": 1}, "text"),
        ({"value": 1}, "flag"),
        ({"value": []}, "object"),
        ({"value": {}}, "array"),
        ({"value": ["valid", 1]}, "strings"),
    ],
)
def test_generator_json_accessors_reject_wrong_shapes(value, method):
    node = JSONObject.parse(value)
    with pytest.raises(ValueError):
        getattr(node, method)("value")


def test_generator_json_accessors_preserve_valid_open_extensions():
    node = JSONObject.parse({"name": "schema", "x-vendor": {"values": ["a", "b"]}})
    assert node.text("name") == "schema"
    assert node.object("x-vendor").strings("values") == ["a", "b"]
    assert node == {"name": "schema", "x-vendor": {"values": ["a", "b"]}}


def test_typed_smoke_receipt_round_trip_retains_recovery_identities():
    receipt = SmokeReceipt(
        base_url="https://api.example.test",
        runs=[ObservedRun(run_id="run-retained")],
        launch_keys={"0": "idempotency-retained"},
    )
    restored = SmokeReceipt.model_validate_json(receipt.model_dump_json(by_alias=True))
    assert restored.runs[0].run_id == "run-retained"
    assert restored.launch_keys == {"0": "idempotency-retained"}
    assert not restored.runs[0].cleanup_complete
    assert not restored.complete
