import json
from types import SimpleNamespace

import pytest

from scripts import smoke


@pytest.mark.parametrize("state", ["failed", "cancelled", "partial", "running"])
def test_output_does_not_substitute_for_successful_execution(state):
    with pytest.raises(RuntimeError, match="did not complete successfully"):
        smoke.require_successful_run(SimpleNamespace(state=state, result_available=True))


def test_completed_run_is_required():
    smoke.require_successful_run(SimpleNamespace(state="completed"))


@pytest.mark.parametrize("case", ["existing", "different_api", "too_many_runs", "legacy_schema"])
async def test_smoke_budget_preflight(tmp_path, monkeypatch, case):
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "synth.public-sdk.smoke.v1"
                if case == "legacy_schema"
                else "synth.public-sdk.smoke.v3",
                "base_url": "https://other.example.test"
                if case == "different_api"
                else "https://api.example.test",
                "runs": [{}] * (3 if case == "too_many_runs" else 0),
            }
        )
    )
    with pytest.raises(ValueError):
        await smoke.run(receipt, resume=case != "existing")
