"""The paid smoke must refuse ambiguous receipt reuse before opening a client."""

import json

import pytest

from scripts.smoke import run


@pytest.mark.parametrize("case", ["existing", "different_api", "too_many_runs"])
async def test_smoke_budget_preflight(tmp_path, monkeypatch, case):
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "base_url": "https://other.example.test"
                if case == "different_api"
                else "https://api.example.test",
                "runs": [{}] * (3 if case == "too_many_runs" else 0),
            }
        )
    )
    with pytest.raises(ValueError):
        await run(receipt, resume=case != "existing")
