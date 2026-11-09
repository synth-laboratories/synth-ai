import re
from pathlib import Path

import httpx
import pytest

from synth_ai import errors


@pytest.mark.parametrize(
    "code,status,kind",
    re.findall(
        r"\| `([^`]+)` \| (\d+) \| `([^`]+)` \|",
        (Path(__file__).parents[1] / "docs/errors.md").read_text(),
    ),
)
def test_documented_problem_codes_keep_typed_fields(code, status, kind):
    problem = {
        "type": "about:blank",
        "title": code,
        "status": int(status),
        "detail": code,
        "instance": "urn:request:test",
        "code": code,
        "service": "synth-backend",
        "stage": "admission",
        "outcome": "not_accepted",
    }
    response = httpx.Response(
        int(status), json=problem, request=httpx.Request("POST", "https://example.test/v1/runs")
    )
    with pytest.raises(getattr(errors, kind)) as caught:
        errors.raise_for_response(response)
    assert caught.value.problem.model_dump(exclude_unset=True) == problem
