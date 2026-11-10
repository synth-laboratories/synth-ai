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


UPGRADE_PROBLEM = {
    "type": "https://docs.usesynth.ai/errors/client_upgrade_required",
    "title": "Upgrade Required",
    "status": 426,
    "detail": "synth-ai 0.31.0 is no longer supported; upgrade to 0.32.0 or newer",
    "instance": "urn:request:test",
    "code": "client_upgrade_required",
    "service": "synth-backend",
    "stage": "public_client_version",
    "outcome": "not_accepted",
    "retryable": False,
    "minimum_version": "0.32.0",
    "client_version": "0.31.0",
}


def test_upgrade_required_is_typed_and_carries_versions():
    response = httpx.Response(
        426, json=UPGRADE_PROBLEM, request=httpx.Request("GET", "https://example.test/v1/account")
    )
    with pytest.raises(errors.UpgradeRequiredError) as caught:
        errors.raise_for_response(response)
    error = caught.value
    assert isinstance(error, errors.SwarmsAPIError)
    assert not isinstance(error, errors.InvalidRequestError)
    assert error.code == "client_upgrade_required" and error.status == 426
    assert error.minimum_version == "0.32.0" and error.client_version == "0.31.0"
    assert error.problem.retryable is False


def test_upgrade_required_code_wins_without_version_members():
    problem = {
        k: v for k, v in UPGRADE_PROBLEM.items() if k not in {"minimum_version", "client_version"}
    }
    with pytest.raises(errors.UpgradeRequiredError) as caught:
        errors.raise_problem(errors.Problem.model_validate(problem))
    assert caught.value.minimum_version is None


def test_upgrade_required_is_never_retried():
    from synth_ai._retry import stream_refusal_retryable

    assert stream_refusal_retryable(errors.Problem.model_validate(UPGRADE_PROBLEM)) is False
