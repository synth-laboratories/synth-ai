"""Pure comparison tests: no client, journey, slot or network execution."""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
from dataclasses import dataclass, replace
from typing import Optional

import pytest

from scripts.acceptance.isolation import classify


@dataclass(frozen=True, slots=True)
class Seen:
    accepted: bool = False
    status: Optional[int] = 404
    code: Optional[str] = "resource_not_found"
    cause: str = ""
    problem: Optional[dict[str, object]] = None


def resource(code="resource_not_found"):
    return Seen(code=code, problem={"code": code, "status": 404})


def test_matching_resource_404s_prove_only_the_resource_case():
    assert classify(resource(), resource(), ()) == "pass"


@pytest.mark.parametrize("code", [None, "", "route_not_found", "not_found", "banana"])
def test_untyped_or_missing_route_404_is_unproven(code):
    outcome = resource(code)
    assert classify(outcome, outcome, ()).startswith("unproven:")


def test_equal_status_without_a_problem_is_unproven():
    assert classify(Seen(), Seen(), ()) == "unproven:resource_problem_missing"


def test_different_existence_details_fail():
    foreign = replace(resource(), problem={"code": "resource_not_found", "detail": "exists"})
    assert classify(foreign, resource(), ()) == "differs"


def test_cross_org_success_is_a_leak():
    assert classify(replace(resource(), accepted=True, status=200), resource(), ()) == "leak"


@pytest.mark.parametrize("status", [401, 403, 422, 503])
def test_identical_wrong_statuses_are_unproven(status):
    outcome = replace(resource(), status=status)
    assert classify(outcome, outcome, ()) == f"unproven:uniform_{status}"


@pytest.mark.parametrize("code", ["usage_export_not_found", "plan_request_unknown"])
def test_matching_account_resource_refusals_prove_the_resource_case(code):
    denial = resource(code)
    assert classify(denial, denial, ()) == "pass"


def test_matching_checkout_refusals_prove_the_resource_case():
    denial = resource("checkout_not_found")
    assert classify(denial, denial, ()) == "pass"
