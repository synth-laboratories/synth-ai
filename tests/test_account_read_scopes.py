"""Account observations require explicit grants; product scopes do not imply them."""

import pytest
from synth_ai.contracts.account import (
    ACCOUNT_READ_OPERATIONS,
    ALERTS,
    BALANCE,
    CONTEXT,
    EVENTS,
    LIMITS,
    PLAN,
    USAGE,
    account_read_scopes,
)
from synth_ai.contracts.operations import Operation
from synth_ai.errors import UnsupportedOperationError


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (CONTEXT, set()),
        (PLAN, {"account:billing:read"}),
        (BALANCE, {"account:billing:read"}),
        (ALERTS, {"account:billing:read"}),
        (USAGE, {"account:billing:read", "account:usage:read"}),
        (EVENTS, {"account:billing:read", "account:usage:read"}),
        (LIMITS, {"account:billing:read", "account:read"}),
    ],
)
def test_account_read_scope_requirements(operation, expected):
    assert account_read_scopes(operation) == expected


def test_unknown_or_modified_operation_cannot_claim_account_grants():
    with pytest.raises(UnsupportedOperationError):
        account_read_scopes(Operation(USAGE.id, "GET", "/other", True, contract="account.v1"))
    assert len({operation.id for operation in ACCOUNT_READ_OPERATIONS}) == 7
