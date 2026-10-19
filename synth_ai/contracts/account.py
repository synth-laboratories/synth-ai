"""Read-only account.v1 routes; see backend account-billing-v023.md.

These observational account routes have their own version. They neither
negotiate nor claim Index/Swarms execution qualification or release pins.
"""

from synth_ai.contracts.operations import Operation
from synth_ai.errors import UnsupportedOperationError
from synth_ai.options import ClientOptions

CONTEXT = Operation(
    "account.context",
    "GET",
    "/account/context",
    True,
    contract="account.v1",
    required_scope=None,
)
PLAN = Operation(
    "account.plan",
    "GET",
    "/account/billing/plan",
    True,
    contract="account.v1",
    required_scope="account:billing:read",
)
BALANCE = Operation(
    "account.balance",
    "GET",
    "/account/billing/balance",
    True,
    contract="account.v1",
    required_scope="account:billing:read",
)
USAGE = Operation(
    "account.usage",
    "GET",
    "/account/usage/summary",
    True,
    contract="account.v1",
    required_scope="account:usage:read",
)
EVENTS = Operation(
    "account.events",
    "GET",
    "/account/usage/events",
    True,
    contract="account.v1",
    required_scope="account:usage:read",
)
LIMITS = Operation(
    "account.limits",
    "GET",
    "/account/limits",
    True,
    contract="account.v1",
    required_scope="account:read",
)
ACCOUNT_READ_OPERATIONS = (CONTEXT, PLAN, BALANCE, USAGE, EVENTS, LIMITS)


def require_account_read(operation: Operation, options: ClientOptions) -> None:
    if operation not in ACCOUNT_READ_OPERATIONS or operation.mutation or operation.may_charge:
        raise UnsupportedOperationError("Unknown account read operation")
    if options.release_manifest_sha256 is not None or options.candidate_manifest_sha256 is not None:
        raise UnsupportedOperationError(
            "Account reads do not negotiate Index/Swarms execution manifests",
            code="account_execution_manifest_not_applicable",
        )
