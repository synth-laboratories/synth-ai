"""Advisory tool discovery; native admission remains authoritative.

See backend deployment/release/ACCOUNT_POLICY_ENGINE.md. Hiding a new-work tool
must not block retained mutation recovery or turn a replay into a fresh launch.
"""

from synth_ai.types.account import AccountEntitlements

POLICY_ACTION_TOOLS = frozenset(
    {"index_search", "index_search_create", "swarms_create", "swarms_preflight"}
)


def visible_policy_tools(
    available: frozenset[str], policy: AccountEntitlements | None
) -> frozenset[str]:
    """Keep lifecycle tools during policy outages; never infer a positive grant."""
    allowed = (
        {row.subproduct for row in policy.subproducts if row.access == "allowed"}
        if policy is not None
        else set()
    )
    visible = available - POLICY_ACTION_TOOLS
    if allowed & {"index.fast", "index.deep"}:
        visible |= available & {"index_search", "index_search_create"}
    if allowed & {"swarms.docker", "swarms.daytona"} and allowed & {
        "swarms.openrouter",
        "swarms.openai",
        "swarms.tinker",
        "swarms.synth",
    }:
        visible |= available & {"swarms_create", "swarms_preflight"}
    return frozenset(visible)
