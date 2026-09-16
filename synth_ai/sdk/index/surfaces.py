"""Which Index operations each customer surface reaches, and why the rest do not.

The typed sync and async clients reach every backend operation a customer
credential can use. The CLI and the MCP server are deliberately smaller: each
operation they omit is listed here with its reason. The release parity gate
checks this table in both directions against the packaged backend contract and
against what each command and tool actually sends, so a new backend operation,
a removed one, or a command that quietly calls something else fails a test.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class SurfaceOperations:
    """Operations one command or tool sends, with and without a credential.

    ``public`` is empty when the command or tool needs an account.
    """

    authenticated: frozenset[str]
    public: frozenset[str] = frozenset()


def _ops(*authenticated: str, public: tuple[str, ...] = ()) -> SurfaceOperations:
    return SurfaceOperations(frozenset(authenticated), frozenset(public))


_READS = {
    "capabilities": _ops("index.capabilities", public=("index.public.capabilities",)),
    "tags": _ops("index.tags.list", public=("index.public.tags.list",)),
    "search": _ops("index.search", public=("index.public.search",)),
    "contents": _ops("index.contents.retrieve", public=("index.public.contents.retrieve",)),
    "contribution": _ops(
        "index.contributions.retrieve", public=("index.public.contributions.retrieve",)
    ),
    "revision": _ops(
        "index.contributions.revisions.retrieve",
        public=("index.public.contributions.revisions.retrieve",),
    ),
    "asset": _ops(
        "index.contributions.assets.retrieve",
        public=("index.public.contributions.assets.retrieve",),
    ),
    "account": _ops("index.me.retrieve"),
    "my_contributions": _ops("index.me.contributions.list"),
    "usage": _ops("index.me.usage"),
    "promo_credit": _ops("index.me.promo_credit"),
}

_RESEARCH_SUBMIT = _ops(
    "index.me.retrieve",
    "index.contributions.research.lookup",
    "index.contributions.research.create",
    "index.contributions.revisions.retrieve",
    "index.contributions.upload.prepare",
    "index.contributions.upload.finalize",
    "index.contributions.submit",
)
_RESEARCH_RECOVER = _ops(
    "index.me.retrieve",
    "index.contributions.research.lookup",
    "index.contributions.revisions.retrieve",
)

CLI_COMMANDS: Mapping[str, SurfaceOperations] = MappingProxyType(
    {
        "index capabilities": _READS["capabilities"],
        "index tags": _READS["tags"],
        "index search": _READS["search"],
        "index contents": _READS["contents"],
        "index contribution": _READS["contribution"],
        "index revision": _READS["revision"],
        "index asset": _READS["asset"],
        "index account": _READS["account"],
        "index my-contributions": _READS["my_contributions"],
        "index usage": _READS["usage"],
        "index promo-credit": _READS["promo_credit"],
        "index research preview": _ops(),
        "index research submit": _RESEARCH_SUBMIT,
        "index research recover-state": _RESEARCH_RECOVER,
    }
)

MCP_TOOLS: Mapping[str, SurfaceOperations] = MappingProxyType(
    {
        "index_capabilities": _READS["capabilities"],
        "index_list_tags": _READS["tags"],
        "index_search": _READS["search"],
        "index_get_contents": _READS["contents"],
        "index_get_contribution": _READS["contribution"],
        "index_contribution_status": _READS["revision"],
        "index_get_asset": _READS["asset"],
        "index_account": _READS["account"],
        "index_my_contributions": _READS["my_contributions"],
        "index_usage": _READS["usage"],
        "index_promo_credit": _READS["promo_credit"],
        "index_contribution_create": _ops("index.contributions.create"),
        "index_contribution_upload": _ops(
            "index.contributions.upload.prepare", "index.contributions.upload.finalize"
        ),
        "index_contribution_submit": _ops("index.contributions.submit"),
    }
)

_REVIEWER = "Reviewer and publisher decisions stay in the typed client and the web app."
_PROGRAM = "Reward and contest programs are operated from the typed client, not agents."
_SHARING = "Collection sharing changes access for other people; typed client and web app only."
_PROFILE = "Profiles are edited and browsed in the web app; the typed client covers scripting."
_REVISION = "Opening a new revision of published work is a deliberate typed-client step."

# Operations neither the CLI nor MCP reaches, and why.
_NOT_ON_AGENT_SURFACES = {
    "index.contributions.publication.create": _REVIEWER,
    "index.contributions.withdrawal.create": _REVIEWER,
    "index.contributions.reviews.create": _REVIEWER,
    "index.contributions.assessments.list": (
        "Assessments already arrive inside the revision read both surfaces expose."
    ),
    "index.reviews.list": _REVIEWER,
    "index.contributions.revisions.create": _REVISION,
    "index.collections.list": _SHARING,
    "index.collections.grants.list": _SHARING,
    "index.collections.grants.create": _SHARING,
    "index.collections.grants.revoke": _SHARING,
    "index.me.rewards.list": _PROGRAM,
    "index.rewards.award": _PROGRAM,
    "index.rewards.reverse": _PROGRAM,
    "index.contests.create": _PROGRAM,
    "index.contests.retrieve": _PROGRAM,
    "index.contests.status.update": _PROGRAM,
    "index.contests.leaderboard": _PROGRAM,
    "index.contests.entries.create": _PROGRAM,
    "index.contests.entries.score": _PROGRAM,
    "index.contests.entries.review": _PROGRAM,
    "index.me.profile.update": _PROFILE,
    "index.me.profile.pins.update": _PROFILE,
    "index.profiles.retrieve": _PROFILE,
    "index.public.profiles.retrieve": _PROFILE,
}

CLI_EXCLUSIONS: Mapping[str, str] = MappingProxyType(
    {
        **_NOT_ON_AGENT_SURFACES,
        "index.contributions.create": (
            "The CLI uploads only verified research conversions (research submit); "
            "a blank draft plus hand-built package is a typed-client or MCP flow."
        ),
    }
)

MCP_EXCLUSIONS: Mapping[str, str] = MappingProxyType(
    {
        **_NOT_ON_AGENT_SURFACES,
        "index.contributions.research.create": (
            "Research intake reads a verified local conversion and keeps resumable "
            "state on disk; it is the CLI's research submit, not an agent tool."
        ),
        "index.contributions.research.lookup": (
            "Only research intake and its saved-state recovery use this lookup."
        ),
    }
)

# Request headers the backend contract declares that customer surfaces never
# send, with the reason. Every other declared header must be sent as declared.
OPERATOR_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        "Synth-Acceptance-Context": (
            "Signed acceptance-run context used by operator qualification tooling."
        ),
        "Synth-Acceptance-Signature": "Signature over Synth-Acceptance-Context.",
    }
)

__all__ = [
    "CLI_COMMANDS",
    "CLI_EXCLUSIONS",
    "MCP_EXCLUSIONS",
    "MCP_TOOLS",
    "OPERATOR_HEADERS",
    "SurfaceOperations",
]
