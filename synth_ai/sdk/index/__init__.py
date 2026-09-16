"""Synth Index clients and contracts; backend routes remain the wire authority."""

from . import surfaces
from .agent_policy import IndexAccessPolicy
from .catalog import (
    Capabilities,
    CollectionGrant,
    CollectionGrantSpec,
    Collections,
    ContestEntry,
    ContestView,
    IndexUsageSummary,
    Leaderboard,
    MyRewards,
    PrivatePromoCredit,
    ProfileView,
    PromoCreditSummary,
    RewardAward,
    TagRegistry,
)
from .contracts import ContributionReference
from .contributions import (
    ContributionDraft,
    ContributionUploadPrepared,
    ContributionUploadSpec,
    ResearchDraftSpec,
    ResearchLookupView,
    ResearchSource,
)
from .errors import IndexErrorCode, index_error_code
from .lifecycle import (
    Assessment,
    ContributionView,
    MeView,
    PublicationSpec,
    PublicationStatus,
    ReviewDecision,
    ReviewSpec,
    RevisionCreateSpec,
    RevisionView,
    WithdrawalSpec,
)
from .package import ContributionPackage
from .public import AsyncPublicIndexClient, PublicIndexClient
from .search import ContentsResult, ContentsSpec, PublicSearchResult, SearchResult, SearchSpec
from .submission import ContributionSubmission, ContributionSubmitSpec, RevisionStatus

__all__ = [
    "Assessment",
    "AsyncPublicIndexClient",
    "Capabilities",
    "CollectionGrant",
    "CollectionGrantSpec",
    "Collections",
    "ContentsResult",
    "ContentsSpec",
    "ContestEntry",
    "ContestView",
    "ContributionDraft",
    "ContributionPackage",
    "ContributionReference",
    "ContributionSubmission",
    "ContributionSubmitSpec",
    "ContributionUploadPrepared",
    "ContributionUploadSpec",
    "ContributionView",
    "IndexAccessPolicy",
    "IndexErrorCode",
    "IndexUsageSummary",
    "Leaderboard",
    "MeView",
    "MyRewards",
    "PrivatePromoCredit",
    "ProfileView",
    "PromoCreditSummary",
    "PublicIndexClient",
    "PublicSearchResult",
    "PublicationSpec",
    "PublicationStatus",
    "ResearchDraftSpec",
    "ResearchLookupView",
    "ResearchSource",
    "ReviewDecision",
    "ReviewSpec",
    "RevisionCreateSpec",
    "RevisionStatus",
    "RevisionView",
    "RewardAward",
    "SearchResult",
    "SearchSpec",
    "TagRegistry",
    "WithdrawalSpec",
    "index_error_code",
    "surfaces",
]
