"""Open Research v1 client + MCP-facing models.

Mirrors the locked HTTP contract at
``backend-open-research-routes`` ↦
``specifications/daily/2026-05-13/open_research_http_contract_v1.md``.

This package is intentionally thin: each MCP tool maps 1:1 onto one
backend endpoint under ``/api/open-research/v1/``. Submissions go
through the same backend review gate as the public composer — there is
no MCP-only bypass.
"""

from synth_ai.managed_research.open_research.client import (
    OPEN_RESEARCH_BASE,
    OpenResearchClient,
)
from synth_ai.managed_research.open_research.errors import (
    OPEN_RESEARCH_ERROR_CLASSES,
    OpenResearchError,
)
from synth_ai.managed_research.open_research.fingerprint import (
    DEFAULT_FINGERPRINT_PATH,
    load_or_create_fingerprint,
)
from synth_ai.managed_research.open_research.models import (
    BundleDownloadResult,
    ListExperimentsResponse,
    ListProjectsResponse,
    ListQueuesResponse,
    MetricTarget,
    SubmissionResponse,
    SubmitQuestionArgs,
)

__all__ = [
    "DEFAULT_FINGERPRINT_PATH",
    "OPEN_RESEARCH_BASE",
    "OPEN_RESEARCH_ERROR_CLASSES",
    "OpenResearchClient",
    "OpenResearchError",
    "BundleDownloadResult",
    "ListExperimentsResponse",
    "ListProjectsResponse",
    "ListQueuesResponse",
    "MetricTarget",
    "SubmissionResponse",
    "SubmitQuestionArgs",
    "load_or_create_fingerprint",
]
