"""Public ``SynthClient().research`` facade over core transport + advanced session."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from synth_ai.core.research.account import ResearchAccountAPI
from synth_ai.core.research.client import Client as CoreResearchClient
from synth_ai.core.research.environments import EnvironmentsAPI
from synth_ai.core.research.experiments import ResearchExperimentsAPI
from synth_ai.core.research.factories import FactoriesAPI
from synth_ai.core.research.image_releases import ImageReleasesAPI
from synth_ai.core.research.knowledge import ResearchKnowledgeAPI
from synth_ai.core.research.projects import ResearchProjectsAPI
from synth_ai.core.research.swarms import ResearchSwarmsAPI
from synth_ai.core.research.wiki import ResearchWikiAPI

if TYPE_CHECKING:
    from synth_ai.core.research.advanced import (
        ResearchAdvancedAPI,
        ResearchSession,
    )
    from synth_ai.core.research.session.files import FilesAPI


class Client:
    """Research entrypoint on ``SynthClient``.

    Obtain via ``SynthClient().research``. The three hero namespaces are
    projects, swarms, and factories.

    Example:
        >>> client = SynthClient()
        >>> research = client.research
        >>> project = research.projects.create(request)
        >>> swarm = research.swarms.create(project.project_id, request=launch)
        >>> swarm.wait()
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._core = CoreResearchClient(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        self._session: ResearchSession | None = None
        self._advanced: ResearchAdvancedAPI | None = None
        self._account: ResearchAccountAPI | None = None
        self._experiments: ResearchExperimentsAPI | None = None
        self._knowledge: ResearchKnowledgeAPI | None = None
        self._wiki: ResearchWikiAPI | None = None

    def _open_session(self) -> ResearchSession:
        if self._session is None:
            from synth_ai.core.research.advanced import open_advanced_session

            self._session = open_advanced_session(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._session

    @property
    def advanced(self) -> ResearchAdvancedAPI:
        """Explicitly unstable operator capabilities outside the hero workflow."""
        if self._advanced is None:
            from synth_ai.core.research.advanced import ResearchAdvancedAPI

            self._advanced = ResearchAdvancedAPI(
                open_session=self._open_session,
                limits=self._core.limits,
                economics=self._core.economics,
            )
        return self._advanced

    @property
    def account(self) -> ResearchAccountAPI:
        """Account-scoped reads and the API-key lifecycle."""
        if self._account is None:
            self._account = ResearchAccountAPI(self._open_session())
        return self._account

    @property
    def experiments(self) -> ResearchExperimentsAPI:
        """Experiment bundles, comparisons, and history."""
        if self._experiments is None:
            self._experiments = ResearchExperimentsAPI(self._open_session())
        return self._experiments

    @property
    def knowledge(self) -> ResearchKnowledgeAPI:
        """Durable typed knowledge carried between research cycles."""
        if self._knowledge is None:
            self._knowledge = ResearchKnowledgeAPI(self._open_session())
        return self._knowledge

    @property
    def wiki(self) -> ResearchWikiAPI:
        """Project wiki reads plus proposal intake."""
        if self._wiki is None:
            self._wiki = ResearchWikiAPI(self._open_session())
        return self._wiki

    @property
    def factories(self) -> FactoriesAPI:
        """Stable Factory lifecycle and typed Efforts."""
        return self._core.factories

    @property
    def environments(self) -> EnvironmentsAPI:
        """Versioned runtime declarations and deterministic preflight."""
        return self._core.environments

    @property
    def image_releases(self) -> ImageReleasesAPI:
        """Immutable customer image-release receipts and actor runtime images."""
        return self._core.image_releases

    @property
    def files(self) -> FilesAPI:
        """Run and project file APIs, including trace-bundle run outputs.

        `FilesAPI` has always existed on the session; it was simply never
        surfaced here, so `client.research.files` raised `AttributeError: files`
        while `session.files` worked. Callers reaching for run outputs — the
        `trace_v5_bundle` collection a `trace_mode = "required"` eval performs
        after a run completes — hit that gap only once the run had already
        succeeded, turning a passing benchmark into a reported failure.
        """
        return self._open_session().files

    @property
    def projects(self) -> ResearchProjectsAPI:
        """Create and configure Research projects through the core client."""
        return self._core.projects

    @property
    def swarms(self) -> ResearchSwarmsAPI:
        """Launch and control typed Research swarms."""
        return self._core.swarms

    @property
    def runs(self) -> ResearchSwarmsAPI:
        """Deprecated alias for :attr:`swarms`."""
        warnings.warn(
            "research.runs is deprecated; use research.swarms instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._core.swarms

    def __getattr__(self, name: str) -> object:
        compatibility = {
            "session": "session",
            "backing_client": "session",
            "efforts": "efforts",
            "limits": "limits",
            "economics": "economics",
            "secrets": "secrets",
            "hosted_artifacts": "artifacts",
            "visuals": "visuals",
            "images": "images",
            "tag": "tag",
        }
        target = compatibility.get(name)
        if target is not None:
            warnings.warn(
                f"research.{name} is deprecated; use research.advanced.{target}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self.advanced, target)
        if name == "get_limits":
            warnings.warn(
                "research.get_limits is deprecated; use research.advanced.limits.retrieve.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.advanced.limits.retrieve
        raise AttributeError(name)

    def close(self) -> None:
        """Close the underlying HTTP session and cached namespace clients."""
        self._core.close()
        if self._session is not None:
            self._session.close()
        self._session = None
        self._advanced = None
        self._account = None
        self._experiments = None
        self._knowledge = None
        self._wiki = None


ResearchClient = Client


__all__ = ["Client", "ResearchClient"]
