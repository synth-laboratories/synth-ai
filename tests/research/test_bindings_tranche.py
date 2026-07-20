"""Bindings-tranche tests: fake sessions only, no network."""

from __future__ import annotations

from typing import Any

import pytest
from synth_ai.managed_research.models.factories import (
    FactoryCostsLimits,
    FactoryPublicVisuals,
    FactoryStatus,
)
from synth_ai.managed_research.sdk.transport import build_http_transport
from synth_ai.research.account import (
    AccountBalance,
    AccountIdentity,
    ResearchAccountAPI,
)
from synth_ai.research.experiments import ResearchExperimentsAPI
from synth_ai.research.factories import ResearchFactoriesAPI
from synth_ai.research.factory_handles import ResearchEffortHandle
from synth_ai.research.knowledge import (
    ResearchKnowledgeAPI,
    ResearchProjectsNotesAPI,
)
from synth_ai.research.project_namespaces import ResearchProjectsGitAPI
from synth_ai.research.wiki import ResearchWikiAPI


class _RecordingSession:
    """Fake session client capturing every ``_request_json`` call."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = responses or {}

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "json_body": json_body,
            }
        )
        return self._responses.get(f"{method} {path}", {})


# ---------------------------------------------------------------------------
# Auth header (transport-level, shared by every account/wiki/git call)
# ---------------------------------------------------------------------------


def test_transport_auth_header_is_bearer_api_key() -> None:
    transport = build_http_transport(
        api_key="sk-test-123",
        backend_base="http://backend.local",
        timeout_seconds=1.0,
    )
    try:
        assert transport.client.headers["authorization"] == "Bearer sk-test-123"
    finally:
        transport.close()


# ---------------------------------------------------------------------------
# Account path construction
# ---------------------------------------------------------------------------


def test_account_paths() -> None:
    session = _RecordingSession(
        responses={
            "GET /api/v1/balance/current": {
                "org_id": "org_1",
                "balance_cents": 250,
                "balance_dollars": 2.5,
                "last_updated": "2026-07-20T00:00:00Z",
            },
            "GET /api/v1/me": {"org_id": "org_1", "user_id": "user_1"},
            "GET /api/v1/usage/tiers": [],
        }
    )
    account = ResearchAccountAPI(session)  # type: ignore[arg-type]

    balance = account.balance.get()
    assert isinstance(balance, AccountBalance)
    assert balance.balance_cents == 250

    account.balance.usage()
    account.credits.transactions(cursor="c1", limit=10)
    account.tiers.list()
    account.tiers.get("free")
    account.usage()
    account.user_limits()
    account.overview(days=7, include_projects=True)
    account.byok.list()
    account.byok.create(provider="openai", encrypted_key_b64="Zm9v")
    account.byok.get("openai")
    account.byok.delete("openai")
    account.byok.validate("openai")
    account.byok.status()
    account.members.list(limit=5, offset=10)
    account.subscription.cancel()
    account.crypto.public_key()
    me = account.me()
    assert isinstance(me, AccountIdentity)
    assert me.org_id == "org_1"

    observed = [(call["method"], call["path"]) for call in session.calls]
    assert observed == [
        ("GET", "/api/v1/balance/current"),
        ("GET", "/api/v1/balance/usage"),
        ("GET", "/api/v1/credits/transactions"),
        ("GET", "/api/v1/usage/tiers"),
        ("GET", "/api/v1/usage/tiers/free"),
        ("GET", "/api/v1/usage"),
        ("GET", "/api/v1/usage/user-limits"),
        ("GET", "/api/v1/usage/overview"),
        ("GET", "/api/v1/byok/keys"),
        ("POST", "/api/v1/byok/keys"),
        ("GET", "/api/v1/byok/keys/openai"),
        ("DELETE", "/api/v1/byok/keys/openai"),
        ("POST", "/api/v1/byok/keys/openai/validate"),
        ("GET", "/api/v1/byok/status"),
        ("GET", "/api/members"),
        ("POST", "/api/v1/subscription/cancel"),
        ("GET", "/api/v1/crypto/public-key"),
        ("GET", "/api/v1/me"),
    ]

    tx_call = session.calls[2]
    assert tx_call["params"] == {"limit": 10, "cursor": "c1"}
    overview_call = session.calls[7]
    assert overview_call["params"] == {"days": 7, "include_projects": True}
    byok_create_call = session.calls[9]
    assert byok_create_call["json_body"] == {
        "provider": "openai",
        "encrypted_key_b64": "Zm9v",
    }
    members_call = session.calls[14]
    assert members_call["params"] == {"limit": 5, "offset": 10}


# ---------------------------------------------------------------------------
# Wiki path construction
# ---------------------------------------------------------------------------


def test_wiki_paths() -> None:
    session = _RecordingSession(
        responses={
            "GET /smr/projects/proj_1/wiki/pages": {
                "state": {"phase": "ready"},
                "wiki_project": {"wiki_project_id": "wiki_1"},
                "pages": [
                    {
                        "page_id": "page_1",
                        "project_id": "proj_1",
                        "slug": "overview",
                        "title": "Overview",
                    }
                ],
            },
        }
    )
    wiki = ResearchWikiAPI(session)  # type: ignore[arg-type]

    wiki.overview("proj_1")
    listing = wiki.pages.list("proj_1")
    wiki.pages.get("proj_1", "overview")
    wiki.search("proj_1", "loss curve", limit=5)
    wiki.context_pack.preview("proj_1", limit=40)
    wiki.proposals.list("proj_1", state="open", limit=25)
    wiki.proposals.create("proj_1", {"title": "Fix stale page"})

    assert listing.pages[0].slug == "overview"
    assert listing.state == {"phase": "ready"}

    observed = [(call["method"], call["path"]) for call in session.calls]
    assert observed == [
        ("GET", "/smr/projects/proj_1/wiki"),
        ("GET", "/smr/projects/proj_1/wiki/pages"),
        ("GET", "/smr/projects/proj_1/wiki/pages/overview"),
        ("GET", "/smr/projects/proj_1/wiki/search"),
        ("GET", "/smr/projects/proj_1/wiki/context-pack/preview"),
        ("GET", "/smr/projects/proj_1/wiki/proposals"),
        ("POST", "/smr/projects/proj_1/wiki/proposals"),
    ]
    assert session.calls[3]["params"] == {"q": "loss curve", "limit": 5}
    assert session.calls[4]["params"] == {"limit": 40}
    assert session.calls[5]["params"] == {"limit": 25, "state": "open"}
    assert session.calls[6]["json_body"] == {"title": "Fix stale page"}


# ---------------------------------------------------------------------------
# Git read paths
# ---------------------------------------------------------------------------


def test_git_read_paths() -> None:
    session = _RecordingSession(
        responses={
            "GET /smr/projects/proj_1/git/status": {
                "project_id": "proj_1",
                "branch": "main",
                "default_branch": "main",
                "head_commit_sha": "abc123",
                "recent_commits": [
                    {
                        "sha": "abc123",
                        "summary": "init",
                        "author_name": "a",
                        "author_email": "a@x",
                        "authored_at": "2026-07-20T00:00:00Z",
                    }
                ],
                "unmerged_branches": [{"name": "feat", "head_commit_sha": "def456"}],
                "tree_paths": ["README.md"],
                "tree_truncated": False,
            },
            "GET /smr/projects/proj_1/git/tree": {"entries": ["src/a.py"]},
            "GET /smr/projects/proj_1/git/pull-requests": {
                "pull_requests": [
                    {
                        "number": 7,
                        "title": "Add thing",
                        "state": "open",
                        "html_url": "https://github.com/x/y/pull/7",
                        "user": {"login": "dev"},
                        "head": {"ref": "feat"},
                        "base": {"ref": "main"},
                    }
                ]
            },
        }
    )
    git = ResearchProjectsGitAPI(session)  # type: ignore[arg-type]

    status = git.status("proj_1", branch="main")
    assert status.head_commit_sha == "abc123"
    assert status.recent_commits[0].sha == "abc123"
    assert status.unmerged_branches[0].name == "feat"

    entries = git.tree("proj_1", ref="main", path_prefix="src")
    assert entries == ("src/a.py",)

    git.file("proj_1", "README.md", ref="main")
    git.diff("proj_1", base_ref="main", head_ref="feat", path="src/a.py")
    prs = git.pull_requests.list("proj_1", state="open", limit=10)
    assert prs[0].number == 7
    assert prs[0].user_login == "dev"
    assert prs[0].head_ref == "feat"

    observed = [(call["method"], call["path"]) for call in session.calls]
    assert observed == [
        ("GET", "/smr/projects/proj_1/git/status"),
        ("GET", "/smr/projects/proj_1/git/tree"),
        ("GET", "/smr/projects/proj_1/git/file"),
        ("GET", "/smr/projects/proj_1/git/diff"),
        ("GET", "/smr/projects/proj_1/git/pull-requests"),
    ]
    assert session.calls[2]["params"] == {"path": "README.md", "ref": "main"}
    assert session.calls[3]["params"] == {
        "base_ref": "main",
        "head_ref": "feat",
        "path": "src/a.py",
    }
    assert session.calls[4]["params"] == {"state": "open", "limit": 10}


# ---------------------------------------------------------------------------
# Experiments typed decode
# ---------------------------------------------------------------------------


class _ExperimentsStubSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def get_experiment_bundle(self, project_id: str, experiment_id: str) -> dict[str, Any]:
        self.calls.append(("bundle", (project_id, experiment_id)))
        return {
            "experiment_id": experiment_id,
            "project_id": project_id,
            "schema_version": "v1",
            "run_ids": ["run_1"],
            "integrity": {"accepted_cycle": True},
            "unexpected_key": {"kept": True},
        }

    def get_experiment_history(self, project_id: str, *, limit: int = 50) -> dict[str, Any]:
        self.calls.append(("history", (project_id, limit)))
        return {
            "project_id": project_id,
            "schema_version": "v1",
            "bundles": [],
            "accepted_cycles": 3,
            "incomplete_cycles": 1,
        }

    def compare_experiments(self, project_id: str, experiment_ids: Any) -> dict[str, Any]:
        self.calls.append(("compare", (project_id, tuple(experiment_ids))))
        return {
            "project_id": project_id,
            "schema_version": "v1",
            "experiment_ids": list(experiment_ids),
            "comparable": True,
            "rows": [{"experiment_id": "exp_1"}],
        }


def test_experiments_typed_decode() -> None:
    session = _ExperimentsStubSession()
    api = ResearchExperimentsAPI(session)  # type: ignore[arg-type]

    bundle = api.bundle("proj_1", "exp_1")
    assert bundle.experiment_id == "exp_1"
    assert bundle.accepted_cycle is True
    assert bundle.run_ids == ("run_1",)
    assert bundle.raw["unexpected_key"] == {"kept": True}

    history = api.history("proj_1", limit=10)
    assert history.accepted_cycles == 3
    assert session.calls[1] == ("history", ("proj_1", 10))

    comparison = api.compare("proj_1", ["exp_1", "exp_2"])
    assert comparison.comparable is True
    assert comparison.experiment_ids == ("exp_1", "exp_2")


# ---------------------------------------------------------------------------
# Notes + knowledge typed decode
# ---------------------------------------------------------------------------


class _NotesStubSession:
    def get_project_notes(self, project_id: str) -> dict[str, Any]:
        return {
            "project_id": project_id,
            "notes": "hello",
            "updated_at": "2026-07-20T01:02:03Z",
        }

    def set_project_notes(self, project_id: str, notes: str) -> dict[str, Any]:
        return {"project_id": project_id, "notes": notes, "updated_at": None}

    def append_project_notes(self, project_id: str, notes: str) -> dict[str, Any]:
        return {"project_id": project_id, "notes": f"hello\n{notes}", "updated_at": None}

    def get_org_knowledge(self) -> dict[str, Any]:
        return {"org_id": "org_1", "content": "facts", "updated_at": None}

    def set_org_knowledge(self, content: str) -> dict[str, Any]:
        return {"org_id": "org_1", "content": content, "updated_at": None}


def test_notes_and_knowledge_typed() -> None:
    session = _NotesStubSession()
    notes_api = ResearchProjectsNotesAPI(session)  # type: ignore[arg-type]
    knowledge_api = ResearchKnowledgeAPI(session)  # type: ignore[arg-type]

    notes = notes_api.get("proj_1")
    assert notes.project_id == "proj_1"
    assert notes.notes == "hello"
    assert notes.updated_at is not None

    assert notes_api.set("proj_1", "replaced").notes == "replaced"
    assert notes_api.append("proj_1", "more").notes == "hello\nmore"

    knowledge = knowledge_api.get()
    assert knowledge.org_id == "org_1"
    assert knowledge.content == "facts"
    assert knowledge_api.set("new facts").content == "new facts"


# ---------------------------------------------------------------------------
# FactoryPublicVisuals / FactoryCostsLimits decode
# ---------------------------------------------------------------------------


def _sample_visuals_payload() -> dict[str, Any]:
    return {
        "schema": "synth.open_research.factory_public_visuals.v1",
        "programme_id": "open-frontier-crafter",
        "current_run": {"run_id": "run_1"},
        "next_scheduled_event": {"kind": "wake"},
        "last_accepted_result": None,
        "latest_report": {"report_id": "rep_1"},
        "score_trend": [{"score": 0.5}, {"score": 0.6}],
        "candidate_rejections": [{"reason": "no_lift"}],
        "evidence_rejections": [{"reason": "stale"}],
        "scheduler_decisions": [{"decision": "wake"}],
        "public_data": ["experiments"],
        "visualizations": [{"key": "score_trend"}],
        "brand_new_backend_key": {"forward": "compatible"},
    }


def test_factory_public_visuals_decode_with_unknown_keys() -> None:
    visuals = FactoryPublicVisuals.from_wire(_sample_visuals_payload())
    assert visuals.schema == "synth.open_research.factory_public_visuals.v1"
    assert visuals.programme_id == "open-frontier-crafter"
    assert visuals.current_run == {"run_id": "run_1"}
    assert visuals.last_accepted_result is None
    assert visuals.score_trend == ({"score": 0.5}, {"score": 0.6})
    assert visuals.candidate_rejections == ({"reason": "no_lift"},)
    assert visuals.evidence_rejections == ({"reason": "stale"},)
    assert visuals.scheduler_decisions == ({"decision": "wake"},)
    assert visuals.public_data == ("experiments",)
    # Unknown keys survive in raw.
    assert visuals.raw["brand_new_backend_key"] == {"forward": "compatible"}


def test_factory_costs_limits_decode() -> None:
    payload = {
        "factory_budget_policy": {"daily_usd": 10},
        "factory_budget_status": {"spent_usd": 2},
        "factory_cap_policy": {"max_runs": 3},
        "effort_budget_policies": {"eff_1": {"daily_usd": 1}},
        "future_key": True,
    }
    costs = FactoryCostsLimits.from_wire(payload)
    assert costs.factory_budget_policy == {"daily_usd": 10}
    assert costs.factory_budget_status == {"spent_usd": 2}
    assert costs.factory_cap_policy == {"max_runs": 3}
    assert costs.effort_budget_policies == {"eff_1": {"daily_usd": 1}}
    assert costs.raw["future_key"] is True


def test_factory_status_typed_visuals_properties() -> None:
    status = FactoryStatus.from_wire(
        {
            "factory": {
                "factory_id": "fac_1",
                "org_id": "org_1",
                "name": "F",
                "kind": "customer",
                "status": "active",
            },
            "public_visuals": _sample_visuals_payload(),
            "costs_limits": {"factory_budget_policy": {"daily_usd": 10}},
        }
    )
    assert status.typed_public_visuals.programme_id == "open-frontier-crafter"
    assert status.typed_costs_limits.factory_budget_policy == {"daily_usd": 10}


# ---------------------------------------------------------------------------
# Factory handle lifecycle delegation
# ---------------------------------------------------------------------------


class _RecordingFactoriesAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def pause(self, factory_id: str) -> str:
        self.calls.append(("pause", factory_id))
        return "paused"

    def resume(self, factory_id: str) -> str:
        self.calls.append(("resume", factory_id))
        return "resumed"

    def archive(self, factory_id: str) -> str:
        self.calls.append(("archive", factory_id))
        return "archived"

    def status(self, factory_id: str) -> str:
        self.calls.append(("status", factory_id))
        return "status"

    def watch_status(self, factory_id: str, **kwargs: Any) -> Any:
        self.calls.append(("watch_status", (factory_id, kwargs)))
        return iter(())

    def list_efforts(self, factory_id: str) -> list[str]:
        self.calls.append(("list_efforts", factory_id))
        return ["effort"]


class _RecordingEffortsAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(self, effort_id: str) -> str:
        self.calls.append(("get", effort_id))
        return "effort"

    def pause(self, effort_id: str) -> str:
        self.calls.append(("pause", effort_id))
        return "paused"

    def resume(self, effort_id: str) -> str:
        self.calls.append(("resume", effort_id))
        return "resumed"


class _FactoriesStubSession:
    def __init__(self) -> None:
        self.factories = _RecordingFactoriesAPI()
        self.efforts = _RecordingEffortsAPI()


def test_factories_api_lifecycle_promotion() -> None:
    session = _FactoriesStubSession()
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]

    api.pause("fac_1")
    api.resume("fac_1")
    api.archive("fac_1")
    api.watch_status("fac_1", poll_interval=1.0, timeout=2.0, stop_when_idle=True)

    assert session.factories.calls == [
        ("pause", "fac_1"),
        ("resume", "fac_1"),
        ("archive", "fac_1"),
        (
            "watch_status",
            ("fac_1", {"poll_interval": 1.0, "timeout": 2.0, "stop_when_idle": True}),
        ),
    ]


def test_factory_handle_delegation() -> None:
    session = _FactoriesStubSession()
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]
    handle = api.open("fac_1")
    assert handle.factory_id == "fac_1"

    assert handle.status() == "status"
    assert handle.pause() == "paused"
    assert handle.resume() == "resumed"
    assert handle.archive() == "archived"
    assert handle.efforts.list() == ["effort"]

    effort_handle = handle.efforts.open("eff_1")
    assert isinstance(effort_handle, ResearchEffortHandle)
    assert effort_handle.effort_id == "eff_1"
    assert effort_handle.get() == "effort"
    assert effort_handle.pause() == "paused"
    assert effort_handle.resume() == "resumed"

    assert session.factories.calls == [
        ("status", "fac_1"),
        ("pause", "fac_1"),
        ("resume", "fac_1"),
        ("archive", "fac_1"),
        ("list_efforts", "fac_1"),
    ]
    assert session.efforts.calls == [
        ("get", "eff_1"),
        ("pause", "eff_1"),
        ("resume", "eff_1"),
    ]


def test_wake_due_rejects_mismatched_preview() -> None:
    session = _FactoriesStubSession()
    api = ResearchFactoriesAPI(session)  # type: ignore[arg-type]
    handle = api.open("fac_1")

    class _Preview:
        factory_id = "fac_other"

    with pytest.raises(ValueError):
        handle.wake_due(preview=_Preview())  # type: ignore[arg-type]
