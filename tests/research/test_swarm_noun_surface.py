"""Guard the Managed Swarm public noun: canonical names, compat aliases, wire mapping."""

from __future__ import annotations

import warnings

import pytest
from synth_ai.research import (
    ResearchSwarm,
    ResearchSwarmHandle,
    ResearchSwarmProgress,
    ResearchSwarmsAPI,
    ResearchSwarmSession,
)
from synth_ai.research.client import ResearchClient


def _client() -> ResearchClient:
    return ResearchClient(api_key="test-key", base_url="http://localhost:1")


def test_canonical_imports_emit_no_deprecation() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import synth_ai.research.swarm_readouts  # noqa: F401
        import synth_ai.research.swarms  # noqa: F401


def test_run_module_shims_alias_swarm_types() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from synth_ai.research.runs import (
            ResearchRunHandle,
            ResearchRunsAPI,
            ResearchRunSession,
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert ResearchRunHandle is ResearchSwarmHandle
    assert ResearchRunSession is ResearchSwarmSession
    assert ResearchRunsAPI is ResearchSwarmsAPI


def test_package_level_run_aliases_point_at_swarm_types() -> None:
    from synth_ai.research import (
        ResearchRun,
        ResearchRunHandle,
        ResearchRunProgress,
        ResearchRunsAPI,
    )

    assert ResearchRun is ResearchSwarm
    assert ResearchRunProgress is ResearchSwarmProgress
    assert ResearchRunHandle is ResearchSwarmHandle
    assert ResearchRunsAPI is ResearchSwarmsAPI


def test_client_swarms_is_canonical_and_runs_warns() -> None:
    client = _client()
    swarms = client.swarms
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = client.runs
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert alias is swarms


def test_handle_exposes_swarm_id_over_wire_run_id() -> None:
    assert isinstance(ResearchSwarmHandle.swarm_id, property)


def test_get_rejects_conflicting_swarm_and_run_ids() -> None:
    api = _client().swarms
    with pytest.raises(ValueError), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        api.get(swarm_id="swarm-a", run_id="swarm-b")


def test_get_accepts_deprecated_run_id_keyword() -> None:
    from synth_ai.research.swarms import _resolve_swarm_id

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _resolve_swarm_id(None, "swarm-a") == "swarm-a"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
