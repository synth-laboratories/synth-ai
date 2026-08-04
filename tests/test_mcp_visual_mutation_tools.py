"""MCP visual MUTATE parity: create/update/promote/unpublish over the SDK.

The MCP layer gains write adapters for the existing SDK visual write surface.
Two invariants are load-bearing:

1. Scope discipline — every mutate tool is registered with WRITE scopes (both
   on the ToolDefinition and in the registry's default-scope table), and the
   destructive SDK operations (delete/restore) are deliberately not exposed.
2. Score-series server authority — no MCP mutation path may create or rewrite
   a Visual claiming a score-series kind; the refusal is typed with error code
   ``visual_score_series_server_authoritative`` and fires before any client
   is constructed.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from synth_ai.mcp.research.registry import (
    _DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME,
    READ_SCOPES,
    WRITE_SCOPES,
    build_tool_registry,
)
from synth_ai.mcp.research.tools.visuals import (
    VISUAL_SCORE_SERIES_ERROR_CODE,
    VisualScoreSeriesServerAuthoritativeError,
    build_visual_tools,
)

MUTATE_TOOL_NAMES = (
    "research_create_visual",
    "research_update_visual",
    "research_promote_visual",
    "research_unpublish_visual",
)


class _WireResult:
    """Stands in for any SDK contract object; MCP handlers call to_wire()."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_wire(self) -> dict[str, Any]:
        return dict(self._payload)


class _RecordingVisualsAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> _WireResult:
        self.calls.append((name, args, kwargs))
        return _WireResult({"op": name})

    def create(self, *args: Any, **kwargs: Any) -> _WireResult:
        return self._record("create", *args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> _WireResult:
        return self._record("update", *args, **kwargs)

    def promote(self, *args: Any, **kwargs: Any) -> _WireResult:
        return self._record("promote", *args, **kwargs)

    def unpublish(self, *args: Any, **kwargs: Any) -> _WireResult:
        return self._record("unpublish", *args, **kwargs)


class _StubClient:
    def __init__(self) -> None:
        self.visuals = _RecordingVisualsAPI()

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


class _StubClientFactory:
    def __init__(self) -> None:
        self.client = _StubClient()
        self.invocations = 0

    def __call__(self, args: dict[str, Any]) -> _StubClient:
        self.invocations += 1
        return self.client


@pytest.fixture()
def factory() -> _StubClientFactory:
    return _StubClientFactory()


def _tools(factory: _StubClientFactory) -> dict[str, Any]:
    return {tool.name: tool for tool in build_visual_tools(factory)}


# ---------------------------------------------------------------------------
# Registration and scopes
# ---------------------------------------------------------------------------


def test_mutate_tools_registered_with_write_scopes(factory: _StubClientFactory) -> None:
    tools = _tools(factory)
    for name in MUTATE_TOOL_NAMES:
        assert name in tools, f"missing mutate tool {name}"
        assert tools[name].required_scopes == WRITE_SCOPES
        assert _DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME[name] == WRITE_SCOPES


def test_read_tools_keep_read_scopes(factory: _StubClientFactory) -> None:
    tools = _tools(factory)
    for name in (
        "research_list_visuals",
        "research_get_visual",
        "research_get_visual_content",
        "research_get_visual_preview",
    ):
        assert tools[name].required_scopes == READ_SCOPES


def test_destructive_visual_operations_deliberately_absent(
    factory: _StubClientFactory,
) -> None:
    names = set(_tools(factory))
    assert "research_delete_visual" not in names
    assert "research_restore_visual" not in names


def test_tools_survive_registry_scope_gate(factory: _StubClientFactory) -> None:
    registry = build_tool_registry(build_visual_tools(factory))
    for name in MUTATE_TOOL_NAMES:
        assert registry[name].required_scopes == WRITE_SCOPES


def test_mutate_tools_advertised_by_stable_stdio_server() -> None:
    from synth_ai.mcp.research.server import ResearchMcpServer

    names = set(ResearchMcpServer(api_key="sk-test").available_tool_names())
    assert set(MUTATE_TOOL_NAMES) <= names


def test_score_series_guard_documented_in_tool_descriptions(
    factory: _StubClientFactory,
) -> None:
    tools = _tools(factory)
    for name in ("research_create_visual", "research_update_visual"):
        description = tools[name].description
        assert "server-authoritative" in description
        assert VISUAL_SCORE_SERIES_ERROR_CODE in description


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_create_requires_project_title_html(factory: _StubClientFactory) -> None:
    create = _tools(factory)["research_create_visual"].handler
    with pytest.raises(ValueError, match="'project_id'"):
        create({"title": "t", "html": "<html></html>"})
    with pytest.raises(ValueError, match="'title'"):
        create({"project_id": "proj_1", "html": "<html></html>"})
    with pytest.raises(ValueError, match="'html'"):
        create({"project_id": "proj_1", "title": "t"})
    assert factory.invocations == 0


def test_create_rejects_malformed_optional_inputs(factory: _StubClientFactory) -> None:
    create = _tools(factory)["research_create_visual"].handler
    base = {"project_id": "proj_1", "title": "t", "html": "<html></html>"}
    with pytest.raises(ValueError, match="metadata"):
        create({**base, "metadata": ["not-an-object"]})
    with pytest.raises(ValueError, match="source_run_ids"):
        create({**base, "source_run_ids": [17]})
    with pytest.raises(ValueError, match="preview_png_base64"):
        create({**base, "preview_png_base64": "not-base64!!"})
    assert factory.invocations == 0


def test_update_requires_at_least_one_field(factory: _StubClientFactory) -> None:
    update = _tools(factory)["research_update_visual"].handler
    with pytest.raises(ValueError, match="at least one"):
        update({"visual_id": "vis_1"})
    assert factory.invocations == 0


def test_promote_requires_slug(factory: _StubClientFactory) -> None:
    promote = _tools(factory)["research_promote_visual"].handler
    with pytest.raises(ValueError, match="'slug'"):
        promote({"visual_id": "vis_1"})
    assert factory.invocations == 0


def test_unpublish_requires_visual_id(factory: _StubClientFactory) -> None:
    unpublish = _tools(factory)["research_unpublish_visual"].handler
    with pytest.raises(ValueError, match="'visual_id'"):
        unpublish({})
    assert factory.invocations == 0


# ---------------------------------------------------------------------------
# Handler -> SDK wiring
# ---------------------------------------------------------------------------


def test_create_wires_arguments_to_sdk(factory: _StubClientFactory) -> None:
    create = _tools(factory)["research_create_visual"].handler
    png = base64.b64encode(b"png-bytes").decode("ascii")
    result = create(
        {
            "project_id": "proj_1",
            "title": "Loss curves",
            "summary": "sweep 4",
            "html": "<html>ok</html>",
            "visual_kind": "research_visual",
            "source_run_ids": ["run_a", "run_b"],
            "metadata": {"k": "v"},
            "root_artifact_id": "art_root",
            "preview_png_base64": png,
        }
    )
    assert result == {"op": "create"}
    ((name, args, kwargs),) = factory.client.visuals.calls
    assert name == "create"
    assert str(args[0]) == "proj_1"
    assert kwargs["title"] == "Loss curves"
    assert kwargs["summary"] == "sweep 4"
    assert kwargs["html"] == "<html>ok</html>"
    assert kwargs["visual_kind"] == "research_visual"
    assert kwargs["source_run_ids"] == ("run_a", "run_b")
    assert kwargs["metadata"] == {"k": "v"}
    assert str(kwargs["root_artifact_id"]) == "art_root"
    assert kwargs["preview_png"] == b"png-bytes"


def test_create_defaults_visual_kind(factory: _StubClientFactory) -> None:
    create = _tools(factory)["research_create_visual"].handler
    create({"project_id": "proj_1", "title": "t", "html": "<html></html>"})
    ((_, _, kwargs),) = factory.client.visuals.calls
    assert kwargs["visual_kind"] == "research_visual"
    assert kwargs["root_artifact_id"] is None
    assert kwargs["preview_png"] is None


def test_update_wires_patch_to_sdk(factory: _StubClientFactory) -> None:
    update = _tools(factory)["research_update_visual"].handler
    result = update({"visual_id": "vis_1", "title": "New title", "summary": "s"})
    assert result == {"op": "update"}
    ((name, args, _),) = factory.client.visuals.calls
    assert name == "update"
    assert str(args[0]) == "vis_1"
    patch = args[1]
    assert patch.to_wire() == {"title": "New title", "summary": "s"}


def test_promote_wires_slug_to_sdk(factory: _StubClientFactory) -> None:
    promote = _tools(factory)["research_promote_visual"].handler
    result = promote({"visual_id": "vis_1", "slug": "loss-curves"})
    assert result == {"op": "promote"}
    ((name, args, _),) = factory.client.visuals.calls
    assert name == "promote"
    assert str(args[0]) == "vis_1"
    assert args[1].to_wire() == {"slug": "loss-curves"}


def test_unpublish_wires_visual_id_to_sdk(factory: _StubClientFactory) -> None:
    unpublish = _tools(factory)["research_unpublish_visual"].handler
    result = unpublish({"visual_id": "vis_1"})
    assert result == {"op": "unpublish"}
    ((name, args, _),) = factory.client.visuals.calls
    assert name == "unpublish"
    assert str(args[0]) == "vis_1"


# ---------------------------------------------------------------------------
# Score-series server authority guard
# ---------------------------------------------------------------------------

SCORE_SERIES_SPELLINGS = (
    "score_series",
    "score-series",
    "Score Series",
    "scoreSeries",
    "SCORE_SERIES",
    "factory.score_series",
    "score_series_v2",
    "banking77_score_series",
)


@pytest.mark.parametrize("kind", SCORE_SERIES_SPELLINGS)
def test_create_refuses_score_series_kinds(factory: _StubClientFactory, kind: str) -> None:
    create = _tools(factory)["research_create_visual"].handler
    with pytest.raises(VisualScoreSeriesServerAuthoritativeError) as excinfo:
        create(
            {
                "project_id": "proj_1",
                "title": "fake scores",
                "html": "<html>forged</html>",
                "visual_kind": kind,
            }
        )
    assert str(excinfo.value.error_code) == VISUAL_SCORE_SERIES_ERROR_CODE
    assert factory.invocations == 0, "guard must fire before any client is built"


@pytest.mark.parametrize("kind", SCORE_SERIES_SPELLINGS)
def test_update_refuses_score_series_kinds(factory: _StubClientFactory, kind: str) -> None:
    update = _tools(factory)["research_update_visual"].handler
    with pytest.raises(VisualScoreSeriesServerAuthoritativeError) as excinfo:
        update({"visual_id": "vis_1", "visual_kind": kind})
    assert str(excinfo.value.error_code) == VISUAL_SCORE_SERIES_ERROR_CODE
    assert factory.invocations == 0


def test_guard_is_typed_and_not_retryable(factory: _StubClientFactory) -> None:
    create = _tools(factory)["research_create_visual"].handler
    with pytest.raises(VisualScoreSeriesServerAuthoritativeError) as excinfo:
        create(
            {
                "project_id": "proj_1",
                "title": "t",
                "html": "<html></html>",
                "visual_kind": "score_series",
            }
        )
    failure = excinfo.value.failure
    assert failure is not None
    assert str(failure.code) == VISUAL_SCORE_SERIES_ERROR_CODE
    assert failure.retry.retryable is False


@pytest.mark.parametrize("kind", ["scoreboard", "research_visual", "score_summary"])
def test_non_score_series_kinds_pass_the_guard(factory: _StubClientFactory, kind: str) -> None:
    create = _tools(factory)["research_create_visual"].handler
    create(
        {
            "project_id": "proj_1",
            "title": "t",
            "html": "<html></html>",
            "visual_kind": kind,
        }
    )
    ((_, _, kwargs),) = factory.client.visuals.calls
    assert kwargs["visual_kind"] == kind
