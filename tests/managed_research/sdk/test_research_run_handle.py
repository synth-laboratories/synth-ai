"""ResearchRunHandle subclasses RunHandle (SYN-2892 #5): inherited surface + wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from synth_ai.managed_research.sdk.runs import RunHandle
from synth_ai.research.runs import ResearchRunHandle

# Public method surface ResearchRunHandle must keep exposing. Acts as a drift guard:
# if a future change drops one of these (e.g. a renamed RunHandle method), this fails.
INHERITED_METHODS = [
    "get",
    "public_state",
    "wait",
    "contract",
    "wait_terminal",
    "explain_blocker",
    "timeline",
    "execution",
    "transcript",
    "stream_events",
    "messages",
    "task_events",
    "objective_events",
    "work_graph",
    "event_log",
    "authority_readouts",
    "operator_evidence",
    "traces",
    "participants",
    "artifact_progress",
    "actor_logs",
    "checkpoints",
    "checkpoint",
    "stop",
    "pause",
    "resume",
    "control_actor",
    "pause_actor",
    "resume_actor",
    "interrupt_actor",
    "work_products",
    "reports",
    "final_report",
    "report_text",
    "actor_usage",
    "cost_summary",
    "resource_limits",
    "progress_toward_resource_limits",
    "artifact_manifest",
    "artifacts",
]
WRAPPER_METHODS = [
    "progress",
    "full_progress",
    "stream_transcript",
    "work_product_content",
    "usage",
    "download_workspace_archive",
    "list_artifacts",
]


class _FakeWorkProducts:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def content(self, work_product_id: str, *, as_text: bool = True) -> str:
        self._calls.append(("work_products.content", work_product_id, as_text))
        return "REPORT TEXT"


class _FakeRuns:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def stream_transcript(self, run_id: str, *, cursor=None, page_size=200, **kw):
        self._calls.append(("runs.stream_transcript", run_id, page_size))
        return iter([{"event": 1}])


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list = []
        self.work_products = _FakeWorkProducts(self.calls)
        self.runs = _FakeRuns(self.calls)

    def get_run_observability_snapshot(self, project_id, run_id, **kwargs):
        self.calls.append(("snapshot", project_id, run_id, kwargs))
        return "SNAPSHOT"

    def get_run_observability_snapshot_full(self, project_id, run_id):
        self.calls.append(("snapshot_full", project_id, run_id))
        return "FULL_SNAPSHOT"

    def get_run_usage(self, run_id):
        self.calls.append(("usage", run_id))
        return "USAGE"

    def download_run_workspace_archive(
        self, project_id, run_id, destination, *, timeout_seconds=None
    ):
        self.calls.append(("download", project_id, run_id, destination, timeout_seconds))
        return {"downloaded": destination}

    def list_run_artifacts(
        self, run_id, *, project_id=None, artifact_type=None, limit=None, cursor=None
    ):
        self.calls.append(("list_run_artifacts", run_id, project_id, artifact_type))
        return [SimpleNamespace(name="a", path="/a"), SimpleNamespace(name="b", path="/b")]


@pytest.fixture
def handle() -> tuple[ResearchRunHandle, _FakeClient]:
    client = _FakeClient()
    return ResearchRunHandle(RunHandle(client, "proj-1", "run-1")), client


def test_is_runhandle_subclass() -> None:
    assert issubclass(ResearchRunHandle, RunHandle)


def test_public_method_surface_preserved() -> None:
    for name in INHERITED_METHODS + WRAPPER_METHODS:
        assert callable(getattr(ResearchRunHandle, name, None)), f"missing public method: {name}"


def test_init_populates_inherited_attributes(handle) -> None:
    rh, _ = handle
    assert rh.project_id == "proj-1"
    assert rh.run_id == "run-1"


def test_progress_dispatches_to_client(handle) -> None:
    rh, client = handle
    assert rh.progress(event_limit=7) == "SNAPSHOT"
    kind, project_id, run_id, kwargs = client.calls[-1]
    assert (kind, project_id, run_id) == ("snapshot", "proj-1", "run-1")
    assert kwargs["event_limit"] == 7


def test_full_progress_dispatches_to_client(handle) -> None:
    rh, client = handle
    assert rh.full_progress() == "FULL_SNAPSHOT"
    assert client.calls[-1] == ("snapshot_full", "proj-1", "run-1")


def test_usage_dispatches_to_client(handle) -> None:
    rh, client = handle
    assert rh.usage() == "USAGE"
    assert client.calls[-1] == ("usage", "run-1")


def test_work_product_content_dispatches_to_client(handle) -> None:
    rh, client = handle
    assert rh.work_product_content("wp-9", as_text=False) == "REPORT TEXT"
    assert client.calls[-1] == ("work_products.content", "wp-9", False)


def test_download_workspace_archive_dispatches_to_client(handle) -> None:
    rh, client = handle
    assert rh.download_workspace_archive("/tmp/out", timeout_seconds=5.0) == {
        "downloaded": "/tmp/out"
    }
    assert client.calls[-1] == ("download", "proj-1", "run-1", "/tmp/out", 5.0)


def test_stream_transcript_dispatches_to_client(handle) -> None:
    rh, client = handle
    list(rh.stream_transcript(page_size=50))
    assert client.calls[-1] == ("runs.stream_transcript", "run-1", 50)


def test_list_artifacts_transforms_to_dicts(handle) -> None:
    rh, client = handle
    result = rh.list_artifacts(artifact_type="log")
    assert result == [{"name": "a", "path": "/a"}, {"name": "b", "path": "/b"}]
    assert client.calls[-1] == ("list_run_artifacts", "run-1", "proj-1", "log")
