from __future__ import annotations

from contextlib import nullcontext

from synth_ai.mcp.managed_research_server import ManagedResearchMcpServer


class _FakeClient:
    def __init__(self) -> None:
        self.upload_calls: list[dict[str, object]] = []
        self.sublinear_calls: list[dict[str, object]] = []
        self.project_message_calls: list[dict[str, object]] = []

    def git_upload_files(self, project_id: str, **kwargs: object) -> dict[str, object]:
        self.upload_calls.append({"project_id": project_id, **kwargs})
        return {
            "project_event_id": "event-upload",
            "event_summary": "Operator uploaded 2 files to branch feat/collab",
        }

    def sublinear_create_task(self, project_id: str, **kwargs: object) -> dict[str, object]:
        self.sublinear_calls.append({"project_id": project_id, **kwargs})
        return {
            "project_event_id": "event-task",
            "event_summary": "Operator created Sublinear task NH-142",
        }

    def post_project_message(self, project_id: str, **kwargs: object) -> dict[str, object]:
        self.project_message_calls.append({"project_id": project_id, **kwargs})
        return {
            "project_event_id": "event-message",
            "event_summary": "Agent codex posted a project message",
            "message_for_agents": "Agent codex posted a project message",
        }


def test_git_upload_tool_uses_upload_endpoint_and_adds_agent_message(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeClient()
    monkeypatch.setattr(server, "_client_from_args", lambda _args: nullcontext(fake_client))

    result = server._tool_git_upload_files(
        {
            "project_id": "project-1",
            "branch": "feat/collab",
            "commit_message": "Upload seed files",
            "files": [
                {"path": "notes/a.txt", "content": "hello"},
                {"path": "notes/b.txt", "content_base64": "aGVsbG8="},
            ],
        }
    )

    assert fake_client.upload_calls == [
        {
            "project_id": "project-1",
            "branch": "feat/collab",
            "commit_message": "Upload seed files",
            "files": [
                {"path": "notes/a.txt", "content": "hello"},
                {"path": "notes/b.txt", "content_base64": "aGVsbG8="},
            ],
            "run_id": None,
            "project_only": False,
            "idempotency_key": None,
        }
    ]
    assert result["project_event_id"] == "event-upload"
    assert result["message_for_agents"] == "Operator uploaded 2 files to branch feat/collab"


def test_sublinear_mutation_tools_add_message_for_agents(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeClient()
    monkeypatch.setattr(server, "_client_from_args", lambda _args: nullcontext(fake_client))

    result = server._tool_sublinear_create_task(
        {
            "project_id": "project-1",
            "title": "Coordinate SFT collaboration",
            "description": "Track the next collaboration slice",
        }
    )

    assert fake_client.sublinear_calls == [
        {
            "project_id": "project-1",
            "title": "Coordinate SFT collaboration",
            "description": "Track the next collaboration slice",
            "run_id": None,
            "branch": None,
            "project_only": False,
            "idempotency_key": None,
        }
    ]
    assert result["project_event_id"] == "event-task"
    assert result["message_for_agents"] == "Operator created Sublinear task NH-142"


def test_project_message_tool_passes_through_source_and_message_for_agents(monkeypatch) -> None:
    server = ManagedResearchMcpServer()
    fake_client = _FakeClient()
    monkeypatch.setattr(server, "_client_from_args", lambda _args: nullcontext(fake_client))

    result = server._tool_post_project_message(
        {
            "project_id": "project-1",
            "body": "I pushed the first collaboration slice.",
            "summary": "Codex shared a project update",
            "branch": "feat/collab",
            "task_id": "task-9",
            "repo": "acme/repo",
            "source": "agent",
            "actor_type": "agent",
            "actor_id": "codex",
        }
    )

    assert fake_client.project_message_calls == [
        {
            "project_id": "project-1",
            "body": "I pushed the first collaboration slice.",
            "summary": "Codex shared a project update",
            "payload": None,
            "run_id": None,
            "branch": "feat/collab",
            "task_id": "task-9",
            "repo": "acme/repo",
            "source": "agent",
            "actor_type": "agent",
            "actor_id": "codex",
            "project_only": False,
            "idempotency_key": None,
        }
    ]
    assert result["project_event_id"] == "event-message"
    assert result["message_for_agents"] == "Agent codex posted a project message"
