"""End-to-end contract tests for Track B Artifacts, Experiments, Research Logs, and Run Review.

Covers SYN-5973 (Hosted Interactive Artifacts), SYN-5974 (Multi-Experiment & Notebook
Research Logs), and SYN-5975 (Run Review Surface) across AsyncSynthClient, SynthClient,
CLI, and canonical OpenAPI 1.43.0 MCP tool dispatch.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic import JsonValue

from synth_ai import (
    ArtifactKind,
    ArtifactOrigin,
    AsyncSynthClient,
    AuthorKind,
    ExactReference,
    LogCategory,
    SynthClient,
    cli,
)
from synth_ai.errors import ForbiddenError, InvalidRequestError, NotFoundError
from synth_ai.mcp.server import call_tool
from synth_ai.research import ReferenceSource
from synth_ai.research_mcp import RESEARCH_MCP_TOOLS, call_research_tool

if TYPE_CHECKING:
    from pathlib import Path

BASE_URL = "https://research.example.test"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_backend() -> httpx.MockTransport:
    artifacts: dict[str, dict[str, JsonValue]] = {}
    artifact_bodies: dict[tuple[str, str], str] = {}
    experiments: dict[str, dict[str, JsonValue]] = {}
    logs: list[dict[str, JsonValue]] = []

    def problem(status: int, code: str, detail: str) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "type": "https://docs.usesynth.ai/errors/" + code,
                "title": code,
                "status": status,
                "instance": "urn:request:track_b_test",
                "code": code,
                "detail": detail,
                "service": "core",
                "stage": "research",
                "retryable": False,
                "outcome": "not_accepted",
            },
            headers={"X-Request-Id": "req_track_b_test"},
        )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        auth = request.headers.get("authorization", "")
        if auth == "Bearer sk_other_org":
            return problem(403, "forbidden", "cross-org access denied")

        if method == "POST" and path == "/v1/artifacts":
            body = json.loads(request.content)
            art_id = f"art_{len(artifacts) + 1:03d}"
            rev_id = f"{art_id}_rev_1"
            digest = _digest(body["content"])
            rev: dict[str, JsonValue] = {
                "revision": 1,
                "revision_id": rev_id,
                "digest_sha256": digest,
                "byte_size": len(body["content"].encode("utf-8")),
                "media_type": "text/html",
                "created_at": "2026-11-24T17:00:00Z",
                "author_kind": "human" if body.get("origin") == "human" else "local_agent",
                "summary": body["summary"],
                "byte_state": "available",
                "provenance": {
                    "source": "artifact",
                    "kind": "artifact_revision",
                    "record_id": art_id,
                    "revision": "1",
                    "digest_sha256": digest,
                },
            }
            record: dict[str, JsonValue] = {
                "artifact_id": art_id,
                "run_id": body.get("run_id"),
                "notebook_id": body["notebook_id"],
                "title": body["title"],
                "kind": body["kind"],
                "origin": body.get("origin", "agent"),
                "current_revision": 1,
                "revisions": [rev],
                "share": None,
                "pinned": False,
            }
            artifacts[art_id] = record
            artifact_bodies[(art_id, rev_id)] = body["content"]
            return httpx.Response(201, json=record)

        if method == "POST" and path.startswith("/v1/artifacts/") and path.endswith("/revisions"):
            art_id = path.split("/")[3]
            if art_id not in artifacts:
                return problem(404, "not_found", "artifact not found")
            body = json.loads(request.content)
            record = artifacts[art_id]
            raw_revs = record["revisions"]
            assert isinstance(raw_revs, list)
            revisions = list(raw_revs)
            next_num = len(revisions) + 1
            rev_id = f"{art_id}_rev_{next_num}"
            digest = _digest(body["content"])
            revisions.append(
                {
                    "revision": next_num,
                    "revision_id": rev_id,
                    "digest_sha256": digest,
                    "byte_size": len(body["content"].encode("utf-8")),
                    "media_type": "text/html",
                    "created_at": "2026-11-24T17:05:00Z",
                    "author_kind": body.get("author_kind") or "local_agent",
                    "summary": body["summary"],
                    "byte_state": "available",
                    "provenance": {
                        "source": "artifact",
                        "kind": "artifact_revision",
                        "record_id": art_id,
                        "revision": str(next_num),
                        "digest_sha256": digest,
                    },
                }
            )
            record["current_revision"] = next_num
            record["revisions"] = revisions
            artifact_bodies[(art_id, rev_id)] = body["content"]
            return httpx.Response(200, json=record)

        if method == "GET" and path.startswith("/v1/artifacts/") and path.endswith("/content"):
            art_id = path.split("/")[3]
            rev_id = request.url.params.get("revision_id")
            if not rev_id and art_id in artifacts:
                raw_revs = artifacts[art_id]["revisions"]
                assert isinstance(raw_revs, list) and raw_revs
                last_rev = raw_revs[-1]
                assert isinstance(last_rev, dict)
                rev_id = str(last_rev["revision_id"])
            content = artifact_bodies.get((art_id, rev_id or ""))
            if content is None:
                return problem(404, "not_found", "artifact content not found")
            return httpx.Response(
                200,
                text=content,
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )

        if path.startswith("/v1/artifacts/") and path.endswith("/share"):
            art_id = path.split("/")[3]
            if art_id not in artifacts:
                return problem(404, "not_found", "artifact not found")
            grant: dict[str, JsonValue]
            if method == "POST":
                body = json.loads(request.content)
                grant = {
                    "share_id": f"shr_{art_id}",
                    "scope": str(body["scope"]),
                    "url": f"https://artifacts.usesynth.ai/s/{art_id}",
                    "revoked": False,
                    "created_at": "2026-11-24T17:10:00Z",
                }
            else:
                grant = {
                    "share_id": f"shr_{art_id}",
                    "scope": "org",
                    "url": f"https://artifacts.usesynth.ai/s/{art_id}",
                    "revoked": True,
                    "created_at": "2026-11-24T17:10:00Z",
                }
            artifacts[art_id]["share"] = grant
            return httpx.Response(200, json=grant)

        if method == "GET" and path.startswith("/v1/artifacts/"):
            art_id = path.split("/")[3]
            if art_id not in artifacts:
                return problem(404, "not_found", "artifact not found")
            return httpx.Response(200, json=artifacts[art_id])

        if method == "GET" and path.startswith("/v1/notebooks/") and path.endswith("/artifacts"):
            notebook_id = path.split("/")[3]
            run_id = request.url.params.get("run_id")
            items = [
                item
                for item in artifacts.values()
                if item["notebook_id"] == notebook_id
                and (run_id is None or item["run_id"] == run_id)
            ]
            return httpx.Response(200, json={"items": items})

        if method == "POST" and path.startswith("/v1/notebooks/") and path.endswith("/experiments"):
            notebook_id = path.split("/")[3]
            body = json.loads(request.content)
            exp_id = f"exp_{len(experiments) + 1:03d}"
            exp_record: dict[str, JsonValue] = {
                "experiment_id": exp_id,
                "notebook_id": notebook_id,
                "run_id": body["run_id"],
                "revision": 1,
                "title": body["title"],
                "specification": body["specification"],
                "hypothesis": body.get("hypothesis"),
                "arms": body["arms"],
                "pinned_inputs": body.get("pinned_inputs", []),
                "stopping_criteria": body.get("stopping_criteria"),
                "provenance": {
                    "source": "research",
                    "kind": "experiment",
                    "record_id": exp_id,
                    "revision": "1",
                    "digest_sha256": _digest(body["specification"]),
                },
                "trials": [],
            }
            experiments[exp_id] = exp_record
            return httpx.Response(201, json=exp_record)

        if method == "POST" and path.startswith("/v1/experiments/") and path.endswith("/revisions"):
            exp_id = path.split("/")[3]
            if exp_id not in experiments:
                return problem(404, "not_found", "experiment not found")
            body = json.loads(request.content)
            exp_record = dict(experiments[exp_id])
            current_rev = exp_record["revision"]
            assert isinstance(current_rev, int)
            next_rev = current_rev + 1
            exp_record["revision"] = next_rev
            for key in ("title", "specification", "hypothesis", "arms", "stopping_criteria"):
                if key in body:
                    exp_record[key] = body[key]
            experiments[exp_id] = exp_record
            return httpx.Response(200, json=exp_record)

        if method == "GET" and path.startswith("/v1/experiments/"):
            exp_id = path.split("/")[3]
            if exp_id not in experiments:
                return problem(404, "not_found", "experiment not found")
            return httpx.Response(200, json=experiments[exp_id])

        if method == "GET" and path.startswith("/v1/notebooks/") and path.endswith("/experiments"):
            notebook_id = path.split("/")[3]
            run_id = request.url.params.get("run_id")
            items = [
                item
                for item in experiments.values()
                if item["notebook_id"] == notebook_id
                and (run_id is None or item["run_id"] == run_id)
            ]
            return httpx.Response(200, json={"items": items})

        if (
            method == "POST"
            and path.startswith("/v1/notebooks/")
            and path.endswith("/research-log/init")
        ):
            notebook_id = path.split("/")[3]
            body = json.loads(request.content)
            return httpx.Response(
                201,
                json={
                    "notebook_id": notebook_id,
                    "title": body["title"],
                    "summary": body.get("summary"),
                    "entry_count": len(logs),
                    "created_at": "2026-11-24T17:00:00Z",
                },
            )

        if method == "POST" and "/research-log/" in path and path.endswith("/corrections"):
            parts = path.split("/")
            notebook_id, entry_id = parts[3], parts[5]
            body = json.loads(request.content)
            entry: dict[str, JsonValue] = {
                "entry_id": f"log_{len(logs) + 1:03d}",
                "notebook_id": notebook_id,
                "run_id": body.get("run_id"),
                "category": "correction",
                "author_kind": "human",
                "author_label": "reviewer",
                "text": body["text"],
                "occurred_at": "2026-11-24T17:15:00Z",
                "recorded_at": "2026-11-24T17:15:00Z",
                "sync_state": "synced",
                "correction_of": entry_id,
                "sources": body.get("sources", []),
                "artifact_ids": body.get("artifact_ids", []),
                "missing_evidence": [],
            }
            logs.append(entry)
            return httpx.Response(201, json=entry)

        if (
            method == "POST"
            and path.startswith("/v1/notebooks/")
            and path.endswith("/research-log")
        ):
            notebook_id = path.split("/")[3]
            body = json.loads(request.content)
            entry = {
                "entry_id": f"log_{len(logs) + 1:03d}",
                "notebook_id": notebook_id,
                "run_id": body.get("run_id"),
                "category": body["category"],
                "author_kind": body.get("author_kind", "local_agent"),
                "author_label": body.get("author_label") or "agent",
                "text": body["text"],
                "occurred_at": "2026-11-24T17:12:00Z",
                "recorded_at": "2026-11-24T17:12:00Z",
                "sync_state": "synced",
                "correction_of": body.get("correction_of"),
                "sources": body.get("sources", []),
                "artifact_ids": body.get("artifact_ids", []),
                "missing_evidence": [],
            }
            logs.append(entry)
            return httpx.Response(201, json=entry)

        if (
            method == "GET"
            and path.startswith("/v1/notebooks/")
            and path.endswith("/research-log/export")
        ):
            notebook_id = path.split("/")[3]
            fmt = request.url.params.get("format", "markdown")
            matching = [item for item in logs if item["notebook_id"] == notebook_id]
            rendered = "\n".join(f"- [{item['category']}] {item['text']}" for item in matching)
            return httpx.Response(
                200,
                json={
                    "notebook_id": notebook_id,
                    "run_id": request.url.params.get("run_id"),
                    "format": fmt,
                    "content": rendered,
                    "entry_count": len(matching),
                },
            )

        if method == "GET" and path.startswith("/v1/notebooks/") and path.endswith("/research-log"):
            notebook_id = path.split("/")[3]
            category = request.url.params.get("category")
            query = request.url.params.get("query")
            items = [
                item
                for item in logs
                if item["notebook_id"] == notebook_id
                and (category is None or item["category"] == category)
                and (query is None or query.lower() in str(item["text"]).lower())
            ]
            return httpx.Response(200, json={"items": items})

        if method == "GET" and path.startswith("/v1/runs/") and path.endswith("/review"):
            run_id = path.split("/")[3]
            return httpx.Response(
                200,
                json={
                    "pacing": {
                        "phase": "explore",
                        "elapsed_seconds": 900,
                        "wall_time_seconds": 3600,
                        "budget_spent_usd": "2.50",
                        "budget_limit_usd": "10.00",
                    },
                    "experiments": [
                        item for item in experiments.values() if item["run_id"] == run_id
                    ],
                    "research_log": [item for item in logs if item["run_id"] == run_id],
                    "artifacts": [item for item in artifacts.values() if item["run_id"] == run_id],
                },
            )

        return problem(404, "not_found", f"unhandled route {method} {path}")

    return httpx.MockTransport(handler)


async def test_async_artifacts_experiments_and_research_logs_lifecycle() -> None:
    transport = _make_backend()
    async with httpx.AsyncClient(transport=transport) as http:
        async with AsyncSynthClient("sk_org_a", base_url=BASE_URL, http_client=http) as client:
            created = await client.artifacts.create_artifact(
                "nb_alpha",
                "Pareto Frontier",
                ArtifactKind.HTML,
                "<main>v1</main>",
                "initial chart",
                origin=ArtifactOrigin.AGENT,
                run_id="run_AAAAAAAAAAAAAAAAAAAAAA",
            )
            assert created.current_revision == 1
            assert created.notebook_id == "nb_alpha"
            updated = await client.artifacts.update_artifact(
                created.artifact_id, "<main>v2</main>", "add uncertainty bands"
            )
            assert updated.current_revision == 2
            assert len(updated.revisions) == 2

            rev2 = updated.revisions[-1]
            raw = await client.artifacts.get_artifact_content(
                created.artifact_id,
                revision_id=rev2.revision_id,
                digest_sha256=rev2.digest_sha256,
            )
            assert raw == "<main>v2</main>"

            inline = await client.artifacts.get_artifact_content_inline(
                created.artifact_id,
                revision_id=rev2.revision_id,
                digest_sha256=rev2.digest_sha256,
            )
            assert inline.encoding == "utf8"
            assert inline.content == "<main>v2</main>"

            grant = await client.artifacts.share_artifact(created.artifact_id, scope="link")
            assert grant.revoked is False
            revoked = await client.artifacts.revoke_artifact_share(created.artifact_id)
            assert revoked.revoked is True

            exp = await client.experiments.create_experiment(
                "nb_alpha",
                "run_AAAAAAAAAAAAAAAAAAAAAA",
                "CISPO vs GRPO",
                "Compare clipped ratio schedules on 8 seeds",
                ["baseline", "cispo-v2"],
                hypothesis="CISPO improves pass@1 per dollar by >=10%",
            )
            assert exp.revision == 1
            exp_v2 = await client.experiments.revise_experiment(
                exp.experiment_id,
                arms=["baseline", "cispo-v2", "cispo-v3"],
                stopping_criteria="Stop if p < 0.01",
            )
            assert exp_v2.revision == 2
            assert len(exp_v2.arms) == 3

            summary = await client.research_logs.create_log("nb_alpha", title="Alpha Lab Book")
            assert summary.notebook_id == "nb_alpha"

            citation = ExactReference(
                source=ReferenceSource.ARTIFACT,
                kind="artifact_revision",
                record_id=created.artifact_id,
                revision="2",
                digest_sha256=rev2.digest_sha256,
            )
            entry = await client.research_logs.append_entry(
                "nb_alpha",
                LogCategory.HYPOTHESIS,
                "Expect CISPO-v3 to dominate on tail latency",
                run_id="run_AAAAAAAAAAAAAAAAAAAAAA",
                author_kind=AuthorKind.LOCAL_AGENT,
                sources=[citation],
                artifact_ids=[created.artifact_id],
            )
            correction = await client.research_logs.correct_entry(
                "nb_alpha",
                entry.entry_id,
                "Tail latency gain came from batch packing, not clipping",
                sources=[citation],
            )
            assert correction.correction_of == entry.entry_id

            found = await client.research_logs.search_entries("nb_alpha", query="packing")
            assert len(found.items) == 1
            exported = await client.research_logs.export_log("nb_alpha")
            assert exported.entry_count == 2
            assert "batch packing" in exported.content

            review = await client.swarms.review("run_AAAAAAAAAAAAAAAAAAAAAA")
            assert review.pacing is not None
            assert review.pacing.phase == "explore"
            assert len(review.artifacts) == 1
            assert len(review.experiments) == 1
            assert len(review.research_log) == 1


async def test_cross_org_access_and_missing_records_fail_closed() -> None:
    transport = _make_backend()
    async with httpx.AsyncClient(transport=transport) as http:
        async with AsyncSynthClient("sk_other_org", base_url=BASE_URL, http_client=http) as other:
            with pytest.raises(ForbiddenError):
                await other.artifacts.get_artifact("art_001")
            with pytest.raises(ForbiddenError):
                await other.research_logs.search_entries("nb_alpha")
        async with AsyncSynthClient("sk_org_a", base_url=BASE_URL, http_client=http) as client:
            with pytest.raises(NotFoundError):
                await client.artifacts.get_artifact("art_missing")
            with pytest.raises(NotFoundError):
                await client.experiments.get_experiment("exp_missing")


def test_sync_client_and_cli_surface(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _make_backend()
    with SynthClient(
        "sk_org_a", base_url=BASE_URL, http_client=httpx.AsyncClient(transport=transport)
    ) as client:
        art = client.artifacts.create_artifact(
            "nb_sync", "Summary Table", "markdown", "# Results\n", "v1 summary"
        )
        assert client.artifacts.list_artifacts("nb_sync").items[0].artifact_id == art.artifact_id
        exp = client.experiments.create_experiment(
            "nb_sync", "run_BBBBBBBBBBBBBBBBBBBBBB", "Sweep", "Spec", ["a", "b"]
        )
        assert client.experiments.get_experiment(exp.experiment_id).title == "Sweep"
        note = client.research_logs.append_entry(
            "nb_sync", "preference", "Prefer compact tables over prose"
        )
        assert note.category == LogCategory.PREFERENCE
        review = client.swarms.review("run_BBBBBBBBBBBBBBBBBBBBBB")
        assert len(review.experiments) == 1

    artifact_file = tmp_path / "chart.svg"
    artifact_file.write_text("<svg><circle r='10'/></svg>", encoding="utf-8")
    monkeypatch.setenv("SYNTH_API_KEY", "sk_org_a")
    code = cli.main(
        [
            "--base-url",
            BASE_URL,
            "artifacts",
            "create",
            "--notebook-id",
            "nb_sync",
            "--title",
            "CLI Chart",
            "--kind",
            "svg",
            "--summary",
            "from cli",
            "--file",
            str(artifact_file),
        ],
        http_client=httpx.AsyncClient(transport=transport),
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["title"] == "CLI Chart"
    assert payload["kind"] == "svg"

    review_code = cli.main(
        [
            "--base-url",
            BASE_URL,
            "inspect",
            "run_BBBBBBBBBBBBBBBBBBBBBB",
            "--view",
            "review",
        ],
        http_client=httpx.AsyncClient(transport=transport),
    )
    assert review_code == 0
    review_payload = json.loads(capsys.readouterr().out)
    assert len(review_payload["experiments"]) == 1


async def test_canonical_openapi_mcp_dispatch_for_track_b() -> None:
    transport = _make_backend()
    async with httpx.AsyncClient(transport=transport) as http:
        async with AsyncSynthClient("sk_org_a", base_url=BASE_URL, http_client=http) as client:
            created = json.loads(
                await call_tool(
                    client,
                    "swarms_artifact_create",
                    {
                        "idempotency_key": "idem_art_1",
                        "body": {
                            "notebook_id": "nb_mcp",
                            "run_id": "run_CCCCCCCCCCCCCCCCCCCCCC",
                            "title": "Canonical MCP Artifact",
                            "kind": "markdown",
                            "origin": "agent",
                            "content": "# Live Review",
                            "summary": "v1",
                        },
                    },
                )
            )
            assert created["notebook_id"] == "nb_mcp"
            content = json.loads(
                await call_tool(
                    client,
                    "swarms_artifact_content",
                    {"artifact_id": created["artifact_id"]},
                )
            )
            assert content["content"] == "# Live Review"
            assert content["encoding"] == "utf8"

            summary = json.loads(
                await call_tool(
                    client,
                    "swarms_research_log_init",
                    {
                        "notebook_id": "nb_mcp",
                        "idempotency_key": "idem_log_init_1",
                        "body": {"title": "MCP Notebook"},
                    },
                )
            )
            assert summary["notebook_id"] == "nb_mcp"

            review = json.loads(
                await call_tool(
                    client,
                    "swarms_run_review",
                    {"run_id": "run_CCCCCCCCCCCCCCCCCCCCCC"},
                )
            )
            assert len(review["artifacts"]) == 1


async def test_research_mcp_catalog_and_dispatch() -> None:
    assert len(RESEARCH_MCP_TOOLS) == 14
    transport = _make_backend()
    async with httpx.AsyncClient(transport=transport) as http:
        async with AsyncSynthClient("sk_org_a", base_url=BASE_URL, http_client=http) as client:
            created_json = await call_research_tool(
                client,
                "artifacts_create",
                {
                    "notebook_id": "nb_mcp",
                    "title": "MCP Widget",
                    "kind": "react",
                    "content": "export default function App() { return null; }",
                    "summary": "initial widget",
                },
            )
            created = json.loads(created_json)
            assert created["current_revision"] == 1

            note_json = await call_research_tool(
                client,
                "research_log_append",
                {
                    "notebook_id": "nb_mcp",
                    "category": "observation",
                    "text": "Widget renders cleanly in opaque sandbox",
                    "sources": [
                        {
                            "source": "artifact",
                            "kind": "react",
                            "record_id": created["artifact_id"],
                            "revision": "1",
                            "digest_sha256": "0" * 64,
                        }
                    ],
                    "artifact_ids": [created["artifact_id"]],
                },
            )
            note_payload = json.loads(note_json)
            assert note_payload["category"] == "observation"
            assert note_payload["sources"][0]["source"] == "artifact"

            with pytest.raises(InvalidRequestError) as refused:
                await call_research_tool(client, "artifacts_create", {"notebook_id": "nb_mcp"})
            assert refused.value.code == "tool_arguments_invalid"
