"""Notebook discovery and Human Artifactory across SDK, CLI and MCP (SYN-6067)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic import JsonValue

from synth_ai import (
    ArtifactoryCategory,
    ArtifactoryHypothesis,
    ArtifactoryOrigin,
    ArtifactoryOriginRequiresHumanCredentialError,
    ArtifactoryPreference,
    ArtifactoryRevisionConflictError,
    AsyncSynthClient,
    SynthClient,
)
from synth_ai._cli_artifactory import execute_artifactory, parse_payload, render_artifactory
from synth_ai._cli_parser import parser
from synth_ai.artifactory import ArtifactoryFinding, ArtifactoryReference, payload_category
from synth_ai.errors import ArtifactoryPayloadError, InvalidRequestError, NotFoundError
from synth_ai.mcp.server import call_tool

if TYPE_CHECKING:
    from pathlib import Path

BASE_URL = "https://artifactory.example.test"
HUMAN_KEY = "sk_personal"
AUTOMATION_KEY = "sk_automation"
DIGEST = "0" * 64
PREFERENCE: dict[str, JsonValue] = {
    "aggregation_mode": "weighted_sum",
    "objectives": [
        {
            "sensor_id": "rhodes.reward",
            "direction": "maximize",
            "weight": 1.0,
            "scale_normalization": 1.0,
            "priority_rank": 1,
        }
    ],
    "hard_constraints": [],
    "risk_tolerance": {},
    "budgets": {"hard_spend_cap_micros": 5_000_000},
}


class Backend:
    """In-memory Core double for the six SYN-6065 operations."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, JsonValue]] = {}
        self.requests: list[httpx.Request] = []

    @staticmethod
    def problem(status: int, code: str) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "type": "https://docs.usesynth.ai/errors/" + code,
                "title": code,
                "status": status,
                "detail": code,
                "instance": "urn:request:artifactory_test",
                "code": code,
                "service": "core",
                "stage": "artifactory",
                "retryable": False,
                "outcome": "not_accepted",
            },
        )

    def _revision(
        self, record_id: str, number: int, body: dict[str, JsonValue], category: str
    ) -> dict[str, JsonValue]:
        revision: dict[str, JsonValue] = {
            "revision": number,
            "revision_id": f"afrev_{record_id}_{number}",
            "digest_sha256": DIGEST,
            "schema_version": f"synth.decision.{category}.v1",
            "author_kind": str(body.get("author_kind", "local_agent")),
            "pinned": bool(body.get("pinned", False)),
            "summary": str(body.get("summary", "")),
            "provenance": {
                "source": "human_artifactory",
                "kind": category,
                "record_id": record_id,
                "revision": str(number),
                "digest_sha256": DIGEST,
            },
            "created_at": "2026-11-27T12:00:00Z",
        }
        if category in body:
            revision[category] = body[category]
        return revision

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        human_credential = request.headers["authorization"] == "Bearer " + HUMAN_KEY
        path, method = request.url.path, request.method
        if method == "GET" and path == "/v1/notebooks":
            cursor = request.url.params.get("cursor")
            item = {
                "notebook_id": "nb_2" if cursor else "nb_1",
                "artifact_count": 1,
                "experiment_count": 0,
                "log_entry_count": 3,
                "artifactory_count": len(self.records),
                "has_research_log": True,
                "updated_at": "2026-11-27T12:00:00Z",
            }
            page: dict[str, JsonValue] = {"items": [item], "has_more": cursor is None}
            if cursor is None:
                page["next_cursor"] = "c2"
            return httpx.Response(200, json=page)
        if path.endswith("/artifactory") and method == "POST":
            body = json.loads(request.content)
            if body["origin"] == "human" and not human_credential:
                return self.problem(403, "artifactory_origin_requires_human_credential")
            record_id = f"afrec_{len(self.records) + 1}"
            category = body["category"]
            revision = self._revision(record_id, 1, body, category)
            record: dict[str, JsonValue] = {
                "record_id": record_id,
                "notebook_id": path.split("/")[3],
                "title": body["title"],
                "category": category,
                "origin": body["origin"],
                "pinned": revision["pinned"],
                "current_revision": 1,
                "current_digest_sha256": DIGEST,
                "schema_version": revision["schema_version"],
                "provenance": revision["provenance"],
                "revisions": [revision],
                "created_at": "2026-11-27T12:00:00Z",
                "updated_at": "2026-11-27T12:00:00Z",
                category: body[category],
            }
            if "run_id" in body:
                record["run_id"] = body["run_id"]
            self.records[record_id] = record
            return httpx.Response(201, json=record)
        if path.endswith("/artifactory") and method == "GET":
            return httpx.Response(200, json={"items": list(self.records.values())})
        if path.endswith("/preference/effective"):
            pinned = [
                r
                for r in self.records.values()
                if r["category"] == "preference" and r["pinned"] is True
            ]
            return httpx.Response(200, json=pinned[0]) if pinned else self.problem(404, "nf")
        if path.endswith("/revisions"):
            record = self.records[path.split("/")[3]]
            body = json.loads(request.content)
            if record["origin"] == "human" and not human_credential:
                return self.problem(403, "artifactory_origin_requires_human_credential")
            current = record["current_revision"]
            assert isinstance(current, int)
            if body.get("expected_revision", current) != current:
                return self.problem(409, "artifactory_revision_conflict")
            category = str(record["category"])
            revision = self._revision(str(record["record_id"]), current + 1, body, category)
            revisions = record["revisions"]
            assert isinstance(revisions, list)
            revisions.append(revision)
            record.update(current_revision=current + 1, pinned=revision["pinned"])
            if category in body:
                record[category] = body[category]
            return httpx.Response(200, json=record)
        if method == "GET" and path.startswith("/v1/artifactory/"):
            record_id = path.rsplit("/", 1)[1]
            if record_id in self.records:
                return httpx.Response(200, json=self.records[record_id])
            return self.problem(404, "artifactory_record_not_found")
        return self.problem(404, "route_not_found")


def http(backend: Backend) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(backend.handler))


async def test_async_lifecycle_and_typed_refusals() -> None:
    backend = Backend()
    preference = ArtifactoryPreference.model_validate(PREFERENCE)
    async with AsyncSynthClient(HUMAN_KEY, base_url=BASE_URL, http_client=http(backend)) as human:
        page = await human.notebooks.list_notebooks(limit=5)
        assert page.items[0].notebook_id == "nb_1"
        assert page.has_more
        assert [n.notebook_id async for n in human.notebooks.iter_notebooks()] == ["nb_1", "nb_2"]

        created = await human.artifactory.create_record(
            "nb_1",
            "Reward per USD",
            preference,
            origin=ArtifactoryOrigin.HUMAN,
            author_kind="human",
            pinned=True,
            idempotency_key="idem_pref_1",
        )
        sent = json.loads(backend.requests[-1].content)
        assert backend.requests[-1].headers["Idempotency-Key"] == "idem_pref_1"
        assert sent["category"] == "preference"
        assert sent["origin"] == "human"
        assert sent["preference"]["budgets"] == {"hard_spend_cap_micros": 5_000_000}
        assert created.category == ArtifactoryCategory.PREFERENCE
        assert created.preference is not None

        await human.artifactory.list_records(
            "nb_1", category="preference", origin="human", run_id="run_1", pinned_only=True
        )
        params = backend.requests[-1].url.params
        assert dict(params) == {
            "category": "preference",
            "origin": "human",
            "run_id": "run_1",
            "pinned": "true",
        }
        await human.artifactory.list_records("nb_1")
        assert "pinned" not in backend.requests[-1].url.params

        effective = await human.artifactory.resolve_preference("nb_1", run_id="run_1")
        assert effective.record_id == created.record_id
        assert backend.requests[-1].url.params["run_id"] == "run_1"

        revised = await human.artifactory.revise_record(
            created.record_id, pinned=False, summary="unpin", expected_revision=1
        )
        assert revised.current_revision == 2
        generated_key = backend.requests[-1].headers.get("Idempotency-Key")
        assert generated_key is not None
        assert generated_key != "idem_pref_1"
        with pytest.raises(ArtifactoryRevisionConflictError) as conflict:
            await human.artifactory.revise_record(created.record_id, expected_revision=1)
        assert conflict.value.problem.status == 409
        assert len((await human.artifactory.get_record(created.record_id)).revisions) == 2

    async with AsyncSynthClient(
        AUTOMATION_KEY, base_url=BASE_URL, http_client=http(backend)
    ) as automation:
        with pytest.raises(ArtifactoryOriginRequiresHumanCredentialError) as refused:
            await automation.artifactory.create_record(
                "nb_1", "Human claim", preference, origin="human"
            )
        assert refused.value.problem.status == 403
        with pytest.raises(ArtifactoryOriginRequiresHumanCredentialError):
            await automation.artifactory.revise_record(created.record_id, summary="agent edit")
        agent = await automation.artifactory.create_record(
            "nb_1", "Batch size", ArtifactoryHypothesis(statement="Batch > 64 does not help.")
        )
        assert agent.origin == ArtifactoryOrigin.AGENT
        assert json.loads(backend.requests[-1].content)["category"] == "hypothesis"


def test_sync_client_and_payload_typing() -> None:
    backend = Backend()
    finding = ArtifactoryFinding(
        summary="lr 3e-4 diverges",
        dead_end=True,
        sources=[
            ArtifactoryReference(
                source="run", kind="output", record_id="out_1", revision="1", digest_sha256=DIGEST
            )
        ],
    )
    with SynthClient(HUMAN_KEY, base_url=BASE_URL, http_client=http(backend)) as client:
        record = client.artifactory.create_record("nb_1", "Divergence", finding, run_id="run_1")
        assert record.category == ArtifactoryCategory.FINDING
        assert record.run_id == "run_1"
        assert client.artifactory.get_record(record.record_id).title == "Divergence"
        assert [n.notebook_id for n in client.notebooks.iter_notebooks(limit=1)] == [
            "nb_1",
            "nb_2",
        ]
    with pytest.raises(ArtifactoryPayloadError):
        payload_category(PREFERENCE)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ArtifactoryPayloadError, match="ArtifactoryHypothesis"):
        parse_payload("hypothesis", json.dumps({"rationale": "missing statement"}))


async def test_cli_commands_render_readable_output(tmp_path: Path) -> None:
    backend = Backend()
    payload = tmp_path / "preference.json"
    payload.write_text(json.dumps(PREFERENCE), encoding="utf-8")
    cli = parser()
    async with AsyncSynthClient(HUMAN_KEY, base_url=BASE_URL, http_client=http(backend)) as client:
        created = await execute_artifactory(
            client,
            cli.parse_args(
                [
                    "artifactory",
                    "create",
                    "--notebook-id",
                    "nb_1",
                    "--title",
                    "Reward per USD",
                    "--category",
                    "preference",
                    "--origin",
                    "human",
                    "--file",
                    str(payload),
                    "--pin",
                ]
            ),
        )
        view = render_artifactory(created)
        assert view is not None
        assert "weighted_sum: maximize rhodes.reward" in view
        assert "pinned    yes" in view

        listing = render_artifactory(
            await execute_artifactory(
                client, cli.parse_args(["artifactory", "list", "--notebook-id", "nb_1", "--pinned"])
            )
        )
        assert listing is not None
        assert listing.splitlines()[0].split() == [
            "RECORD",
            "CATEGORY",
            "ORIGIN",
            "PINNED",
            "REV",
            "TITLE",
        ]
        assert backend.requests[-1].url.params["pinned"] == "true"

        notebooks = render_artifactory(
            await execute_artifactory(client, cli.parse_args(["notebooks", "list"]))
        )
        assert notebooks is not None
        assert "more: --cursor c2" in notebooks

        revised = await execute_artifactory(
            client,
            cli.parse_args(
                ["artifactory", "revise", "afrec_1", "--unpin", "--expected-revision", "1"]
            ),
        )
        assert json.loads(backend.requests[-1].content) == {
            "pinned": False,
            "expected_revision": 1,
        }
        assert render_artifactory(revised) is not None

        with pytest.raises(ArtifactoryPayloadError, match="--file requires --category"):
            await execute_artifactory(
                client, cli.parse_args(["artifactory", "revise", "afrec_1", "--file", str(payload)])
            )
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"statement": 7}), encoding="utf-8")
        with pytest.raises(ArtifactoryPayloadError, match="invalid ArtifactoryHypothesis"):
            await execute_artifactory(
                client,
                cli.parse_args(
                    [
                        "artifactory",
                        "create",
                        "--notebook-id",
                        "nb_1",
                        "--title",
                        "t",
                        "--category",
                        "hypothesis",
                        "--origin",
                        "agent",
                        "--file",
                        str(bad),
                    ]
                ),
            )


async def test_mcp_tools_default_agent_origin_and_surface_refusals() -> None:
    backend = Backend()
    async with AsyncSynthClient(
        AUTOMATION_KEY, base_url=BASE_URL, http_client=http(backend)
    ) as client:
        created = json.loads(
            await call_tool(
                client,
                "swarms_artifactory_record_create",
                {
                    "notebook_id": "nb_1",
                    "idempotency_key": "idem_mcp_1",
                    "body": {
                        "title": "Batch size",
                        "category": "hypothesis",
                        "hypothesis": {"statement": "Batch > 64 does not help."},
                    },
                },
            )
        )
        assert created["origin"] == "agent"
        assert json.loads(backend.requests[-1].content)["origin"] == "agent"

        listed = json.loads(
            await call_tool(
                client, "swarms_artifactory_records_list", {"notebook_id": "nb_1", "pinned": "true"}
            )
        )
        assert [r["record_id"] for r in listed["items"]] == [created["record_id"]]
        assert backend.requests[-1].url.params["pinned"] == "true"
        got = json.loads(
            await call_tool(
                client, "swarms_artifactory_record_get", {"record_id": created["record_id"]}
            )
        )
        assert got["title"] == "Batch size"
        revised = json.loads(
            await call_tool(
                client,
                "swarms_artifactory_record_revise",
                {
                    "record_id": created["record_id"],
                    "idempotency_key": "idem_mcp_2",
                    "body": {"expected_revision": 1, "pinned": True},
                },
            )
        )
        assert revised["current_revision"] == 2
        notebooks = json.loads(await call_tool(client, "swarms_notebooks_list", {}))
        assert notebooks["items"][0]["notebook_id"] == "nb_1"
        assert backend.requests[-1].url.params["limit"] == "20"

        with pytest.raises(InvalidRequestError, match="category does not match"):
            await call_tool(
                client,
                "swarms_artifactory_record_create",
                {
                    "notebook_id": "nb_1",
                    "idempotency_key": "idem_mcp_3",
                    "body": {"title": "x", "category": "finding", "hypothesis": {"statement": "s"}},
                },
            )
        with pytest.raises(ArtifactoryOriginRequiresHumanCredentialError):
            await call_tool(
                client,
                "swarms_artifactory_record_create",
                {
                    "notebook_id": "nb_1",
                    "idempotency_key": "idem_mcp_4",
                    "body": {
                        "title": "Human claim",
                        "category": "preference",
                        "origin": "human",
                        "preference": PREFERENCE,
                    },
                },
            )
        with pytest.raises(NotFoundError):
            await call_tool(
                client,
                "swarms_artifactory_preference_resolve",
                {"notebook_id": "nb_1"},
            )
