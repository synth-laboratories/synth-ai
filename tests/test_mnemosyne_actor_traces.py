"""Actor MCP trace download/compare clients against an in-process fake session.

The fake mirrors the backend owner's ``chunk_response`` and compare result
shapes (services/orchestra_port/mnemosyne_program_owner.py and
services/mnemosyne/project_trace_compare.py). No monkeypatch.
"""

import asyncio
import base64
import hashlib
import json

import pytest
from pydantic import ValidationError
from synth_ai.errors import CapabilityUnavailableError, ContractError
from synth_ai.sdk.mnemosyne.actor_tools import (
    COMPARE_PROJECT_TRACES_TOOL,
    PROJECT_TRACE_ARCHIVE_BYTES_MAX,
    READ_PROJECT_TRACE_TOOL,
)
from synth_ai.sdk.mnemosyne.trace_chunks import (
    RetainedTraceSource,
    adownload_project_trace,
    download_project_trace,
    partial_path,
)
from synth_ai.sdk.mnemosyne.trace_compare import (
    PinnedProjectTrace,
    ProjectTraceComparison,
    acompare_project_traces,
    compare_project_traces,
)

ASSOCIATION = "mna_" + "a" * 64
ARCHIVE = bytes(range(256)) * 41  # 10_496 bytes; not chunk aligned
DIGEST = hashlib.sha256(ARCHIVE).hexdigest()


class LinkLostError(Exception):
    pass


def _envelope(result):
    return {
        "content": [{"type": "text", "text": json.dumps(result)}],
        "structuredContent": result,
        "isError": False,
    }


class FakeActorSession:
    def __init__(
        self,
        archive=ARCHIVE,
        *,
        tools=(READ_PROJECT_TRACE_TOOL,),
        fail_after=None,
        corrupt_at=None,
        oversize=False,
        served_digest=None,
    ):
        self.archive = archive
        self.tools = tools
        self.fail_after = fail_after
        self.corrupt_at = corrupt_at
        self.oversize = oversize
        self.served_digest = served_digest or hashlib.sha256(archive).hexdigest()
        self.calls = []

    def tool_names(self):
        return self.tools

    def call_tool(self, name, arguments):
        if name not in self.tools:
            raise AssertionError(f"unadvertised tool called: {name}")
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise LinkLostError("connection lost")
        self.calls.append((name, arguments))
        if name == COMPARE_PROJECT_TRACES_TOOL:
            return _envelope(compare_result(arguments["traces"]))
        window = arguments["range"]
        assert window["archive_sha256"] == self.served_digest
        start = window["offset"]
        length = window["length"] * (2 if self.oversize else 1)
        data = self.archive[start : start + length]
        end = start + len(data)
        digest = hashlib.sha256(data).hexdigest()
        if self.corrupt_at == start:
            data = bytes([data[0] ^ 1]) + data[1:]
        return _envelope(
            {
                "schema_version": "synth.mnemosyne.actor-project-trace-chunk.v1",
                "association_id": arguments["association_id"],
                "rollout_id": "rollout-1",
                "execution_epoch": "epoch-1",
                "archive_sha256": self.served_digest,
                "archive_size_bytes": len(self.archive),
                "offset": start,
                "byte_count": len(data),
                "chunk_sha256": digest,
                "next_offset": None if end == len(self.archive) else end,
                "complete": end == len(self.archive),
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            }
        )


class AsyncFake(FakeActorSession):
    async def call_tool(self, name, arguments):
        return FakeActorSession.call_tool(self, name, arguments)


def source(archive=ARCHIVE):
    return RetainedTraceSource(
        association_id=ASSOCIATION,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        archive_size_bytes=len(archive),
    )


def test_download_streams_verified_windows(tmp_path):
    session = FakeActorSession()
    target = tmp_path / "trace.zip"
    receipt = download_project_trace(session, source(), target, chunk_bytes=4096)
    assert target.read_bytes() == ARCHIVE
    assert [c[1]["range"]["offset"] for c in session.calls] == [0, 4096, 8192]
    assert session.calls[-1][1]["range"]["length"] == len(ARCHIVE) - 8192
    assert receipt.chunks_fetched == 3 and receipt.resumed_from == 0
    assert receipt.rollout_id == "rollout-1"
    assert not partial_path(source(), target).exists()


def test_resume_after_interruption_continues_from_partial(tmp_path):
    target = tmp_path / "trace.zip"
    with pytest.raises(LinkLostError):
        download_project_trace(FakeActorSession(fail_after=2), source(), target, chunk_bytes=4096)
    partial = partial_path(source(), target)
    assert partial.stat().st_size == 8192 and not target.exists()
    resumed = FakeActorSession()
    receipt = download_project_trace(resumed, source(), target, chunk_bytes=4096)
    assert [c[1]["range"]["offset"] for c in resumed.calls] == [8192]
    assert receipt.resumed_from == 8192 and receipt.chunks_fetched == 1
    assert target.read_bytes() == ARCHIVE


def test_async_resume_from_a_non_aligned_partial(tmp_path):
    target = tmp_path / "trace.zip"
    partial_path(source(), target).write_bytes(ARCHIVE[:1000])
    session = AsyncFake()
    receipt = asyncio.run(adownload_project_trace(session, source(), target, chunk_bytes=4096))
    assert session.calls[0][1]["range"]["offset"] == 1000
    assert receipt.resumed_from == 1000 and target.read_bytes() == ARCHIVE


def test_chunk_digest_mismatch_refuses_and_keeps_verified_prefix(tmp_path):
    target = tmp_path / "trace.zip"
    with pytest.raises(ContractError) as refused:
        download_project_trace(
            FakeActorSession(corrupt_at=4096), source(), target, chunk_bytes=4096
        )
    assert refused.value.code == "actor_trace_chunk_digest_mismatch"
    assert refused.value.stage == "chunk@4096"
    assert partial_path(source(), target).read_bytes() == ARCHIVE[:4096]
    assert not target.exists()


def test_full_digest_mismatch_removes_partial(tmp_path):
    target = tmp_path / "trace.zip"
    partial = partial_path(source(), target)
    partial.write_bytes(b"\x00" * 8192)  # wrong but chunk-plausible prefix
    with pytest.raises(ContractError) as refused:
        download_project_trace(FakeActorSession(), source(), target, chunk_bytes=4096)
    assert refused.value.code == "actor_trace_archive_digest_mismatch"
    assert not partial.exists() and not target.exists()


def test_listed_digest_disagreeing_with_bytes_refuses(tmp_path):
    session = FakeActorSession(served_digest="b" * 64)
    listed = RetainedTraceSource(
        association_id=ASSOCIATION, archive_sha256="b" * 64, archive_size_bytes=len(ARCHIVE)
    )
    with pytest.raises(ContractError) as refused:
        download_project_trace(session, listed, tmp_path / "t.zip", chunk_bytes=4096)
    assert refused.value.code == "actor_trace_archive_digest_mismatch"


def test_size_bounds(tmp_path):
    with pytest.raises(ValidationError):
        RetainedTraceSource(
            association_id=ASSOCIATION,
            archive_sha256=DIGEST,
            archive_size_bytes=PROJECT_TRACE_ARCHIVE_BYTES_MAX + 1,
        )
    with pytest.raises(ContractError) as refused:
        download_project_trace(
            FakeActorSession(), source(), tmp_path / "t.zip", chunk_bytes=1024 * 1024 + 1
        )
    assert refused.value.code == "actor_trace_chunk_too_large"
    with pytest.raises(ContractError) as oversized:
        download_project_trace(
            FakeActorSession(oversize=True), source(), tmp_path / "t.zip", chunk_bytes=4096
        )
    assert oversized.value.code == "actor_trace_chunk_too_large"
    big = tmp_path / "big.zip"
    partial_path(source(), big).write_bytes(ARCHIVE + b"x")
    with pytest.raises(ContractError) as partial:
        download_project_trace(FakeActorSession(), source(), big, chunk_bytes=4096)
    assert partial.value.code == "actor_trace_partial_invalid"


def test_download_is_capability_gated(tmp_path):
    session = FakeActorSession(tools=("mnemosyne_list_project_traces",))
    with pytest.raises(CapabilityUnavailableError) as refused:
        download_project_trace(session, source(), tmp_path / "t.zip")
    assert refused.value.code == "actor_trace_tool_not_advertised" and session.calls == []


def compare_result(traces):
    archives = [
        {
            "index": index,
            "association_id": pin["association_id"],
            "rollout_id": f"rollout-{index}",
            "execution_epoch": "epoch-1",
            "archive_sha256": pin["archive_sha256"],
            "archive_size_bytes": 100,
            "entry_count": 1,
            "expanded_bytes": 12,
            "manifest_sha256": "c" * 64,
            "manifest": {
                "truncated": False,
                "entries": [{"path": "trace.jsonl", "size_bytes": 12, "sha256": "d" * 64}],
            },
            "traces": [
                {
                    "path": "trace.jsonl",
                    "format": "jsonl",
                    "step_count": 3,
                    "steps_complete": True,
                    "reward_steps": 0,
                    "reward_total": None,
                    "final_score": 1 if index else None,
                    "terminal_status": "unknown",
                }
            ],
            "traces_not_summarized": 0,
        }
        for index, pin in enumerate(traces)
    ]
    listing = {"total": 0, "truncated": False, "items": []}
    return {
        "schema_version": "synth.mnemosyne.actor-project-trace-compare.v1",
        "archive_count": len(traces),
        "archives": archives,
        "entries": {"common": listing, "changed": listing, "only_in": listing},
        "trace_steps": {
            "total": 1,
            "truncated": False,
            "items": [
                {
                    "path": "trace.jsonl",
                    "step_counts": [3] * len(traces),
                    "first_divergence_index": None,
                    "comparable": True,
                }
            ],
        },
    }


def pins(count):
    return ProjectTraceComparison(
        traces=tuple(
            PinnedProjectTrace(association_id=f"mna_{i:064x}", archive_sha256=f"{i:064x}")
            for i in range(1, count + 1)
        )
    )


def test_compare_bounds_are_two_to_eight_distinct():
    for count in (1, 9):
        with pytest.raises(ValidationError):
            pins(count)
    pin = PinnedProjectTrace(association_id=ASSOCIATION, archive_sha256=DIGEST)
    with pytest.raises(ValidationError):
        ProjectTraceComparison(traces=(pin, pin))
    assert len(pins(8).traces) == 8


def test_compare_is_capability_gated():
    session = FakeActorSession(tools=(READ_PROJECT_TRACE_TOOL,))
    with pytest.raises(CapabilityUnavailableError):
        compare_project_traces(session, pins(2))
    assert session.calls == []


def test_compare_returns_typed_result_bound_to_pins():
    session = FakeActorSession(tools=(COMPARE_PROJECT_TRACES_TOOL,))
    result = compare_project_traces(session, pins(3))
    assert [a.index for a in result.archives] == [0, 1, 2]
    assert result.archives[0].traces[0].final_score is None
    assert result.trace_steps.items[0].first_divergence_index is None
    assert session.calls[0][1] == pins(3).model_dump(mode="json")
    asynchronous = AsyncFake(tools=(COMPARE_PROJECT_TRACES_TOOL,))
    assert asyncio.run(acompare_project_traces(asynchronous, pins(2))).archive_count == 2


def test_compare_refuses_a_result_for_other_pins():
    class Swapped(FakeActorSession):
        def call_tool(self, name, arguments):
            reply = super().call_tool(name, arguments)
            result = dict(reply["structuredContent"])
            result["archives"] = list(reversed(result["archives"]))
            return _envelope(result)

    with pytest.raises(ContractError) as refused:
        compare_project_traces(Swapped(tools=(COMPARE_PROJECT_TRACES_TOOL,)), pins(2))
    assert refused.value.code == "actor_trace_compare_mismatch"


def test_tool_error_is_a_typed_refusal():
    class Refusing(FakeActorSession):
        def call_tool(self, name, arguments):
            return {"content": [{"type": "text", "text": "trace_compare_changed"}], "isError": True}

    with pytest.raises(ContractError) as refused:
        compare_project_traces(Refusing(tools=(COMPARE_PROJECT_TRACES_TOOL,)), pins(2))
    assert refused.value.code == "actor_trace_tool_error"
    assert "trace_compare_changed" in refused.value.cause
